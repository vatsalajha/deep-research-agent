"""Agent nodes for query analysis, search, synthesis, and report generation."""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from src.state import AgentState
from src.templates import TEMPLATES
from src.tools import WebSearchTool


def create_query_analyzer(api_key: str):
    """Create the query analysis node.

    This is the first node in the graph. It takes the user's research
    question and breaks it into 3-5 targeted search queries.

    Args:
        api_key: Anthropic API key.

    Returns:
        A node function that accepts and returns AgentState.
    """
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        temperature=0.7,
    )

    def query_analyzer(state: AgentState) -> dict:
        """Analyze the user query and produce search queries."""
        user_query = state["query"]

        prompt = f"""Analyze this research query and determine the best search strategy.

Query: {user_query}

Provide 3-5 specific search queries that would gather comprehensive information about this topic.
Consider: definitions, current trends, examples, data, expert opinions.

Respond ONLY with valid JSON (no markdown fencing):
{{
    "strategy": "brief explanation of your research approach",
    "search_queries": ["query1", "query2", "query3"]
}}"""

        response = llm.invoke([HumanMessage(content=prompt)])

        try:
            result = json.loads(response.content)
            search_queries = result["search_queries"]
            strategy = result["strategy"]
        except (json.JSONDecodeError, KeyError):
            search_queries = [user_query]
            strategy = "Direct search"

        return {
            "search_queries": search_queries,
            "messages": [
                AIMessage(
                    content=f"Strategy: {strategy}\nSearch queries: {', '.join(search_queries)}"
                )
            ],
        }

    return query_analyzer


def create_web_searcher(search_tool: WebSearchTool):
    """Create the web search execution node.

    Runs searches for any queries that haven't been searched yet,
    then merges the new results into the existing search_results dict.

    Args:
        search_tool: An initialized WebSearchTool instance.

    Returns:
        A node function that accepts and returns AgentState.
    """

    def web_searcher(state: AgentState) -> dict:
        """Execute web searches for new queries."""
        queries_to_search = state["search_queries"]
        existing_results = state.get("search_results", {})

        # Only search queries we haven't seen before
        new_queries = [q for q in queries_to_search if q not in existing_results]

        if new_queries:
            print(f"\nSearching for: {new_queries}")
            new_results = search_tool.batch_search(new_queries, max_results=5)
            existing_results = {**existing_results, **new_results}

        total_sources = sum(len(v) for v in existing_results.values())

        return {
            "search_results": existing_results,
            "messages": [
                AIMessage(content=f"Found {total_sources} relevant sources")
            ],
        }

    return web_searcher


def create_synthesizer(api_key: str):
    """Create the synthesis and gap-analysis node.

    After searches complete, this node asks Claude to evaluate whether
    the collected results are sufficient or if additional searches are
    needed. It controls the research loop.

    Args:
        api_key: Anthropic API key.

    Returns:
        A node function that accepts and returns AgentState.
    """
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        temperature=0.5,
    )

    def synthesizer(state: AgentState) -> dict:
        """Analyze search results and decide if more research is needed."""
        search_results = state["search_results"]
        original_query = state["query"]
        iterations = state["iterations"]
        max_iterations = state["max_iterations"]

        # Format results for the LLM
        formatted_results = ""
        for query, results in search_results.items():
            formatted_results += f"\n## Results for '{query}':\n"
            for i, result in enumerate(results, 1):
                formatted_results += (
                    f"{i}. {result['title']}\n"
                    f"   {result['content'][:200]}...\n"
                )

        analysis_prompt = f"""Review these research results for the query: "{original_query}"

{formatted_results}

Analyze:
1. Is there sufficient information to write a comprehensive report?
2. What important aspects are still missing?
3. What additional searches would help?

Respond ONLY with valid JSON (no markdown fencing):
{{
    "sufficiency": "sufficient" or "needs_more_research",
    "coverage_summary": "what topics are covered so far",
    "gaps": ["gap1", "gap2"],
    "additional_queries": ["query1", "query2"]
}}"""

        response = llm.invoke([HumanMessage(content=analysis_prompt)])

        try:
            analysis = json.loads(response.content)
        except (json.JSONDecodeError, KeyError):
            analysis = {
                "sufficiency": "sufficient",
                "coverage_summary": "Analysis complete",
                "gaps": [],
                "additional_queries": [],
            }

        research_complete = (
            analysis["sufficiency"] == "sufficient"
            or iterations >= max_iterations - 1
        )

        update: dict = {
            "iterations": iterations + 1,
            "research_complete": research_complete,
            "messages": [
                AIMessage(
                    content=f"Synthesis (iteration {iterations + 1}): "
                    f"{analysis.get('coverage_summary', 'In progress')}"
                )
            ],
        }

        # Queue additional searches if research isn't done yet
        if not research_complete and analysis.get("additional_queries"):
            update["search_queries"] = (
                state["search_queries"] + analysis["additional_queries"]
            )

        return update

    return synthesizer


def create_report_generator(api_key: str):
    """Create the final report generation node.

    This is the terminal node. It takes all accumulated search results,
    compiles a numbered source list, and asks Claude to produce a
    structured, citation-rich research report.

    Args:
        api_key: Anthropic API key.

    Returns:
        A node function that accepts and returns AgentState.
    """
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        temperature=0.7,
    )

    def report_generator(state: AgentState) -> dict:
        """Generate a comprehensive, cited report from all research."""
        original_query = state["query"]
        search_results = state["search_results"]

        # Build numbered source list
        sources: list[dict] = []
        for query, results in search_results.items():
            for result in results:
                sources.append(
                    {
                        "id": len(sources) + 1,
                        "title": result["title"],
                        "url": result["url"],
                        "search_query": query,
                    }
                )

        sources_text = "\n".join(
            f"[{s['id']}] {s['title']} ({s['url']})" for s in sources
        )

        # Format full research content
        research_text = ""
        for query, results in search_results.items():
            research_text += f"\n## Search: {query}\n"
            for result in results:
                research_text += f"\n{result['title']}\n{result['content']}\n"

        # Select report template based on style
        style = state.get("report_style", "detailed")
        template_instructions = TEMPLATES.get(style, TEMPLATES["detailed"])

        report_prompt = f"""You are an expert research analyst. Based on the following research, write a report answering the query: "{original_query}"

RESEARCH DATA:
{research_text}

AVAILABLE SOURCES FOR CITATIONS:
{sources_text}

{template_instructions}"""

        response = llm.invoke([HumanMessage(content=report_prompt)])
        report = response.content

        # Append full source list
        report += "\n\n## Sources\n"
        for source in sources:
            report += f"[{source['id']}] {source['title']}\n{source['url']}\n\n"

        return {
            "final_report": report,
            "sources": sources,
            "messages": [AIMessage(content=report)],
        }

    return report_generator

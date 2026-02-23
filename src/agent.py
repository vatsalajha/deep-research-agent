"""Main agent orchestration using LangGraph."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.nodes import (
    create_query_analyzer,
    create_report_generator,
    create_synthesizer,
    create_web_searcher,
)
from src.state import AgentState
from src.templates import VALID_STYLES
from src.tools import WebSearchTool


def create_llm(
    groq_api_key: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.7,
) -> BaseChatModel:
    """Create a LangChain chat model.

    Args:
        groq_api_key: Groq API key.
        model: Model identifier (default: llama-3.3-70b-versatile).
        temperature: Sampling temperature.

    Returns:
        A LangChain chat model instance.
    """
    return ChatGroq(
        model=model,
        api_key=groq_api_key,
        temperature=temperature,
    )


class DeepResearchAgent:
    """Orchestrates the full research pipeline via a LangGraph state graph.

    Flow:
        analyze_query -> search -> synthesize -+-> generate_report -> END
                           ^                   |
                           +-------------------+  (needs more research)

    Args:
        groq_api_key: Groq API key for LLM inference.
        tavily_key: Tavily web-search API key.
        max_iterations: Maximum search-synthesize cycles (default 3).
        model: LLM model identifier (default: llama-3.3-70b-versatile).
    """

    def __init__(
        self,
        groq_api_key: str,
        tavily_key: str,
        max_iterations: int = 3,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.max_iterations = max_iterations
        self.search_tool = WebSearchTool(tavily_key)
        self.llm = create_llm(groq_api_key, model=model)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph execution graph."""
        # Create node functions — all share the same LLM instance
        query_analyzer = create_query_analyzer(self.llm)
        web_searcher = create_web_searcher(self.search_tool)
        synthesizer = create_synthesizer(self.llm)
        report_generator = create_report_generator(self.llm)

        # Assemble graph
        graph = StateGraph(AgentState)

        graph.add_node("analyze_query", query_analyzer)
        graph.add_node("search", web_searcher)
        graph.add_node("synthesize", synthesizer)
        graph.add_node("generate_report", report_generator)

        # Edges
        graph.set_entry_point("analyze_query")
        graph.add_edge("analyze_query", "search")
        graph.add_edge("search", "synthesize")

        def should_continue(state: AgentState) -> str:
            """Route to report generation or another search loop."""
            if state.get("research_complete", False):
                return "generate_report"
            return "search"

        graph.add_conditional_edges(
            "synthesize",
            should_continue,
            {
                "generate_report": "generate_report",
                "search": "search",
            },
        )

        graph.add_edge("generate_report", END)

        return graph.compile()

    def run(self, query: str, report_style: str = "detailed") -> dict:
        """Execute the research agent on a query.

        Args:
            query: The user's research question.
            report_style: Report format — "detailed", "summary", or "academic".

        Returns:
            The final agent state containing the report, sources, and messages.
        """
        if report_style not in VALID_STYLES:
            raise ValueError(
                f"Unknown report_style '{report_style}'. "
                f"Choose from: {VALID_STYLES}"
            )

        initial_state: AgentState = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "search_queries": [],
            "search_results": {},
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "sources": [],
            "report_style": report_style,
        }

        print(f"\nStarting research on: {query}\n")
        print("=" * 70)

        final_state = None
        for step in self.graph.stream(initial_state):
            node_name = list(step.keys())[0]
            print(f"  Completed: {node_name}")
            final_state = step[node_name]

        print("=" * 70)
        print("Research complete.\n")

        return final_state

    def get_report(self, state: dict) -> str:
        """Extract the final report from the agent state."""
        return state.get("final_report", "No report generated.")

    def save_report(self, report: str, filepath: str = "research_report.md") -> None:
        """Save a report to a markdown file.

        Creates parent directories if they don't exist.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(report)
        print(f"Report saved to {filepath}")

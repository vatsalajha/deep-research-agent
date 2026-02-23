"""Main agent orchestration using LangGraph."""

import os
import warnings

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama
# from langchain_xai import ChatXAI
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


# ── Provider configuration ────────────────────────────────────────────────
# First model in each list is the default for that provider.

PROVIDER_MODELS: dict[str, list[str]] = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "anthropic": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
    "google": ["gemini-2.0-flash", "gemini-2.5-pro-preview-06-05"],
    # "ollama": ["llama3", "mistral", "phi3"],
    # "deepseek": ["deepseek-chat", "deepseek-coder"],
    # "xai": ["grok-2-latest"],
    # "perplexity": ["llama-3.1-sonar-large-128k-online"],
}

PROVIDER_ENV_KEYS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    # "deepseek": "DEEPSEEK_API_KEY",
    # "xai": "XAI_API_KEY",
    # "perplexity": "PERPLEXITY_API_KEY",
    # "ollama": None,
}


def create_llm(
    provider: str = "groq",
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
) -> BaseChatModel:
    """Create a LangChain chat model for the given provider.

    Args:
        provider: LLM provider name (groq, openai, anthropic, google).
        api_key: Provider API key.  Falls back to the corresponding env var.
        model: Model identifier.  Defaults to the first model in PROVIDER_MODELS.
        temperature: Sampling temperature.

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If the provider is unknown or no API key is available.
    """
    provider = provider.lower()

    if provider not in PROVIDER_MODELS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {list(PROVIDER_MODELS.keys())}"
        )

    # Resolve API key: explicit > env var
    if not api_key:
        env_var = PROVIDER_ENV_KEYS.get(provider)
        if env_var:
            api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"No API key for provider '{provider}'. "
            f"Pass api_key= or set {PROVIDER_ENV_KEYS.get(provider, '???')}."
        )

    # Default model for provider
    if not model:
        model = PROVIDER_MODELS[provider][0]

    if provider == "groq":
        return ChatGroq(model=model, api_key=api_key, temperature=temperature)

    if provider == "openai":
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)

    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=api_key, temperature=temperature)

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=temperature,
        )

    # ── Commented-out providers (uncomment + install to enable) ─────────
    # if provider == "deepseek":
    #     return ChatOpenAI(
    #         model=model, api_key=api_key, temperature=temperature,
    #         base_url="https://api.deepseek.com/v1",
    #     )
    #
    # if provider == "perplexity":
    #     return ChatOpenAI(
    #         model=model, api_key=api_key, temperature=temperature,
    #         base_url="https://api.perplexity.ai",
    #     )
    #
    # if provider == "ollama":
    #     return ChatOllama(model=model, temperature=temperature)
    #
    # if provider == "xai":
    #     return ChatXAI(model=model, api_key=api_key, temperature=temperature)

    raise ValueError(f"Provider '{provider}' matched models but has no constructor.")


class DeepResearchAgent:
    """Orchestrates the full research pipeline via a LangGraph state graph.

    Flow:
        analyze_query -> search -> synthesize -+-> generate_report -> END
                           ^                   |
                           +-------------------+  (needs more research)

    Args:
        llm_provider: LLM provider name (groq, openai, anthropic, google).
        llm_api_key: API key for the chosen LLM provider.
        llm_model: Model identifier (defaults to provider's first model).
        search_api_key: Tavily API key.
        max_iterations: Maximum search-synthesize cycles (default 3).
        max_results: Number of search results per query (default 3).
        system_prompt: Optional system prompt prepended to LLM calls.
        temperature: Sampling temperature (default 0.7).

    Deprecated kwargs (still work for backward compatibility):
        groq_api_key: Use llm_api_key + llm_provider="groq" instead.
        tavily_key: Use search_api_key instead.
        model: Use llm_model instead.
    """

    def __init__(
        self,
        llm_provider: str = "groq",
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        search_api_key: str | None = None,
        max_iterations: int = 3,
        max_results: int = 3,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        # ── Deprecated kwargs for backward compatibility ──
        groq_api_key: str | None = None,
        tavily_key: str | None = None,
        model: str | None = None,
    ) -> None:
        # ── Map deprecated kwargs to new params ──
        if groq_api_key is not None:
            warnings.warn(
                "groq_api_key is deprecated. Use llm_provider='groq' and llm_api_key=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if llm_api_key is None:
                llm_api_key = groq_api_key
                llm_provider = "groq"

        if tavily_key is not None:
            warnings.warn(
                "tavily_key is deprecated. Use search_api_key=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if search_api_key is None:
                search_api_key = tavily_key

        if model is not None:
            warnings.warn(
                "model is deprecated. Use llm_model=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if llm_model is None:
                llm_model = model

        self.llm_provider = llm_provider
        self.max_iterations = max_iterations
        self.max_results = max_results
        self.system_prompt = system_prompt
        self.search_tool = WebSearchTool(search_api_key)
        self.llm = create_llm(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
            temperature=temperature,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph execution graph."""
        # Create node functions — all share the same LLM instance
        query_analyzer = create_query_analyzer(self.llm, system_prompt=self.system_prompt)
        web_searcher = create_web_searcher(self.search_tool, max_results=self.max_results)
        synthesizer = create_synthesizer(self.llm, system_prompt=self.system_prompt)
        report_generator = create_report_generator(self.llm, system_prompt=self.system_prompt)

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

        final_state = self.graph.invoke(initial_state)

        print("=" * 70)
        print("Research complete.\n")

        return final_state

    def stream_run(self, query: str, report_style: str = "detailed"):
        """Execute the research agent, yielding structured progress events.

        Yields dicts with keys:
            event: "start", "node_complete", "done", or "error"
            node:  node name (for node_complete events)
            data:  node output state delta
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

        yield {"event": "start", "node": None, "data": initial_state}

        final_state = None
        try:
            for step in self.graph.stream(initial_state):
                node_name = list(step.keys())[0]
                node_output = step[node_name]
                final_state = node_output
                yield {
                    "event": "node_complete",
                    "node": node_name,
                    "data": node_output,
                }
        except Exception as e:
            yield {"event": "error", "node": None, "data": {"error": str(e)}}
            return

        yield {"event": "done", "node": None, "data": final_state or {}}

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

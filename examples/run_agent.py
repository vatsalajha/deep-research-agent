"""Example script to run the Deep Research Agent.

Usage:
    source .venv/bin/activate
    python examples/run_agent.py
"""

import re
import sys

from src.agent import DeepResearchAgent, PROVIDER_MODELS, PROVIDER_ENV_KEYS
from src.utils import load_environment


# Example queries covering different research types
QUERIES = [
    "What are the latest developments in quantum computing in 2024-2025?",
    "How is AI being used to combat climate change?",
    "What are the key differences between LangChain and LangGraph for building AI agents?",
]


def make_safe_filename(query: str) -> str:
    """Turn a query string into a filesystem-safe filename."""
    slug = re.sub(r"[^a-z0-9]+", "_", query[:40].lower()).strip("_")
    return f"reports/{slug}.md"


def main() -> None:
    """Run the agent on each example query and save reports."""
    env = load_environment()

    # Auto-detect the first available LLM provider
    provider = None
    api_key = None
    for p in PROVIDER_MODELS:
        key_name = f"{p}_api_key"
        if env.get(key_name):
            provider = p
            api_key = env[key_name]
            break

    if not provider:
        print("Error: No LLM provider API key found. Set at least one in .env.")
        sys.exit(1)

    tavily_key = env.get("tavily_api_key")
    if not tavily_key:
        print("Error: No Tavily API key found. Set TAVILY_API_KEY in .env.")
        sys.exit(1)

    print(f"Using LLM provider: {provider} ({PROVIDER_MODELS[provider][0]})")
    print(f"Using search provider: Tavily\n")

    agent = DeepResearchAgent(
        llm_provider=provider,
        llm_api_key=api_key,
        search_api_key=tavily_key,
        max_iterations=3,
    )

    # If a query is passed as a CLI argument, use that instead
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else QUERIES

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")
        print(f"{'=' * 70}\n")

        final_state = agent.run(query)
        report = agent.get_report(final_state)

        print("\nREPORT:\n")
        print(report)

        filepath = make_safe_filename(query)
        agent.save_report(report, filepath)


if __name__ == "__main__":
    main()

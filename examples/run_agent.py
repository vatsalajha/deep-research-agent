"""Example script to run the Deep Research Agent.

Usage:
    source .venv/bin/activate
    python examples/run_agent.py
"""

import re
import sys

from src.agent import DeepResearchAgent
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

    agent = DeepResearchAgent(
        anthropic_key=env["anthropic_api_key"],
        tavily_key=env["tavily_api_key"],
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

import os
from dotenv import load_dotenv


def load_environment() -> dict[str, str]:
    """Load environment variables from .env file.

    Returns:
        Dictionary with anthropic_api_key and tavily_api_key.

    Raises:
        ValueError: If required API keys are missing.
    """
    load_dotenv()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY not found in environment")

    return {
        "anthropic_api_key": anthropic_key,
        "tavily_api_key": tavily_key,
    }


def format_search_results(results: list[dict]) -> str:
    """Format search results into readable text with source numbers."""
    formatted = ""
    for i, result in enumerate(results, 1):
        formatted += f"\n[Source {i}]\n"
        formatted += f"Title: {result.get('title', 'N/A')}\n"
        formatted += f"URL: {result.get('url', 'N/A')}\n"
        formatted += f"Content: {result.get('content', 'N/A')}\n"
    return formatted

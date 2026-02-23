import os
from dotenv import load_dotenv


# Maps provider name -> environment variable holding its API key
PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    # "deepseek": "DEEPSEEK_API_KEY",
    # "xai": "XAI_API_KEY",
    # "perplexity": "PERPLEXITY_API_KEY",
    # "ollama": None,  # Ollama runs locally, no key needed
}

SEARCH_ENV_KEYS = {
    "tavily": "TAVILY_API_KEY",
    # "exa": "EXA_API_KEY",
    # "serpapi": "SERPAPI_API_KEY",
}


def load_environment() -> dict[str, str | None]:
    """Load environment variables from .env file.

    Returns all provider API keys found in the environment.
    Missing keys are returned as None.

    Returns:
        Dictionary with keys like ``groq_api_key``, ``openai_api_key``, etc.
        Only keys that are present in the environment have non-None values.

    Raises:
        ValueError: If no LLM provider key *and* no search provider key is found.
    """
    load_dotenv()

    keys: dict[str, str | None] = {}

    # LLM provider keys
    for provider, env_var in PROVIDER_ENV_KEYS.items():
        if env_var is None:
            continue
        keys[f"{provider}_api_key"] = os.getenv(env_var) or None

    # Search provider keys
    for provider, env_var in SEARCH_ENV_KEYS.items():
        keys[f"{provider}_api_key"] = os.getenv(env_var) or None

    has_llm_key = any(
        keys.get(f"{p}_api_key") for p in PROVIDER_ENV_KEYS
    )
    has_search_key = any(
        keys.get(f"{p}_api_key") for p in SEARCH_ENV_KEYS
    )

    if not has_llm_key and not has_search_key:
        raise ValueError(
            "No API keys found. Set at least one LLM provider key "
            "(GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY) "
            "and one search key (TAVILY_API_KEY) in your .env file."
        )

    if not has_llm_key:
        raise ValueError(
            "No LLM provider key found. Set at least one of: "
            "GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY."
        )

    if not has_search_key:
        raise ValueError(
            "No search provider key found. Set TAVILY_API_KEY in your .env file."
        )

    return keys


def format_search_results(results: list[dict]) -> str:
    """Format search results into readable text with source numbers."""
    formatted = ""
    for i, result in enumerate(results, 1):
        formatted += f"\n[Source {i}]\n"
        formatted += f"Title: {result.get('title', 'N/A')}\n"
        formatted += f"URL: {result.get('url', 'N/A')}\n"
        formatted += f"Content: {result.get('content', 'N/A')}\n"
    return formatted

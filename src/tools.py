"""Web search tool wrapper for the Deep Research Agent."""

from tavily import TavilyClient


class WebSearchTool:
    """Wrapper around the Tavily web search API.

    Provides single and batch search methods with error handling
    and consistent result formatting.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize with a Tavily API key.

        Args:
            api_key: Tavily API key from https://tavily.com
        """
        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> list[dict]:
        """Execute a single web search.

        Args:
            query: Search query string.
            max_results: Number of results to return.
            search_depth: "basic" or "advanced".

        Returns:
            List of result dicts with keys: title, url, content, raw_content.
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                include_raw_content=True,
            )

            results = []
            for item in response.get("results", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                        "raw_content": item.get("raw_content", ""),
                    }
                )
            return results

        except Exception as e:
            print(f"Search error for '{query}': {e}")
            return []

    def batch_search(
        self, queries: list[str], max_results: int = 3
    ) -> dict[str, list[dict]]:
        """Execute multiple searches and collect results.

        Args:
            queries: List of search query strings.
            max_results: Number of results per query.

        Returns:
            Mapping of query string to its list of results.
        """
        results: dict[str, list[dict]] = {}
        for query in queries:
            results[query] = self.search(query, max_results)
        return results


# ── Alternative search providers (uncomment + install to enable) ──────────
#
# class ExaSearchTool:
#     """Wrapper around the Exa search API.
#
#     pip install exa-py
#     Requires EXA_API_KEY environment variable.
#     """
#
#     def __init__(self, api_key: str) -> None:
#         from exa_py import Exa
#         self.client = Exa(api_key=api_key)
#
#     def search(self, query: str, max_results: int = 5, **kwargs) -> list[dict]:
#         try:
#             response = self.client.search_and_contents(
#                 query, num_results=max_results, text=True,
#             )
#             return [
#                 {
#                     "title": r.title or "",
#                     "url": r.url or "",
#                     "content": r.text or "",
#                     "raw_content": r.text or "",
#                 }
#                 for r in response.results
#             ]
#         except Exception as e:
#             print(f"Exa search error for '{query}': {e}")
#             return []
#
#     def batch_search(self, queries: list[str], max_results: int = 3) -> dict[str, list[dict]]:
#         results: dict[str, list[dict]] = {}
#         for query in queries:
#             results[query] = self.search(query, max_results)
#         return results
#
#
# class SerpAPISearchTool:
#     """Wrapper around the SerpAPI Google Search API.
#
#     pip install google-search-results
#     Requires SERPAPI_API_KEY environment variable.
#     """
#
#     def __init__(self, api_key: str) -> None:
#         self.api_key = api_key
#
#     def search(self, query: str, max_results: int = 5, **kwargs) -> list[dict]:
#         try:
#             from serpapi import GoogleSearch
#             params = {
#                 "q": query,
#                 "num": max_results,
#                 "api_key": self.api_key,
#             }
#             search = GoogleSearch(params)
#             data = search.get_dict()
#             return [
#                 {
#                     "title": r.get("title", ""),
#                     "url": r.get("link", ""),
#                     "content": r.get("snippet", ""),
#                     "raw_content": r.get("snippet", ""),
#                 }
#                 for r in data.get("organic_results", [])[:max_results]
#             ]
#         except Exception as e:
#             print(f"SerpAPI search error for '{query}': {e}")
#             return []
#
#     def batch_search(self, queries: list[str], max_results: int = 3) -> dict[str, list[dict]]:
#         results: dict[str, list[dict]] = {}
#         for query in queries:
#             results[query] = self.search(query, max_results)
#         return results


def create_search_tool(provider: str = "tavily", api_key: str | None = None):
    """Factory to create a search tool for the given provider.

    Args:
        provider: Search provider name (currently only "tavily" is active).
        api_key: API key for the search provider.

    Returns:
        A search tool instance with .search() and .batch_search() methods.

    Raises:
        ValueError: If the provider is unknown or unsupported.
    """
    provider = provider.lower()

    if provider == "tavily":
        return WebSearchTool(api_key)

    # Uncomment the following blocks when the corresponding provider is enabled:
    # if provider == "exa":
    #     return ExaSearchTool(api_key)
    #
    # if provider == "serpapi":
    #     return SerpAPISearchTool(api_key)

    raise ValueError(
        f"Unknown search provider '{provider}'. Supported: tavily"
    )

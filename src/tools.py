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

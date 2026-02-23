"""Validation tests for the Deep Research Agent."""


def test_imports():
    """Test that all project modules import successfully."""
    print("Testing imports...")
    modules = {
        "src.state (AgentState)": lambda: __import__("src.state", fromlist=["AgentState"]),
        "src.utils (load_environment)": lambda: __import__("src.utils", fromlist=["load_environment"]),
        "src.tools (WebSearchTool)": lambda: __import__("src.tools", fromlist=["WebSearchTool"]),
        "src.nodes (all nodes)": lambda: __import__(
            "src.nodes",
            fromlist=[
                "create_query_analyzer",
                "create_web_searcher",
                "create_synthesizer",
                "create_report_generator",
            ],
        ),
        "src.agent (DeepResearchAgent)": lambda: __import__("src.agent", fromlist=["DeepResearchAgent"]),
    }

    all_ok = True
    for name, importer in modules.items():
        try:
            importer()
            print(f"  \u2705 {name}")
        except Exception as e:
            print(f"  \u274c {name} — {e}")
            all_ok = False

    print()
    return all_ok


def test_environment():
    """Test that API keys load from .env."""
    print("Testing environment loading...")
    from src.utils import load_environment

    try:
        env = load_environment()
        masked_anthropic = "***" + env["anthropic_api_key"][-4:]
        masked_tavily = "***" + env["tavily_api_key"][-4:]
        print(f"  \u2705 Environment loaded")
        print(f"     Anthropic key: {masked_anthropic}")
        print(f"     Tavily key:    {masked_tavily}")
        print()
        return True
    except ValueError as e:
        print(f"  \u274c {e}")
        print()
        return False


def test_tools():
    """Test that web search returns results."""
    print("Testing web search tool...")
    from src.tools import WebSearchTool
    from src.utils import load_environment

    try:
        env = load_environment()
        tool = WebSearchTool(env["tavily_api_key"])
        results = tool.search("Python programming", max_results=2)

        if results:
            print(f"  \u2705 Search returned {len(results)} results")
            for r in results:
                print(f"     - {r['title'][:60]}")
        else:
            print("  \u26a0\ufe0f  Search returned no results (API may be rate-limited)")

        print()
        return True
    except Exception as e:
        print(f"  \u274c Search failed — {e}")
        print()
        return False


def test_agent_init():
    """Test that the agent initializes and the graph compiles."""
    print("Testing agent initialization...")
    from src.agent import DeepResearchAgent
    from src.utils import load_environment

    try:
        env = load_environment()
        agent = DeepResearchAgent(
            anthropic_key=env["anthropic_api_key"],
            tavily_key=env["tavily_api_key"],
            max_iterations=2,
        )
        print(f"  \u2705 DeepResearchAgent created")
        print(f"  \u2705 Graph compiled ({len(agent.graph.nodes)} nodes)")
        print()
        return True
    except Exception as e:
        print(f"  \u274c Agent init failed — {e}")
        print()
        return False


def test_templates():
    """Test that report templates load and agent validates styles."""
    print("Testing report templates...")
    from src.templates import TEMPLATES, VALID_STYLES

    all_ok = True
    for style in ["detailed", "summary", "academic"]:
        if style in TEMPLATES and style in VALID_STYLES:
            print(f"  \u2705 '{style}' template available")
        else:
            print(f"  \u274c '{style}' template missing")
            all_ok = False

    # Verify invalid style is rejected
    from src.agent import DeepResearchAgent
    from src.utils import load_environment

    env = load_environment()
    agent = DeepResearchAgent(
        anthropic_key=env["anthropic_api_key"],
        tavily_key=env["tavily_api_key"],
    )
    try:
        agent.run("test", report_style="nonexistent")
        print("  \u274c Invalid style was not rejected")
        all_ok = False
    except ValueError:
        print("  \u2705 Invalid style correctly rejected")

    print()
    return all_ok


def main():
    print("=" * 70)
    print("DEEP RESEARCH AGENT — VALIDATION TESTS")
    print("=" * 70 + "\n")

    results = [
        test_imports(),
        test_environment(),
        test_tools(),
        test_agent_init(),
        test_templates(),
    ]

    print("=" * 70)
    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"\u2705 All {total} tests passed!")
    else:
        print(f"\u26a0\ufe0f  {passed}/{total} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()

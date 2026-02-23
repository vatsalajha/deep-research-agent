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
        "src.agent (DeepResearchAgent)": lambda: __import__(
            "src.agent",
            fromlist=["DeepResearchAgent", "PROVIDER_MODELS", "PROVIDER_ENV_KEYS"],
        ),
    }

    all_ok = True
    for name, importer in modules.items():
        try:
            importer()
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name} -- {e}")
            all_ok = False

    print()
    return all_ok


def test_environment():
    """Test that API keys load from .env."""
    print("Testing environment loading...")
    from src.utils import load_environment

    try:
        env = load_environment()
        found = []
        for key_name, value in sorted(env.items()):
            if value:
                masked = "***" + value[-4:]
                print(f"  [OK] {key_name}: {masked}")
                found.append(key_name)
            else:
                print(f"  [--] {key_name}: not set")

        if not found:
            print("  [FAIL] No API keys found at all")
            print()
            return False

        print(f"\n  Found {len(found)} key(s)")
        print()
        return True
    except ValueError as e:
        print(f"  [FAIL] {e}")
        print()
        return False


def test_tools():
    """Test that web search returns results."""
    print("Testing web search tool...")
    from src.tools import WebSearchTool
    from src.utils import load_environment

    try:
        env = load_environment()
        tavily_key = env.get("tavily_api_key")
        if not tavily_key:
            print("  [SKIP] No Tavily API key found")
            print()
            return True

        tool = WebSearchTool(tavily_key)
        results = tool.search("Python programming", max_results=2)

        if results:
            print(f"  [OK] Search returned {len(results)} results")
            for r in results:
                print(f"     - {r['title'][:60]}")
        else:
            print("  [WARN] Search returned no results (API may be rate-limited)")

        print()
        return True
    except Exception as e:
        print(f"  [FAIL] Search failed -- {e}")
        print()
        return False


def test_agent_init():
    """Test that the agent initializes and the graph compiles."""
    print("Testing agent initialization...")
    from src.agent import DeepResearchAgent, PROVIDER_MODELS, PROVIDER_ENV_KEYS
    from src.utils import load_environment

    try:
        env = load_environment()

        # Auto-detect first available LLM provider
        provider = None
        api_key = None
        for p in PROVIDER_MODELS:
            key_name = f"{p}_api_key"
            if env.get(key_name):
                provider = p
                api_key = env[key_name]
                break

        if not provider:
            print("  [FAIL] No LLM provider key found")
            print()
            return False

        tavily_key = env.get("tavily_api_key")
        if not tavily_key:
            print("  [FAIL] No Tavily API key found")
            print()
            return False

        print(f"  Using provider: {provider}")

        agent = DeepResearchAgent(
            llm_provider=provider,
            llm_api_key=api_key,
            search_api_key=tavily_key,
            max_iterations=2,
        )
        print(f"  [OK] DeepResearchAgent created")
        print(f"  [OK] Graph compiled ({len(agent.graph.nodes)} nodes)")

        # Also verify backward-compat constructor works
        if env.get("groq_api_key"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                compat_agent = DeepResearchAgent(
                    groq_api_key=env["groq_api_key"],
                    tavily_key=tavily_key,
                )
            print(f"  [OK] Backward-compatible constructor works")

        print()
        return True
    except Exception as e:
        print(f"  [FAIL] Agent init failed -- {e}")
        print()
        return False


def test_templates():
    """Test that report templates load and agent validates styles."""
    print("Testing report templates...")
    from src.templates import TEMPLATES, VALID_STYLES

    all_ok = True
    for style in ["detailed", "summary", "academic"]:
        if style in TEMPLATES and style in VALID_STYLES:
            print(f"  [OK] '{style}' template available")
        else:
            print(f"  [FAIL] '{style}' template missing")
            all_ok = False

    # Verify invalid style is rejected
    from src.agent import DeepResearchAgent, PROVIDER_MODELS
    from src.utils import load_environment

    env = load_environment()

    # Auto-detect provider for this test
    provider = None
    api_key = None
    for p in PROVIDER_MODELS:
        key_name = f"{p}_api_key"
        if env.get(key_name):
            provider = p
            api_key = env[key_name]
            break

    tavily_key = env.get("tavily_api_key")
    if not provider or not tavily_key:
        print("  [SKIP] No provider/search key for style rejection test")
        print()
        return all_ok

    agent = DeepResearchAgent(
        llm_provider=provider,
        llm_api_key=api_key,
        search_api_key=tavily_key,
    )
    try:
        agent.run("test", report_style="nonexistent")
        print("  [FAIL] Invalid style was not rejected")
        all_ok = False
    except ValueError:
        print("  [OK] Invalid style correctly rejected")

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
        print(f"All {total} tests passed!")
    else:
        print(f"{passed}/{total} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Streamlit UI for the Deep Research Agent."""

import time
from contextlib import redirect_stdout
from io import StringIO

import streamlit as st

from src.agent import DeepResearchAgent, PROVIDER_MODELS, PROVIDER_ENV_KEYS
from src.templates import VALID_STYLES
from src.utils import load_environment

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="DRA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sharp / angular theme — no rounded corners
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Kill all border-radius globally */
    *, *::before, *::after {
        border-radius: 0 !important;
    }

    /* Buttons: flat, sharp edges */
    .stButton > button {
        border-radius: 0 !important;
        border: 1px solid #444;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Inputs, text areas, selects */
    .stTextArea textarea,
    .stTextInput input,
    .stSelectbox > div > div,
    .stSlider > div {
        border-radius: 0 !important;
    }

    /* Tabs: sharp underline style */
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 0 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        border-radius: 0 !important;
    }

    /* Status containers, alerts, info boxes */
    .stAlert, .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #333;
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 0 !important;
        border: 1px solid #444;
    }

    /* Code blocks */
    .stCodeBlock, code {
        border-radius: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "analyze_query": "Analyzing Query",
    "search": "Searching the Web",
    "synthesize": "Synthesizing Results",
    "generate_report": "Generating Report",
}

NODE_DESCRIPTIONS = {
    "analyze_query": "Breaking down your question into targeted search queries...",
    "search": "Executing web searches via Tavily...",
    "synthesize": "Evaluating research coverage and identifying gaps...",
    "generate_report": "Writing the final cited report...",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "env_loaded": False,
        "env_keys": {},
        "progress_log": [],
        "final_report": None,
        "sources": [],
        "research_running": False,
        "test_results": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests():
    """Import and run each test function, capturing stdout."""
    from test_agent import (
        test_agent_init,
        test_environment,
        test_imports,
        test_templates,
        test_tools,
    )

    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Web Search Tool", test_tools),
        ("Agent Init", test_agent_init),
        ("Templates", test_templates),
    ]

    results = []
    for name, fn in tests:
        buf = StringIO()
        try:
            with redirect_stdout(buf):
                passed = fn()
            results.append({"name": name, "passed": bool(passed), "output": buf.getvalue()})
        except Exception as e:
            results.append({"name": name, "passed": False, "output": f"Exception: {e}"})

    st.session_state.test_results = results


def render_test_results():
    """Display test results in the sidebar."""
    results = st.session_state.test_results
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    if passed == total:
        st.success(f"All {total} tests passed")
    else:
        st.warning(f"{passed}/{total} tests passed")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        with st.expander(f"[{status}] {r['name']}"):
            st.code(r["output"], language="text")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _available_providers(env_keys: dict) -> list[str]:
    """Return providers that have an API key loaded."""
    available = []
    for provider in PROVIDER_MODELS:
        key_name = f"{provider}_api_key"
        if env_keys.get(key_name):
            available.append(provider)
    return available


def render_sidebar():
    with st.sidebar:
        st.title("Configuration")

        # --- API Keys ---
        st.subheader("API Keys")
        if not st.session_state.env_loaded:
            if st.button("Load API Keys from .env"):
                try:
                    env = load_environment()
                    st.session_state.env_keys = env
                    st.session_state.env_loaded = True
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        else:
            st.success("API keys loaded")
            env = st.session_state.env_keys
            for provider in list(PROVIDER_ENV_KEYS.keys()) + ["tavily"]:
                key_name = f"{provider}_api_key"
                val = env.get(key_name)
                if val:
                    masked = "***" + val[-4:]
                    st.text(f"{provider.capitalize():10s} {masked}")

        st.divider()

        # --- LLM Provider ---
        st.subheader("LLM Provider")
        available = _available_providers(st.session_state.env_keys) if st.session_state.env_loaded else []

        if available:
            llm_provider = st.selectbox("Provider", available, index=0)
        else:
            llm_provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()), index=0, disabled=True)

        # --- Model (dynamic based on provider) ---
        models_for_provider = PROVIDER_MODELS.get(llm_provider, [])
        llm_model = st.selectbox("Model", models_for_provider, index=0) if models_for_provider else None

        # --- Temperature ---
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)

        st.divider()

        # --- Search API ---
        st.subheader("Search API")
        search_api = st.selectbox("Search provider", ["Tavily"], index=0)

        # --- Results per query ---
        max_results = st.slider("Results per query", 1, 10, 3)

        st.divider()

        # --- Agent Settings ---
        st.subheader("Agent Settings")
        max_iterations = st.slider("Max iterations", 1, 5, 2)
        report_style = st.selectbox("Report style", VALID_STYLES, index=0)

        # --- System Prompt ---
        with st.expander("System Prompt"):
            system_prompt = st.text_area(
                "Custom system prompt (optional)",
                value="",
                height=100,
                placeholder="e.g. You are a senior analyst specializing in...",
            )

        st.divider()

        # --- Test Runner ---
        st.subheader("Validation Tests")
        if st.button("Run Tests", disabled=st.session_state.research_running):
            with st.spinner("Running tests..."):
                run_tests()
            st.rerun()

        if st.session_state.test_results is not None:
            render_test_results()

    return {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "temperature": temperature,
        "max_results": max_results,
        "max_iterations": max_iterations,
        "report_style": report_style,
        "system_prompt": system_prompt.strip() if system_prompt else None,
    }


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def build_node_detail(node: str, data: dict) -> str:
    """Extract a human-readable detail string from a node's output."""
    if node == "analyze_query":
        queries = data.get("search_queries", [])
        return f"Generated {len(queries)} search queries: {', '.join(queries)}"

    if node == "search":
        results = data.get("search_results", {})
        total = sum(len(v) for v in results.values())
        return f"Found {total} sources across {len(results)} queries"

    if node == "synthesize":
        iteration = data.get("iterations", "?")
        complete = data.get("research_complete", False)
        if complete:
            return f"Iteration {iteration}: Research sufficient — proceeding to report"
        return f"Iteration {iteration}: Needs more research — queuing additional searches"

    if node == "generate_report":
        sources = data.get("sources", [])
        report_len = len(data.get("final_report", ""))
        return f"Report generated — {report_len:,} chars, {len(sources)} cited sources"

    return "Completed"


def log_entry(node, message):
    st.session_state.progress_log.append({
        "node": node,
        "message": message,
        "timestamp": time.strftime("%H:%M:%S"),
    })


# ---------------------------------------------------------------------------
# Research execution
# ---------------------------------------------------------------------------

def execute_research(query, settings):
    """Run the agent and update the UI in real time."""
    st.session_state.research_running = True
    st.session_state.progress_log = []
    st.session_state.final_report = None
    st.session_state.sources = []

    env = st.session_state.env_keys
    provider = settings["llm_provider"]
    api_key = env.get(f"{provider}_api_key")

    agent = DeepResearchAgent(
        llm_provider=provider,
        llm_api_key=api_key,
        llm_model=settings["llm_model"],
        search_api_key=env.get("tavily_api_key"),
        max_iterations=settings["max_iterations"],
        max_results=settings["max_results"],
        system_prompt=settings["system_prompt"],
        temperature=settings["temperature"],
    )

    status_container = st.status("Researching...", expanded=True)

    with status_container:
        for event in agent.stream_run(query, report_style=settings["report_style"]):
            if event["event"] == "start":
                st.write(f"Starting research on: **{query}**")
                log_entry("start", "Research started")

            elif event["event"] == "node_complete":
                node = event["node"]
                data = event["data"]
                label = NODE_LABELS.get(node, node)
                detail = build_node_detail(node, data)
                st.write(f"**{label}** — {detail}")
                log_entry(node, detail)

            elif event["event"] == "error":
                err = event["data"].get("error", "Unknown error")
                st.error(f"Error: {err}")
                log_entry("error", err)

            elif event["event"] == "done":
                data = event["data"]
                st.session_state.final_report = data.get("final_report", "")
                st.session_state.sources = data.get("sources", [])
                st.write("Research complete!")
                log_entry("done", "Research complete")

        status_container.update(label="Research complete", state="complete")

    st.session_state.research_running = False


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_progress_log():
    """Render the accumulated progress log."""
    log = st.session_state.progress_log
    if not log:
        st.info("No research has been run yet. Enter a query above and click **Start Research**.")
        return

    for entry in log:
        node = entry["node"]
        label = NODE_LABELS.get(node, node)
        st.markdown(f"`{entry['timestamp']}` **{label}** — {entry['message']}")


def render_report():
    """Render the final report tab."""
    report = st.session_state.final_report
    if report is None:
        st.info("Report will appear here after research completes.")
        return

    st.download_button(
        label="Download Report (.md)",
        data=report,
        file_name="research_report.md",
        mime="text/markdown",
    )

    st.divider()
    st.markdown(report)

    sources = st.session_state.sources
    if sources:
        with st.expander(f"All Sources ({len(sources)})", expanded=False):
            for s in sources:
                st.markdown(f"**[{s['id']}]** [{s['title']}]({s['url']})")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def render_main(settings):
    st.title("Deep Research Agent")

    # Dynamic caption based on selected provider + model
    provider = settings["llm_provider"].capitalize()
    model = settings["llm_model"] or "default"
    st.caption(f"LangGraph-powered iterative research with {provider} ({model}) + Tavily")

    query = st.text_area(
        "Research query",
        placeholder="e.g. What are the latest developments in quantum computing?",
        height=80,
    )

    run_clicked = st.button(
        "Start Research",
        type="primary",
        disabled=(
            not st.session_state.env_loaded
            or not query.strip()
            or st.session_state.research_running
        ),
    )

    tab_progress, tab_report = st.tabs(["Live Progress", "Report"])

    if run_clicked:
        with tab_progress:
            execute_research(query, settings)

    with tab_progress:
        render_progress_log()

    with tab_report:
        render_report()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_session_state()
    settings = render_sidebar()
    render_main(settings)


if __name__ == "__main__":
    main()

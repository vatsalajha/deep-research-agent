"""Streamlit UI for the Deep Research Agent."""

import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO

import streamlit as st

from src.agent import DeepResearchAgent, PROVIDER_MODELS, PROVIDER_ENV_KEYS
from src.sessions import SessionManager
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
# Sharp / angular theme — no rounded corners + sidebar session button tweaks
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

    /* Sidebar session buttons: compact, left-aligned, no uppercase */
    .session-entry .stButton > button {
        text-transform: none !important;
        letter-spacing: normal !important;
        font-weight: 400 !important;
        text-align: left !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.85rem !important;
        border: none !important;
        width: 100% !important;
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
        # History / persistence
        "session_manager": SessionManager(),
        "active_session_id": None,
        "history_search_query": "",
        "research_query": None,
        "research_duration": None,
        "research_settings": None,
        # Rerun support
        "rerun_query": None,
        "rerun_settings": None,
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
# History sidebar helpers
# ---------------------------------------------------------------------------

def _format_session_time(timestamp_str: str) -> str:
    """Return a short human-readable time string."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return timestamp_str
    now = datetime.now(timezone.utc)
    if dt.date() == now.date():
        return dt.strftime("%-I:%M %p")
    return dt.strftime("%b %-d")


def _group_sessions_by_date(sessions: list[dict]) -> dict[str, list[dict]]:
    """Group sessions into Starred, Today, Yesterday, This Week, Earlier."""
    now = datetime.now(timezone.utc)
    groups: dict[str, list[dict]] = {
        "Starred": [],
        "Today": [],
        "Yesterday": [],
        "This Week": [],
        "Earlier": [],
    }
    for s in sessions:
        if s.get("starred"):
            groups["Starred"].append(s)
            continue
        try:
            dt = datetime.fromisoformat(s["timestamp"])
        except (ValueError, TypeError):
            groups["Earlier"].append(s)
            continue
        delta = (now.date() - dt.date()).days
        if delta == 0:
            groups["Today"].append(s)
        elif delta == 1:
            groups["Yesterday"].append(s)
        elif delta < 7:
            groups["This Week"].append(s)
        else:
            groups["Earlier"].append(s)
    return groups


def _load_session(session_id: str) -> None:
    """Load a stored session into the current view."""
    sm: SessionManager = st.session_state.session_manager
    session = sm.get_session(session_id)
    if session is None:
        return
    st.session_state.final_report = session["report"]
    st.session_state.sources = session.get("sources", [])
    st.session_state.active_session_id = session_id
    st.session_state.research_query = session["query"]
    st.session_state.research_duration = session.get("duration_seconds")
    st.session_state.research_settings = session.get("settings")
    st.session_state.progress_log = []


def render_history_sidebar():
    """Render the session history panel at the top of the sidebar."""
    sm: SessionManager = st.session_state.session_manager

    with st.sidebar:
        st.title("History")

        search_q = st.text_input(
            "Search sessions",
            value=st.session_state.history_search_query,
            placeholder="Filter by query...",
            key="history_search_input",
            label_visibility="collapsed",
        )
        st.session_state.history_search_query = search_q

        if search_q.strip():
            sessions = sm.search_sessions(search_q.strip())
        else:
            sessions = sm.list_sessions()

        st.caption(f"{len(sessions)} session{'s' if len(sessions) != 1 else ''}")

        if not sessions:
            st.info("No sessions yet. Run a research query to get started.")
        else:
            groups = _group_sessions_by_date(sessions)
            for group_name, group_sessions in groups.items():
                if not group_sessions:
                    continue
                st.markdown(f"**{group_name}**")
                for s in group_sessions:
                    _render_session_entry(s)

            st.divider()
            if st.button("Clear All History", type="secondary"):
                count = sm.clear_all()
                st.session_state.active_session_id = None
                st.info(f"Cleared {count} session{'s' if count != 1 else ''}")
                st.experimental_rerun()


def _render_session_entry(session: dict) -> None:
    """Render a single session entry with load, star, and delete controls."""
    sm: SessionManager = st.session_state.session_manager
    sid = session["id"]
    query_preview = session["query"][:50] + ("..." if len(session["query"]) > 50 else "")
    time_str = _format_session_time(session["timestamp"])
    settings = session.get("settings", {})
    provider = settings.get("llm_provider", "")
    model = settings.get("llm_model", "")
    badge = f"{provider}/{model}" if provider else ""

    is_active = st.session_state.active_session_id == sid
    star_icon = "S" if session.get("starred") else "s"

    cols = st.columns([7, 1, 1])

    with cols[0]:
        label = f"{'> ' if is_active else ''}{query_preview}"
        if st.button(label, key=f"load_{sid}", use_container_width=True):
            _load_session(sid)
            st.experimental_rerun()
        st.caption(f"{time_str} | {badge}")

    with cols[1]:
        if st.button(star_icon, key=f"star_{sid}"):
            sm.toggle_star(sid)
            st.experimental_rerun()

    with cols[2]:
        if st.button("X", key=f"del_{sid}"):
            sm.delete_session(sid)
            if st.session_state.active_session_id == sid:
                st.session_state.active_session_id = None
                st.session_state.final_report = None
                st.session_state.sources = []
            st.experimental_rerun()


# ---------------------------------------------------------------------------
# Sidebar (Configuration)
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
        with st.expander("Configuration", expanded=False):
            # --- API Keys ---
            st.subheader("API Keys")
            if not st.session_state.env_loaded:
                if st.button("Load API Keys from .env"):
                    try:
                        env = load_environment()
                        st.session_state.env_keys = env
                        st.session_state.env_loaded = True
                        st.experimental_rerun()
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

            # --- Deep Browse ---
            deep_browse = st.checkbox(
                "Deep Browse (extract full pages)",
                value=False,
                help="After searching, fetch full page content from each result URL via Tavily Extract. Slower but produces richer reports.",
            )

            st.divider()

            # --- Agent Settings ---
            st.subheader("Agent Settings")
            max_iterations = st.slider("Max iterations", 1, 5, 2)
            report_style = st.selectbox("Report style", VALID_STYLES, index=0)

            # --- System Prompt ---
            system_prompt = st.text_area(
                "Custom system prompt (optional)",
                value="",
                height=100,
                placeholder="e.g. You are a senior analyst specializing in...",
            )

        # --- Test Runner (outside expander to avoid nested expanders) ---
        st.divider()
        st.subheader("Validation Tests")
        if st.button("Run Tests", disabled=st.session_state.research_running):
            with st.spinner("Running tests..."):
                run_tests()
            st.experimental_rerun()

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
        "deep_browse": deep_browse,
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
    st.session_state.active_session_id = None

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
        deep_browse=settings.get("deep_browse", False),
    )

    start_time = time.time()
    progress_placeholder = st.empty()

    with progress_placeholder.expander("Researching...", expanded=True):
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
                st.success("Research complete!")
                log_entry("done", "Research complete")

    duration = time.time() - start_time
    st.session_state.research_running = False

    # Auto-save on success
    if st.session_state.final_report:
        sm: SessionManager = st.session_state.session_manager
        session_id = sm.save_session(
            query=query,
            report=st.session_state.final_report,
            settings=settings,
            sources=st.session_state.sources,
            duration_seconds=duration,
        )
        st.session_state.active_session_id = session_id
        st.session_state.research_query = query
        st.session_state.research_duration = duration
        st.session_state.research_settings = settings


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

    sm: SessionManager = st.session_state.session_manager
    active_id = st.session_state.active_session_id

    # --- Session metadata caption ---
    query = st.session_state.research_query
    duration = st.session_state.research_duration
    settings = st.session_state.research_settings
    source_count = len(st.session_state.sources)
    if query:
        meta_parts = [f"Query: {query}"]
        if settings:
            meta_parts.append(f"{settings.get('llm_provider', '')}/{settings.get('llm_model', '')}")
        if duration is not None:
            meta_parts.append(f"{duration:.1f}s")
        meta_parts.append(f"{source_count} sources")
        st.caption(" | ".join(meta_parts))

    # --- Action bar ---
    act_cols = st.columns(4)
    with act_cols[0]:
        st.download_button(
            label="Download Report",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
        )
    with act_cols[1]:
        if active_id:
            export_md = sm.export_session_md(active_id)
            if export_md:
                st.download_button(
                    label="Export Full Session",
                    data=export_md,
                    file_name="research_session.md",
                    mime="text/markdown",
                )
    with act_cols[2]:
        if active_id:
            session = sm.get_session(active_id)
            star_label = "Unstar" if session and session.get("starred") else "Star"
            if st.button(star_label, key="report_star"):
                sm.toggle_star(active_id)
                st.experimental_rerun()
    with act_cols[3]:
        if st.session_state.research_query:
            if st.button("Rerun", key="report_rerun"):
                st.session_state.rerun_query = st.session_state.research_query
                st.session_state.rerun_settings = st.session_state.research_settings
                st.experimental_rerun()

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

    # Handle rerun flow
    rerun_query = st.session_state.get("rerun_query")
    default_query = rerun_query if rerun_query else ""

    query = st.text_area(
        "Research query",
        value=default_query,
        placeholder="e.g. What are the latest developments in quantum computing?",
        height=80,
    )

    # Determine which settings to use (rerun settings or current sidebar settings)
    rerun_settings = st.session_state.get("rerun_settings")
    effective_settings = rerun_settings if rerun_settings else settings

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
        # Clear rerun state
        st.session_state.rerun_query = None
        st.session_state.rerun_settings = None
        with tab_progress:
            execute_research(query, effective_settings)

    with tab_progress:
        render_progress_log()

    with tab_report:
        render_report()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_session_state()
    render_history_sidebar()
    settings = render_sidebar()
    render_main(settings)


if __name__ == "__main__":
    main()

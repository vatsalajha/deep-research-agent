"""Session persistence for the Deep Research Agent.

Stores research sessions as JSON at ./data/sessions.json.
No external dependencies beyond the standard library.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_PATH = Path("./data/sessions.json")


class SessionManager:
    """CRUD manager for persisted research sessions."""

    def __init__(self, storage_path: str | Path = _DEFAULT_PATH):
        self._path = Path(storage_path)

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load_data(self) -> dict:
        if not self._path.exists():
            return {"version": 1, "sessions": []}
        text = self._path.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {"version": 1, "sessions": []}

    def _save_data(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_session(
        self,
        query: str,
        report: str,
        settings: dict,
        sources: list[dict],
        duration_seconds: float,
    ) -> str:
        """Save a completed research session. Returns the session ID."""
        session_id = uuid.uuid4().hex
        session = {
            "id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "report": report,
            "settings": settings,
            "sources": sources,
            "duration_seconds": round(duration_seconds, 1),
            "starred": False,
        }
        data = self._load_data()
        data["sessions"].insert(0, session)  # newest first
        self._save_data(data)
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        for s in self._load_data()["sessions"]:
            if s["id"] == session_id:
                return s
        return None

    def list_sessions(self) -> list[dict]:
        return self._load_data()["sessions"]

    def delete_session(self, session_id: str) -> bool:
        data = self._load_data()
        before = len(data["sessions"])
        data["sessions"] = [s for s in data["sessions"] if s["id"] != session_id]
        if len(data["sessions"]) < before:
            self._save_data(data)
            return True
        return False

    def toggle_star(self, session_id: str) -> bool | None:
        """Toggle the starred flag. Returns new value, or None if not found."""
        data = self._load_data()
        for s in data["sessions"]:
            if s["id"] == session_id:
                s["starred"] = not s["starred"]
                self._save_data(data)
                return s["starred"]
        return None

    def search_sessions(self, query: str) -> list[dict]:
        """Case-insensitive substring search on the query field."""
        q = query.lower()
        return [s for s in self._load_data()["sessions"] if q in s["query"].lower()]

    def clear_all(self) -> int:
        """Delete all sessions. Returns count deleted."""
        data = self._load_data()
        count = len(data["sessions"])
        data["sessions"] = []
        self._save_data(data)
        return count

    def export_session_md(self, session_id: str) -> str | None:
        """Return a full markdown export of a session with metadata header."""
        session = self.get_session(session_id)
        if session is None:
            return None

        settings = session.get("settings", {})
        sources = session.get("sources", [])
        ts = session["timestamp"]
        dur = session.get("duration_seconds", 0)

        lines = [
            f"# Research Session",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| **Query** | {session['query']} |",
            f"| **Date** | {ts} |",
            f"| **Provider** | {settings.get('llm_provider', 'N/A')} |",
            f"| **Model** | {settings.get('llm_model', 'N/A')} |",
            f"| **Temperature** | {settings.get('temperature', 'N/A')} |",
            f"| **Report Style** | {settings.get('report_style', 'N/A')} |",
            f"| **Duration** | {dur:.1f}s |",
            f"| **Sources** | {len(sources)} |",
            "",
            "---",
            "",
            session["report"],
            "",
            "---",
            "",
            "## Sources",
            "",
        ]

        for s in sources:
            lines.append(f"- **[{s.get('id', '?')}]** [{s.get('title', 'Untitled')}]({s.get('url', '')})")

        return "\n".join(lines)

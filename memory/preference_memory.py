"""Thread-scoped preference memory stored alongside existing long-term memory."""

from pathlib import Path
import re
import sqlite3
import time

from security.pii_detector import SECRET_VALUE, mask_text


PREFERENCE_PATTERN = re.compile(
    r"(?i)\b(?:i\s+prefer|my\s+preference\s+is|please\s+always)\s+(.{3,200})"
)
SENSITIVE = re.compile(
    r"(?i)(password|passwd|pin|cvv|cvc|otp|api[_ -]?key|token|secret|private[_ -]?key|credential|bank[_ -]?account|iban)"
)


class PreferenceMemory:
    """Persistent preferences isolated by the caller-provided thread ID."""

    def __init__(self, database_path: str | Path = "memory/long_term.sqlite3"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS preferences ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL, "
            "preference TEXT NOT NULL, created REAL NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_preferences_thread ON preferences(thread_id, created)"
        )
        self.connection.commit()

    def _safe(self, text: str) -> bool:
        return not (SENSITIVE.search(text or "") or SECRET_VALUE.search(text or ""))

    def extract_and_store(self, thread_id: str, text: str) -> list[str]:
        if not self._safe(text):
            return []
        extracted: list[str] = []
        for match in PREFERENCE_PATTERN.finditer(text or ""):
            preference = mask_text(match.group(1).strip().rstrip(".!?"))
            if preference and self._safe(preference):
                self.connection.execute(
                    "INSERT INTO preferences(thread_id, preference, created) VALUES (?, ?, ?)",
                    (thread_id, preference, time.time()),
                )
                extracted.append(preference)
        if extracted:
            self.connection.commit()
        return extracted

    def retrieve(self, thread_id: str, query: str, limit: int = 4) -> list[str]:
        rows = self.connection.execute(
            "SELECT preference FROM preferences WHERE thread_id = ? ORDER BY created DESC LIMIT 100",
            (thread_id,),
        ).fetchall()
        query_words = set(re.findall(r"\w+", query.lower()))
        ranked = sorted(
            (
                len(query_words & set(re.findall(r"\w+", preference.lower()))),
                preference,
            )
            for (preference,) in rows
        )
        return [preference for score, preference in reversed(ranked) if score > 0][:limit]

    def close(self) -> None:
        self.connection.close()

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple


class TranscriptStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )
            conn.commit()

    def add_message(self, session_id: str, role: str, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO transcripts (session_id, role, message) VALUES (?, ?, ?)",
                (session_id, role, message),
            )
            conn.commit()

    def get_recent_messages(self, session_id: str, limit: int = 6) -> List[Tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT role, message
                FROM transcripts
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                ''',
                (session_id, limit),
            ).fetchall()
        return list(reversed(rows))

    def format_context(self, session_id: str, limit: int = 6) -> List[str]:
        return [f"{role}: {message}" for role, message in self.get_recent_messages(session_id, limit)]

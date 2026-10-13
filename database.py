"""
Add SQLite module with event throttling and incident logging

This is the Database module
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "safeguard.db"


def _get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=3)
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


def init_db():
    """Create the violations table if it doesn't exist."""
    conn = _get_connection()
    try:
        conn.execute("PRAGMA journal_mode=WAL")  # persistent, set once
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details    TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def log_violation(event_type: str, details: str = ""):
    """Insert one violation row."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO violations (timestamp, event_type, details) VALUES (?, ?, ?)",
            (ts, event_type, details),
        )
        conn.commit()
    finally:
        conn.close()


init_db()

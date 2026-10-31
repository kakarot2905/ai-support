import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    status TEXT NOT NULL,            -- Processing | Processed | Failed
    error TEXT,
    chunk_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    customer_email TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',   -- active | escalated
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,              -- user | agent
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',      -- JSON list of filenames
    ticket_id INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    summary TEXT NOT NULL,
    customer_name TEXT NOT NULL,
    customer_email TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Open',     -- Open | In Progress | Resolved
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def query(sql: str, params: tuple = ()) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> int:
    """Run an INSERT/UPDATE and return the last inserted row id."""
    with connect() as conn:
        return conn.execute(sql, params).lastrowid


def add_message(conversation_id: int, role: str, content: str,
                sources: list[str] | None = None, ticket_id: int | None = None) -> None:
    execute(
        "INSERT INTO messages (conversation_id, role, content, sources, ticket_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (conversation_id, role, content, json.dumps(sources or []), ticket_id, now()),
    )


def get_messages(conversation_id: int) -> list[dict]:
    rows = query("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,))
    for r in rows:
        r["sources"] = json.loads(r["sources"])
    return rows

# Importing required libraries for SQLite session and memory management
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Configuring logging for database operations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = "devassist.db"
MEMORY_PREFIX = "kaydet:"
ROLLING_SUMMARY_THRESHOLD = 10


# Initializing SQLite database and creating tables if not exists
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Creating sessions table for chat history management
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            summary TEXT
        )
    """)

    # Creating messages table for full conversation storage
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            attribution_json TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    # Creating memories table for user-saved persistent notes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            session_id TEXT
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Initializing SQLite database and verifying table schema...")


# Generating unique session ID using timestamp
def generate_session_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


# Creating new chat session and persisting to SQLite
def create_session(first_message: str) -> str:
    session_id = generate_session_id()
    title = first_message[:40] + ("..." if len(first_message) > 40 else "")
    now = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, title, now, now)
    )
    conn.commit()
    conn.close()

    logger.info(f"Creating new session: {session_id} — '{title}'")
    return session_id


# Saving message to SQLite with optional attribution metadata
def save_message(
    session_id: str,
    role: str,
    content: str,
    attribution: Optional[List[Dict[str, Any]]] = None
) -> None:
    now = datetime.utcnow().isoformat()
    attribution_json = json.dumps(attribution) if attribution else None

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO messages (session_id, role, content, attribution_json, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, role, content, attribution_json, now)
    )
    cursor.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?",
        (now, session_id)
    )
    conn.commit()
    conn.close()
    logger.info(f"Saving {role} message to session: {session_id[:16]}...")


# Loading all messages for session as Ollama-compatible message list
def load_session_messages(session_id: str) -> List[Dict[str, str]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    messages = [{"role": row[0], "content": row[1]} for row in rows]
    logger.info(f"Loading {len(messages)} messages for session: {session_id[:16]}...")
    return messages


# Loading full message history with attribution for UI rendering
def load_session_full(session_id: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT role, content, attribution_json, timestamp
           FROM messages WHERE session_id = ? ORDER BY timestamp ASC""",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    messages = []
    for row in rows:
        messages.append({
            "role": row[0],
            "content": row[1],
            "attribution": json.loads(row[2]) if row[2] else None,
            "timestamp": row[3]
        })
    logger.info(f"Loading full session history: {session_id[:16]}...")
    return messages


# Fetching all sessions ordered by most recently updated
def get_all_sessions() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    sessions = [
        {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3]
        }
        for row in rows
    ]
    logger.info(f"Fetching {len(sessions)} sessions from database...")
    return sessions


# Detecting and extracting memory content from user message
def extract_memory(content: str) -> Optional[str]:
    lower = content.strip().lower()
    if lower.startswith(MEMORY_PREFIX):
        memory = content.strip()[len(MEMORY_PREFIX):].strip()
        return memory if memory else None
    return None


# Saving persistent memory note to SQLite memories table
def save_memory(content: str, session_id: Optional[str] = None) -> None:
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO memories (content, created_at, session_id) VALUES (?, ?, ?)",
        (content, now, session_id)
    )
    conn.commit()
    conn.close()
    logger.info(f"Saving persistent memory: '{content[:50]}...'")


# Loading all saved memories for context injection
def load_memories() -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT content FROM memories ORDER BY created_at ASC"
    )
    rows = cursor.fetchall()
    conn.close()

    memories = [row[0] for row in rows]
    logger.info(f"Loading {len(memories)} persistent memories for injection...")
    return memories


# Saving rolling summary for session context compression
def save_summary(session_id: str, summary: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sessions SET summary = ? WHERE id = ?",
        (summary, session_id)
    )
    conn.commit()
    conn.close()
    logger.info(f"Saving rolling summary for session: {session_id[:16]}...")


# Loading existing summary for session context injection
def load_summary(session_id: str) -> Optional[str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT summary FROM sessions WHERE id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


# Counting total messages in session for rolling summary trigger
def count_messages(session_id: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (session_id,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


# Deleting session and all associated messages from database
def delete_session(session_id: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    logger.info(f"Deleting session and messages: {session_id[:16]}...")


if __name__ == "__main__":
    # Running standalone database initialization test
    logger.info("Starting database initialization test...")
    init_db()

    session_id = create_session("What does the add function do?")
    save_message(session_id, "user", "What does the add function do?")
    save_message(session_id, "assistant", "The add function takes a and b and returns their sum.")

    save_memory("RAG pipeline uses BM25+Dense+RRF", session_id)

    sessions = get_all_sessions()
    messages = load_session_messages(session_id)
    memories = load_memories()

    print("\n--- Database Test ---")
    print(f"Sessions : {len(sessions)}")
    print(f"Messages : {len(messages)}")
    print(f"Memories : {len(memories)}")
    print(f"Memory   : {memories[0]}")
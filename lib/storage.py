import os
import sqlite3
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from passlib.hash import pbkdf2_sha256

DB_PATH = os.path.join("data", "app.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def init_db() -> None:
    con = _conn()
    cur = con.cursor()

    # Users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)

    # Sessions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    )
    """)

    # Conversations
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL,
        system_prompt TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    # Messages
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id)
    )
    """)

    # Attachments (images/files linked to a message)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attachments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL,
        kind TEXT NOT NULL,               -- 'image' | 'file'
        filename TEXT NOT NULL,
        mime TEXT NOT NULL,
        data BLOB NOT NULL,
        text_content TEXT,                -- extracted text for non-image files (optional)
        created_at TEXT NOT NULL,
        FOREIGN KEY(message_id) REFERENCES messages(id)
    )
    """)

    # Settings
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)

    con.commit()

    # migrations (safe)
    cols = _table_columns(con, "conversations")
    if "title" not in cols:
        cur.execute("ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT ''")
    if "updated_at" not in cols:
        cur.execute("ALTER TABLE conversations ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")
        cur.execute("UPDATE conversations SET updated_at = created_at WHERE updated_at = ''")

    con.commit()
    con.close()


# ---------------- Attachments ----------------
def add_attachment(
    message_id: int,
    kind: str,
    filename: str,
    mime: str,
    data: bytes,
    text_content: Optional[str] = None,
) -> int:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO attachments(message_id, kind, filename, mime, data, text_content, created_at) "
        "VALUES(?,?,?,?,?,?,?)",
        (message_id, kind, filename, mime, sqlite3.Binary(data), text_content, utc_now()),
    )
    att_id = cur.lastrowid
    con.commit()
    con.close()
    return int(att_id)


def list_attachments_for_message_ids(message_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    """Return mapping message_id -> list of attachment dicts.

    Attachment dict includes raw bytes under key 'data'.
    """
    if not message_ids:
        return {}

    con = _conn()
    cur = con.cursor()
    placeholders = ",".join(["?"] * len(message_ids))
    cur.execute(
        f"SELECT id, message_id, kind, filename, mime, data, text_content, created_at "
        f"FROM attachments WHERE message_id IN ({placeholders}) ORDER BY id ASC",
        message_ids,
    )
    rows = cur.fetchall()
    con.close()

    out: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        mid = int(r["message_id"])
        out.setdefault(mid, []).append(
            {
                "id": int(r["id"]),
                "message_id": mid,
                "kind": r["kind"],
                "filename": r["filename"],
                "mime": r["mime"],
                "data": bytes(r["data"]),
                "text_content": r["text_content"],
                "created_at": r["created_at"],
            }
        )
    return out


# ---------------- Settings ----------------
def get_setting(key: str, default: str = "") -> str:
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cur.fetchone()
    con.close()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    con.commit()
    con.close()


# ---------------- Users ----------------
def user_count() -> int:
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM users")
    row = cur.fetchone()
    con.close()
    return int(row["c"])


def any_admin_exists() -> bool:
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1")
    row = cur.fetchone()
    con.close()
    return row is not None


def upsert_user(user_id: str, password: str, role: str) -> None:
    pw_hash = pbkdf2_sha256.hash(password)
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO users(user_id, password_hash, role) VALUES(?,?,?) "
        "ON CONFLICT(user_id) DO UPDATE SET password_hash=excluded.password_hash, role=excluded.role",
        (user_id, pw_hash, role),
    )
    con.commit()
    con.close()


def verify_user(user_id: str, password: str) -> Optional[Dict[str, str]]:
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT user_id, password_hash, role FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    if pbkdf2_sha256.verify(password, row["password_hash"]):
        return {"user_id": row["user_id"], "role": row["role"]}
    return None


def list_users(limit: int = 200) -> List[sqlite3.Row]:
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT user_id, role FROM users ORDER BY role, user_id LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    return rows


# ---------------- Sessions ----------------
def create_session(user_id: str, role: str, days: int = 7) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    exp = now + timedelta(days=days)

    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO sessions(token, user_id, role, created_at, expires_at) VALUES(?,?,?,?,?)",
        (token, user_id, role, now.isoformat() + "Z", exp.isoformat() + "Z"),
    )
    con.commit()
    con.close()
    return token


def get_session(token: str) -> Optional[Dict[str, str]]:
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT token, user_id, role, expires_at FROM sessions WHERE token = ?", (token,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None

    exp = datetime.fromisoformat(row["expires_at"].replace("Z", ""))
    if datetime.utcnow() > exp:
        delete_session(token)
        return None

    return {"user_id": row["user_id"], "role": row["role"]}


def delete_session(token: str) -> None:
    con = _conn()
    cur = con.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    con.commit()
    con.close()


# ---------------- Conversations ----------------
def create_conversation(user_id: str, role: str, title: str, model: str, system_prompt: str) -> int:
    now = utc_now()
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO conversations(user_id, role, title, model, system_prompt, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?,?)",
        (user_id, role, title, model, system_prompt, now, now),
    )
    conv_id = cur.lastrowid
    con.commit()
    con.close()
    return int(conv_id)


def touch_conversation(
    conversation_id: int,
    title: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    fields = []
    params: List[Any] = []

    if title is not None:
        fields.append("title = ?")
        params.append(title)
    if model is not None:
        fields.append("model = ?")
        params.append(model)
    if system_prompt is not None:
        fields.append("system_prompt = ?")
        params.append(system_prompt)

    fields.append("updated_at = ?")
    params.append(utc_now())
    params.append(conversation_id)

    con = _conn()
    cur = con.cursor()
    cur.execute(f"UPDATE conversations SET {', '.join(fields)} WHERE id = ?", params)
    con.commit()
    con.close()


def list_conversations_for_user(user_id: str, limit: int = 200) -> List[sqlite3.Row]:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
        (user_id, limit),
    )
    rows = cur.fetchall()
    con.close()
    return rows


def list_conversations_with_counts_for_user(user_id: str, limit: int = 200) -> List[sqlite3.Row]:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        """
        SELECT c.*,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
        FROM conversations c
        WHERE c.user_id = ?
        ORDER BY c.updated_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cur.fetchall()
    con.close()
    return rows


def list_conversations_admin(
    user_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    limit: int = 200,
) -> List[sqlite3.Row]:
    con = _conn()
    cur = con.cursor()

    q = "SELECT * FROM conversations WHERE 1=1"
    params: List[Any] = []

    if user_filter:
        q += " AND user_id LIKE ?"
        params.append(f"%{user_filter}%")
    if role_filter:
        q += " AND role = ?"
        params.append(role_filter)
    if model_filter:
        q += " AND model = ?"
        params.append(model_filter)

    q += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    cur.execute(q, params)
    rows = cur.fetchall()
    con.close()
    return rows


# ---------------- Messages ----------------
def add_message(conversation_id: int, role: str, content: str) -> int:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO messages(conversation_id, role, content, created_at) VALUES(?,?,?,?)",
        (conversation_id, role, content, utc_now()),
    )
    msg_id = cur.lastrowid
    con.commit()
    con.close()
    return int(msg_id)


def get_conversation_messages(conversation_id: int) -> List[sqlite3.Row]:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    )
    rows = cur.fetchall()
    con.close()
    return rows


def update_message(message_id: int, new_content: str) -> None:
    con = _conn()
    cur = con.cursor()
    cur.execute("UPDATE messages SET content = ? WHERE id = ?", (new_content, message_id))
    con.commit()
    con.close()


def delete_messages_after(conversation_id: int, message_id: int) -> None:
    con = _conn()
    cur = con.cursor()

    # Delete attachments linked to the messages we are about to delete.
    cur.execute(
        "DELETE FROM attachments WHERE message_id IN ("
        "  SELECT id FROM messages WHERE conversation_id = ? AND id > ?"
        ")",
        (conversation_id, message_id),
    )
    cur.execute(
        "DELETE FROM messages WHERE conversation_id = ? AND id > ?",
        (conversation_id, message_id),
    )
    con.commit()
    con.close()

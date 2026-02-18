import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from passlib.hash import pbkdf2_sha256

from . import supabase_storage

# Default local SQLite path (used when DATABASE_URL is not set)
DB_PATH = os.path.join("data", "app.db")


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read from Streamlit secrets if available, else env vars."""
    val = os.environ.get(key)
    try:
        import streamlit as st  # type: ignore

        if key in st.secrets:
            val = st.secrets.get(key)  # type: ignore
    except Exception:
        pass

    if val is None:
        return default
    val = str(val).strip()
    return val if val else default


def _db_url() -> Optional[str]:
    return _get_secret("DATABASE_URL")


def _using_postgres() -> bool:
    return bool(_db_url())


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ---------------- Connections ----------------

def _sqlite_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def _pg_conn():
    # Import lazily so local SQLite users don't need psycopg installed.
    import psycopg
    from psycopg.rows import dict_row

    url = _db_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")

    # Supabase pooler transaction mode does not support prepared statements.
    # Disable prepares to avoid errors when using pooler port 6543.
    return psycopg.connect(url, row_factory=dict_row, prepare_threshold=0)


def _conn():
    return _pg_conn() if _using_postgres() else _sqlite_conn()


def _table_columns_sqlite(con: sqlite3.Connection, table: str) -> List[str]:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


# ---------------- Schema ----------------

def init_db() -> None:
    """Initialize DB schema.

    - If DATABASE_URL is set: creates tables in Postgres (Supabase).
    - Otherwise: uses local SQLite.
    """
    if _using_postgres():
        _init_postgres()
    else:
        _init_sqlite()


def _init_sqlite() -> None:
    con = _sqlite_conn()
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
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
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
        """
    )

    # Attachments: allow either storing bytes in DB (data) OR storing in Supabase (storage_*).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            filename TEXT NOT NULL,
            mime TEXT NOT NULL,
            storage_bucket TEXT,
            storage_path TEXT,
            data BLOB,
            text_content TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(message_id) REFERENCES messages(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    con.commit()

    # migrations (safe)
    try:
        cols = _table_columns_sqlite(con, "conversations")
        if "title" not in cols:
            cur.execute("ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT ''")
        if "updated_at" not in cols:
            cur.execute("ALTER TABLE conversations ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")
            cur.execute("UPDATE conversations SET updated_at = created_at WHERE updated_at = ''")

        # attachments migrations (older installs may not have storage_* columns)
        cols_att = _table_columns_sqlite(con, "attachments")
        if "storage_bucket" not in cols_att:
            cur.execute("ALTER TABLE attachments ADD COLUMN storage_bucket TEXT")
        if "storage_path" not in cols_att:
            cur.execute("ALTER TABLE attachments ADD COLUMN storage_path TEXT")
        if "data" not in cols_att:
            cur.execute("ALTER TABLE attachments ADD COLUMN data BLOB")
    except Exception:
        pass

    con.commit()
    con.close()


def _init_postgres() -> None:
    con = _pg_conn()
    cur = con.cursor()

    # Note: we store timestamps as ISO strings for compatibility with the existing app logic.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            id BIGSERIAL PRIMARY KEY,
            message_id BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            filename TEXT NOT NULL,
            mime TEXT NOT NULL,
            storage_bucket TEXT,
            storage_path TEXT,
            data BYTEA,
            text_content TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_attachments_message_id ON attachments(message_id, id)")

    con.commit()
    con.close()


# ---------------- Settings ----------------

def get_setting(key: str, default: str = "") -> str:
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
            row = cur.fetchone()
            return (row or {}).get("value", default) if row else default

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cur.fetchone()
    con.close()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO settings(key, value) VALUES(%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (key, value),
            )
            con.commit()
        return

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    con.commit()
    con.close()


# ---------------- Users ----------------

def user_count() -> int:
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) AS c FROM users")
            row = cur.fetchone()
            return int((row or {}).get("c", 0))

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM users")
    row = cur.fetchone()
    con.close()
    return int(row["c"])  # type: ignore[index]


def any_admin_exists() -> bool:
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1")
            return cur.fetchone() is not None

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1")
    row = cur.fetchone()
    con.close()
    return row is not None


def upsert_user(user_id: str, password: str, role: str) -> None:
    pw_hash = pbkdf2_sha256.hash(password)
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO users(user_id, password_hash, role) VALUES(%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET password_hash = EXCLUDED.password_hash,
                    role = EXCLUDED.role
                """,
                (user_id, pw_hash, role),
            )
            con.commit()
        return

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO users(user_id, password_hash, role) VALUES(?,?,?) "
        "ON CONFLICT(user_id) DO UPDATE SET password_hash=excluded.password_hash, role=excluded.role",
        (user_id, pw_hash, role),
    )
    con.commit()
    con.close()


def verify_user(user_id: str, password: str) -> Optional[Dict[str, str]]:
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute("SELECT user_id, password_hash, role FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
        if not row:
            return None
        if pbkdf2_sha256.verify(password, row["password_hash"]):
            return {"user_id": row["user_id"], "role": row["role"]}
        return None

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id, password_hash, role FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    if pbkdf2_sha256.verify(password, row["password_hash"]):
        return {"user_id": row["user_id"], "role": row["role"]}
    return None


def list_users(limit: int = 200):
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute("SELECT user_id, role FROM users ORDER BY role, user_id LIMIT %s", (limit,))
            return cur.fetchall()

    con = _sqlite_conn()
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

    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO sessions(token, user_id, role, created_at, expires_at) VALUES(%s,%s,%s,%s,%s)",
                (token, user_id, role, now.isoformat() + "Z", exp.isoformat() + "Z"),
            )
            con.commit()
        return token

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO sessions(token, user_id, role, created_at, expires_at) VALUES(?,?,?,?,?)",
        (token, user_id, role, now.isoformat() + "Z", exp.isoformat() + "Z"),
    )
    con.commit()
    con.close()
    return token


def get_session(token: str) -> Optional[Dict[str, str]]:
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute("SELECT token, user_id, role, expires_at FROM sessions WHERE token = %s", (token,))
            row = cur.fetchone()
        if not row:
            return None
        exp = datetime.fromisoformat(str(row["expires_at"]).replace("Z", ""))
        if datetime.utcnow() > exp:
            delete_session(token)
            return None
        return {"user_id": row["user_id"], "role": row["role"]}

    con = _sqlite_conn()
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
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM sessions WHERE token = %s", (token,))
            con.commit()
        return

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    con.commit()
    con.close()


# ---------------- Conversations ----------------

def create_conversation(user_id: str, role: str, title: str, model: str, system_prompt: str) -> int:
    now = utc_now()
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO conversations(user_id, role, title, model, system_prompt, created_at, updated_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (user_id, role, title, model, system_prompt, now, now),
            )
            row = cur.fetchone()
            con.commit()
            return int(row["id"])

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO conversations(user_id, role, title, model, system_prompt, created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
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
        fields.append("title = %s" if _using_postgres() else "title = ?")
        params.append(title)
    if model is not None:
        fields.append("model = %s" if _using_postgres() else "model = ?")
        params.append(model)
    if system_prompt is not None:
        fields.append("system_prompt = %s" if _using_postgres() else "system_prompt = ?")
        params.append(system_prompt)

    fields.append("updated_at = %s" if _using_postgres() else "updated_at = ?")
    params.append(utc_now())

    if _using_postgres():
        params.append(conversation_id)
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(f"UPDATE conversations SET {', '.join(fields)} WHERE id = %s", params)
            con.commit()
        return

    params.append(conversation_id)
    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute(f"UPDATE conversations SET {', '.join(fields)} WHERE id = ?", params)
    con.commit()
    con.close()


def list_conversations_for_user(user_id: str, limit: int = 200):
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM conversations WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
                (user_id, limit),
            )
            return cur.fetchall()

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
        (user_id, limit),
    )
    rows = cur.fetchall()
    con.close()
    return rows


def list_conversations_with_counts_for_user(user_id: str, limit: int = 200):
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT c.*,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                FROM conversations c
                WHERE c.user_id = %s
                ORDER BY c.updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            return cur.fetchall()

    con = _sqlite_conn()
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
):
    if _using_postgres():
        q = "SELECT * FROM conversations WHERE 1=1"
        params: List[Any] = []
        if user_filter:
            q += " AND user_id ILIKE %s"
            params.append(f"%{user_filter}%")
        if role_filter:
            q += " AND role = %s"
            params.append(role_filter)
        if model_filter:
            q += " AND model = %s"
            params.append(model_filter)
        q += " ORDER BY updated_at DESC LIMIT %s"
        params.append(limit)

        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(q, params)
            return cur.fetchall()

    con = _sqlite_conn()
    cur = con.cursor()
    q = "SELECT * FROM conversations WHERE 1=1"
    params2: List[Any] = []
    if user_filter:
        q += " AND user_id LIKE ?"
        params2.append(f"%{user_filter}%")
    if role_filter:
        q += " AND role = ?"
        params2.append(role_filter)
    if model_filter:
        q += " AND model = ?"
        params2.append(model_filter)
    q += " ORDER BY updated_at DESC LIMIT ?"
    params2.append(limit)
    cur.execute(q, params2)
    rows = cur.fetchall()
    con.close()
    return rows


# ---------------- Messages ----------------

def add_message(conversation_id: int, role: str, content: str) -> int:
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES(%s,%s,%s,%s) RETURNING id",
                (conversation_id, role, content, utc_now()),
            )
            row = cur.fetchone()
            con.commit()
            return int(row["id"])

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO messages(conversation_id, role, content, created_at) VALUES(?,?,?,?)",
        (conversation_id, role, content, utc_now()),
    )
    msg_id = cur.lastrowid
    con.commit()
    con.close()
    return int(msg_id)


def get_conversation_messages(conversation_id: int):
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT id, role, content, created_at FROM messages WHERE conversation_id = %s ORDER BY id ASC",
                (conversation_id,),
            )
            return cur.fetchall()

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    )
    rows = cur.fetchall()
    con.close()
    return rows


def update_message(message_id: int, new_content: str) -> None:
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute("UPDATE messages SET content = %s WHERE id = %s", (new_content, message_id))
            con.commit()
        return

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute("UPDATE messages SET content = ? WHERE id = ?", (new_content, message_id))
    con.commit()
    con.close()


def delete_messages_after(conversation_id: int, message_id: int) -> None:
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM messages WHERE conversation_id = %s AND id > %s", (conversation_id, message_id))
            con.commit()
        return

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute("DELETE FROM messages WHERE conversation_id = ? AND id > ?", (conversation_id, message_id))
    con.commit()
    con.close()


# ---------------- Attachments ----------------

def add_attachment(
    *,
    message_id: int,
    kind: str,
    filename: str,
    mime: str,
    data: bytes,
    text_content: Optional[str] = None,
    user_id: Optional[str] = None,
    conversation_id: Optional[int] = None,
) -> int:
    """Persist an attachment.

    If Supabase Storage is configured, uploads to Storage and stores storage_bucket/path.
    Otherwise stores raw bytes in DB (SQLite BLOB or Postgres BYTEA).

    NOTE: For Storage uploads, you should pass user_id and conversation_id to generate structured paths.
    """
    now = utc_now()

    storage_bucket: Optional[str] = None
    storage_path: Optional[str] = None
    blob: Optional[bytes] = data

    if supabase_storage.is_enabled():
        if user_id is None or conversation_id is None:
            raise ValueError("user_id and conversation_id are required when Supabase Storage is enabled")
        storage_bucket, storage_path = supabase_storage.upload_bytes(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            filename=filename,
            mime=mime,
            data=data,
        )
        blob = None  # do not duplicate bytes in DB

    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO attachments(message_id, kind, filename, mime, storage_bucket, storage_path, data, text_content, created_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (message_id, kind, filename, mime, storage_bucket, storage_path, blob, text_content, now),
            )
            row = cur.fetchone()
            con.commit()
            return int(row["id"])

    con = _sqlite_conn()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO attachments(message_id, kind, filename, mime, storage_bucket, storage_path, data, text_content, created_at)
        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            message_id,
            kind,
            filename,
            mime,
            storage_bucket,
            storage_path,
            sqlite3.Binary(blob) if blob is not None else None,
            text_content,
            now,
        ),
    )
    att_id = cur.lastrowid
    con.commit()
    con.close()
    return int(att_id)


def list_attachments_for_message_ids(message_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    """Return mapping message_id -> list of attachment dicts.

    Attachment dict includes raw bytes under key 'data'. If bytes are stored in Supabase Storage,
    this function downloads them server-side.
    """
    if not message_ids:
        return {}

    rows: List[Any]
    if _using_postgres():
        with _pg_conn() as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT id, message_id, kind, filename, mime, storage_bucket, storage_path, data, text_content, created_at
                FROM attachments
                WHERE message_id = ANY(%s::bigint[])
                ORDER BY id ASC
                """,
                (message_ids,),
            )
            rows = cur.fetchall()
    else:
        con = _sqlite_conn()
        cur = con.cursor()
        placeholders = ",".join(["?"] * len(message_ids))
        cur.execute(
            f"""
            SELECT id, message_id, kind, filename, mime, storage_bucket, storage_path, data, text_content, created_at
            FROM attachments
            WHERE message_id IN ({placeholders})
            ORDER BY id ASC
            """,
            message_ids,
        )
        rows = cur.fetchall()
        con.close()

    out: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        mid = int(r["message_id"]) if isinstance(r, dict) else int(r["message_id"])  # sqlite Row acts like dict
        bucket = r.get("storage_bucket") if isinstance(r, dict) else r["storage_bucket"]
        path = r.get("storage_path") if isinstance(r, dict) else r["storage_path"]
        blob = r.get("data") if isinstance(r, dict) else r["data"]

        data_bytes: bytes = b""
        if blob is not None:
            data_bytes = bytes(blob)
        elif bucket and path and supabase_storage.is_enabled():
            try:
                data_bytes = supabase_storage.download_bytes(bucket, path)
            except Exception:
                data_bytes = b""

        out.setdefault(mid, []).append(
            {
                "id": int(r["id"]),
                "message_id": mid,
                "kind": r["kind"],
                "filename": r["filename"],
                "mime": r["mime"],
                "storage_bucket": bucket,
                "storage_path": path,
                "data": data_bytes,
                "text_content": r.get("text_content") if isinstance(r, dict) else r["text_content"],
                "created_at": r["created_at"],
            }
        )

    return out

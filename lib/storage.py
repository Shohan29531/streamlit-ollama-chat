import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from passlib.hash import bcrypt

# Optional: psycopg (Postgres) for Supabase Postgres
try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover
    psycopg = None
    dict_row = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit first, then env
    try:
        import streamlit as st  # type: ignore

        if key in st.secrets:
            v = st.secrets.get(key)
            return str(v) if v is not None else default
    except Exception:
        pass
    return os.environ.get(key, default)


DATABASE_URL = _get_secret("DATABASE_URL")
DB_PATH = _get_secret("DB_PATH", "./app.db")

_USE_PG = bool(DATABASE_URL)

# Supabase Storage (optional)
SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _get_secret("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = _get_secret("SUPABASE_STORAGE_BUCKET", "ds330-chat-uploads")
USE_SUPABASE_STORAGE = (
    _get_secret("USE_SUPABASE_STORAGE", "true").lower() == "true"
    and bool(SUPABASE_URL)
    and bool(SUPABASE_SERVICE_ROLE_KEY)
)


# ---------------- Connection helpers ----------------


@contextmanager
def _pg_conn():
    if not psycopg:
        raise RuntimeError(
            "psycopg is required for Postgres. Add 'psycopg[binary]' to requirements.txt"
        )
    assert DATABASE_URL
    conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)  # type: ignore
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


_SQLITE_SINGLETON: Optional[sqlite3.Connection] = None


def _sqlite_singleton() -> sqlite3.Connection:
    global _SQLITE_SINGLETON
    if _SQLITE_SINGLETON is None:
        _SQLITE_SINGLETON = _sqlite_conn()
    return _SQLITE_SINGLETON


def _rows_to_dicts(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        elif isinstance(r, sqlite3.Row):
            out.append({k: r[k] for k in r.keys()})
        else:
            try:
                out.append(dict(r))
            except Exception:
                out.append({})
    return out


def _exec(sql: str, params: Tuple[Any, ...] = (), fetch: str = "none") -> Any:
    """Execute SQL against the configured backend.

    fetch: 'none' | 'one' | 'all'
    """
    if _USE_PG:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                if fetch == "one":
                    row = cur.fetchone()
                    return row
                if fetch == "all":
                    return cur.fetchall()
                return None
    else:
        conn = _sqlite_singleton()
        cur = conn.cursor()
        cur.execute(sql, params)
        if fetch == "one":
            row = cur.fetchone()
            conn.commit()
            return row
        if fetch == "all":
            rows = cur.fetchall()
            conn.commit()
            return rows
        conn.commit()
        return None


def _has_column(table: str, col: str) -> bool:
    if _USE_PG:
        row = _exec(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s
            LIMIT 1
            """,
            (table, col),
            fetch="one",
        )
        return bool(row)
    else:
        rows = _exec(f"PRAGMA table_info({table})", fetch="all")
        for r in rows:
            d = {k: r[k] for k in r.keys()}  # type: ignore
            if d.get("name") == col:
                return True
        return False


def _add_column(table: str, col: str, col_type_sql: str) -> None:
    if _has_column(table, col):
        return
    _exec(f"ALTER TABLE {table} ADD COLUMN {col} {col_type_sql}")


# ---------------- Schema / init ----------------


def init_db() -> None:
    """Create tables + apply small migrations.

    Safe to call on every Streamlit rerun.
    """
    # users
    _exec(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
        """
    )

    # sessions
    _exec(
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

    # settings
    _exec(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    # assignments
    _exec(
        """
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            prompt TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
        if not _USE_PG
        else """
        CREATE TABLE IF NOT EXISTS assignments (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            prompt TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
    )

    # conversations
    _exec(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            title TEXT,
            model TEXT,
            system_prompt TEXT,
            base_prompt TEXT,
            assignment_id INTEGER,
            assignment_name TEXT,
            assignment_prompt TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
        if not _USE_PG
        else """
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            title TEXT,
            model TEXT,
            system_prompt TEXT,
            base_prompt TEXT,
            assignment_id INTEGER,
            assignment_name TEXT,
            assignment_prompt TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
    )

    # messages
    _exec(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
        if not _USE_PG
        else """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
    )

    # attachments (either stored inline or in Supabase Storage)
    _exec(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY,
            message_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            filename TEXT,
            mime TEXT,
            data BLOB,
            text_content TEXT,
            bucket TEXT,
            path TEXT,
            created_at TEXT NOT NULL
        )
        """
        if not _USE_PG
        else """
        CREATE TABLE IF NOT EXISTS attachments (
            id SERIAL PRIMARY KEY,
            message_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            filename TEXT,
            mime TEXT,
            data BYTEA,
            text_content TEXT,
            bucket TEXT,
            path TEXT,
            created_at TEXT NOT NULL
        )
        """,
    )

    # Lightweight migrations for older schemas
    # conversations new columns
    _add_column("conversations", "base_prompt", "TEXT")
    _add_column("conversations", "assignment_id", "INTEGER")
    _add_column("conversations", "assignment_name", "TEXT")
    _add_column("conversations", "assignment_prompt", "TEXT")

    # assignments new column (if table existed without prompt)
    _add_column("assignments", "prompt", "TEXT")

    # attachments storage metadata
    _add_column("attachments", "bucket", "TEXT")
    _add_column("attachments", "path", "TEXT")

    # Back-compat migration: move old global prompt into base prompt
    if get_setting("base_system_prompt", None) is None:
        old = get_setting("global_system_prompt", None)
        if old is not None:
            set_setting("base_system_prompt", old)

    # Ensure at least one assignment + active assignment selection
    _ensure_default_assignment()

    # Backfill: older conversations may have no assignment.* columns populated.
    # Starting now we assume every conversation belongs to an assignment.
    _backfill_conversation_assignments()


def _backfill_conversation_assignments() -> None:
    """Assign all existing conversations to Assignment 1 if missing."""

    # Ensure Assignment 1 exists (created by _ensure_default_assignment()).
    row = _exec(
        "SELECT id, name, prompt FROM assignments WHERE name = ?" if not _USE_PG else
        "SELECT id, name, prompt FROM assignments WHERE name = %s",
        ("Assignment 1",),
        fetch="one",
    )

    if not row:
        # Fallback: take the first assignment.
        row = _exec(
            "SELECT id, name, prompt FROM assignments ORDER BY id LIMIT 1",
            fetch="one",
        )
        if not row:
            return

    a_id = int(row["id"] if isinstance(row, dict) else row[0])
    a_name = (row["name"] if isinstance(row, dict) else row[1]) or "Assignment 1"
    a_prompt = (row["prompt"] if isinstance(row, dict) else row[2]) or ""

    # Only fill when missing; do NOT overwrite system_prompt snapshots.
    if _USE_PG:
        _exec(
            """
            UPDATE conversations
            SET assignment_id = COALESCE(assignment_id, %s),
                assignment_name = COALESCE(NULLIF(assignment_name, ''), %s),
                assignment_prompt = COALESCE(NULLIF(assignment_prompt, ''), %s)
            WHERE assignment_id IS NULL
               OR assignment_name IS NULL OR assignment_name = ''
               OR assignment_prompt IS NULL OR assignment_prompt = ''
            """,
            (a_id, a_name, a_prompt),
        )
    else:
        _exec(
            """
            UPDATE conversations
            SET assignment_id = COALESCE(assignment_id, ?),
                assignment_name = COALESCE(NULLIF(assignment_name, ''), ?),
                assignment_prompt = COALESCE(NULLIF(assignment_prompt, ''), ?)
            WHERE assignment_id IS NULL
               OR assignment_name IS NULL OR assignment_name = ''
               OR assignment_prompt IS NULL OR assignment_prompt = ''
            """,
            (a_id, a_name, a_prompt),
        )


def _ensure_default_assignment() -> None:
    rows = list_assignments()
    if rows:
        # active assignment exists?
        aid = get_setting("active_assignment_id", None)
        if aid is None:
            set_setting("active_assignment_id", str(rows[0]["id"]))
        return

    # Create a default assignment
    now = _now_iso()
    if _USE_PG:
        row = _exec(
            """
            INSERT INTO assignments (name, prompt, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            ("Assignment 1", "", now, now),
            fetch="one",
        )
        new_id = int(row["id"])  # type: ignore
    else:
        _exec(
            """
            INSERT INTO assignments (name, prompt, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            ("Assignment 1", "", now, now),
        )
        row = _exec("SELECT last_insert_rowid() AS id", fetch="one")
        new_id = int(row["id"])  # type: ignore

    set_setting("active_assignment_id", str(new_id))


# ---------------- Settings ----------------


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    row = _exec("SELECT value FROM settings WHERE key = ?" if not _USE_PG else "SELECT value FROM settings WHERE key = %s", (key,), fetch="one")
    if not row:
        return default
    if isinstance(row, dict):
        return row.get("value")
    return row[0]


def set_setting(key: str, value: str) -> None:
    if _USE_PG:
        _exec(
            """
            INSERT INTO settings (key, value) VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (key, value),
        )
    else:
        _exec(
            """
            INSERT INTO settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


# ---------------- Users / Auth ----------------


def user_count() -> int:
    row = _exec("SELECT COUNT(*) AS c FROM users", fetch="one")
    if isinstance(row, dict):
        return int(row.get("c") or 0)
    return int(row[0] if row else 0)


def any_admin_exists() -> bool:
    row = _exec(
        "SELECT 1 FROM users WHERE role = ? LIMIT 1" if not _USE_PG else "SELECT 1 FROM users WHERE role = %s LIMIT 1",
        ("admin",),
        fetch="one",
    )
    return bool(row)


def upsert_user(user_id: str, password: str, role: str) -> None:
    pw_hash = bcrypt.hash(password)
    if _USE_PG:
        _exec(
            """
            INSERT INTO users (user_id, password_hash, role)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET password_hash = EXCLUDED.password_hash, role = EXCLUDED.role
            """,
            (user_id, pw_hash, role),
        )
    else:
        _exec(
            """
            INSERT INTO users (user_id, password_hash, role)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET password_hash = excluded.password_hash, role = excluded.role
            """,
            (user_id, pw_hash, role),
        )


def verify_user(user_id: str, password: str) -> Optional[Dict[str, str]]:
    row = _exec(
        "SELECT user_id, password_hash, role FROM users WHERE user_id = ?" if not _USE_PG else "SELECT user_id, password_hash, role FROM users WHERE user_id = %s",
        (user_id,),
        fetch="one",
    )
    if not row:
        return None
    d = row if isinstance(row, dict) else {"user_id": row[0], "password_hash": row[1], "role": row[2]}
    if bcrypt.verify(password, d["password_hash"]):
        return {"user_id": d["user_id"], "role": d["role"]}
    return None


def list_users() -> List[Dict[str, Any]]:
    rows = _exec("SELECT user_id, role FROM users ORDER BY user_id", fetch="all")
    return _rows_to_dicts(rows)


def create_session(user_id: str, role: str, hours: int = 12) -> str:
    token = secrets.token_urlsafe(24)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=hours)
    _exec(
        """
        INSERT INTO sessions (token, user_id, role, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?)
        """ if not _USE_PG else
        """
        INSERT INTO sessions (token, user_id, role, created_at, expires_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (token, user_id, role, now.isoformat(), expires.isoformat()),
    )
    return token


def get_session(token: str) -> Optional[Dict[str, Any]]:
    row = _exec(
        "SELECT token, user_id, role, expires_at FROM sessions WHERE token = ?" if not _USE_PG else "SELECT token, user_id, role, expires_at FROM sessions WHERE token = %s",
        (token,),
        fetch="one",
    )
    if not row:
        return None
    d = row if isinstance(row, dict) else {"token": row[0], "user_id": row[1], "role": row[2], "expires_at": row[3]}
    try:
        exp = datetime.fromisoformat(d["expires_at"])
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < datetime.now(timezone.utc):
            delete_session(token)
            return None
    except Exception:
        pass
    return {"token": d["token"], "user_id": d["user_id"], "role": d["role"]}


def delete_session(token: str) -> None:
    _exec(
        "DELETE FROM sessions WHERE token = ?" if not _USE_PG else "DELETE FROM sessions WHERE token = %s",
        (token,),
    )


# ---------------- Assignments / Prompts ----------------


def list_assignments() -> List[Dict[str, Any]]:
    rows = _exec("SELECT id, name, prompt FROM assignments ORDER BY id", fetch="all")
    return _rows_to_dicts(rows)


def add_assignment(name: str, prompt: str = "") -> int:
    now = _now_iso()
    if _USE_PG:
        row = _exec(
            """
            INSERT INTO assignments (name, prompt, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (name, prompt or "", now, now),
            fetch="one",
        )
        return int(row["id"])  # type: ignore
    else:
        _exec(
            """
            INSERT INTO assignments (name, prompt, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, prompt or "", now, now),
        )
        row = _exec("SELECT last_insert_rowid() AS id", fetch="one")
        return int(row["id"])  # type: ignore


def update_assignment_prompt(assignment_id: int, prompt: str) -> None:
    now = _now_iso()
    _exec(
        "UPDATE assignments SET prompt = ?, updated_at = ? WHERE id = ?" if not _USE_PG else "UPDATE assignments SET prompt = %s, updated_at = %s WHERE id = %s",
        (prompt or "", now, assignment_id),
    )


def get_assignment(assignment_id: int) -> Optional[Dict[str, Any]]:
    row = _exec(
        "SELECT id, name, prompt FROM assignments WHERE id = ?" if not _USE_PG else "SELECT id, name, prompt FROM assignments WHERE id = %s",
        (assignment_id,),
        fetch="one",
    )
    if not row:
        return None
    d = row if isinstance(row, dict) else {"id": row[0], "name": row[1], "prompt": row[2]}
    return d


def get_active_assignment() -> Dict[str, Any]:
    aid_str = get_setting("active_assignment_id", None)
    aid = int(aid_str) if aid_str and aid_str.isdigit() else None
    if aid is not None:
        a = get_assignment(aid)
        if a:
            return a

    # fallback to first assignment
    rows = list_assignments()
    if not rows:
        _ensure_default_assignment()
        rows = list_assignments()

    a = rows[0]
    set_setting("active_assignment_id", str(a["id"]))
    return a


def set_active_assignment(assignment_id: int) -> None:
    set_setting("active_assignment_id", str(int(assignment_id)))


def get_base_system_prompt(default: str) -> str:
    v = get_setting("base_system_prompt", None)
    return v if (v is not None and v.strip() != "") else default


def set_base_system_prompt(prompt: str) -> None:
    set_setting("base_system_prompt", prompt or "")


# ---------------- Conversations ----------------


def create_conversation(
    user_id: str,
    role: str,
    title: str,
    model: str,
    system_prompt: str,
    base_prompt: Optional[str] = None,
    assignment_id: Optional[int] = None,
    assignment_name: Optional[str] = None,
    assignment_prompt: Optional[str] = None,
) -> int:
    # Backstop: ensure assignment metadata is always present.
    if assignment_id is None or assignment_name is None:
        try:
            a = get_active_assignment()
            assignment_id = assignment_id if assignment_id is not None else int(a["id"])
            assignment_name = assignment_name if assignment_name is not None else a.get("name")
            if assignment_prompt is None:
                assignment_prompt = a.get("prompt") or ""
        except Exception:
            assignment_id = assignment_id if assignment_id is not None else None

    now = _now_iso()
    if _USE_PG:
        row = _exec(
            """
            INSERT INTO conversations (
                user_id, role, title, model, system_prompt,
                base_prompt, assignment_id, assignment_name, assignment_prompt,
                created_at, updated_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                user_id,
                role,
                title,
                model,
                system_prompt,
                base_prompt,
                assignment_id,
                assignment_name,
                assignment_prompt,
                now,
                now,
            ),
            fetch="one",
        )
        return int(row["id"])  # type: ignore
    else:
        _exec(
            """
            INSERT INTO conversations (
                user_id, role, title, model, system_prompt,
                base_prompt, assignment_id, assignment_name, assignment_prompt,
                created_at, updated_at
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                user_id,
                role,
                title,
                model,
                system_prompt,
                base_prompt,
                assignment_id,
                assignment_name,
                assignment_prompt,
                now,
                now,
            ),
        )
        row = _exec("SELECT last_insert_rowid() AS id", fetch="one")
        return int(row["id"])  # type: ignore


def touch_conversation(
    conversation_id: int,
    title: Optional[str] = None,
    model: Optional[str] = None,
    # system_prompt is optional: by default we DO NOT overwrite snapshot prompts.
    system_prompt: Optional[str] = None,
) -> None:
    now = _now_iso()
    sets = ["updated_at = ?" if not _USE_PG else "updated_at = %s"]
    params: List[Any] = [now]

    if title is not None:
        sets.append("title = ?" if not _USE_PG else "title = %s")
        params.append(title)
    if model is not None:
        sets.append("model = ?" if not _USE_PG else "model = %s")
        params.append(model)
    if system_prompt is not None:
        sets.append("system_prompt = ?" if not _USE_PG else "system_prompt = %s")
        params.append(system_prompt)

    params.append(conversation_id)

    sql = (
        f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?"
        if not _USE_PG
        else f"UPDATE conversations SET {', '.join(sets)} WHERE id = %s"
    )
    _exec(sql, tuple(params))


def get_conversation(conversation_id: int) -> Optional[Dict[str, Any]]:
    row = _exec(
        "SELECT * FROM conversations WHERE id = ?" if not _USE_PG else "SELECT * FROM conversations WHERE id = %s",
        (conversation_id,),
        fetch="one",
    )
    if not row:
        return None
    return row if isinstance(row, dict) else {k: row[k] for k in row.keys()}  # type: ignore


def list_conversations_for_user(user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    rows = _exec(
        """
        SELECT id, title, updated_at, model, assignment_name
        FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
        LIMIT ?
        """ if not _USE_PG else
        """
        SELECT id, title, updated_at, model, assignment_name
        FROM conversations
        WHERE user_id = %s
        ORDER BY updated_at DESC
        LIMIT %s
        """,
        (user_id, limit),
        fetch="all",
    )
    return _rows_to_dicts(rows)


def list_conversations_with_counts_for_user(user_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    rows = _exec(
        """
        SELECT c.id, c.title, c.updated_at, c.model, COUNT(m.id) AS msg_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.user_id = ?
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT ?
        """ if not _USE_PG else
        """
        SELECT c.id, c.title, c.updated_at, c.model, COUNT(m.id) AS msg_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.user_id = %s
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT %s
        """,
        (user_id, limit),
        fetch="all",
    )
    return _rows_to_dicts(rows)


def list_conversations_admin(
    user_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    assignment_id_filter: Optional[int] = None,
    limit: int = 300,
) -> List[Dict[str, Any]]:
    where = []
    params: List[Any] = []

    if user_filter:
        where.append("user_id LIKE ?" if not _USE_PG else "user_id ILIKE %s")
        params.append(f"%{user_filter}%")
    if role_filter:
        where.append("role = ?" if not _USE_PG else "role = %s")
        params.append(role_filter)
    if model_filter:
        where.append("model = ?" if not _USE_PG else "model = %s")
        params.append(model_filter)

    if assignment_id_filter is not None:
        where.append("assignment_id = ?" if not _USE_PG else "assignment_id = %s")
        params.append(int(assignment_id_filter))

    wsql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = (
        f"SELECT id, user_id, role, title, model, assignment_id, assignment_name, updated_at FROM conversations {wsql} ORDER BY updated_at DESC LIMIT ?"
        if not _USE_PG
        else f"SELECT id, user_id, role, title, model, assignment_id, assignment_name, updated_at FROM conversations {wsql} ORDER BY updated_at DESC LIMIT %s"
    )

    params.append(limit)
    rows = _exec(sql, tuple(params), fetch="all")
    return _rows_to_dicts(rows)


# ---------------- Messages ----------------


def get_conversation_messages(conversation_id: int) -> List[Dict[str, Any]]:
    rows = _exec(
        "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC" if not _USE_PG
        else "SELECT id, role, content, created_at FROM messages WHERE conversation_id = %s ORDER BY id ASC",
        (conversation_id,),
        fetch="all",
    )
    return _rows_to_dicts(rows)


def add_message(conversation_id: int, role: str, content: str) -> int:
    now = _now_iso()
    if _USE_PG:
        row = _exec(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (%s,%s,%s,%s)
            RETURNING id
            """,
            (conversation_id, role, content, now),
            fetch="one",
        )
        return int(row["id"])  # type: ignore
    else:
        _exec(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?,?,?,?)
            """,
            (conversation_id, role, content, now),
        )
        row = _exec("SELECT last_insert_rowid() AS id", fetch="one")
        return int(row["id"])  # type: ignore


def update_message(message_id: int, content: str) -> None:
    _exec(
        "UPDATE messages SET content = ? WHERE id = ?" if not _USE_PG else "UPDATE messages SET content = %s WHERE id = %s",
        (content, message_id),
    )


def delete_messages_after(conversation_id: int, message_id: int) -> None:
    # Delete attachments for deleted messages first
    rows = _exec(
        "SELECT id FROM messages WHERE conversation_id = ? AND id > ?" if not _USE_PG
        else "SELECT id FROM messages WHERE conversation_id = %s AND id > %s",
        (conversation_id, message_id),
        fetch="all",
    )
    msg_ids = [int(r["id"]) if isinstance(r, dict) else int(r[0]) for r in rows]  # type: ignore
    if msg_ids:
        placeholders = ",".join(["?"] * len(msg_ids)) if not _USE_PG else ",".join(["%s"] * len(msg_ids))
        _exec(
            f"DELETE FROM attachments WHERE message_id IN ({placeholders})" if not _USE_PG else f"DELETE FROM attachments WHERE message_id IN ({placeholders})",
            tuple(msg_ids),
        )

    _exec(
        "DELETE FROM messages WHERE conversation_id = ? AND id > ?" if not _USE_PG
        else "DELETE FROM messages WHERE conversation_id = %s AND id > %s",
        (conversation_id, message_id),
    )


# ---------------- Attachments ----------------


def _supabase_client():
    if not USE_SUPABASE_STORAGE:
        return None
    try:
        from lib import supabase_storage  # type: ignore

        return supabase_storage
    except Exception:
        return None


def add_attachment(
    message_id: int,
    kind: str,
    filename: str,
    mime: str,
    data: bytes,
    text_content: Optional[str] = None,
) -> None:
    """Persist an attachment.

    If Supabase Storage is configured, binary data is uploaded there and only metadata is stored in Postgres.
    Otherwise, binary data is stored inline (SQLite BLOB / Postgres BYTEA).
    """
    now = _now_iso()

    bucket = None
    path = None
    blob = data

    sb = _supabase_client()
    if sb is not None:
        # upload under a deterministic prefix
        safe_name = filename.replace("/", "_")
        path = f"m{message_id}/{secrets.token_urlsafe(8)}-{safe_name}"
        sb.upload_bytes(
            supabase_url=SUPABASE_URL,
            service_role_key=SUPABASE_SERVICE_ROLE_KEY,
            bucket=SUPABASE_STORAGE_BUCKET,
            path=path,
            data=data,
            content_type=mime or "application/octet-stream",
        )
        bucket = SUPABASE_STORAGE_BUCKET
        blob = None

    _exec(
        """
        INSERT INTO attachments (message_id, kind, filename, mime, data, text_content, bucket, path, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """ if not _USE_PG else
        """
        INSERT INTO attachments (message_id, kind, filename, mime, data, text_content, bucket, path, created_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (message_id, kind, filename, mime, blob, text_content, bucket, path, now),
    )


def list_attachments_for_message_ids(message_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    if not message_ids:
        return {}

    placeholders = ",".join(["?"] * len(message_ids)) if not _USE_PG else ",".join(["%s"] * len(message_ids))
    rows = _exec(
        f"SELECT id, message_id, kind, filename, mime, data, text_content, bucket, path FROM attachments WHERE message_id IN ({placeholders}) ORDER BY id ASC",
        tuple(message_ids),
        fetch="all",
    )

    out: Dict[int, List[Dict[str, Any]]] = {}
    sb = _supabase_client()

    for r in _rows_to_dicts(rows):
        mid = int(r["message_id"])
        data = r.get("data")

        # psycopg returns memoryview for bytea
        if isinstance(data, memoryview):
            data = data.tobytes()

        if data is None and sb is not None and r.get("bucket") and r.get("path"):
            try:
                data = sb.download_bytes(
                    supabase_url=SUPABASE_URL,
                    service_role_key=SUPABASE_SERVICE_ROLE_KEY,
                    bucket=r["bucket"],
                    path=r["path"],
                )
            except Exception:
                data = None

        out.setdefault(mid, []).append(
            {
                "id": r.get("id"),
                "message_id": mid,
                "kind": r.get("kind"),
                "filename": r.get("filename"),
                "mime": r.get("mime"),
                "data": data,
                "text_content": r.get("text_content"),
            }
        )

    return out
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import sqlite3
import os
from datetime import datetime

TOKEN = os.environ.get("REMOTE_SYNC_TOKEN", "")

DB = "server_logs.db"

app = FastAPI()

def now():
    return datetime.utcnow().isoformat() + "Z"

def conn():
    c = sqlite3.connect(DB)
    c.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        role TEXT,
        model TEXT,
        system_prompt TEXT,
        local_conversation_id INTEGER,
        messages_json TEXT,
        created_at TEXT
    )
    """)
    return c

class Ingest(BaseModel):
    conversation: Dict
    messages: List[Dict]

@app.post("/ingest")
def ingest(payload: Ingest, authorization: Optional[str] = Header(None)):
    if not TOKEN:
        raise HTTPException(status_code=500, detail="Server missing REMOTE_SYNC_TOKEN env var")
    if not authorization or authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    c = conn()
    c.execute(
        "INSERT INTO logs(user_id, role, model, system_prompt, local_conversation_id, messages_json, created_at) VALUES (?,?,?,?,?,?,?)",
        (
            payload.conversation.get("user_id"),
            payload.conversation.get("role"),
            payload.conversation.get("model"),
            payload.conversation.get("system_prompt"),
            payload.conversation.get("local_conversation_id"),
            str(payload.messages),
            now(),
        ),
    )
    c.commit()
    c.close()
    return {"ok": True}

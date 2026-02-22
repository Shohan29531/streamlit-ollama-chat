# DS330 Chat (Streamlit) — Deployment & Admin Guide

A classroom-oriented, **ChatGPT-like** Streamlit app for DS330 that supports:
- **Ollama Cloud** model selection (dropdown populated from Ollama Cloud)
- **Text + image + file uploads** in the chat input
- **Assignments** (admin selects an active assignment; students see it)
- **Two-part system prompts**: Base prompt + Assignment-specific prompt
- **Supabase Postgres** persistence (recommended) with optional **Supabase Storage** for uploads
- Student-friendly UX (copy message / copy whole conversation) and admin tools (browse conversations, download transcript)

> **Roles**
> - **Admin**: manages assignments, prompts, users, and conversation browser
> - **Student**: chats only (no conversation editing)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)  
2. [Local Development Setup](#local-development-setup)  
3. [Deployment on Streamlit Community Cloud](#deployment-on-streamlit-community-cloud)  
4. [Secrets & Configuration](#secrets--configuration)  
5. [Supabase Postgres Setup](#supabase-postgres-setup)  
6. [Optional Supabase Storage for Uploads](#optional-supabase-storage-for-uploads)  
7. [Ollama Cloud Setup](#ollama-cloud-setup)  
8. [Assignments & Prompts (Admin)](#assignments--prompts-admin)  
9. [Conversation Browser & Transcripts (Admin)](#conversation-browser--transcripts-admin)  
10. [Clipboard Copy UX (Students + Admin Chat)](#clipboard-copy-ux-students--admin-chat)  
11. [Troubleshooting](#troubleshooting)  
12. [Security Notes](#security-notes)  
13. [Repo Layout](#repo-layout)

---

## Architecture Overview

### High-level flow
1. User logs in (admin or student).
2. On chat send:
   - Text is collected from the chat input.
   - Uploaded files/images (if any) are processed:
     - **Images** → base64 and attached to the final user message in the Ollama payload via `messages[].images`.
     - **Files** → text is extracted and appended to the message content as a structured attachment block.
3. The app builds:
   - **Base system prompt**
   - **Assignment-specific prompt**
   - User message(s)
4. The app calls **Ollama Cloud** `/api/chat`.
5. Messages + metadata are persisted in **Supabase Postgres** (recommended).

### Where state lives
- **Auth sessions**: Streamlit session state + cookies (encrypted)
- **Persistent data**: Supabase Postgres
- **Uploads**: Prefer Supabase Storage; fallback is DB bytes (not recommended long-term)

---

## Local Development Setup

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure secrets locally
Create `.streamlit/secrets.toml` (do not commit this file):
```toml
OLLAMA_HOST = "https://ollama.com"
OLLAMA_API_KEY = "YOUR_OLLAMA_KEY"

# Recommended: Supabase Postgres
DATABASE_URL = "postgres://..."
SUPABASE_URL = "https://<project-ref>.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "YOUR_SERVICE_ROLE_KEY"

# Optional Storage
SUPABASE_BUCKET = "chat-uploads"

# Auth/cookies
COOKIE_SECRET = "a-long-random-string"
BOOTSTRAP_PASSWORD = "admin-bootstrap-password"

# Optional: restrict model choices (comma-separated)
# MODEL_ALLOWLIST = "gpt-oss:120b,qwen3-vl:235b"
```

### 4) Run Streamlit
```bash
streamlit run app.py
```

---

## Deployment on Streamlit Community Cloud

Streamlit Community Cloud deploys directly from a **GitHub repository**.

### 1) Push code to GitHub
From your project root:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

### 2) Create the app in Streamlit Cloud
- Streamlit Community Cloud → **Create app**
- Select repo + branch (`main`)
- Main file: `app.py`

### 3) Add secrets in Streamlit Cloud
In your app → **Settings → Secrets**, paste the TOML values you use locally (see [Secrets & Configuration](#secrets--configuration)).

### 4) Reboot after changes
Whenever you change dependencies or secrets, use:
- **Manage app → Reboot app**
- If dependencies seem stuck, **Clear cache** then reboot.

---

## Secrets & Configuration

All configuration is read from Streamlit secrets and environment variables.

### Required (Ollama Cloud)
- `OLLAMA_HOST` (usually `https://ollama.com`)
- `OLLAMA_API_KEY`

### Recommended (Persistence)
- `DATABASE_URL` (Supabase Postgres connection string)

### Recommended (Uploads)
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_BUCKET` (e.g., `chat-uploads`)

### Auth
- `COOKIE_SECRET` (long random string)
- `BOOTSTRAP_PASSWORD` (admin bootstrap)

### Optional
- `MODEL_ALLOWLIST` (comma-separated model names). If set, only these models appear in the dropdown.

---

## Supabase Postgres Setup

### 1) Create a Supabase project
In Supabase dashboard:
- Create a project
- Save:
  - **Database connection string** (pooler preferred)
  - **Project URL** (`SUPABASE_URL`)
  - **Service role key** (`SUPABASE_SERVICE_ROLE_KEY`)

### 2) Connection string tip (pooler)
If you use Supabase pooler, it’s safest for hosted apps and avoids connection issues.

**Set `DATABASE_URL`** to the pooler connection string from the Supabase “Connect” panel.

### 3) App schema creation
On first run, the app initializes required tables automatically (users, assignments, conversations, messages, attachments, etc.).

> If you previously used SQLite and want to migrate, see Troubleshooting → Migration.

---

## Optional Supabase Storage for Uploads

Storing raw file bytes in Postgres is not ideal. Supabase Storage is recommended.

### 1) Create a bucket
Supabase → Storage → Create bucket:
- Name: `chat-uploads` (or any name)
- Private: recommended

### 2) Provide secrets
Set:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_BUCKET`

### 3) Behavior
- Uploads are stored in the bucket.
- DB stores a reference (path, mime, size) and the Storage key.
- Transcript downloads are text-only; images are excluded by design.

---

## Ollama Cloud Setup

### 1) API key
Create an Ollama API key in your Ollama account settings.

### 2) Host
Use:
- `OLLAMA_HOST = "https://ollama.com"`

### 3) Model dropdown
The app populates model choices by calling:
- `GET /api/tags` on the host

If you prefer to restrict student choices, set:
- `MODEL_ALLOWLIST = "model1,model2,model3"`

### Multimodal (images)
Ollama Cloud chat expects:
- `messages[].content` as **string**
- `messages[].images` as **list of base64 strings**

The app is implemented accordingly.

---

## Assignments & Prompts (Admin)

The Admin dashboard provides:
- **Assignments & Prompts** tab:
  - Choose active assignment
  - Create new assignment
  - Edit:
    - **Base system prompt** (applies to all assignments)
    - **Assignment-specific prompt** (applies only to selected assignment)

### Student UX
Students see:
- The active **Model**
- The active **Assignment** label in the left panel

---

## Conversation Browser & Transcripts (Admin)

Admin can browse:
- Any user’s conversations
- Full conversation view “as-is” (chat bubble format)
- **Download transcript** button:
  - Exports a `.txt` transcript (text only; images excluded)

> Student copy features are separate from admin transcript downloads.

---

## Clipboard Copy UX (Students + Admin Chat)

In chat view (for students and admin’s own chat):
- Per-message **copy icon** copies that message’s text to clipboard
- “copy whole conversation” copies the conversation text (images excluded)

No downloads are triggered for student copy actions.

---

## Troubleshooting

### A) `ModuleNotFoundError: psycopg`
Add to `requirements.txt`:
```txt
psycopg[binary]==3.3.2
```
Then commit/push and reboot Streamlit Cloud.

### B) `OperationalError: failed to resolve host @@db...`
Your `DATABASE_URL` is malformed (often an extra `@` or a password containing `@` that isn’t URL-encoded).
- Re-copy the connection string from Supabase dashboard.
- If password contains special chars, URL-encode them.

### C) `400 json: cannot unmarshal array into ... content of type string`
You sent OpenAI-style multimodal blocks to Ollama. Fix is:
- `messages[].content` must be a string
- images must be in `messages[].images` as base64

This project implements the correct Ollama schema.

### D) Password hash errors (`not a valid bcrypt hash`, 72-byte password)
If you migrated users from older versions:
- ensure the latest `lib/storage.py` is in place
- requirements pin passlib/bcrypt versions compatible with Streamlit Cloud

### E) Streamlit Cloud “redacted error”
Use the app’s built-in error panel (non-redacted snippet) and Streamlit “Manage app” logs.

### F) Migrating from SQLite → Supabase
If you previously ran local SQLite (`data/app.db`) and want to migrate:
- ask for the migration script, and we will generate a one-off safe migrator that copies:
  - users
  - assignments
  - conversations
  - messages
  - attachments metadata (and optionally uploads)

---

## Security Notes

- Do **not** commit:
  - `.streamlit/secrets.toml`
  - database files (`*.db`)
  - keys or tokens
- Keep Supabase Storage **private** for student content.
- Use Service Role Key only server-side (Streamlit secrets), never client-side.
- Consider rotating keys each semester.

---

## Repo Layout

Typical structure:

```
.
├── app.py
├── requirements.txt
├── README.md
└── lib
    ├── __init__.py
    ├── storage.py
    ├── attachments.py
    ├── supabase_storage.py
    ├── ollama_api.py
    └── render.py
```

---

## License / Attribution

Internal classroom tool for DS330. Customize as needed for your course policies.

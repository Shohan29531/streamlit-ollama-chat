import os
import streamlit as st
from streamlit_cookies_manager_ext import EncryptedCookieManager

from lib.ollama_api import chat_stream, list_models
from lib.render import render_chat_text
from lib.attachments import (
    is_image,
    image_bytes_to_b64,
    extract_text_from_file,
    truncate_text,
)
from lib.storage import (
    init_db,
    user_count,
    any_admin_exists,
    upsert_user,
    verify_user,
    list_users,
    get_setting,
    set_setting,
    create_session,
    get_session,
    delete_session,
    create_conversation,
    touch_conversation,
    list_conversations_for_user,
    list_conversations_with_counts_for_user,
    list_conversations_admin,
    get_conversation_messages,
    add_message,
    add_attachment,
    list_attachments_for_message_ids,
    update_message,
    delete_messages_after,
)

# ---------------- Config ----------------
st.set_page_config(page_title="Ollama Chat (Threads + History)", layout="wide")
init_db()

OLLAMA_HOST = st.secrets.get("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.environ.get("OLLAMA_API_KEY", None))
BOOTSTRAP_PASSWORD = st.secrets.get("BOOTSTRAP_PASSWORD", "change-this-now")

ACTIVE_MODEL_KEY = "active_model"
SYSTEM_PROMPT_KEY = "global_system_prompt"

DEFAULT_SYSTEM_PROMPT = "You are a helpful tutor. Be clear, concise, and accurate."
DEFAULT_MODEL_FALLBACKS = ["gpt-oss:20b", "gpt-oss:20b-cloud"]
MODEL_ALLOWLIST = st.secrets.get("MODEL_ALLOWLIST", os.environ.get("MODEL_ALLOWLIST", "")).strip()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_model_choices(host: str, api_key: str | None):
    """Cache /api/tags results to avoid re-fetching on every rerun."""
    return list_models(host, api_key)


# Attachments
MAX_ATTACHMENTS_PER_MESSAGE = 5
MAX_TOTAL_UPLOAD_MB = 20
MAX_FILE_TEXT_CHARS = 12_000
INCLUDE_FILE_TEXT_LAST_N_USER_MSGS = 3
INCLUDE_IMAGES_LAST_N_USER_MSGS = 1

# ---------------- Cookies ----------------
COOKIES_PASSWORD = st.secrets.get(
    "COOKIES_PASSWORD",
    os.environ.get("COOKIES_PASSWORD", "change-this-now-too"),
)
cookies = EncryptedCookieManager(prefix="streamlit-ollama-chat/", password=COOKIES_PASSWORD)
if not cookies.ready():
    st.stop()
COOKIE_SESSION_KEY = "session_token"


def load_active_model() -> str:
    return get_setting(ACTIVE_MODEL_KEY, DEFAULT_MODEL_FALLBACKS[0])


def load_system_prompt() -> str:
    return get_setting(SYSTEM_PROMPT_KEY, DEFAULT_SYSTEM_PROMPT)


def logout():
    tok = st.session_state.get("session_token") or cookies.get(COOKIE_SESSION_KEY)
    if tok:
        delete_session(tok)
    try:
        if cookies.get(COOKIE_SESSION_KEY) is not None:
            del cookies[COOKIE_SESSION_KEY]
            cookies.save()
    except Exception:
        pass

    st.session_state.auth = None
    st.session_state.session_token = None
    st.session_state.chat = []
    st.session_state.conversation_id = None
    st.session_state.draft_title = ""
    st.session_state.attachments_uploader = []
    st.session_state.editing_msg_id = None
    st.session_state.editing_idx = None
    st.session_state.editing_text = ""
    st.rerun()


def _render_attachments(attachments, key_prefix: str = "att"):
    """Render image/file attachments under a chat message."""
    if not attachments:
        return

    images = [a for a in attachments if a.get("kind") == "image"]
    files = [a for a in attachments if a.get("kind") == "file"]

    if images:
        for j, a in enumerate(images):
            st.image(a.get("data"), caption=a.get("filename"), use_column_width=True)

    if files:
        for j, a in enumerate(files):
            fname = a.get("filename")
            mime = a.get("mime")
            size_kb = round(len(a.get("data") or b"") / 1024)
            st.markdown(f"📎 **{fname}**  \\n+_({mime}, {size_kb} KB)_")

            # Download
            st.download_button(
                label=f"Download {fname}",
                data=a.get("data") or b"",
                file_name=fname,
                mime=mime or "application/octet-stream",
                key=f"{key_prefix}_dl_{j}_{fname}",
            )

            # Preview extracted text if available
            txt = a.get("text_content")
            if txt:
                preview, truncated = truncate_text(txt, 2000)
                with st.expander(f"Preview extracted text: {fname}", expanded=False):
                    st.code(preview)
                    if truncated:
                        st.caption("(Preview truncated)")


def _build_payload_messages(chat, system_prompt: str):
    """Convert session chat (with attachments) into Ollama /api/chat messages.

    Policy:
      - Always send full text history.
      - Only include extracted file text for the last N *user* messages.
      - Only include images for the last N *user* messages.
    """
    msgs = [{"role": "system", "content": system_prompt}]

    # Identify the last N user messages for file text / images.
    user_idxs = [i for i, m in enumerate(chat) if m.get("role") == "user"]
    file_text_allowed = set(user_idxs[-INCLUDE_FILE_TEXT_LAST_N_USER_MSGS :])
    images_allowed = set(user_idxs[-INCLUDE_IMAGES_LAST_N_USER_MSGS :])

    for i, m in enumerate(chat):
        role = m.get("role")
        content = m.get("content") or ""

        if role != "user":
            msgs.append({"role": role, "content": content})
            continue

        atts = m.get("attachments") or []
        file_blocks = []
        image_b64 = []
        for a in atts:
            if a.get("kind") == "file" and i in file_text_allowed:
                txt = a.get("text_content")
                if txt:
                    clipped, _ = truncate_text(txt, MAX_FILE_TEXT_CHARS)
                    file_blocks.append(f"[Attached file: {a.get('filename')}]\n{clipped}")
            if a.get("kind") == "image" and i in images_allowed:
                try:
                    image_b64.append(image_bytes_to_b64(a.get("data") or b""))
                except Exception:
                    pass

        if file_blocks:
            content = content + "\n\n" + "\n\n".join(file_blocks)

        msg_obj = {"role": "user", "content": content}
        if image_b64:
            msg_obj["images"] = image_b64
        msgs.append(msg_obj)

    return msgs


# ---------------- Session State ----------------
if "auth" not in st.session_state:
    st.session_state.auth = None

if "session_token" not in st.session_state:
    st.session_state.session_token = None

if "chat" not in st.session_state:
    # each item: {"id": int, "role": str, "content": str}
    st.session_state.chat = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "draft_title" not in st.session_state:
    st.session_state.draft_title = ""

# pending uploads (file_uploader stores its value in session_state)
if "attachments_uploader" not in st.session_state:
    st.session_state.attachments_uploader = []

# in-place editing state
if "editing_msg_id" not in st.session_state:
    st.session_state.editing_msg_id = None
if "editing_idx" not in st.session_state:
    st.session_state.editing_idx = None
if "editing_text" not in st.session_state:
    st.session_state.editing_text = ""


# ---------------- UI: Title ----------------
st.title("Ollama Chat")

# ---------------- First-time bootstrap ----------------
if user_count() == 0 and not any_admin_exists():
    st.subheader("First-time setup (Create Admin)")
    st.info("No users exist yet. Create the first admin account.")

    bootstrap_pw = st.text_input("Bootstrap password (from secrets.toml)", type="password")
    admin_id = st.text_input("New admin ID")
    admin_pass = st.text_input("New admin password", type="password")

    if st.button("Create Admin"):
        if bootstrap_pw != BOOTSTRAP_PASSWORD:
            st.error("Bootstrap password is incorrect.")
            st.stop()
        if not admin_id.strip() or not admin_pass:
            st.error("Admin ID and password are required.")
            st.stop()

        upsert_user(admin_id.strip(), admin_pass, "admin")

        if not get_setting(SYSTEM_PROMPT_KEY, ""):
            set_setting(SYSTEM_PROMPT_KEY, DEFAULT_SYSTEM_PROMPT)
        if not get_setting(ACTIVE_MODEL_KEY, ""):
            set_setting(ACTIVE_MODEL_KEY, DEFAULT_MODEL_FALLBACKS[0])

        st.success("Admin created. Please login now.")
    st.stop()

# ---------------- Auto-login from cookie ----------------
if st.session_state.auth is None:
    tok = cookies.get(COOKIE_SESSION_KEY)
    if tok:
        auth = get_session(tok)
        if auth:
            st.session_state.auth = auth
            st.session_state.session_token = tok
        else:
            try:
                del cookies[COOKIE_SESSION_KEY]
                cookies.save()
            except Exception:
                pass

# ---------------- Login gate ----------------
if st.session_state.auth is None:
    st.subheader("Login")
    user_id = st.text_input("ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        auth = verify_user(user_id.strip(), password)
        if auth:
            st.session_state.auth = auth
            tok = create_session(auth["user_id"], auth["role"], days=7)
            st.session_state.session_token = tok
            cookies[COOKIE_SESSION_KEY] = tok
            cookies.save()

            st.session_state.chat = []
            st.session_state.conversation_id = None
            st.session_state.draft_title = ""
            st.session_state.attachments_uploader = []
            st.session_state.editing_msg_id = None
            st.session_state.editing_idx = None
            st.session_state.editing_text = ""
            st.rerun()
        else:
            st.error("Invalid ID or password")
    st.stop()

current_user = st.session_state.auth["user_id"]
current_role = st.session_state.auth["role"]

# ---------------- Top bar ----------------
colL, colR = st.columns([4, 1])
with colL:
    st.caption(f"Logged in as: **{current_user}** • Role: **{current_role}**")
with colR:
    if st.button("Logout"):
        logout()

# ---------------- Sidebar: Navigation ----------------
st.sidebar.title("Navigation")
if current_role == "student":
    page = st.sidebar.radio("Page", ["Chat", "My History"], index=0)
else:
    page = st.sidebar.radio("Page", ["Chat", "Admin Dashboard"], index=0)

# ---------------- Sidebar: Admin controls ----------------
st.sidebar.caption(f"Model host: {OLLAMA_HOST}")

if ("ollama.com" in OLLAMA_HOST.lower()) and (not OLLAMA_API_KEY):
    st.sidebar.error("Missing OLLAMA_API_KEY for Ollama Cloud. Add it in Streamlit Secrets.")
    model_choices = DEFAULT_MODEL_FALLBACKS
else:
    try:
        model_choices = _cached_model_choices(OLLAMA_HOST, OLLAMA_API_KEY) or DEFAULT_MODEL_FALLBACKS
    except Exception as e:
        if "localhost" in OLLAMA_HOST or "127.0.0.1" in OLLAMA_HOST:
            st.sidebar.warning(
                "Could not reach the local Ollama server. If this is a Streamlit Cloud deployment, "
                'set OLLAMA_HOST to "https://ollama.com" and provide OLLAMA_API_KEY in Secrets.'
            )
        else:
            st.sidebar.warning(f"Could not fetch model list: {e}")
        model_choices = DEFAULT_MODEL_FALLBACKS

# Optional: restrict dropdown to a comma-separated allowlist (useful for classroom deployments)
if MODEL_ALLOWLIST:
    allowed = {m.strip() for m in MODEL_ALLOWLIST.split(",") if m.strip()}
    model_choices = [m for m in model_choices if m in allowed] or model_choices


active_model = load_active_model()
system_prompt = load_system_prompt()

if current_role == "admin":
    st.sidebar.divider()
    st.sidebar.subheader("Active Model (Admin-only)")
    new_model = st.sidebar.selectbox(
        "Model for everyone",
        options=model_choices,
        index=model_choices.index(active_model) if active_model in model_choices else 0,
    )
    if st.sidebar.button("Save active model"):
        set_setting(ACTIVE_MODEL_KEY, new_model)
        st.toast("Saved active model", icon="✅")
        st.rerun()

    st.sidebar.subheader("System Prompt (Admin-only)")
    new_prompt = st.sidebar.text_area("Edit system prompt", value=system_prompt, height=180)
    if st.sidebar.button("Save system prompt"):
        set_setting(SYSTEM_PROMPT_KEY, new_prompt)
        st.toast("Saved system prompt", icon="✅")
        st.rerun()

    if st.sidebar.button("Reset prompt to code default"):
        set_setting(SYSTEM_PROMPT_KEY, DEFAULT_SYSTEM_PROMPT)
        st.toast("Reset to code default", icon="🧹")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("User Management")
    with st.sidebar.expander("Create / Reset User", expanded=False):
        u_id = st.text_input("User ID", key="um_user")
        u_pw = st.text_input("Password", type="password", key="um_pass")
        u_role = st.selectbox("Role", ["student", "admin"], key="um_role")
        if st.button("Create/Update user"):
            if not u_id.strip() or not u_pw:
                st.sidebar.error("User ID + password required")
            else:
                upsert_user(u_id.strip(), u_pw, u_role)
                st.sidebar.success("User saved")

    with st.sidebar.expander("List users", expanded=False):
        rows = list_users()
        for r in rows:
            st.write(f"- {r['user_id']} ({r['role']})")
else:
    st.sidebar.divider()
    st.sidebar.subheader("Active Model")
    st.sidebar.write(f"**{active_model}**")

# ---------------- Sidebar: Thread selector (Chat only) ----------------
if page == "Chat":
    st.sidebar.divider()
    st.sidebar.subheader("Conversation Threads")

    threads = list_conversations_for_user(current_user, limit=200)
    thread_options = ["➕ New conversation"] + [
        f"#{t['id']} • {t['title'] or '(untitled)'} • {t['updated_at']}"
        for t in threads
    ]

    default_index = 0
    if st.session_state.conversation_id is not None:
        match = [i for i, t in enumerate(threads, start=1) if t["id"] == st.session_state.conversation_id]
        if match:
            default_index = match[0]

    selected = st.sidebar.selectbox("Select a thread", thread_options, index=default_index)

    if selected.startswith("➕"):
        if st.sidebar.button("Start fresh thread"):
            st.session_state.conversation_id = None
            st.session_state.chat = []
            st.session_state.draft_title = ""
            st.session_state.attachments_uploader = []
            st.session_state.editing_msg_id = None
            st.session_state.editing_idx = None
            st.session_state.editing_text = ""
            st.rerun()
    else:
        selected_id = int(selected.split("•")[0].strip().lstrip("#"))
        if selected_id != st.session_state.conversation_id:
            msgs = get_conversation_messages(selected_id)
            mid_list = [int(m["id"]) for m in msgs]
            att_map = list_attachments_for_message_ids(mid_list)
            st.session_state.conversation_id = selected_id
            st.session_state.chat = [
                {
                    "id": m["id"],
                    "role": m["role"],
                    "content": m["content"],
                    "attachments": att_map.get(int(m["id"]), []),
                }
                for m in msgs
            ]
            st.session_state.draft_title = ""
            st.session_state.attachments_uploader = []
            st.session_state.editing_msg_id = None
            st.session_state.editing_idx = None
            st.session_state.editing_text = ""
            st.rerun()

    st.sidebar.caption("Students cannot change the model. Admin controls it globally.")

# ---------------- Page: Chat ----------------
if page == "Chat":
    active_model = load_active_model()
    system_prompt = load_system_prompt()

    st.caption(f"Ollama host: {OLLAMA_HOST} • Active model: {active_model}")

    st.subheader("Thread Title")
    st.session_state.draft_title = st.text_input(
        "Optional (helps you find it later)",
        value=st.session_state.draft_title,
        placeholder="e.g., Week 3 Homework Help",
    )

    # Render chat with small "Edit" action on USER messages
    for i, msg in enumerate(st.session_state.chat):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_chat_text(msg["content"])
            else:
                st.markdown(msg["content"])
                _render_attachments(msg.get("attachments"), key_prefix=f"m{msg.get('id')}")

            if msg["role"] == "user" and msg.get("id") is not None and st.session_state.conversation_id is not None:
                cols = st.columns([1, 9])
                with cols[0]:
                    if st.button("✏️ Edit", key=f"edit_btn_{msg['id']}"):
                        st.session_state.editing_msg_id = msg["id"]
                        st.session_state.editing_idx = i
                        st.session_state.editing_text = msg["content"]
                        st.rerun()

    # --- Edit panel (ChatGPT-like behavior: modify in place, delete later turns, regenerate) ---
    if st.session_state.editing_msg_id is not None:
        st.divider()
        st.subheader("Editing a past prompt")
        st.caption("Saving will overwrite that prompt, delete all later messages in this thread, and regenerate from there.")

        new_text = st.text_area(
            "Edit your prompt",
            value=st.session_state.editing_text,
            height=160,
            key="edit_panel_text",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Save & Regenerate", type="primary"):
                conv_id = st.session_state.conversation_id
                msg_id = st.session_state.editing_msg_id
                idx = st.session_state.editing_idx

                # 1) Update the message in DB
                update_message(msg_id, new_text)

                # 2) Delete everything after it in DB
                delete_messages_after(conv_id, msg_id)

                # 3) Truncate session chat and overwrite content in place
                st.session_state.chat[idx]["content"] = new_text
                st.session_state.chat = st.session_state.chat[: idx + 1]

                # 4) Regenerate assistant from truncated history
                payload_messages = _build_payload_messages(st.session_state.chat, system_prompt)

                with st.chat_message("assistant"):
                    ph = st.empty()
                    full = ""
                    try:
                        for chunk in chat_stream(
                            host=OLLAMA_HOST,
                            api_key=OLLAMA_API_KEY,
                            model=active_model,
                            messages=payload_messages,
                            stream=True,
                        ):
                            full += chunk
                            ph.empty()
                            with ph.container():
                                render_chat_text(full)
                    except Exception as e:
                        st.error(f"Chat failed: {e}")
                        full = ""

                if full.strip():
                    asst_id = add_message(conv_id, "assistant", full)
                    st.session_state.chat.append({"id": asst_id, "role": "assistant", "content": full})
                    touch_conversation(
                        conv_id,
                        title=st.session_state.draft_title.strip() or None,
                        model=active_model,
                        system_prompt=system_prompt,
                    )
                    st.toast("Updated + regenerated ✅", icon="✅")

                # clear edit state
                st.session_state.editing_msg_id = None
                st.session_state.editing_idx = None
                st.session_state.editing_text = ""
                st.rerun()

        with c2:
            if st.button("Cancel"):
                st.session_state.editing_msg_id = None
                st.session_state.editing_idx = None
                st.session_state.editing_text = ""
                st.rerun()

    # Normal input (disable while editing to avoid confusion)
    if st.session_state.editing_msg_id is None:
        submission = st.chat_input(
            "Type your message…",
            accept_file="multiple",
            file_type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
                "gif",
                "pdf",
                "docx",
                "txt",
                "md",
                "csv",
                "json",
            ],
            key="chat_input_box",
        )
    else:
        submission = None

    if submission:
        text = (getattr(submission, 'text', '') or '').strip()
        uploads = list(getattr(submission, 'files', []) or [])
        if (not text) and uploads:
            text = 'Please analyze the attached file(s).'

        # ---- Collect attachments selected for this message ----
        if len(uploads) > MAX_ATTACHMENTS_PER_MESSAGE:
            st.error(f"Too many attachments. Max is {MAX_ATTACHMENTS_PER_MESSAGE}.")
            st.stop()

        total_bytes = 0
        pending_attachments = []
        for uf in uploads:
            data = uf.getvalue()
            total_bytes += len(data)
            mime = getattr(uf, "type", None) or "application/octet-stream"
            filename = getattr(uf, "name", "uploaded")

            if is_image(mime, filename):
                pending_attachments.append(
                    {
                        "kind": "image",
                        "filename": filename,
                        "mime": mime,
                        "data": data,
                        "text_content": None,
                    }
                )
            else:
                txt = extract_text_from_file(filename, mime, data)
                pending_attachments.append(
                    {
                        "kind": "file",
                        "filename": filename,
                        "mime": mime,
                        "data": data,
                        "text_content": txt,
                    }
                )

        if total_bytes > MAX_TOTAL_UPLOAD_MB * 1024 * 1024:
            st.error(f"Attachments are too large. Total must be <= {MAX_TOTAL_UPLOAD_MB} MB.")
            st.stop()

        if st.session_state.conversation_id is None:
            title = st.session_state.draft_title.strip()
            if not title:
                title = (text[:60] or "(untitled)")
            st.session_state.conversation_id = create_conversation(
                user_id=current_user,
                role=current_role,
                title=title,
                model=active_model,
                system_prompt=system_prompt,
            )

        conv_id = st.session_state.conversation_id

        touch_conversation(
            conv_id,
            title=st.session_state.draft_title.strip() or None,
            model=active_model,
            system_prompt=system_prompt,
        )

        user_msg_id = add_message(conv_id, "user", text)

        # Persist attachments (if any)
        saved_atts = []
        for a in pending_attachments:
            try:
                add_attachment(
                    message_id=user_msg_id,
                    kind=a["kind"],
                    filename=a["filename"],
                    mime=a["mime"],
                    data=a["data"],
                    text_content=a.get("text_content"),
                    user_id=current_user,
                    conversation_id=conv_id,
                )
            except Exception:
                # If persistence fails, still allow the chat to proceed.
                pass
            saved_atts.append(a)

        st.session_state.chat.append(
            {"id": user_msg_id, "role": "user", "content": text, "attachments": saved_atts}
        )

        with st.chat_message("user"):
            st.markdown(text)
            _render_attachments(saved_atts, key_prefix=f"send_{user_msg_id}")

        payload_messages = _build_payload_messages(st.session_state.chat, system_prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            try:
                for chunk in chat_stream(
                    host=OLLAMA_HOST,
                    api_key=OLLAMA_API_KEY,
                    model=active_model,
                    messages=payload_messages,
                    stream=True,
                ):
                    full += chunk
                    placeholder.empty()
                    with placeholder.container():
                        render_chat_text(full)
            except Exception as e:
                st.error(f"Chat failed: {e}")
                full = ""

        if full.strip():
            asst_id = add_message(conv_id, "assistant", full)
            st.session_state.chat.append({"id": asst_id, "role": "assistant", "content": full})
            touch_conversation(
                conv_id,
                title=st.session_state.draft_title.strip() or None,
                model=active_model,
                system_prompt=system_prompt,
            )
            st.toast("Auto-saved ✅", icon="💾")

        # Clear uploader state after send
        st.session_state.attachments_uploader = []

# ---------------- Page: Student History ----------------
if page == "My History":
    st.header("My History")
    rows = list_conversations_with_counts_for_user(current_user, limit=500)
    if not rows:
        st.info("No conversations yet. Go to Chat and start one.")
    else:
        options = [
            f"#{r['id']} • {r['title'] or '(untitled)'} • msgs:{r['msg_count']} • updated:{r['updated_at']}"
            for r in rows
        ]
        sel = st.selectbox("Choose a thread", options)

        conv_id = int(sel.split("•")[0].strip().lstrip("#"))
        msgs = get_conversation_messages(conv_id)
        mid_list = [int(m["id"]) for m in msgs]
        att_map = list_attachments_for_message_ids(mid_list)

        st.subheader(f"Thread #{conv_id}")
        export_lines = []
        for m in msgs:
            with st.chat_message(m["role"]):
                if m["role"] == "assistant":
                    render_chat_text(m["content"])
                else:
                    st.markdown(m["content"])
                    _render_attachments(att_map.get(int(m["id"]), []), key_prefix=f"hist_{m['id']}")
            export_lines.append(f"{m['role'].upper()}: {m['content']}")

        st.download_button(
            label="Download transcript (.txt)",
            data="\n\n".join(export_lines),
            file_name=f"{current_user}_conversation_{conv_id}.txt",
            mime="text/plain",
        )

        if st.button("Open this thread in Chat"):
            st.session_state.conversation_id = conv_id
            mid_list = [int(m["id"]) for m in msgs]
            att_map = list_attachments_for_message_ids(mid_list)
            st.session_state.chat = [
                {
                    "id": m["id"],
                    "role": m["role"],
                    "content": m["content"],
                    "attachments": att_map.get(int(m["id"]), []),
                }
                for m in msgs
            ]
            st.session_state.draft_title = ""
            st.session_state.attachments_uploader = []
            st.session_state.editing_msg_id = None
            st.session_state.editing_idx = None
            st.session_state.editing_text = ""
            st.rerun()

# ---------------- Page: Admin Dashboard ----------------
if page == "Admin Dashboard":
    st.header("Admin Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        ufilter = st.text_input("Filter user contains", value="")
    with col2:
        rfilter = st.selectbox("Role", ["", "student", "admin"], index=0)
    with col3:
        mfilter = st.text_input("Model (exact)", value="")

    rows = list_conversations_admin(
        user_filter=ufilter or None,
        role_filter=rfilter or None,
        model_filter=mfilter or None,
        limit=300,
    )

    if not rows:
        st.write("No saved chats found.")
    else:
        options = [
            f"#{r['id']} • {r['user_id']} • {r['role']} • {r['title'] or '(untitled)'} • {r['model']} • {r['updated_at']}"
            for r in rows
        ]
        sel = st.selectbox("Pick a conversation", options)
        conv_id = int(sel.split("•")[0].strip().lstrip("#"))

        msgs = get_conversation_messages(conv_id)
        mid_list = [int(m["id"]) for m in msgs]
        att_map = list_attachments_for_message_ids(mid_list)
        export_lines = []
        for m in msgs:
            with st.chat_message(m["role"]):
                if m["role"] == "assistant":
                    render_chat_text(m["content"])
                else:
                    st.markdown(m["content"])
                    _render_attachments(att_map.get(int(m["id"]), []), key_prefix=f"adm_{m['id']}")
            export_lines.append(f"{m['role'].upper()}: {m['content']}")

        st.download_button(
            label="Download transcript (.txt)",
            data="\n\n".join(export_lines),
            file_name=f"conversation_{conv_id}.txt",
            mime="text/plain",
        )

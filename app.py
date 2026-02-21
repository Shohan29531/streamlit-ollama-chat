import os
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from streamlit_cookies_manager_ext import EncryptedCookieManager

from lib.ollama_api import chat_stream, list_models
from lib.render import render_chat_text
from lib.attachments import (
    detect_kind,
    extract_text_from_bytes,
    is_image_mime,
    to_data_url,
)
from lib.storage import (
    init_db,
    any_admin_exists,
    upsert_user,
    verify_user,
    create_session,
    get_session,
    delete_session,
    list_users,
    get_setting,
    set_setting,
    get_base_system_prompt,
    set_base_system_prompt,
    list_assignments,
    add_assignment,
    get_active_assignment,
    set_active_assignment,
    update_assignment_prompt,
    get_assignment,
    create_conversation,
    touch_conversation,
    get_conversation,
    list_conversations_for_user,
    list_conversations_admin,
    get_conversation_messages,
    add_message,
    update_message,
    delete_messages_after,
    add_attachment,
    list_attachments_for_message_ids,
)

# ---------------- App config ----------------

APP_NAME = "DS330 Chat"

DEFAULT_BASE_PROMPT = """You are a helpful assistant for DS330. Follow the course rules and be concise, correct, and student-friendly."""

# If the model supports it (e.g., gpt-oss), enable extended thinking by default.
DEFAULT_THINK = "high"  # gpt-oss: low|medium|high


def _secret(key: str, default: Optional[str] = None) -> Optional[str]:
    if key in st.secrets:
        v = st.secrets.get(key)
        return str(v) if v is not None else default
    return os.environ.get(key, default)


OLLAMA_HOST = _secret("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_KEY = _secret("OLLAMA_API_KEY")

MODEL_SETTING_KEY = "active_model"

st.set_page_config(page_title=APP_NAME, page_icon="💬", layout="wide")

# Tighten sidebar spacing + slightly widen it
st.markdown(
    """
<style>
  /* Reduce extra top padding */
  section[data-testid="stSidebar"] > div {
    padding-top: 0.25rem;
  }

  /* Make the sidebar a bit wider on desktop (no manual margin-left hacks) */
  @media (min-width: 900px) {
    section[data-testid="stSidebar"] {
      width: 360px !important;
      min-width: 360px !important;
      max-width: 360px !important;
    }
    section[data-testid="stSidebar"] > div {
      width: 360px !important;
    }
  }

  /* Use the full main-area width (removes large empty gap on wide screens) */
  div.block-container {
    max-width: 100% !important;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
  }
</style>
""",
    unsafe_allow_html=True,
)
# ---------------- Clipboard button (per-message + whole convo) ----------------


def _copy_button(text: str, key: str, tooltip: str = "Copy") -> None:
    """Render a small copy-to-clipboard button (no downloads)."""
    # Primary: st-copy-button component (best UX)
    try:
        from st_copy_button import st_copy_button  # type: ignore
        import inspect

        sig = inspect.signature(st_copy_button)
        kwargs = dict(
            text=text,
            before_copy_label="📋",
            after_copy_label="✅",
            show_text=False,
        )
        if "key" in sig.parameters:
            kwargs["key"] = key
        if "help" in sig.parameters:
            kwargs["help"] = tooltip
        st_copy_button(**kwargs)
        return
    except Exception:
        pass

    # Fallback: lightweight JS copy button (still copies to clipboard; no downloads)
    try:
        import streamlit.components.v1 as components
        import json as _json
        payload = _json.dumps(text)
        html = f"""
        <div style="display:flex;align-items:center;justify-content:flex-end;">
          <button id="{key}" title="{tooltip}"
            style="border:none;background:transparent;cursor:pointer;padding:0;margin:0;font-size:16px;line-height:1;">
            📋
          </button>
        </div>
        <script>
          const btn = document.getElementById("{key}");
          btn.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText({payload});
              const old = btn.textContent;
              btn.textContent = "✅";
              setTimeout(() => btn.textContent = old, 900);
            }} catch (e) {{
              console.error(e);
            }}
          }});
        </script>
        """
        components.html(html, height=26)
    except Exception:
        # As a last resort, render nothing (never offer a download fallback).
        return


# ---------------- Cookies / Auth ----------------


cookies = EncryptedCookieManager(
    prefix="ds330_chat",
    password=_secret("COOKIE_PASSWORD", "change-me"),
)

if not cookies.ready():
    st.stop()


def _login(user_id: str, role: str) -> None:
    token = create_session(user_id, role)
    cookies["session_token"] = token
    cookies.save()
    st.session_state["user_id"] = user_id
    st.session_state["role"] = role
    st.session_state["session_token"] = token


def _logout() -> None:
    token = st.session_state.get("session_token") or cookies.get("session_token")
    if token:
        try:
            delete_session(token)
        except Exception:
            pass
    cookies["session_token"] = ""
    cookies.save()
    for k in ["user_id", "role", "session_token", "conversation_id", "messages", "conversation_meta"]:
        st.session_state.pop(k, None)


# ---------------- DB init ----------------

init_db()


# ---------------- Session restore ----------------

if "user_id" not in st.session_state:
    token = cookies.get("session_token")
    if token:
        sess = get_session(token)
        if sess:
            st.session_state["user_id"] = sess["user_id"]
            st.session_state["role"] = sess["role"]
            st.session_state["session_token"] = sess["token"]


# ---------------- Model list (Ollama Cloud) ----------------

@st.cache_data(ttl=60, show_spinner=False)
def _cached_models() -> List[str]:
    try:
        return list_models(OLLAMA_HOST, OLLAMA_API_KEY)
    except Exception:
        return []


def _load_active_model(models: List[str]) -> str:
    saved = get_setting(MODEL_SETTING_KEY, None)
    if saved and saved in models:
        return saved
    if models:
        set_setting(MODEL_SETTING_KEY, models[0])
        return models[0]
    return ""


# ---------------- Prompts / Assignments ----------------


def _combined_prompt(base_prompt: str, assignment_prompt: str) -> str:
    base = (base_prompt or "").strip()
    ap = (assignment_prompt or "").strip()
    if not base:
        base = DEFAULT_BASE_PROMPT.strip()
    return base + ("\n\n" + ap if ap else "")


def _active_assignment_label() -> str:
    a = get_active_assignment()
    return a.get("name") or "(no assignment)"


# ---------------- Helpers ----------------


def _load_conversation_into_state(conversation_id: int) -> None:
    conv = get_conversation(conversation_id)
    if not conv:
        st.session_state.pop("conversation_id", None)
        st.session_state["messages"] = []
        st.session_state["conversation_meta"] = {}
        return

    msgs = get_conversation_messages(conversation_id)
    # Attachments
    mids = [m["id"] for m in msgs]
    att_map = list_attachments_for_message_ids(mids)

    ui_msgs: List[Dict[str, Any]] = []
    for m in msgs:
        ui_msgs.append(
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "attachments": att_map.get(m["id"], []),
            }
        )

    st.session_state["conversation_id"] = conversation_id
    st.session_state["messages"] = ui_msgs
    st.session_state["conversation_meta"] = conv


def _conversation_to_text(messages: List[Dict[str, Any]]) -> str:
    """Plain-text transcript (no images)."""
    lines: List[str] = []
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_payload_messages(conversation_id: int) -> List[Dict[str, Any]]:
    """Build messages payload for Ollama /api/chat."""
    conv = st.session_state.get("conversation_meta") or get_conversation(conversation_id) or {}

    # Prefer per-conversation snapshot prompt (for reproducibility)
    base_prompt = (conv.get("base_prompt") or "").strip()
    ap = (conv.get("assignment_prompt") or "").strip()
    sys_prompt = (conv.get("system_prompt") or "").strip()

    if base_prompt or ap:
        system_prompt = _combined_prompt(base_prompt, ap)
    elif sys_prompt:
        system_prompt = sys_prompt
    else:
        # fallback to current global settings
        base = get_base_system_prompt(DEFAULT_BASE_PROMPT)
        ap2 = get_active_assignment().get("prompt") or ""
        system_prompt = _combined_prompt(base, ap2)

    payload: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    for m in st.session_state.get("messages", []):
        role = m["role"]
        content = m.get("content") or ""

        if role == "user":
            # Add images inline for the model
            blocks: List[Dict[str, Any]] = []
            if content.strip():
                blocks.append({"type": "text", "text": content})

            for att in (m.get("attachments") or []):
                if att.get("kind") == "image" and att.get("data"):
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": to_data_url(att["mime"], att["data"])},
                        }
                    )
                elif att.get("kind") == "file" and att.get("text_content"):
                    blocks.append(
                        {
                            "type": "text",
                            "text": f"\n\n[Attached file: {att.get('filename','file')}\n{att.get('text_content','')}]\n",
                        }
                    )

            if blocks:
                payload.append({"role": "user", "content": blocks})
            else:
                payload.append({"role": "user", "content": ""})
        else:
            payload.append({"role": "assistant", "content": content})

    return payload


# ---------------- Login screen ----------------


def _render_login() -> None:
    st.title(APP_NAME)

    if not any_admin_exists():
        st.info("No admin account exists yet. Create the first admin below.")
        with st.form("bootstrap_admin"):
            user_id = st.text_input("Admin user ID")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create admin")
        if submitted:
            if not user_id or not password:
                st.error("User ID and password are required.")
            else:
                upsert_user(user_id, password, "admin")
                _login(user_id, "admin")
                st.rerun()
        return

    st.subheader("Sign in")
    with st.form("login_form"):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        auth = verify_user(user_id, password)
        if not auth:
            st.error("Invalid credentials")
            return
        _login(auth["user_id"], auth["role"])
        st.rerun()


# ---------------- Sidebar ----------------


def _sidebar(models: List[str]) -> Tuple[str, str, Dict[str, Any]]:
    user_id = st.session_state["user_id"]
    role = st.session_state["role"]
    is_admin = role == "admin"

    st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.caption(f"Signed in as **{user_id}** ({role})")

    # Active assignment (students can see; admin can change)
    active_assignment = get_active_assignment()
    st.sidebar.caption(f"**Active assignment:** {active_assignment.get('name')}")

    # Model selection
    active_model = _load_active_model(models)
    if is_admin:
        if models:
            sel = st.sidebar.selectbox("Active model", models, index=models.index(active_model))
            if sel != active_model:
                set_setting(MODEL_SETTING_KEY, sel)
                active_model = sel
        else:
            st.sidebar.warning("No models found. Check OLLAMA_HOST / API key.")

        # Assignment selection (admin only)
        assignments = list_assignments()
        if assignments:
            id_to_name = {int(a["id"]): a["name"] for a in assignments}
            ids = list(id_to_name.keys())
            active_id = int(active_assignment["id"])
            idx = ids.index(active_id) if active_id in ids else 0
            new_id = st.sidebar.selectbox(
                "Set active assignment",
                ids,
                format_func=lambda i: id_to_name.get(int(i), str(i)),
                index=idx,
            )
            if int(new_id) != active_id:
                set_active_assignment(int(new_id))
                st.rerun()

    else:
        st.sidebar.caption(f"**Active model:** {active_model}")

    st.sidebar.divider()
    page = st.sidebar.radio("", ["Chat", "Admin Dashboard"] if is_admin else ["Chat"], index=0, key="nav_page")

    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        _logout()
        st.rerun()

    return page, active_model, active_assignment


# ---------------- Chat UI ----------------


def _render_attachments(attachments: List[Dict[str, Any]]) -> None:
    if not attachments:
        return

    images = [a for a in attachments if a.get("kind") == "image" and a.get("data")]
    files = [a for a in attachments if a.get("kind") == "file"]

    if images:
        st.caption("Attachments")
        for img in images:
            st.image(img["data"], caption=img.get("filename"), use_container_width=True)

    if files:
        st.caption("Attachments")
        for f in files:
            name = f.get("filename") or "file"
            st.markdown(f"- **{name}**")


def _render_message(role: str, content: str) -> None:
    if role == "assistant":
        render_chat_text(content)
    else:
        st.markdown(content)


def _chat_page(active_model: str, active_assignment: Dict[str, Any]) -> None:
    user_id = st.session_state["user_id"]
    role = st.session_state["role"]
    is_admin = role == "admin"

    st.title(APP_NAME)

    # Thread title (saved per conversation)
    conv_id_existing = st.session_state.get("conversation_id")
    if conv_id_existing:
        meta = st.session_state.get("conversation_meta") or {}
        current_title = meta.get("title") or ""
        tcols = st.columns([0.75, 0.25])
        with tcols[0]:
            new_title = st.text_input("Thread title", value=current_title, key="thread_title")
        with tcols[1]:
            if st.button("Save title", key="save_title"):
                touch_conversation(int(conv_id_existing), title=new_title)
                st.session_state["conversation_meta"] = get_conversation(int(conv_id_existing)) or {}
                st.rerun()

        # Snapshot metadata
        if meta.get("assignment_name"):
            st.caption(f"This thread uses assignment: **{meta.get('assignment_name')}** · Model: **{meta.get('model')}**")


    # Ensure state
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_meta", {})

    # Sidebar thread picker
    with st.sidebar:
        st.markdown("#### Conversations")
        convs = list_conversations_for_user(user_id)
        options = [None] + [c["id"] for c in convs]
        labels = {None: "➕ New conversation"}
        for c in convs:
            title = c.get("title") or f"Conversation {c['id']}"
            # Optional: include assignment tag in list (helps admins)
            if c.get("assignment_name"):
                title = f"{title} · {c['assignment_name']}"
            labels[c["id"]] = title

        current = st.session_state.get("conversation_id")
        idx = options.index(current) if current in options else 0
        picked = st.selectbox("", options, index=idx, format_func=lambda x: labels.get(x, str(x)))
        if picked != current:
            if picked is None:
                st.session_state.pop("conversation_id", None)
                st.session_state["messages"] = []
                st.session_state["conversation_meta"] = {}
            else:
                _load_conversation_into_state(int(picked))
            st.rerun()

    # Render chat history
    msgs = st.session_state.get("messages", [])
    last_assistant_idx = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)

    for i, m in enumerate(msgs):
        with st.chat_message(m["role"]):
            _render_message(m["role"], m.get("content") or "")
            _render_attachments(m.get("attachments") or [])

            # Per-message copy button (one per message)
            ccols = st.columns([0.92, 0.08])
            with ccols[1]:
                _copy_button(m.get("content") or "", key=f"copy_msg_{m.get('id','x')}", tooltip="Copy this message")

            # Copy whole conversation (bottom of last assistant response)
            if i == last_assistant_idx:
                tcols = st.columns([0.80, 0.20])
                with tcols[1]:
                    _copy_button(
                        _conversation_to_text(msgs),
                        key=f"copy_conv_{st.session_state.get('conversation_id','new')}",
                        tooltip="Copy the full conversation (text only)",
                    )

            # Conversation edit controls — ADMIN ONLY
            if is_admin and m["role"] == "user":
                edit_key = f"edit_btn_{m['id']}"
                if st.button("✏️ Edit", key=edit_key, help="Edit this user message and regenerate from here"):
                    st.session_state["editing"] = {
                        "message_id": m["id"],
                        "original": m.get("content") or "",
                        "draft": m.get("content") or "",
                    }
                    st.rerun()

    # Admin edit panel (admin only)
    if is_admin and st.session_state.get("editing"):
        ed = st.session_state["editing"]
        st.info("Admin edit mode: update the message and regenerate from that point.")
        ed["draft"] = st.text_area("Edit user message", value=ed["draft"], height=140)
        bcols = st.columns([0.25, 0.25, 0.5])
        if bcols[0].button("Save & Regenerate", type="primary"):
            conv_id = st.session_state.get("conversation_id")
            if not conv_id:
                st.session_state["editing"] = None
                st.rerun()

            update_message(ed["message_id"], ed["draft"])
            delete_messages_after(int(conv_id), int(ed["message_id"]))

            # Reload from DB
            _load_conversation_into_state(int(conv_id))

            # Regenerate assistant response
            with st.chat_message("assistant"):
                ph = st.empty()
                full = ""
                payload = _build_payload_messages(int(conv_id))
                for chunk in chat_stream(
                    host=OLLAMA_HOST,
                    api_key=OLLAMA_API_KEY,
                    model=active_model,
                    messages=payload,
                    options=None,
                    think=DEFAULT_THINK,
                ):
                    full += chunk
                    ph.markdown(full)

            add_message(int(conv_id), "assistant", full)
            touch_conversation(int(conv_id), model=active_model)
            _load_conversation_into_state(int(conv_id))
            st.session_state["editing"] = None
            st.rerun()

        if bcols[1].button("Cancel"):
            st.session_state["editing"] = None
            st.rerun()

    # Chat input with files (ChatGPT-style)
    prompt_val = st.chat_input(
        f"Message {APP_NAME}",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "pdf", "txt", "md", "csv", "json", "docx"],
    )

    if not prompt_val:
        return

    # Streamlit returns either string (older versions) or ChatInputValue (newer)
    if isinstance(prompt_val, str):
        user_text = prompt_val
        files = []
    else:
        user_text = (prompt_val.text or "")
        files = list(prompt_val.files or [])

    # Ensure conversation exists
    conv_id = st.session_state.get("conversation_id")
    if not conv_id:
        base_prompt = get_base_system_prompt(DEFAULT_BASE_PROMPT)
        assignment_prompt = active_assignment.get("prompt") or ""
        sys_prompt = _combined_prompt(base_prompt, assignment_prompt)

        conv_id = create_conversation(
            user_id=user_id,
            role=role,
            title="New conversation",
            model=active_model,
            system_prompt=sys_prompt,
            base_prompt=base_prompt,
            assignment_id=int(active_assignment["id"]),
            assignment_name=active_assignment.get("name"),
            assignment_prompt=assignment_prompt,
        )
        st.session_state["conversation_id"] = conv_id
        st.session_state["conversation_meta"] = get_conversation(int(conv_id)) or {}

    # Persist user message
    user_msg_id = add_message(int(conv_id), "user", user_text)

    # Handle files -> attachments
    attachments: List[Dict[str, Any]] = []
    for f in files:
        raw = f.getvalue()
        mime = getattr(f, "type", "application/octet-stream")
        filename = getattr(f, "name", "file")
        kind = "image" if is_image_mime(mime) else "file"

        text_content = None
        if kind == "file":
            try:
                text_content = extract_text_from_bytes(filename, raw)
            except Exception:
                text_content = None

        add_attachment(
            message_id=user_msg_id,
            kind=kind,
            filename=filename,
            mime=mime,
            data=raw,
            text_content=text_content,
        )

        attachments.append(
            {
                "kind": kind,
                "filename": filename,
                "mime": mime,
                "data": raw,
                "text_content": text_content,
            }
        )

    # Add to UI state
    st.session_state["messages"].append(
        {"id": user_msg_id, "role": "user", "content": user_text, "attachments": attachments}
    )

    # Render user message
    with st.chat_message("user"):
        st.markdown(user_text)
        _render_attachments(attachments)
        # per message copy
        ccols = st.columns([0.92, 0.08])
        with ccols[1]:
            _copy_button(user_text, key=f"copy_msg_{user_msg_id}")

    # Build payload and stream assistant
    payload = _build_payload_messages(int(conv_id))
    with st.chat_message("assistant"):
        ph = st.empty()
        full = ""
        for chunk in chat_stream(
            host=OLLAMA_HOST,
            api_key=OLLAMA_API_KEY,
            model=active_model,
            messages=payload,
            options=None,
            think=DEFAULT_THINK,
        ):
            full += chunk
            ph.markdown(full)

    asst_id = add_message(int(conv_id), "assistant", full)

    # Update UI state
    st.session_state["messages"].append(
        {"id": asst_id, "role": "assistant", "content": full, "attachments": []}
    )

    touch_conversation(int(conv_id), model=active_model)
    st.rerun()


# ---------------- Admin dashboard ----------------


def _admin_dashboard(active_model: str) -> None:
    st.title("Admin Dashboard")

    # Assignment & prompt editor
    st.subheader("Assignments & System Prompts")

    active = get_active_assignment()
    assignments = list_assignments()

    name_by_id = {int(a["id"]): a["name"] for a in assignments}
    ids = list(name_by_id.keys())
    idx = ids.index(int(active["id"])) if int(active["id"]) in ids else 0

    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        new_active = st.selectbox(
            "Active assignment",
            ids,
            index=idx,
            format_func=lambda i: name_by_id.get(int(i), str(i)),
        )
        if int(new_active) != int(active["id"]):
            set_active_assignment(int(new_active))
            st.rerun()

        st.caption(f"Currently editing prompts for: **{get_active_assignment().get('name')}**")

        with st.expander("➕ Add new assignment", expanded=False):
            new_name = st.text_input("Assignment name", placeholder="Assignment 2")
            new_prompt = st.text_area("Assignment-specific prompt (optional)", height=160)
            if st.button("Create assignment"):
                if not new_name.strip():
                    st.error("Assignment name is required")
                else:
                    aid = add_assignment(new_name.strip(), new_prompt or "")
                    set_active_assignment(aid)
                    st.success(f"Created and activated '{new_name.strip()}'")
                    st.rerun()

    with c2:
        base_prompt = get_base_system_prompt(DEFAULT_BASE_PROMPT)
        base_edit = st.text_area(
            "Base system prompt (applies to ALL assignments)",
            value=base_prompt,
            height=320,
        )
        if st.button("Save base prompt", type="primary"):
            set_base_system_prompt(base_edit)
            st.success("Saved base system prompt")

    # Assignment-specific prompt (full width)
    active = get_active_assignment()
    ap_edit = st.text_area(
        f"Assignment-specific prompt — {active.get('name')}",
        value=active.get("prompt") or "",
        height=320,
    )
    if st.button("Save assignment prompt"):
        update_assignment_prompt(int(active["id"]), ap_edit)
        st.success("Saved assignment prompt")

    st.divider()

    # User management (existing)
    st.subheader("Users")
    users = list_users()
    st.dataframe(users, use_container_width=True, hide_index=True)

    with st.expander("Add / Update user"):
        uid = st.text_input("User ID", key="new_uid")
        pw = st.text_input("Password", type="password", key="new_pw")
        role = st.selectbox("Role", ["student", "admin"], key="new_role")
        if st.button("Save user"):
            if not uid or not pw:
                st.error("User ID and password are required.")
            else:
                upsert_user(uid, pw, role)
                st.success("User saved")
                st.rerun()

    st.divider()

    # Conversation browser
    st.subheader("Conversation browser")
    user_filter = st.text_input("Filter by user_id (contains)", "")
    model_filter = st.text_input("Filter by model (exact)", "")
    role_filter = st.selectbox("Filter by role", ["", "student", "admin"], index=0)

    convs = list_conversations_admin(
        user_filter=user_filter or None,
        role_filter=role_filter or None,
        model_filter=model_filter or None,
        limit=200,
    )

    if not convs:
        st.caption("No conversations match filters.")
        return

    conv_ids = [c["id"] for c in convs]
    labels = {}
    for c in convs:
        cid = c["id"]
        title = c.get("title") or f"Conversation {cid}"
        labels[cid] = f"{c.get('updated_at','')} · {c.get('user_id','')} · {title}"

    picked = st.selectbox("Select conversation", conv_ids, format_func=lambda x: labels.get(x, str(x)))
    conv = get_conversation(int(picked)) or {}
    msgs = get_conversation_messages(int(picked))

    st.caption(
        f"Model: **{conv.get('model')}** · Assignment: **{conv.get('assignment_name') or '—'}**"
    )

    transcript = _conversation_to_text([{**m, "attachments": []} for m in msgs])
    st.text_area("Transcript (text only)", value=transcript, height=260)
    _copy_button(transcript, key=f"copy_admin_conv_{picked}", tooltip="Copy transcript")


# ---------------- Main routing ----------------


def main() -> None:
    if "user_id" not in st.session_state:
        _render_login()
        return

    models = _cached_models()
    page, active_model, active_assignment = _sidebar(models)

    if page == "Admin Dashboard" and st.session_state.get("role") == "admin":
        _admin_dashboard(active_model)
    else:
        _chat_page(active_model, active_assignment)


if __name__ == "__main__":
    main()

import os
import re
import uuid
from functools import lru_cache
from typing import Optional, Tuple

import requests


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


def is_enabled() -> bool:
    return bool(_get_secret("SUPABASE_URL") and _get_secret("SUPABASE_SERVICE_ROLE_KEY") and _get_secret("SUPABASE_BUCKET"))


def _cfg() -> Tuple[str, str, str]:
    url = (_get_secret("SUPABASE_URL") or "").rstrip("/")
    key = _get_secret("SUPABASE_SERVICE_ROLE_KEY") or ""
    bucket = _get_secret("SUPABASE_BUCKET") or ""
    if not url or not key or not bucket:
        raise RuntimeError("Supabase Storage not configured: need SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_BUCKET")
    return url, key, bucket


_filename_safe_re = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_filename(name: str) -> str:
    name = (name or "file").strip()
    name = name.replace("\\", "/").split("/")[-1]
    name = _filename_safe_re.sub("_", name)
    return name[:180] if len(name) > 180 else name


def make_object_path(user_id: str, conversation_id: int, message_id: int, filename: str) -> str:
    # Keep paths predictable but unique.
    stem = uuid.uuid4().hex
    return f"{user_id}/c{conversation_id}/m{message_id}/{stem}_{_safe_filename(filename)}"


def upload_bytes(
    *,
    user_id: str,
    conversation_id: int,
    message_id: int,
    filename: str,
    mime: str,
    data: bytes,
    upsert: bool = True,
) -> Tuple[str, str]:
    """Upload bytes to Supabase Storage. Returns (bucket, path)."""
    url, key, bucket = _cfg()
    path = make_object_path(user_id, conversation_id, message_id, filename)

    endpoint = f"{url}/storage/v1/object/{bucket}/{path}"
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": mime or "application/octet-stream",
    }
    if upsert:
        headers["x-upsert"] = "true"

    # Standard upload (small/medium files). For very large files, resumable uploads are better.
    resp = requests.post(endpoint, headers=headers, data=data, timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Supabase upload failed ({resp.status_code}): {resp.text[:300]}")

    # Clear any cached old bytes for this object.
    _download_cached.cache_clear()
    return bucket, path


@lru_cache(maxsize=256)
def _download_cached(url: str, key: str, bucket: str, path: str) -> bytes:
    endpoint = f"{url}/storage/v1/object/{bucket}/{path}"
    headers = {"Authorization": f"Bearer {key}", "apikey": key}
    resp = requests.get(endpoint, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Supabase download failed ({resp.status_code}): {resp.text[:300]}")
    return resp.content


def download_bytes(bucket: str, path: str) -> bytes:
    """Download bytes from Supabase Storage (server-side)."""
    url, key, _bucket_default = _cfg()
    b = bucket or _bucket_default
    if not b or not path:
        raise ValueError("bucket and path are required")
    return _download_cached(url, key, b, path)

import json
from typing import Any, Dict, Iterable, List, Optional

import requests


def _normalize_host(host: str) -> str:
    host = (host or "").strip()
    if not host:
        return host
    if not host.startswith("http://") and not host.startswith("https://"):
        # Ollama Cloud endpoints are HTTPS; local Ollama also supports HTTP.
        host = "https://" + host
    return host.rstrip("/")


def _base_api_url(host: str) -> str:
    return _normalize_host(host) + "/api"


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def list_models(host: str, api_key: Optional[str]) -> List[str]:
    """List available models from Ollama-compatible /api/tags."""
    url = _base_api_url(host) + "/tags"
    r = requests.get(url, headers=_headers(api_key), timeout=30)
    r.raise_for_status()
    data = r.json()

    names: List[str] = []
    for m in data.get("models", []):
        nm = m.get("name") or m.get("model")
        if nm:
            names.append(nm)
    return sorted(set(names))


def _raise_with_details(r: requests.Response, url: str) -> None:
    """Raise an HTTPError with safe, useful details (no secrets)."""
    snippet = ""
    try:
        # Response bodies from Ollama are typically safe JSON error messages.
        snippet = (r.text or "").strip()
    except Exception:
        snippet = ""
    if len(snippet) > 1500:
        snippet = snippet[:1500] + "…"
    msg = f"Ollama API request failed ({r.status_code}) at {url}\n\n{snippet}"
    raise requests.HTTPError(msg, response=r)


def chat_stream(
    host: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, Any]],
    stream: bool = True,
    options: Optional[Dict[str, Any]] = None,
    think: Optional[str] = None,
) -> Iterable[str]:
    """Stream chat tokens from Ollama-compatible /api/chat.

    - `think` is an optional vendor extension (e.g., some cloud models).
    - If the server rejects `think` (400/422), we retry once without it.
    """
    url = _base_api_url(host) + "/chat"

    def _do_post(payload: Dict[str, Any]) -> requests.Response:
        return requests.post(
            url,
            headers=_headers(api_key),
            data=json.dumps(payload),
            stream=stream,
            timeout=300,
        )

    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if options:
        payload["options"] = options
    if think:
        payload["think"] = think

    r = _do_post(payload)

    # If think isn't supported, retry once without it.
    if r.status_code in (400, 422) and think:
        try:
            body = (r.text or "").lower()
        except Exception:
            body = ""
        # Retry on common validation messages, or just retry once on 400/422.
        if ("think" in body) or True:
            r.close()
            payload.pop("think", None)
            r = _do_post(payload)

    if r.status_code >= 400:
        _raise_with_details(r, url)

    if not stream:
        data = r.json()
        msg = (data.get("message") or {}).get("content", "")
        yield msg
        return

    # Streamed responses are newline-delimited JSON objects.
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            part = json.loads(line)
        except Exception:
            continue
        chunk = (part.get("message") or {}).get("content", "")
        if chunk:
            yield chunk

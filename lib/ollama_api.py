import json
from typing import Dict, Iterable, Iterator, List, Optional, Union

import requests


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def list_models(host: str, api_key: Optional[str] = None) -> List[str]:
    """Return available model names from the configured Ollama host."""
    url = host.rstrip("/") + "/api/tags"
    r = requests.get(url, headers=_headers(api_key), timeout=30)
    r.raise_for_status()
    data = r.json() or {}
    models = data.get("models") or []
    names = [m.get("name") for m in models if m.get("name")]
    # Stable, predictable ordering
    return sorted(set(names))


ThinkParam = Union[bool, str]


def chat_stream(
    host: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict],
    stream: bool = True,
    options: Optional[Dict] = None,
    think: Optional[ThinkParam] = "high",
) -> Iterator[str]:
    """Stream assistant content chunks.

    Notes:
    - Some models (e.g., gpt-oss) support a `think` parameter controlling reasoning effort.
    - When `think` is enabled, Ollama may stream `message.thinking` chunks. We intentionally
      ignore those and only yield `message.content`.

    Ollama API reference: /api/chat
    """
    url = host.rstrip("/") + "/api/chat"

    body: Dict = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if options:
        body["options"] = options

    # Enable extended thinking when possible.
    # - gpt-oss expects think: "low"|"medium"|"high"
    # - some others may accept think: true
    if think is not None:
        body["think"] = think

    with requests.post(url, headers=_headers(api_key), data=json.dumps(body), stream=True, timeout=300) as r:
        r.raise_for_status()
        # Ollama streams JSON lines
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # ignore server-side progress
            if obj.get("done"):
                break

            msg = obj.get("message") or {}
            # When thinking is enabled, message.thinking may be present.
            # We DO NOT surface it to students.
            content = msg.get("content")
            if content:
                yield content


def chat_once(
    host: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict],
    options: Optional[Dict] = None,
    think: Optional[ThinkParam] = "high",
) -> str:
    full = ""
    for chunk in chat_stream(
        host=host,
        api_key=api_key,
        model=model,
        messages=messages,
        stream=True,
        options=options,
        think=think,
    ):
        full += chunk
    return full

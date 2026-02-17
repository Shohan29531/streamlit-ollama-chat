import json
from typing import Any, Dict, Iterable, List, Optional
import requests


def _base_api_url(host: str) -> str:
    return host.rstrip("/") + "/api"


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def list_models(host: str, api_key: Optional[str]) -> List[str]:
    url = _base_api_url(host) + "/tags"
    r = requests.get(url, headers=_headers(api_key), timeout=30)
    r.raise_for_status()
    data = r.json()

    names = []
    for m in data.get("models", []):
        nm = m.get("name") or m.get("model")
        if nm:
            names.append(nm)

    return sorted(set(names))


def chat_stream(
    host: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, Any]],
    stream: bool = True,
    options: Optional[Dict] = None,
) -> Iterable[str]:
    url = _base_api_url(host) + "/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if options:
        payload["options"] = options

    with requests.post(
        url,
        headers=_headers(api_key),
        data=json.dumps(payload),
        stream=stream,
        timeout=300,
    ) as r:
        r.raise_for_status()

        if not stream:
            data = r.json()
            msg = (data.get("message") or {}).get("content", "")
            yield msg
            return

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            part = json.loads(line)
            chunk = (part.get("message") or {}).get("content", "")
            if chunk:
                yield chunk

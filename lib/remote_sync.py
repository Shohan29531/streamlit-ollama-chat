from typing import Dict, List, Optional
import requests


def sync_conversation(
    remote_url: Optional[str],
    token: Optional[str],
    conversation_meta: Dict,
    messages: List[Dict],
) -> bool:
    if not remote_url or not token:
        return False

    payload = {"conversation": conversation_meta, "messages": messages}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    try:
        r = requests.post(remote_url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return True
    except Exception:
        return False

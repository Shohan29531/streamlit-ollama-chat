import base64
import io
import json
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ExtractedFile:
    filename: str
    mime: str
    size_bytes: int
    kind: str  # 'image' | 'file'
    data: bytes
    text_content: Optional[str] = None  # only for kind='file'


IMAGE_MIME_PREFIX = "image/"


# ---- Compatibility layer (used by app.py imports) ----

def is_image_mime(mime: str) -> bool:
    """Return True if the mime type is an image."""
    return (mime or "").lower().startswith(IMAGE_MIME_PREFIX)


def detect_kind(filename: str, mime: str) -> str:
    """Return 'image' or 'file' based on mime and filename."""
    fn = (filename or "").lower()
    mime_l = (mime or "").lower()
    if is_image_mime(mime_l) or fn.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return "image"
    return "file"


def to_data_url(mime: str, data: bytes) -> str:
    """Convert raw bytes to a data: URL so vision models can receive images inline."""
    mt = (mime or "").strip() or "application/octet-stream"
    b64 = base64.b64encode(data or b"").decode("utf-8")
    return f"data:{mt};base64,{b64}"


def extract_text_from_bytes(filename: str, data: bytes, mime: str = "") -> str:
    """Best-effort text extraction; mime is optional for backward compatibility."""
    return extract_text_from_file(filename=filename, mime=mime, data=data)


# ---- Internals ----

def is_image(mime: str, filename: str) -> bool:
    """Legacy helper: True if mime/filename indicates an image."""
    return detect_kind(filename, mime) == "image"


def image_bytes_to_b64(data: bytes) -> str:
    """Base64 encode raw bytes (legacy helper)."""
    return base64.b64encode(data).decode("utf-8")


def _safe_decode_text(data: bytes) -> str:
    # Try utf-8 first; fall back to latin-1 without throwing.
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin-1", errors="replace")


def extract_text_from_file(filename: str, mime: str, data: bytes) -> str:
    """Best-effort text extraction for common classroom file types.

    - txt/md/code/json/csv -> decode
    - pdf -> extract via pypdf (if installed)
    - docx -> extract via python-docx (if installed)

    Any extraction errors return a short placeholder message.
    """

    fn = (filename or "").lower()
    mime_l = (mime or "").lower()

    # Plain text-ish
    if mime_l.startswith("text/") or fn.endswith(
        (
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".csv",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
        )
    ):
        if fn.endswith(".json"):
            # Pretty-print JSON when possible.
            try:
                obj = json.loads(_safe_decode_text(data))
                return json.dumps(obj, indent=2, ensure_ascii=False)
            except Exception:
                return _safe_decode_text(data)
        return _safe_decode_text(data)

    # PDF
    if mime_l == "application/pdf" or fn.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(io.BytesIO(data))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            text = "\n".join(parts).strip()
            return text or "[PDF had no extractable text.]"
        except Exception:
            return "[Could not extract PDF text in this environment.]"

    # DOCX
    if (
        mime_l
        in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        )
        or fn.endswith(".docx")
    ):
        try:
            import docx  # type: ignore

            doc = docx.Document(io.BytesIO(data))
            lines = [p.text for p in doc.paragraphs if p.text]
            text = "\n".join(lines).strip()
            return text or "[DOCX had no extractable text.]"
        except Exception:
            return "[Could not extract DOCX text in this environment.]"

    return "[Unsupported file type for text extraction. If you want the model to use it, upload a text/PDF/DOCX file.]"


def truncate_text(text: str, max_chars: int) -> Tuple[str, bool]:
    if text is None:
        return "", False
    if len(text) <= max_chars:
        return text, False
    return text[: max_chars - 1] + "…", True

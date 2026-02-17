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


def is_image(mime: str, filename: str) -> bool:
    mime = (mime or "").lower()
    if mime.startswith(IMAGE_MIME_PREFIX):
        return True
    fn = (filename or "").lower()
    return fn.endswith((".png", ".jpg", ".jpeg", ".webp"))


def image_bytes_to_b64(data: bytes) -> str:
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

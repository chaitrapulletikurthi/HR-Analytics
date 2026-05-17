from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional

def save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def load_text_any(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext in [".txt", ".md"]:
        return p.read_text(encoding="utf-8", errors="ignore")

    if ext == ".docx":
        from docx import Document
        doc = Document(str(p))
        return "\n".join([para.text for para in doc.paragraphs])

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {p}. Install pypdf. Error: {e}")

    raise ValueError(f"Unsupported file type: {ext}")

# -------------------------
# PII Redaction (prototype)
# -------------------------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

def redact_pii(text: str) -> tuple[str, int]:
    if not text:
        return text, 0
    redacted = text
    count = 0
    for rx, token in [
        (EMAIL_RE, "[REDACTED_EMAIL]"),
        (PHONE_RE, "[REDACTED_PHONE]"),
        (SSN_RE, "[REDACTED_SSN]"),
    ]:
        redacted, n = rx.subn(token, redacted)
        count += n
    return redacted, count

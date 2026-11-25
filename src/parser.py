"""Extract raw text from PDF, DOCX and TXT resumes."""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

Source = Union[str, Path, bytes, BinaryIO]


def extract_text(source: Source, filename: str | None = None) -> str:
    """Return plain text from a path, bytes or an uploaded file object."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:  # file-like (e.g. Streamlit UploadedFile)
        filename = filename or getattr(source, "name", "")
        data = source.read()

    ext = Path(filename or "").suffix.lower()
    if ext == ".pdf":
        return _read_pdf(data)
    if ext == ".docx":
        return _read_docx(data)
    return data.decode("utf-8", errors="ignore")


def _read_pdf(data: bytes) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)

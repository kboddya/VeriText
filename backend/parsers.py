from __future__ import annotations

import io
import os

from .preprocessing import split_paragraphs


def extract_text_from_file(filename: str, content: bytes) -> tuple[str, list[str]]:
    """Extract full text content and a list of paragraphs from .md, .docx, .pdf, and .txt files."""
    ext = os.path.splitext(filename)[1].lower()
    full_text = ""

    if ext in {".md", ".markdown", ".txt"}:
        try:
            full_text = content.decode("utf-8")
        except UnicodeDecodeError:
            full_text = content.decode("latin-1", errors="replace")

    elif ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted and extracted.strip():
                    pages.append(extracted.strip())
            full_text = "\n\n".join(pages)
        except Exception as e:
            raise ValueError(f"Failed to read PDF document: {str(e)}") from e

    elif ext in {".docx", ".doc"}:
        if ext == ".doc":
            raise ValueError("Legacy .doc format is not supported. Please convert the file to .docx.")
        try:
            from docx import Document

            doc = Document(io.BytesIO(content))
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paras)
        except Exception as e:
            raise ValueError(f"Failed to read DOCX document: {str(e)}") from e
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .md, .docx, .pdf, .txt")

    full_text = full_text.strip()
    if not full_text:
        raise ValueError("Document contains no readable text.")

    raw_paras = [p.strip() for p in split_paragraphs(full_text) if len(p.strip()) > 20]
    if len(raw_paras) <= 1:
        lines = [line.strip() for line in full_text.split("\n") if len(line.strip()) > 35]
        raw_paras = lines if lines else [full_text]

    return full_text, raw_paras

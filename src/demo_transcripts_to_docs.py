# demo_transcript_to_docs.py

import os
import json
from pathlib import Path
from llama_index.core import Document
from collections import defaultdict
import logging

from dotenv import load_dotenv
from src.logger import configure_logging, get_logger
from src.models.metadata import load_course_metadata
load_dotenv()  # This will load the variables from the .env file
configure_logging()
log = logging.getLogger(__name__)

# TRANSCRIPT_PATH = os.environ['TRANSCRIPT_PATH']
# METADATA_FILE = os.environ['METADATA_FILE']

def _resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(base_dir, path)


def _read_text_file(p: str) -> str:
    try:
        # Special handling for SRT: strip indices and timestamps
        if p.lower().endswith('.srt'):
            return _read_srt_text(p)
        with open(p, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        log.exception("Failed reading text file: %s", p)
        return ""

def _make_document(text: str, meta: dict) -> Document:
    return Document(text=text, metadata=meta)

def _read_pdf_text(p: str) -> str:
    try:
        import PyPDF2
    except Exception:
        log.warning("PyPDF2 not available; cannot read PDF: %s", p)
        return ""
    try:
        text_parts = []
        with open(p, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages) if hasattr(reader, 'pages') else 0
            log.debug("Reading PDF %s (pages=%d)", p, num_pages)
            for idx, page in enumerate(reader.pages):
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    log.exception("Failed extracting text from PDF %s page %s", p, idx)
                    text_parts.append("")
        combined = "\n\n".join([t for t in text_parts if t])
        if not combined.strip():
            log.warning("No text extracted from PDF: %s", p)
        return combined
    except Exception:
        log.exception("PDF read error: %s", p)
        return ""

def _read_srt_text(p: str) -> str:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        out_lines = []
        for line in lines:
            s = line.strip()
            # Skip sequence numbers (pure digits) and timestamp lines
            if s.isdigit():
                continue
            if ('-->' in s) and (':' in s):
                continue
            if not s:
                continue
            out_lines.append(s)
        text = "\n".join(out_lines)
        if not text.strip():
            log.warning("Empty SRT after parsing: %s", p)
        return text
    except Exception:
        log.exception("SRT read error: %s", p)
        return ""

def _log_doc_added(kind: str, path: str, text: str, meta: dict) -> None:
    try:
        size = os.path.getsize(path) if os.path.exists(path) else -1
    except Exception:
        size = -1
    log.debug("Added %s doc: path=%s size=%sB chars=%s meta_keys=%s",
              kind, path, size, len(text or ""), sorted((meta or {}).keys()))

def _create_documents_from_course(course, module_index: int) -> list:
    documents = []
    module = course.select_module_by_index(module_index)
    log.info("Scanning module '%s' (slug=%s) for lesson items", module.module_name, module.module_slug)
    for lesson in module.lessons:
        for item in lesson.items:
            # transcripts (plain text)
            for c in item.content:
                if c.content_type == "transcript" and c.path and c.path.lower().endswith(('.txt', '.srt')):
                    if not os.path.exists(c.path):
                        log.warning("Transcript path not found: %s", c.path)
                        continue
                    try:
                        text = _read_text_file(c.path)
                    except Exception:
                        text = ""
                    if not text.strip():
                        log.warning("Empty transcript text for: %s", c.path)
                        continue
                    documents.append(_make_document(text, {
                        "course_name": course.course_name,
                        "course_slug": course.course_slug,
                        "module_name": module.module_name,
                        "module_slug": module.module_slug,
                        "lesson_name": lesson.lesson_name,
                        "lesson_slug": lesson.lesson_slug,
                        "item_name": item.name,
                        "item_slug": item.transformed_slug,
                        "transcript_file": c.path,
                        "type": "transcript",
                    }))
                    _log_doc_added("transcript", c.path, text, documents[-1].metadata)
            # slides (PDF) as full-text documents when readable
            for c in item.content:
                if c.content_type == "slides" and c.path and c.path.lower().endswith('.pdf') and os.path.exists(c.path):
                    pdf_text = _read_pdf_text(c.path)
                    if pdf_text and pdf_text.strip():
                        documents.append(_make_document(pdf_text, {
                            "course_name": course.course_name,
                            "course_slug": course.course_slug,
                            "module_name": module.module_name,
                            "module_slug": module.module_slug,
                            "lesson_name": lesson.lesson_name,
                            "lesson_slug": lesson.lesson_slug,
                            "item_name": item.name,
                            "item_slug": item.transformed_slug,
                            "slides_file": c.path,
                            "type": "slides",
                        }))
                        _log_doc_added("slides", c.path, pdf_text, documents[-1].metadata)
                    else:
                        log.warning("Skipping slides with no extracted text: %s", c.path)
            # extra notes (Markdown/text/PDF) as full-text documents when readable
            for c in item.content:
                if c.content_type == "extra-notes" and c.path and c.path.lower().endswith(('.md', '.txt', '.pdf')) and os.path.exists(c.path):
                    try:
                        if c.path.lower().endswith('.pdf'):
                            note_text = _read_pdf_text(c.path)
                        else:
                            note_text = _read_text_file(c.path)
                    except Exception:
                        note_text = ""
                    if note_text and note_text.strip():
                        documents.append(_make_document(note_text, {
                            "course_name": course.course_name,
                            "course_slug": course.course_slug,
                            "module_name": module.module_name,
                            "module_slug": module.module_slug,
                            "lesson_name": lesson.lesson_name,
                            "lesson_slug": lesson.lesson_slug,
                            "item_name": item.name,
                            "item_slug": item.transformed_slug,
                            "extra_notes_file": c.path,
                            "type": "extra_notes",
                        }))
                        _log_doc_added("extra_notes", c.path, note_text, documents[-1].metadata)
                    else:
                        log.warning("Skipping extra-notes with no readable text: %s", c.path)
    return documents

def transcripts_to_docs(transcript_path: str, metadata_file: str, topic_number: int, project_dir: str = ".."):
    """Generate documents for a module using pydantic metadata models.

    project_dir is used to resolve relative content paths from metadata, if needed.
    """
    course = load_course_metadata(metadata_file, project_dir)
    documents = _create_documents_from_course(course, topic_number)

    # Summaries and samples (metadata only)
    log.info("Created %d documents for module_index=%s", len(documents), topic_number)
    try:
        from collections import Counter
        type_counts = Counter((d.metadata or {}).get("type", "unknown") for d in documents)
        log.info("Document type summary: %s", dict(type_counts))
    except Exception:
        pass
    # Log only metadata keys, never text content; one JSON object per line
    try:
        import json as _json
        log.info("Sample docs (metadata only):")
        for d in documents:
            md = d.metadata or {}
            line = {
                "course_slug": md.get("course_slug", ""),
                "module_slug": md.get("module_slug", ""),
                "lesson_slug": md.get("lesson_slug", ""),
                "item_slug": md.get("item_slug", ""),
                "type": md.get("type", ""),
                "has_text": bool(d.text),
            }
            log.info(_json.dumps(line))
    except Exception:
        pass

    return documents

# if __name__ == "__main__":
#     # Define paths to be used (can be configured or passed as arguments)
#     documents = transcripts_to_docs(TRANSCRIPT_PATH, METADATA_FILE)

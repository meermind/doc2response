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
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()

def _make_document(text: str, meta: dict) -> Document:
    return Document(text=text, metadata=meta)

def _read_pdf_text(p: str) -> str:
    try:
        import PyPDF2
    except Exception:
        # PyPDF2 not available; return empty so we still create refs
        return ""
    try:
        text_parts = []
        with open(p, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    text_parts.append("")
        return "\n\n".join([t for t in text_parts if t])
    except Exception:
        return ""

def _create_documents_from_course(course, module_index: int) -> list:
    documents = []
    module = course.select_module_by_index(module_index)
    for lesson in module.lessons:
        for item in lesson.items:
            # transcripts (plain text)
            for c in item.content:
                if c.content_type == "transcript" and c.path and c.path.endswith('.txt'):
                    try:
                        text = _read_text_file(c.path)
                    except Exception:
                        text = ""
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
            # slides (PDF) as full-text documents when readable
            for c in item.content:
                if c.content_type == "slides" and c.path and c.path.lower().endswith('.pdf') and os.path.exists(c.path):
                    pdf_text = _read_pdf_text(c.path)
                    if pdf_text:
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
            # extra notes (Markdown/text) as full-text documents when readable
            for c in item.content:
                if c.content_type == "extra-notes" and c.path and c.path.lower().endswith(('.md', '.txt')) and os.path.exists(c.path):
                    try:
                        note_text = _read_text_file(c.path)
                    except Exception:
                        note_text = ""
                    if note_text:
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

# demo_transcript_to_docs.py

import os
import json
from pathlib import Path
from llama_index.core import Document
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()  # This will load the variables from the .env file

# TRANSCRIPT_PATH = os.environ['TRANSCRIPT_PATH']
# METADATA_FILE = os.environ['METADATA_FILE']

def _resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(base_dir, path)


def load_metadata(metadata_file: str) -> dict:
    """Load metadata from a JSON file."""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_file}")
        return {}

def load_module_content_paths(metadata: dict, topic_number: int) -> dict:
    """Extract content paths for a given module index (1-based)."""
    modules = metadata.get("modules", [])
    if topic_number < 1 or topic_number > len(modules):
        return {}

    module = modules[topic_number - 1]

    content = {
        "transcripts": [],
        "slides": [],
        "extra_notes": [],
    }

    for lesson in module.get("lessons", []):
        for item in lesson.get("items", []):
            for c in item.get("content", []):
                ctype = c.get("content_type")
                cpath = c.get("path")
                if not ctype or not cpath:
                    continue
                if ctype == "transcript" and cpath.endswith(".txt"):
                    content["transcripts"].append(cpath)
                elif ctype == "slides" and cpath.endswith(".pdf"):
                    content["slides"].append(cpath)
                elif ctype == "extra-notes" and cpath.endswith(".md"):
                    content["extra_notes"].append(cpath)

    return content

def create_documents_with_metadata(metadata: dict, module_index: int) -> list:
    """Create Document objects enriched with metadata for a single module using paths from metadata."""
    documents = []

    def get_parent_dir(file_path: str) -> str:
        """Extract the parent directory path."""
        return os.path.dirname(file_path)

    modules = metadata.get("modules", [])
    if module_index < 1 or module_index > len(modules):
        return documents

    module = modules[module_index - 1]
    module_name = module["module_name"]
    module_slug = module["module_slug"]

    for lesson in module.get("lessons", []):
        lesson_name = lesson["lesson_name"]
        lesson_slug = lesson["lesson_slug"]

        for item in lesson.get("items", []):
            item_name = item["name"]
            item_slug = item["transformed_slug"]

            # Inline transcripts directly from content path
            for content in item.get("content", []):
                ctype = content.get("content_type")
                cpath = content.get("path")
                if ctype == "transcript" and cpath and cpath.endswith(".txt"):
                    with open(cpath, 'r', encoding='utf-8') as f:
                        content_text = f.read()

                    document = Document(
                        text=content_text,
                        metadata={
                            "course_name": metadata.get("course_name", ""),
                            "course_slug": metadata.get("course_slug", ""),
                            "module_name": module_name,
                            "module_slug": module_slug,
                            "lesson_name": lesson_name,
                            "lesson_slug": lesson_slug,
                            "item_name": item_name,
                            "item_slug": item_slug,
                            "transcript_file": cpath,
                        }
                    )
                    documents.append(document)

            # Also create lightweight reference docs for slides and extra notes paths
            slide_paths = [c["path"] for c in item.get("content", []) if c.get("content_type") == "slides"]
            note_paths = [c["path"] for c in item.get("content", []) if c.get("content_type") == "extra-notes"]
            for spath in slide_paths:
                documents.append(Document(text="", metadata={
                    "type": "slides_ref",
                    "path": spath,
                    "course_slug": metadata.get("course_slug", ""),
                    "module_slug": module_slug,
                    "lesson_slug": lesson_slug,
                    "item_slug": item_slug,
                }))
            for npath in note_paths:
                documents.append(Document(text="", metadata={
                    "type": "extra_notes_ref",
                    "path": npath,
                    "course_slug": metadata.get("course_slug", ""),
                    "module_slug": module_slug,
                    "lesson_slug": lesson_slug,
                    "item_slug": item_slug,
                }))
    return documents

def transcripts_to_docs(transcript_path: str, metadata_file: str, topic_number: int, project_dir: str = ".."):
    """Generate documents for a module using content paths embedded in metadata.

    project_dir is used to resolve relative content paths from metadata, if needed.
    """
    metadata = load_metadata(metadata_file)

    # Normalize paths in metadata to absolute or project-relative for robustness
    for module in metadata.get("modules", []):
        for lesson in module.get("lessons", []):
            for item in lesson.get("items", []):
                for c in item.get("content", []):
                    p = c.get("path")
                    if p:
                        c["path"] = _resolve_path(p, project_dir)

    documents = create_documents_with_metadata(metadata, topic_number)

    print(f"Created {len(documents)} documents.")
    for doc in documents[:5]:
        print(f"Content: {doc.text[:50]}... | Metadata: {doc.metadata}")

    return documents

# if __name__ == "__main__":
#     # Define paths to be used (can be configured or passed as arguments)
#     documents = transcripts_to_docs(TRANSCRIPT_PATH, METADATA_FILE)

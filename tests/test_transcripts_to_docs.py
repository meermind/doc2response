import json
import sys
import types
from pathlib import Path


def test_transcripts_to_docs_minimal(monkeypatch):
    # Prepare test lake directories
    project_dir = Path(__file__).resolve().parents[1]
    inputs_dir = project_dir / "tests" / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Arrange: tiny transcript file in inputs/
    transcript_file = inputs_dir / "t1.txt"
    transcript_file.write_text("Hello world transcript.", encoding="utf-8")

    # Minimal metadata pointing to absolute transcript path
    metadata = {
        "course_name": "Tiny Course",
        "course_slug": "tiny-course",
        "modules": [
            {
                "module_name": "Topic 1 Tiny",
                "module_slug": "topic-1",
                "lessons": [
                    {
                        "lesson_name": "L1",
                        "lesson_slug": "l1",
                        "items": [
                            {
                                "name": "I1",
                                "transformed_slug": "i1",
                                "content": [
                                    {"content_type": "transcript", "path": str(transcript_file)},
                                    {"content_type": "slides", "path": "/abs/path/slides.pdf"},
                                    {"content_type": "extra-notes", "path": "/abs/path/notes.md"},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    metadata_file = inputs_dir / "meta.json"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    # Stub llama_index.core.Document if llama_index not installed
    if "llama_index.core" not in sys.modules:
        core_mod = types.ModuleType("llama_index.core")

        class Document:  # minimal stub
            def __init__(self, text: str, metadata: dict):
                self.text = text
                self.metadata = metadata

        core_mod.Document = Document
        pkg_mod = types.ModuleType("llama_index")
        sys.modules["llama_index"] = pkg_mod
        sys.modules["llama_index.core"] = core_mod

    # Ensure import path and import target function
    monkeypatch.chdir(project_dir)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from src.demo_transcripts_to_docs import transcripts_to_docs

    # Act: topic_number=1
    docs = transcripts_to_docs(
        transcript_path=None,
        metadata_file=str(metadata_file),
        topic_number=1,
        project_dir=str(project_dir),
    )

    # Assert
    assert len(docs) >= 1
    first = docs[0]
    assert "Hello world transcript." in first.text
    assert first.metadata["course_name"] == "Tiny Course"
    assert first.metadata["module_slug"] == "topic-1"
    # Ensure the lightweight refs for slides/notes are also present
    paths = [d.metadata.get("path") for d in docs if d.metadata.get("type") in {"slides_ref", "extra_notes_ref"}]
    assert "/abs/path/slides.pdf" in paths
    assert "/abs/path/notes.md" in paths



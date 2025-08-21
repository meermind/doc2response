import json
import os
import sys
from pathlib import Path

import pytest


def test_orchestrator_end_to_end(monkeypatch, request):
    """End-to-end test: orchestrate -> mock AI -> generate LaTeX -> verify output."""
    # Change CWD to project dir so relative paths match scripts' expectations
    project_dir = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_dir)

    # Use inputs/ lake for metadata
    inputs_dir = project_dir / "tests" / "inputs"
    metadata_file = inputs_dir / "meta.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    # No need to clean assistant_latex; outputs are isolated per test

    # Ensure project dir is importable and import orchestrator
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from src.orchestrator import orchestrate_pipeline
    from src import orchestrator

    # Mock the orchestrator step functions
    def fake_load_docs(metadata_file, topic_number, project_dir=".."):
        return None

    def fake_call_llamaindex(module_name):
        return None

    def fake_generate_latex(course, module, module_name):
        base_dir = Path(os.getenv("INPUT_BASE_DIR", project_dir / "assistant_latex"))
        base = base_dir / course / module_name
        base.mkdir(parents=True, exist_ok=True)
        section_path = base / "Introduction.tex"
        subsection_path = base / "Motivation.tex"
        section_path.write_text("\\section{Introduction}\nThis is the intro.", encoding="utf-8")
        subsection_path.write_text("\\subsection{Motivation}\nWhy this matters.", encoding="utf-8")
        metadata_json = {
            "sections": [
                {"order": 0, "type": "section", "title": "Introduction", "path": str(section_path)},
                {"order": 1, "type": "subsection", "title": "Motivation", "path": str(subsection_path)},
            ]
        }
        (base / "metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
        from src.latex_merger.generate_latex_doc import execute
        execute(course, module, module_name)

    monkeypatch.setattr(orchestrator, "run_load_docs", fake_load_docs)
    monkeypatch.setattr(orchestrator, "run_call_llamaindex", fake_call_llamaindex)
    monkeypatch.setattr(orchestrator, "run_generate_latex", fake_generate_latex)
    # Avoid interactive overwrite prompts in generator
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    # Route outputs to tests/outputs
    outputs_base = project_dir / "tests" / "outputs" / request.node.name
    outputs_base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("D2R_OUTPUT_BASE", str(outputs_base))
    monkeypatch.setenv("INPUT_BASE_DIR", str(outputs_base))

    # Run with all steps; fakes avoid DB/LLM and synthesize outputs
    orchestrate_pipeline(
        run_load=True,
        run_call=True,
        run_generate=True,
        metadata_file=str(metadata_file),
        topic_number=1,
    )

    # Verify final LaTeX output exists and contains our content
    course_name = metadata["course_name"]
    module_code = "Topic 1"
    module_name = metadata["modules"][0]["module_name"]
    out_dir = outputs_base / course_name / "Lecture Notes" / module_code / module_name
    out_file = out_dir / f"{module_name}.tex"

    assert out_file.exists(), f"Expected LaTeX file not found: {out_file}"
    content = out_file.read_text(encoding="utf-8")
    assert "This is the intro." in content
    assert "Why this matters." in content
    assert content.strip().endswith("\\end{document}")



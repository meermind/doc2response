import json
from pathlib import Path


def test_generate_latex_execute_with_prebuilt_metadata(monkeypatch, request):
    # Arrange project dir for imports and relative paths
    project_dir = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_dir)
    import sys
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    # Create assistant_latex structure and files
    course = "Test Course"
    module = "Topic 1"
    module_name = "Topic 1 Unit"
    base = project_dir / "assistant_latex" / course / module_name
    base.mkdir(parents=True, exist_ok=True)

    section_path = base / "Intro.tex"
    subsection_path = base / "Basics.tex"
    section_path.write_text("\\section{Intro}\nIntro text.", encoding="utf-8")
    subsection_path.write_text("\\subsection{Basics}\nBasics text.", encoding="utf-8")

    metadata = {
        "sections": [
            {"order": 0, "type": "section", "title": "Intro", "path": str(section_path)},
            {"order": 1, "type": "subsection", "title": "Basics", "path": str(subsection_path)},
        ]
    }
    (base / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Avoid interactive overwrite prompt
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "y")

    # Route outputs to tests/outputs/<testname>
    outputs_base = project_dir / "tests" / "outputs" / request.node.name
    outputs_base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("D2R_OUTPUT_BASE", str(outputs_base))

    # Act
    from src.latex_merger.generate_latex_doc import execute
    execute(course, module, module_name)

    # Assert final file
    out_dir = outputs_base / course / "Lecture Notes" / module / module_name
    out_file = out_dir / f"{module_name}.tex"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Intro text." in content
    assert "Basics text." in content
    assert content.strip().endswith("\\end{document}")



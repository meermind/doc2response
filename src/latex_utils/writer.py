import os
from typing import List, Dict, Tuple, Optional
from src.latex_utils.sanitizer import LatexSanitizer
from src.latex_utils.metadata_store import MetadataStore


def sanitize_latex(text: str) -> str:
    return LatexSanitizer.sanitize(text)


class LatexFileWriter:
    """Low-level writer utilities for single section/subsection files."""

    @staticmethod
    def write_section(title: str, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"\\section{{{title}}}\n")

    @staticmethod
    def write_subsection(title: str, content: str, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sanitized = sanitize_latex(content)
        if "\\subsection" not in sanitized:
            sanitized = f"\\subsection{{{title}}}\n" + sanitized
        with open(path, 'w', encoding='utf-8') as f:
            f.write(sanitized)


class LatexSectionWriter:
    """High-level writer for section/subsection files with heading normalization."""

    def ensure_heading(self, kind: str, title: str, text: str) -> str:
        sanitized = sanitize_latex(text)
        try:
            import re as _re
            sanitized = _re.sub(r"^(?:\s*\\(?:section|subsection)\{[^}]*\}\s*\n)+", "", sanitized)
        except Exception:
            pass
        needs = "\\section" if kind == "section" else "\\subsection"
        if not sanitized.lstrip().startswith(needs):
            heading = f"\\section{{{title}}}\n" if kind == "section" else f"\\subsection{{{title}}}\n"
            sanitized = heading + sanitized
        return sanitized

    def write_section(self, intro_title: str, path: str) -> None:
        LatexFileWriter.write_section(intro_title, path)

    def write_section_content(self, title: str, text: str, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = self.ensure_heading("section", title, text)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def write_subsection(self, title: str, text: str, path: str) -> None:
        content = self.ensure_heading("subsection", title, text)
        LatexFileWriter.write_subsection(title, content, path)


class LatexAssembler:
    def __init__(self, output_base_dir: str, course_name: str, module_name: str) -> None:
        from src.models.paths import OutputPaths  # local import to avoid cycles
        self.paths = OutputPaths(base_dir=output_base_dir, course_name=course_name, module_name=module_name)
        self.module_dir = self.paths.module_dir()
        self.output_base_dir = output_base_dir
        self.course_name = course_name
        self.module_name = module_name
        self.writer = LatexSectionWriter()

    def write_intro(self, intro_title: str) -> str:
        intro_path = self.paths.intro_path(intro_title)
        self.writer.write_section(intro_title, intro_path)
        return intro_path

    def write_subsection(self, title: str, summary_latex: Optional[str]) -> str:
        sub_path = self.paths.subsection_path(title)
        text = summary_latex or f"\\subsection{{{title}}}\n"
        self.writer.write_subsection(title, text, sub_path)
        return sub_path

    def update_metadata(self, sections: List[Dict[str, str]]) -> None:
        MetadataStore.save_sections(self.module_dir, sections)

    def collate(self, section_refs: List[Dict[str, str]], intro_path: str, out_path: str) -> None:
        parts: List[str] = []
        try:
            with open(intro_path, 'r', encoding='utf-8') as f:
                parts.append(sanitize_latex(f.read()))
        except Exception:
            pass
        for sref in section_refs:
            if sref.get('type') != 'subsection':
                continue
            try:
                with open(sref.get('path', ''), 'r', encoding='utf-8') as f:
                    parts.append(sanitize_latex(f.read()))
            except Exception:
                continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join([p for p in parts if p.strip()]))

    def collate_from_metadata(self, input_base_dir: str, out_path: str) -> None:
        sections, intro_path = MetadataStore.load_sections(input_base_dir, self.course_name, self.module_name)
        self.collate(sections, intro_path, out_path)


def collate_assistant_message(section_refs: List[Dict[str, str]], intro_path: str, out_path: str) -> None:
    parts: List[str] = []
    try:
        with open(intro_path, 'r', encoding='utf-8') as f:
            parts.append(sanitize_latex(f.read()))
    except Exception:
        pass
    for sref in section_refs:
        if sref.get('type') != 'subsection':
            continue
        try:
            with open(sref.get('path', ''), 'r', encoding='utf-8') as f:
                parts.append(sanitize_latex(f.read()))
        except Exception:
            continue
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join([p for p in parts if p.strip()]))


def load_metadata_sections(input_base_dir: str, course: str, module_name: str) -> Tuple[List[Dict[str, str]], str]:
    return MetadataStore.load_sections(input_base_dir, course, module_name)


def collate_from_metadata(input_base_dir: str, course: str, module_name: str, out_path: str) -> None:
    assembler = LatexAssembler(input_base_dir, course, module_name)
    assembler.collate_from_metadata(input_base_dir, out_path)


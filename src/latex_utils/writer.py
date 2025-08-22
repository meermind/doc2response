import os
from typing import List, Dict, Tuple, Optional


def sanitize_latex(text: str) -> str:
    """Normalize and sanitize LaTeX text conservatively.

    Centralizes common fixes to avoid duplication across modules.
    """
    try:
        import re as _re
        # Strip code fences
        text = _re.sub(r"```+\s*latex|```+", "", text)
        # Convert literal escaped newlines/tabs
        text = text.replace("\\n", "\n").replace("\\t", "\t")
        # Remove literal tab characters and stray 'latex' lines
        text = text.replace("\t", " ")
        text = _re.sub(r"(?m)^[ \t]*latex[ \t]*$", "", text)
        # Common command typos
        text = _re.sub(r"(?<!\\)extbf\{", r"\\textbf{", text)
        text = _re.sub(r"(?<!\\)extit\{", r"\\textit{", text)
        text = _re.sub(r"(?<!\\)textrightarrow", r"\\rightarrow", text)
        text = _re.sub(r"(?<!\\)imes\b", r"\\times", text)
        # Remove stray \t macro if not used with braces (invalid in our outputs)
        text = _re.sub(r"\\t(?!\{)", " ", text)
        # Escape underscores outside math
        def _escape_underscores(s: str) -> str:
            out = []
            for line in s.splitlines():
                if ("$" not in line) and ("\\(" not in line) and ("\\[" not in line):
                    line = line.replace("_", "\\_")
                out.append(line)
            return "\n".join(out)
        text = _escape_underscores(text)
        # Escape stray & outside common environments (simple heuristic)
        def _escape_ampersands(s: str) -> str:
            out_lines = []
            env_depth = 0
            for line in s.splitlines():
                if "\\begin{" in line:
                    env_depth += 1
                if env_depth == 0:
                    line = _re.sub(r"(?<!\\)&", r"\\&", line)
                if "\\end{" in line and env_depth > 0:
                    env_depth -= 1
                out_lines.append(line)
            return "\n".join(out_lines)
        text = _escape_ampersands(text)
        return text.strip()
    except Exception:
        return text


def write_intro_file(intro_title: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"\\section{{{intro_title}}}\n")


def write_subsection_file(title: str, content: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sanitized = sanitize_latex(content)
    if "\\subsection" not in sanitized:
        sanitized = f"\\subsection{{{title}}}\n" + sanitized
    with open(path, 'w', encoding='utf-8') as f:
        f.write(sanitized)


class LatexSectionWriter:
    def ensure_heading(self, kind: str, title: str, text: str) -> str:
        """Ensure the LaTeX text starts with the correct heading and sanitize it.

        kind: "section" | "subsection"
        """
        sanitized = sanitize_latex(text)
        # Remove any leading heading line (either \section or \subsection) to avoid duplicates
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = self.ensure_heading("section", intro_title, "")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def write_subsection(self, title: str, text: str, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = self.ensure_heading("subsection", title, text)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)


class LatexAssembler:
    def __init__(self, output_base_dir: str, course_name: str, module_name: str) -> None:
        from src.models.paths import OutputPaths  # local import to avoid cycles
        self.paths = OutputPaths(base_dir=output_base_dir, course_name=course_name, module_name=module_name)
        self.module_dir = self.paths.module_dir()
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
        import json as _json
        meta_path = os.path.join(self.module_dir, "metadata.json")
        os.makedirs(self.module_dir, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            _json.dump({"sections": sections}, f, indent=2)


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
    """Load metadata.json for a module and return section refs and intro path.

    If an enhanced version of a subsection exists under '<module_dir>/enhanced',
    prefer that path over the original.
    """
    import json as _json
    module_dir = os.path.join(input_base_dir, course, module_name)
    metadata_path = os.path.join(module_dir, "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = _json.load(f)
    sections = meta.get('sections', [])
    enhanced_dir = os.path.join(module_dir, 'enhanced')
    # Prefer enhanced subsection file if present
    for s in sections:
        p = s.get('path')
        if not p:
            continue
        if s.get('type') == 'subsection':
            candidate = os.path.join(enhanced_dir, os.path.basename(p))
            if os.path.exists(candidate):
                s['path'] = candidate
    # Find intro (first section)
    intro_path = ""
    for s in sections:
        if s.get('type') == 'section':
            intro_path = s.get('path', '')
            break
    return sections, intro_path


def collate_from_metadata(input_base_dir: str, course: str, module_name: str, out_path: str) -> None:
    sections, intro_path = load_metadata_sections(input_base_dir, course, module_name)
    collate_assistant_message(sections, intro_path, out_path)


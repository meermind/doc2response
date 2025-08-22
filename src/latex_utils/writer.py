import os
from typing import List, Dict


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


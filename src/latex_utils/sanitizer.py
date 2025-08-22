from typing import List


class LatexSanitizer:
    @staticmethod
    def sanitize(text: str) -> str:
        try:
            import re as _re
            # Strip code fences
            text = _re.sub(r"```+\s*latex|```+", "", text)
            # Convert literal escaped newlines/tabs
            text = text.replace("\\n", "\n").replace("\\t", "\t")
            # Remove literal tab characters and stray 'latex' lines
            text = text.replace("\t", " ")
            text = _re.sub(r"(?m)^[ \t]*latex[ \t]*$", "", text)
            # Convert accidental html-like mdframed tags
            text = text.replace("<mdframed>", "\\begin{mdframed}").replace("</mdframed>", "\\end{mdframed}")
            # Common command typos (handle bare and mis-escaped forms, allow spaces)
            text = _re.sub(r"(?<!\\)extbf\s*\{", r"\\textbf{", text)
            text = _re.sub(r"\\extbf\s*\{", r"\\textbf{", text)
            text = _re.sub(r"(?<!\\)extit\s*\{", r"\\textit{", text)
            text = _re.sub(r"\\extit\s*\{", r"\\textit{", text)
            text = _re.sub(r"(?<!\\)textrightarrow", r"\\rightarrow", text)
            # Remove stray \t macro only when not starting a real command (avoid stripping '\\textbf')
            text = _re.sub(r"\\t(?![A-Za-z])", " ", text)
            # Unicode and common symbols
            replacements = {
                "π": "\\pi", "−": "-", "μ": "\\mu", "σ": "\\sigma", "ρ": "\\rho",
                "≈": "\\approx", "≤": "\\leq", "≥": "\\geq", "∑": "\\sum", "∫": "\\int",
                "∈": "\\in", "√": "\\sqrt{}", "∞": "\\infty",
            }
            for k, v in replacements.items():
                text = text.replace(k, v)
            # Remove combining macron (often stray)
            text = text.replace("\u0304", "")
            # Layout typos
            text = _re.sub(r"p\{\s*([0-9.]+)\s*extwidth\s*\}", r"p{\1\\textwidth}", text)
            text = _re.sub(r"(?m)^\s*oindent", r"\\noindent", text)
            text = _re.sub(r"(?mi)^mdframe suggestions.*$", r"% mdframe suggestions", text)
            # Escape underscores outside math
            def _escape_underscores(s: str) -> str:
                out = []
                for line in s.splitlines():
                    if ("$" not in line) and ("\\(" not in line) and ("\\[" not in line):
                        # Avoid double-escaping underscores already escaped
                        # Replace single underscores not preceded by a backslash
                        import re as _re_inner
                        line = _re_inner.sub(r"(?<!\\)_", r"\\_", line)
                    out.append(line)
                return "\n".join(out)
            text = _escape_underscores(text)
            # Escape stray & everywhere except in alignment/table environments
            def _escape_ampersands(s: str) -> str:
                out_lines: List[str] = []
                env_stack: List[str] = []
                begin_re = _re.compile(r"\\begin\{([^}]+)\}")
                end_re = _re.compile(r"\\end\{([^}]+)\}")
                allowed_amp_envs = {
                    "tabular", "tabularx", "array",
                    "align", "align*", "aligned", "alignedat",
                    "split", "gather", "gather*", "flalign", "flalign*", "alignat"
                }
                for raw in s.splitlines():
                    line = raw
                    # Track environment stack
                    for m in begin_re.finditer(line):
                        env_stack.append(m.group(1))
                    for m in end_re.finditer(line):
                        if env_stack and env_stack[-1] == m.group(1):
                            env_stack.pop()
                    # Decide whether to escape & on this line
                    in_allowed = any(env in allowed_amp_envs for env in env_stack)
                    if not in_allowed:
                        line = _re.sub(r"(?<!\\)&", r"\\&", line)
                    out_lines.append(line)
                return "\n".join(out_lines)
            text = _escape_ampersands(text)
            # Math de-noising and operatorname wrapping (outside math)
            def _wrap_operatorname(s: str) -> str:
                out = []
                for line in s.splitlines():
                    if ("$" in line) or ("\\(" in line) or ("\\[" in line):
                        out.append(line)
                    else:
                        out.append(_re.sub(r"(?<![\$\\])\\operatorname\{([^}]+)\}", r"\\(\\operatorname{\1}\\)", line))
                return "\n".join(out)
            text = _wrap_operatorname(text)
            text = _re.sub(r"\\\(\s*\\\(", r"\\(", text)
            text = _re.sub(r"\)\s*\\\(", r"(", text)
            text = _re.sub(r"\\\(\s*\(operatorname", r"\\(\\operatorname", text)
            # Heuristic: wrap consecutive \item lines into itemize if not within a list env
            def _wrap_lonely_items(s: str) -> str:
                lines = s.splitlines()
                env_stack: List[str] = []
                begin_re = _re.compile(r"^\\begin\{([^}]+)\}")
                end_re = _re.compile(r"^\\end\{([^}]+)\}")
                def in_list_env() -> bool:
                    return any(env in ("itemize", "enumerate", "description") for env in env_stack)
                out: List[str] = []
                i = 0
                while i < len(lines):
                    m_b = begin_re.match(lines[i])
                    m_e = end_re.match(lines[i])
                    if m_b:
                        env_stack.append(m_b.group(1)); out.append(lines[i]); i += 1; continue
                    if m_e:
                        if env_stack and env_stack[-1] == m_e.group(1):
                            env_stack.pop()
                        out.append(lines[i]); i += 1; continue
                    if not in_list_env() and lines[i].lstrip().startswith("\\item"):
                        block = []
                        while i < len(lines) and lines[i].lstrip().startswith("\\item"):
                            block.append(lines[i]); i += 1
                        out.append("\\begin{itemize}"); out.extend(block); out.append("\\end{itemize}")
                        continue
                    out.append(lines[i]); i += 1
                return "\n".join(out)
            text = _wrap_lonely_items(text)
            # Heuristic: flatten nested itemize/enumerate directly nested without text between
            def _flatten_nested_lists(s: str) -> str:
                # Remove redundant immediate nested begin/end pairs
                s = _re.sub(r"(?s)\\begin\{itemize\}\s*\\begin\{itemize\}", r"\\begin{itemize}", s)
                s = _re.sub(r"(?s)\\end\{itemize\}\s*\\end\{itemize\}", r"\\end{itemize}", s)
                s = _re.sub(r"(?s)\\begin\{enumerate\}\s*\\begin\{enumerate\}", r"\\begin{enumerate}", s)
                s = _re.sub(r"(?s)\\end\{enumerate\}\s*\\end\{enumerate\}", r"\\end{enumerate}", s)
                return s
            text = _flatten_nested_lists(text)
            # Replace \\times outside math with literal 'x'
            def _times_outside_math(s: str) -> str:
                out = []
                for line in s.splitlines():
                    if ("$" in line) or ("\\(" in line) or ("\\[" in line):
                        out.append(line)
                    else:
                        out.append(line.replace("\\times", "x"))
                return "\n".join(out)
            text = _times_outside_math(text)
            return text.strip()
        except Exception:
            return text



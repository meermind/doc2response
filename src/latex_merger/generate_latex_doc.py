import argparse
import os
import json
import shutil
from src.logger import configure_logging, get_logger
from src.models.latex import LatexMetadata, SectionRef, SectionContent
from src.models.paths import OutputPaths

# Utility function to sort sections/subsections by their order
def sort_by_order(item):
    return item['order']

def _normalize_common_command_typos(text: str) -> str:
    """Conservatively fix common LaTeX command typos near validation.

    Only replace when used as a command (immediately before '{') to avoid
    over-correcting regular words. Handles both bare and already-escaped forms.
    """
    try:
        import re as _re
        # textbf
        text = _re.sub(r"(?<!\\)extbf\{", r"\\textbf{", text)
        text = _re.sub(r"\\extbf\{", r"\\textbf{", text)
        # textit
        text = _re.sub(r"(?<!\\)extit\{", r"\\textit{", text)
        text = _re.sub(r"\\extit\{", r"\\textit{", text)
        return text
    except Exception:
        return text

def execute(course, module, module_name, input_base_dir=None, overwrite: bool | None = None, assistant_message_path: str | None = None):
    """
    Main function to generate a LaTeX document for a given module.
    """
    # Allow tests to control where metadata is read from via env
    if input_base_dir is None:
        input_base_dir = os.getenv("INPUT_BASE_DIR", "assistant_latex")
    # Load metadata.json to get the order of sections/subsections
    metadata_path = os.path.join(input_base_dir, course, module_name, "metadata.json")
    configure_logging()
    log = get_logger(__name__)
    log.info(f"[green]Merging[/] course={course} module={module} module_name={module_name}")
    with open(metadata_path, 'r') as file:
        raw_meta = json.load(file)

    # Normalize section file paths to this module's base directory when relative/inconsistent
    base_dir = os.path.join(input_base_dir, course, module_name)
    for sec in raw_meta.get('sections', []):
        p = sec.get('path')
        if not p:
            continue
        if not os.path.isabs(p):
            # if path already looks like assistant_latex/... but wrong course, force to current base with basename
            sec['path'] = os.path.join(base_dir, os.path.basename(p))
        else:
            # absolute but not under base_dir; leave as-is
            pass
    # Validate metadata
    metadata_model = LatexMetadata.model_validate(raw_meta)

    # Sort the sections and subsections by their order
    sorted_sections = sorted([s.model_dump() for s in metadata_model.sections], key=sort_by_order)

    # Read the start text
    with open(os.path.join('src/latex_merger', 'start.txt'), 'r') as file:
        start_text = file.read()

    # Replace placeholders in the start text
    replace_dict = {
        'TEMPLATE_COURSE_NAME': course,
        'TEMPLATE_MODULE_NAME': module_name,
        'TEMPLATE_LESSON_CODE': module
    }
    for key, value in replace_dict.items():
        start_text = start_text.replace(key, value)

    # Initialize the LaTeX content with the start text
    latex_content = start_text + '\n\n'

    # If an enhancer-collated assistant_message.tex exists (or is provided), prefer it directly
    provided_assistant = assistant_message_path
    if provided_assistant and not os.path.isabs(provided_assistant):
        provided_assistant = os.path.join(base_dir, os.path.basename(provided_assistant))
    assistant_path = provided_assistant or os.path.join(base_dir, "assistant_message.tex")
    if assistant_path and os.path.exists(assistant_path):
        log.info("[success]Using assistant_message.tex[/]: %s", assistant_path)
        with open(assistant_path, 'r') as af:
            assistant_content = af.read()
        # Normalize common command typos prior to validation (best-effort)
        assistant_content = _normalize_common_command_typos(assistant_content)
        latex_content += assistant_content + '\n\n'
    else:
        # Iterate through the sorted sections and append their content
        for section in sorted_sections:
            log.info(f"[cyan]Processing[/] {section['type']}: {section['title']}")

            # Prefer enhanced version if present and newer
            enhanced_path = None
            try:
                base_dir_for_sec = os.path.dirname(section['path'])
                enhanced_dir = os.path.join(base_dir_for_sec, 'enhanced')
                candidate = os.path.join(enhanced_dir, os.path.basename(section['path']))
                if os.path.exists(candidate):
                    enhanced_path = candidate
            except Exception:
                enhanced_path = None

            chosen_path = enhanced_path or section['path']
            if enhanced_path:
                log.info(f"[success]Using enhanced[/]: {os.path.basename(chosen_path)}")
            with open(chosen_path, 'r') as file:
                content = file.read()
            # Normalize common command typos prior to validation
            content = _normalize_common_command_typos(content)
            # Validate each section's content; self-heal known issues
            try:
                SectionContent(ref=SectionRef(**section), text=content)
                healed = content
            except Exception as e:
                # Self-heal minimal cases: wrap common math macros outside math, and ensure environments are closed
                healed = content
                # wrap math-only macros not already in math; and line-level wrapping when needed
                try:
                    import re as _re
                    macros = r"operatorname|mathbb|mathcal|frac|sum|int|lim|sqrt|hat"
                    def _wrap_inline(s: str) -> str:
                        return _re.sub(r"(?<![\$\\])\\operatorname\{([^}]+)\}", r"\\(\\operatorname{\1}\\)", s)
                    def _wrap_lines(s: str) -> str:
                        out = []
                        for line in s.splitlines():
                            if (not any(md in line for md in ("$", "\\(", "\\["))) and _re.search(r"\\(" + macros + r")(\\b|\\{)", line):
                                out.append(f"\\({line}\\)")
                            else:
                                out.append(line)
                        return "\n".join(out)
                    healed = _wrap_inline(_wrap_lines(healed))

                    # Heuristic for lonely \item: wrap consecutive \item lines not inside a list env
                    def _wrap_lonely_items(s: str) -> str:
                        lines = s.splitlines()
                        env_stack = []
                        begin_re = _re.compile(r"^\\begin\{([^}]+)\}")
                        end_re = _re.compile(r"^\\end\{([^}]+)\}")
                        def in_list_env() -> bool:
                            return any(env in ("itemize", "enumerate", "description") for env in env_stack)
                        out = []
                        i = 0
                        while i < len(lines):
                            m_b = begin_re.match(lines[i])
                            m_e = end_re.match(lines[i])
                            if m_b:
                                env_stack.append(m_b.group(1))
                                out.append(lines[i])
                                i += 1
                                continue
                            if m_e:
                                if env_stack and env_stack[-1] == m_e.group(1):
                                    env_stack.pop()
                                out.append(lines[i])
                                i += 1
                                continue
                            if not in_list_env() and lines[i].lstrip().startswith("\\item"):
                                # start collecting consecutive item lines
                                block = []
                                while i < len(lines) and lines[i].lstrip().startswith("\\item"):
                                    block.append(lines[i])
                                    i += 1
                                out.append("\\begin{itemize}")
                                out.extend(block)
                                out.append("\\end{itemize}")
                                continue
                            out.append(lines[i])
                            i += 1
                        return "\n".join(out)
                    healed = _wrap_lonely_items(healed)

                    # Escape underscores outside math on lines without math delimiters
                    def _escape_underscores(s: str) -> str:
                        esc = []
                        for line in s.splitlines():
                            if ("$" not in line) and ("\\(" not in line) and ("\\[" not in line):
                                line = line.replace("_", "\\_")
                            esc.append(line)
                        return "\n".join(esc)
                    healed = _escape_underscores(healed)

                    # Normalize common command typos
                    healed = _normalize_common_command_typos(healed)
                    # Remove literal tabs and stray 'latex' lines
                    healed = healed.replace("\t", " ")
                    healed = _re.sub(r"\\t\s*", " ", healed)
                    healed = _re.sub(r"(?m)^[ \t]*latex[ \t]*$", "", healed)

                    # Escape stray & characters outside alignment/table environments
                    def _escape_misplaced_ampersands(s: str) -> str:
                        # Only skip escaping inside alignment/table envs; escape elsewhere
                        align_envs = {"tabular", "tabularx", "array", "tabu", "align", "aligned", "alignat", "eqnarray"}
                        out_lines = []
                        env_stack = []
                        begin_re = _re.compile(r"\\begin\{([^}]+)\}")
                        end_re = _re.compile(r"\\end\{([^}]+)\}")
                        for line in s.splitlines():
                            m_b = begin_re.search(line)
                            m_e = end_re.search(line)
                            if m_b:
                                env_stack.append(m_b.group(1))
                            in_align = any(e in align_envs for e in env_stack)
                            if not in_align:
                                line = _re.sub(r"(?<!\\)&", r"\\&", line)
                            if m_e and env_stack and env_stack[-1] == m_e.group(1):
                                env_stack.pop()
                            out_lines.append(line)
                        return "\n".join(out_lines)
                    healed = _escape_misplaced_ampersands(healed)

                    # Math de-noising: fix duplicated inline math delimiters and operatorname typos
                    healed = _re.sub(r"\\\(\s*\\\(", r"\\(", healed)
                    healed = _re.sub(r"\)\s*\\\(", r"(", healed)
                    healed = _re.sub(r"\\\(\s*\(operatorname", r"\\(\\operatorname", healed)
                except Exception:
                    pass
                # Additional sanitization: unicode math, mdframed tags, missing headings, and code fences
                try:
                    import re as _re
                    def _sanitize_unicode(s: str) -> str:
                        rep = {
                            "π": "\\pi",
                            "−": "-",
                            "μ": "\\mu",
                            "σ": "\\sigma",
                            "ρ": "\\rho",
                            "≈": "\\approx",
                            "≤": "\\leq",
                            "≥": "\\geq",
                            "∑": "\\sum",
                            "∫": "\\int",
                            "∈": "\\in",
                            "√": "\\sqrt{}",
                            "∞": "\\infty",
                        }
                        for k, v in rep.items():
                            s = s.replace(k, v)
                        return s
                    healed = _sanitize_unicode(healed)
                    # Remove combining macron if present (often stray): U+0304
                    healed = healed.replace("\u0304", "")
                    # Fix common command typos/missing backslashes and bad tokens
                    healed = _re.sub(r"(?<!\\)textrightarrow", r"\\rightarrow", healed)
                    healed = _re.sub(r"(?<!\\)imes", r"\\times", healed)
                    healed = healed.replace("\t", " ")
                    healed = healed.replace("\u0009", " ")
                    healed = _re.sub(r"p\{\s*([0-9.]+)\s*extwidth\s*\}", r"p{\1\\textwidth}", healed)
                    healed = _re.sub(r"(?m)^\s*oindent", r"\\noindent", healed)
                    healed = _re.sub(r"(?mi)^mdframe suggestions.*$", r"% mdframe suggestions", healed)
                    # Convert accidental html-like mdframed tags
                    healed = healed.replace("<mdframed>", "\\begin{mdframed}").replace("</mdframed>", "\\end{mdframed}")
                    # Strip accidental code fences
                    healed = _re.sub(r"```+\\s*latex|```+", "", healed)
                    # Prepend missing heading if needed
                    if section['type'] == 'subsection' and "\\subsection" not in healed:
                        healed = f"\\subsection{{{section['title']}}}\n" + healed
                    if section['type'] == 'section' and "\\section" not in healed:
                        healed = f"\\section{{{section['title']}}}\n" + healed
                except Exception:
                    pass
                # Try validation again; if still failing, drop a .err.txt and continue
                try:
                    SectionContent(ref=SectionRef(**section), text=healed)
                except Exception as e2:
                    err_path = section['path'] + '.err.txt'
                    with open(err_path, 'w') as ef:
                        ef.write(str(e2))
                    log.warning(f"Validation failed for {section['title']} -> wrote error: {err_path}")
            
            latex_content += healed + '\n\n'

    # Add the end document tag
    latex_content += '\\end{document}\n'

    # Define the output path (allow override via env for tests)
    base_out = os.getenv('D2R_OUTPUT_BASE', '../tmp_latex_docs')
    save_path = os.path.join(base_out, course, 'Lecture Notes', module)
    designer_folder = '-'.join([module_name])
    output_filepath = os.path.join(save_path, designer_folder, f"{designer_folder}.tex")

    # Check if the output file already exists
    if overwrite is None:
        overwrite = bool(os.getenv('D2R_OVERWRITE', '0') == '1')
    module_output_dir = os.path.join(save_path, designer_folder)
    if overwrite and os.path.isdir(module_output_dir):
        try:
            shutil.rmtree(module_output_dir)
            log.info(f"[yellow]Cleared merged output[/]: {module_output_dir}")
        except Exception as e:
            log.warning(f"Could not clear merged output {module_output_dir}: {e}")
    elif os.path.exists(output_filepath):
        if not overwrite:
            ans = input(f"File {output_filepath} already exists. Do you want to overwrite? (y/n): ")
            if ans.lower() != 'y':
                print("Operation cancelled.")
                return

    # Create necessary directories and save the concatenated LaTeX content
    os.makedirs(os.path.join(save_path, designer_folder), exist_ok=True)

    with open(output_filepath, 'w') as file:
        file.write(latex_content)

    log.info(f"[bold green]LaTeX generated[/]: {output_filepath}")

def main():
    """
    Entry point for the script, allowing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a LaTeX document for a specific module.")
    parser.add_argument("--course", required=True, help="Course to process.")
    parser.add_argument("--module", required=True, help="Module code to process.")
    parser.add_argument("--module_name", required=True, help="Module name to process.")
    parser.add_argument(
        "--input_base_dir",
        default=os.getenv("INPUT_BASE_DIR", "assistant_latex"),
        help="Base directory where the writer saved outputs (default: assistant_latex or $INPUT_BASE_DIR)",
    )
    parser.add_argument(
        "--assistant_message_path",
        required=False,
        help="Path to assistant_message.tex to wrap into final document (optional)",
    )
    args = parser.parse_args()

    # Execute the function with the passed module
    execute(args.course, args.module, args.module_name, input_base_dir=args.input_base_dir, assistant_message_path=args.assistant_message_path)

if __name__ == '__main__':
    # Mock MODULE_NAME for debugging
    # DEBUG_COURSE = "CM2025 Computer Security"
    # DEBUG_MODULE = "Topic 1"
    # DEBUG_MODULE_NAME = "Topic 1 Malware analysis"

    # Call the execute function directly with the mocked MODULE_NAME
    # execute(DEBUG_COURSE, DEBUG_MODULE, DEBUG_MODULE_NAME)
    main()


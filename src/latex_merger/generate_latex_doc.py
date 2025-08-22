import argparse
import os
import json
import shutil
from src.latex_utils.metadata_store import MetadataStore
from src.logger import configure_logging, get_logger
from src.models.latex import LatexMetadata
from src.models.paths import OutputPaths

# Utility function to sort sections/subsections by their order
def sort_by_order(item):
    return item['order']

## Note: Do not sanitize or heal content here. Enhanced outputs should already be clean.

def execute(course, module, module_name, input_base_dir=None, overwrite: bool | None = None, assistant_message_path: str | None = None, lesson_slug: str | None = None):
    """
    Main function to generate a LaTeX document for a given module.
    """
    # Allow tests to control where metadata is read from via env
    if input_base_dir is None:
        input_base_dir = os.getenv("INPUT_BASE_DIR", "assistant_latex")
    # Resolve module directories consistently via MetadataStore
    module_dir = MetadataStore.module_dir(input_base_dir, course, module_name, lesson_slug)
    metadata_dir = MetadataStore.metadata_dir(module_dir)
    configure_logging()
    log = get_logger(__name__)
    log.info(f"[green]Merging[/] course={course} module={module} module_name={module_name}")
    # Load sections via MetadataStore (prefers enhanced subsection files when present)
    sections, _intro_path = MetadataStore.load_sections(input_base_dir, course, module_name, lesson_slug)
    raw_meta = {"sections": sections}
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

    # Assume an assistant_message_path is always provided
    if not assistant_message_path:
        log.error("assistant_message_path is required but was not provided")
        return
    assistant_path = assistant_message_path
    if not os.path.isabs(assistant_path):
        # If the provided relative path already starts with input_base_dir (e.g., 'assistant_latex/...'), use as-is
        starts_with_base = assistant_path.startswith(input_base_dir + os.sep)
        starts_with_module = assistant_path.startswith(module_dir + os.sep) or (assistant_path == module_dir)
        if not (starts_with_base or starts_with_module):
            assistant_path = os.path.join(module_dir, assistant_path)
    if not os.path.exists(assistant_path):
        log.error("assistant_message not found at: %s", assistant_path)
        return
    log.info("[success]Using assistant_message.tex[/]: %s", assistant_path)
    with open(assistant_path, 'r') as af:
        assistant_content = af.read()
    latex_content += assistant_content + '\n\n'

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


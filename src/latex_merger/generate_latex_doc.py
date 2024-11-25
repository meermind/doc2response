import argparse
import os
import json

# Utility function to sort sections/subsections by their order
def sort_by_order(item):
    return item['order']

def execute(course, module, module_name):
    """
    Main function to generate a LaTeX document for a given module.
    """
    # Load metadata.json to get the order of sections/subsections
    metadata_path = f"assistant_latex/{course}/{module_name}/metadata.json"
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    # Sort the sections and subsections by their order
    sorted_sections = sorted(metadata['sections'], key=sort_by_order)

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

    # Iterate through the sorted sections and append their content
    for section in sorted_sections:
        print(f"Processing {section['type']}: {section['title']}")

        with open(section['path'], 'r') as file:
            content = file.read()

        latex_content += content + '\n\n'

    # Add the end document tag
    latex_content += '\\end{document}\n'

    # Define the output path
    save_path = os.path.join('../tmp_latex_docs', course, 'Lecture Notes', module)
    designer_folder = '-'.join([module_name])
    output_filepath = os.path.join(save_path, designer_folder, f"{designer_folder}.tex")

    # Check if the output file already exists
    if os.path.exists(output_filepath):
        overwrite = input(f"File {output_filepath} already exists. Do you want to overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation cancelled.")
            return

    # Create necessary directories and save the concatenated LaTeX content
    os.makedirs(os.path.join(save_path, designer_folder), exist_ok=True)

    with open(output_filepath, 'w') as file:
        file.write(latex_content)

    print(f"LaTeX document successfully generated at: {output_filepath}")

def main():
    """
    Entry point for the script, allowing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a LaTeX document for a specific module.")
    parser.add_argument("--course", required=True, help="Course to process.")
    parser.add_argument("--module", required=True, help="Module code to process.")
    parser.add_argument("--module_name", required=True, help="Module name to process.")
    args = parser.parse_args()

    # Execute the function with the passed module
    execute(args.course, args.module, args.module_name)

if __name__ == '__main__':
    # Mock MODULE_NAME for debugging
    # DEBUG_COURSE = "CM2025 Computer Security"
    # DEBUG_MODULE = "Topic 1"
    # DEBUG_MODULE_NAME = "Topic 1 Malware analysis"

    # Call the execute function directly with the mocked MODULE_NAME
    # execute(DEBUG_COURSE, DEBUG_MODULE, DEBUG_MODULE_NAME)
    main()


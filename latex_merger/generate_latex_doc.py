import os
import re
import json

from dotenv import load_dotenv
load_dotenv()  # This will load the variables from the .env file

COURSE = os.environ['COURSE']
MODULE = os.environ['MODULE']
MODULE_NAME = os.environ['MODULE_NAME']
SECTION_NAME = os.environ['SECTION_NAME']

# Utility function to sort sections/subsections by their order
def sort_by_order(item):
    return item['order']

def execute(module):
    # Load metadata.json to get the order of sections/subsections
    metadata_path = f"assistant_latex/{COURSE}/{MODULE_NAME}/metadata.json"
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    # Sort the sections and subsections by their order
    sorted_sections = sorted(metadata['sections'], key=sort_by_order)

    # Read the start text
    with open(os.path.join('latex_merger', 'start.txt'), 'r') as file:
        start_text = file.read()

    # Replace placeholders in the start text
    replace_dict = {
        'TEMPLATE_COURSE_NAME': COURSE,
        'TEMPLATE_MODULE_NAME': MODULE_NAME,
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
    save_path = os.path.join('../tmp_latex_docs', COURSE, 'Lecture Notes', module)
    designer_folder = '-'.join([SECTION_NAME])
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

if __name__ == '__main__':
    execute(MODULE)

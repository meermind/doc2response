# demo_transcript_to_docs.py

import os
import json
from pathlib import Path
from llama_index.core import Document
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()  # This will load the variables from the .env file

TRANSCRIPT_PATH = os.environ['TRANSCRIPT_PATH']
METADATA_FILE = os.environ['METADATA_FILE']

def load_metadata(metadata_file: str) -> dict:
    """Load metadata from a JSON file."""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_file}")
        return {}

def get_txt_files(directory: str) -> defaultdict:
    """Recursively fetch all .txt files from the transcript path."""
    txt_files = defaultdict(list)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                base_name = os.path.splitext(file)[0]
                txt_files[base_name].append(os.path.join(root, file))
    return txt_files

def create_documents_with_metadata(metadata: dict, transcript_files: defaultdict) -> list:
    """Create Document objects enriched with metadata."""
    documents = []

    def get_parent_dir(file_path: str) -> str:
        """Extract the parent directory path."""
        return os.path.dirname(file_path)

    for module in metadata.get("modules", []):
        module_name = module["module_name"]
        module_slug = module["module_slug"]

        for lesson in module.get("lessons", []):
            lesson_name = lesson["lesson_name"]
            lesson_slug = lesson["lesson_slug"]

            for item in lesson.get("items", []):
                item_name = item["name"]
                item_slug = item["transformed_slug"]

                for content in item.get("content", []):
                    if content["content_type"] == "transcript":
                        srt_parent_dir = get_parent_dir(content["path"])
                        srt_base_name = os.path.splitext(os.path.basename(srt_parent_dir))[0]

                        matching_txt_files = [
                            txt_file for txt_file in transcript_files["transcript"]
                            if get_parent_dir(txt_file).endswith(srt_base_name)
                        ]

                        for txt_file in matching_txt_files:
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                content_text = f.read()

                            document = Document(
                                text=content_text,
                                metadata={
                                    "course_name": metadata.get("course_name", ""),
                                    "course_slug": metadata.get("course_slug", ""),
                                    "module_name": module_name,
                                    "module_slug": module_slug,
                                    "lesson_name": lesson_name,
                                    "lesson_slug": lesson_slug,
                                    "item_name": item_name,
                                    "item_slug": item_slug,
                                    "srt_file": content["path"],
                                    "txt_file": txt_file,
                                }
                            )
                            documents.append(document)
    return documents

def transcripts_to_docs(transcript_path: str, metadata_file: str):
    """Main function to generate documents with metadata."""
    metadata = load_metadata(metadata_file)
    transcript_files = get_txt_files(transcript_path)
    documents = create_documents_with_metadata(metadata, transcript_files)

    print(f"Created {len(documents)} documents.")
    for doc in documents[:5]:
        print(f"Content: {doc.text[:50]}... | Metadata: {doc.metadata}")

    return documents

if __name__ == "__main__":
    # Define paths to be used (can be configured or passed as arguments)
    documents = transcripts_to_docs(TRANSCRIPT_PATH, METADATA_FILE)

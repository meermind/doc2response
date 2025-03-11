import os
import sys
import subprocess
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_topic_data(metadata_file, topic_number):
    """
    Extracts necessary fields for a specific topic by checking if the module_name starts with 'Topic X'.

    :param metadata_file: Path to the JSON metadata file.
    :param topic_number: The topic number to filter (e.g., "6" for "Topic 6").
    :return: List of dictionaries containing metadata for the specified topic.
    """
    with open(metadata_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    course_name = data["course_name"]
    course_slug = data["course_slug"]
    base_transcript_path = f"../course-crawler/outputs/structured_transcripts/dl_coursera/{course_slug}"

    filtered_data = []

    for module in data["modules"]:
        module_name = module["module_name"]

        if module_name.startswith(f"Topic {topic_number}"):
            transcript_path = f"{base_transcript_path}/{module['module_slug']}"
            filtered_data.append({
                "metadata_file": metadata_file,
                "transcript_path": transcript_path,
                "course": course_name,
                "module": f"Topic {topic_number}",
                "module_name": module_name,
            })

    return filtered_data  # Returns a list (even if only one match is found)

# Define pipeline functions
def run_load_docs(transcript_path, metadata_file):
    """
    Run the document loading pipeline.
    """
    print("Running: demo_load_docs_to_llamaindex.py")
    subprocess.run([
        sys.executable, "src/demo_load_docs_to_llamaindex.py",
        "--transcript_path", transcript_path,
        "--metadata_file", metadata_file
    ], check=True)

def run_call_llamaindex(module_name):
    """
    Run the LlamaIndex processing pipeline.
    """
    print("Running: demo_call_llamaindex.py")
    subprocess.run([
        sys.executable, "src/demo_call_llamaindex.py",
        "--module_name", module_name
    ], check=True)

def run_generate_latex(course, module, module_name):
    """
    Run the LaTeX generation pipeline.
    """
    print("Running: generate_latex_doc.py")
    subprocess.run([
        sys.executable, "src/latex_merger/generate_latex_doc.py",
        "--course", course,
        "--module", module,
        "--module_name", module_name
    ], check=True)

def orchestrate_pipeline(run_load=True, run_call=True, run_generate=True):
    """
    Orchestrate the pipeline based on the provided flags.
    """
    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "test_data/01@topic-1-introduction-to-computer-securit"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 1'
    # module_name = "Topic 1 Introduction to computer security and malware"

    # # Extract environment variables
    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "test_data/02@topic-1-malware-analysis"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 1'
    # module_name = "Topic 1 Malware analysis"

    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/03@topic-2-network-security-dos-attacks-and"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 2'
    # module_name = "Topic 2 Network security - DoS attacks and botnets"

    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/04@topic-2-network-security-defence-with-fi"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 2'
    # module_name = "Topic 2 Network security - defence with firewalls and intrusion detection systems"

    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/05@topic-3-operating-system-security-filesy"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 3'
    # module_name = "Topic 3 Operating system security - filesystems and windows"

    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/06@topic-3-operating-system-security-gnu-li"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 3'
    # module_name = "Topic 3 Operating system security - GNU, Linux, Android and containerisation"

    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/07@topic-4-understanding-cryptography-histo"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 4'
    # module_name = "Topic 4 Understanding Cryptography - History of Assymetric and Symmetric Cryptography"

    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/08@topic-4-understanding-cryptography-trans"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 4'
    # module_name = "Topic 4 Understanding cryptography - Transposition and Substitution"

    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/09@topic-5-rsa-public-key-cryptography-prim"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 5'
    # module_name = "Topic 5 RSA public-key cryptography - primes, Phi and security"

    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/10@topic-5-rsa-public-key-cryptography-proo"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 5'
    # module_name = "Topic 5 RSA public-key cryptography - proof that RSA works"

    # metadata_file = "../course-crawler/crawled_metadata/youtube/openmined.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/youtube/openmined/01@topic01-privacy-preserving-ai/01@lesson01-privacy-preserving-ai/01@privacy-preserving-ai"
    # course = 'Openmined'
    # module = 'Topic 1'
    # module_name = "Topic01 Privacy Preserving Ai"

    # Example usage
    metadata_path = "/Users/datagero/Documents/offline_repos/course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    topic = 7  # Example: Fetch data for "Topic 6"
    topic_data = extract_topic_data(metadata_path, topic)

    for topic in topic_data:
        metadata_file = topic["metadata_file"]
        transcript_path = topic["transcript_path"]
        course = topic["course"]
        module = topic["module"]
        module_name = topic["module_name"]

        if not transcript_path or not metadata_file or not module_name:
            print("Error: TRANSCRIPT_PATH, METADATA_FILE, and MODULE_NAME must be set in the .env file.")
            return

        try:
            if run_load:
                run_load_docs(transcript_path, metadata_file)
            if run_call:
                run_call_llamaindex(module_name)
            if run_generate:
                run_generate_latex(course, module, module_name)
        except subprocess.CalledProcessError as e:
            print(f"Pipeline step failed: {e}")

if __name__ == "__main__":
    # Example DAG execution: All steps
    orchestrate_pipeline(run_load=True, run_call=True, run_generate=True)

    # Uncomment for specific step execution
    # orchestrate_pipeline(run_load=False, run_call=True, run_generate=False)

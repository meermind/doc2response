import os
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define pipeline functions
def run_load_docs(transcript_path, metadata_file):
    """
    Run the document loading pipeline.
    """
    print("Running: demo_load_docs_to_llamaindex.py")
    subprocess.run([
        "python", "src/demo_load_docs_to_llamaindex.py",
        "--transcript_path", transcript_path,
        "--metadata_file", metadata_file
    ], check=True)

def run_call_llamaindex(module_name):
    """
    Run the LlamaIndex processing pipeline.
    """
    print("Running: demo_call_llamaindex.py")
    subprocess.run([
        "python", "src/demo_call_llamaindex.py",
        "--module_name", module_name
    ], check=True)

def run_generate_latex(course, module, module_name):
    """
    Run the LaTeX generation pipeline.
    """
    print("Running: generate_latex_doc.py")
    subprocess.run([
        "python", "src/latex_merger/generate_latex_doc.py",
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

    metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/05@topic-3-operating-system-security-filesy"
    course = 'CM2025 Computer Security'
    module = 'Topic 3'
    module_name = "Topic 3 Operating system security - filesystems and windows"

    # metadata_file = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
    # transcript_path = "../course-crawler/outputs/structured_transcripts/dl_coursera/uol-cm2025-computer-security/06@topic-3-operating-system-security-gnu-li"
    # course = 'CM2025 Computer Security'
    # module = 'Topic 3'
    # module_name = "Topic 3 Operating system security - GNU, Linux, Android and containerisation"

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
    orchestrate_pipeline(run_load=False, run_call=True, run_generate=True)

    # Uncomment for specific step execution
    # orchestrate_pipeline(run_load=False, run_call=True, run_generate=False)

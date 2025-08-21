import os
import sys
import subprocess
import json
import argparse
from dotenv import load_dotenv
from src.models.metadata import load_course_metadata
from src.logger import configure_logging, get_logger

# Load environment variables and logging
load_dotenv()
configure_logging()
log = get_logger(__name__)

def extract_topic_data(metadata_file, topic_number):
    """
    Extracts necessary fields for a specific topic by checking if the module_name starts with 'Topic X'.

    :param metadata_file: Path to the JSON metadata file.
    :param topic_number: The topic number to filter (e.g., "6" for "Topic 6").
    :return: List of dictionaries containing metadata for the specified topic.
    """
    # Load and validate metadata using pydantic models; resolve relative paths if needed
    project_dir = os.getenv("PROJECT_DIR")
    course = load_course_metadata(metadata_file, project_dir)
    module = course.select_module_by_index(topic_number)

    return [{
        "metadata_file": metadata_file,
        "course": course.course_name,
        "module": f"Topic {topic_number}",
        "module_name": module.module_name,
        "module_index": topic_number,
        "module_slug": module.module_slug,
    }]

# Define pipeline functions
def run_load_docs(metadata_file, topic_number, project_dir="..", overwrite=False):
    """
    Run the document loading pipeline by calling the module directly.
    """
    log.info("[bold cyan]Running[/] (direct): demo_load_docs_to_llamaindex.main")
    from src.demo_load_docs_to_llamaindex import main as load_main
    # Call programmatically to avoid subprocess; this still requires proper env/DB when enabled
    load_main(metadata_file=metadata_file, topic_number=topic_number, project_dir=project_dir, overwrite=overwrite)

def run_call_llamaindex(module_name, module_slug=None, output_base_dir=None):
    """
    Run the LlamaIndex processing pipeline by calling the module directly.
    """
    log.info("[bold cyan]Running[/] (direct): demo_call_llamaindex.main")
    from src.demo_call_llamaindex import main as call_main
    args = ["--module_name", module_name]
    if module_slug:
        args += ["--module_slug", module_slug]
    if output_base_dir:
        args += ["--output_base_dir", output_base_dir]
    call_main(args=args)

def run_build_skeleton(module_name, module_slug=None, output_base_dir=None, list_top_k=None, course_name=None):
    """
    Build the lecture-notes skeleton (skeleton.json, metadata.json, subsection_mapping.json).
    """
    log.info("[bold cyan]Running[/] (direct): skeleton_builder.build_skeleton")
    from src.skeleton_builder import build_skeleton
    if output_base_dir is None:
        output_base_dir = os.getenv("D2R_OUTPUT_BASE") or "assistant_latex"
    if list_top_k is None:
        list_top_k = int(os.getenv("D2R_LIST_TOP_K_SKELETON", "500"))
    if course_name is None:
        course_name = os.getenv("COURSE_NAME", "Course")
    return build_skeleton(module_name, module_slug, output_base_dir, list_top_k, course_name)

def run_enhance_subsections(course_name, module_name, module_slug=None, output_base_dir=None, top_k_item=None):
    """
    Enhance subsections by generating per-subsection LaTeX files using the skeleton.
    """
    log.info("[bold cyan]Running[/] (direct): subsection_enhancer.enhance_subsections")
    from src.subsection_enhancer import enhance_subsections
    if output_base_dir is None:
        output_base_dir = os.getenv("D2R_OUTPUT_BASE") or "assistant_latex"
    if top_k_item is None:
        top_k_item = int(os.getenv("D2R_TOP_K_ITEM", "10"))
    enhance_subsections(course_name, module_name, module_slug, output_base_dir, top_k_item)

def run_generate_latex(course, module, module_name, input_base_dir=None, overwrite=False):
    """
    Run the LaTeX generation pipeline by calling the module directly.
    """
    log.info("[bold cyan]Running[/] (direct): latex_merger.generate_latex_doc.execute")
    from src.latex_merger.generate_latex_doc import execute
    if input_base_dir:
        execute(course, module, module_name, input_base_dir=input_base_dir, overwrite=overwrite)
    else:
        execute(course, module, module_name, overwrite=overwrite)

def orchestrate_pipeline(run_load=True, run_call=True, run_generate=True, metadata_file=None, topic_number=None, overwrite=False, run_skeleton=None, run_enhance=None):
    """
    Orchestrate the pipeline based on the provided flags.

    Parameters may be provided directly or via environment variables:
    - metadata_file: path to metadata JSON (env: METADATA_FILE)
    - topic_number: 1-based topic/module index (env: TOPIC_NUMBER)
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
    # Resolve inputs from args/env with sensible defaults for local debugging
    metadata_path = metadata_file or os.getenv(
        "METADATA_FILE",
        "/Users/matias.vizcaino/Documents/datagero_repos/meermind/course-crawler/crawled_metadata/gatech/simulation.json",
    )
    topic = int(topic_number or os.getenv("TOPIC_NUMBER", "1"))
    log.info(f"[green]Input[/] METADATA_FILE={metadata_path} TOPIC_NUMBER={topic}")
    topic_data = extract_topic_data(metadata_path, topic)

    for topic in topic_data:
        metadata_file = topic["metadata_file"]
        course = topic["course"]
        module = topic["module"]
        module_name = topic["module_name"]
        module_index = topic["module_index"]
        module_slug = topic.get("module_slug")

        if not metadata_file or not module_name:
            print("Error: METADATA_FILE and MODULE_NAME must be set.")
            return

        try:
            if run_load:
                run_load_docs(metadata_file, module_index, overwrite=overwrite)
            # Determine two-phase flags
            # Default now: build skeleton only; enhancer is disabled unless explicitly enabled
            do_skeleton = run_skeleton if run_skeleton is not None else bool(run_call)
            enhancer_enabled = os.getenv("D2R_ENABLE_ENHANCER", "0") == "1"
            do_enhance = run_enhance if run_enhance is not None else (bool(run_call) and enhancer_enabled)
            if not do_enhance:
                log.info("[yellow]Enhancer disabled[/] (set D2R_ENABLE_ENHANCER=1 or pass run_enhance=True to enable)")
            if do_skeleton:
                run_build_skeleton(module_name, module_slug=module_slug, course_name=course)
            if do_enhance:
                run_enhance_subsections(course, module_name, module_slug=module_slug)
            if run_generate:
                # Call without extra args to preserve monkeypatched test fakes
                run_generate_latex(course, module, module_name, overwrite=overwrite)
        except subprocess.CalledProcessError as e:
            log.error(f"[red]Pipeline step failed[/]: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run doc2response orchestration pipeline")
    parser.add_argument("--metadata_file", required=False, help="Path to metadata JSON")
    parser.add_argument("--topic_number", type=int, required=False, help="1-based topic index to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data and outputs without prompting")
    parser.add_argument("--skip_load", action="store_true", help="Skip load docs step")
    parser.add_argument("--skip_call", action="store_true", help="Skip LLM call step")
    parser.add_argument("--skip_generate", action="store_true", help="Skip LaTeX generation step")
    args = parser.parse_args()

    orchestrate_pipeline(
        run_load=not args.skip_load,
        run_call=not args.skip_call,
        run_generate=not args.skip_generate,
        metadata_file=args.metadata_file,
        topic_number=args.topic_number,
        overwrite=bool(args.overwrite or os.getenv("D2R_OVERWRITE", "0") == "1"),
    )

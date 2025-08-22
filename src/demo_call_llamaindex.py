import argparse
from dotenv import load_dotenv
from src.logger import configure_logging, get_logger
import os
import json
import logging
import sys
from typing import List, Dict
import shutil
from sqlalchemy import URL
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterCondition

from models.config import TiDBSettings
from src.models.settings import Settings
from src.models.context import CourseContext
from src.models.paths import OutputPaths
from models.latex import LatexMetadata, SectionRef

# Load environment variables
load_dotenv()
configure_logging()
log = get_logger(__name__)

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class SubsectionPlan(BaseModel):
    title: str
    enhanced_latex: str


class WriterResult(BaseModel):
    course_name: str
    module_name: str
    intro_section_latex: str
    subsections: List[SubsectionPlan]

def load_prompt(file_name):
    """Helper: Load a prompt from a file."""
    with open(os.path.join("prompts", file_name), "r", encoding="utf-8") as f:
        return f.read().strip()

def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate enhanced LaTeX content with a typed agent.")
    parser.add_argument("--module_name", help="Name of the module to process.")
    parser.add_argument("--module_slug", required=False, help="Slug of the module; used for robust filtering and diagnostics.")
    parser.add_argument("--vector_table_name", default="demo_load_docs_to_llamaindex", help="Vector table name.")
    parser.add_argument(
        "--output_base_dir",
        default=os.getenv("OUTPUT_BASE_DIR") or os.getenv("D2R_OUTPUT_BASE") or "assistant_latex",
        help="Base directory where LaTeX outputs are written (default: assistant_latex or $OUTPUT_BASE_DIR/$D2R_OUTPUT_BASE)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, clear the module output directory before writing.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable low-cost test mode (limit retrieval and optionally truncate context).",
    )
    parser.add_argument(
        "--per_item",
        action="store_true",
        help="(Planned) Process each item_slug independently and then aggregate.",
    )
    parser.add_argument(
        "--phase",
        choices=["skeleton", "enhance", "both"],
        default=os.getenv("D2R_PHASE", "both"),
        help="Which phase to run: build outline skeleton, enhance subsections, or both.",
    )
    cli_args = parser.parse_args(args)

    # Override environment variables with CLI arguments if provided
    module_name = cli_args.module_name or os.getenv("MODULE_NAME")
    vector_table_name = cli_args.vector_table_name
    is_test_mode = cli_args.test or os.getenv("D2R_TEST_MODE", "0") == "1"
    per_item_mode = bool(cli_args.per_item)
    module_slug_arg = cli_args.module_slug
    phase = cli_args.phase
    overwrite = bool(cli_args.overwrite or os.getenv("D2R_OVERWRITE", "0") == "1")
    context = CourseContext(
        course_name=os.getenv("COURSE_NAME", "Course"),
        module_name=module_name,
        module_slug=module_slug_arg,
        output_base=cli_args.output_base_dir,
        input_base=cli_args.output_base_dir,
        list_top_k_skeleton=int(os.getenv("D2R_LIST_TOP_K_SKELETON", "500")),
        list_top_k_item=int(os.getenv("D2R_LIST_TOP_K_ITEM", "50")),
        top_k=int(os.getenv("D2R_TOP_K", "20")),
        top_k_item=int(os.getenv("D2R_TOP_K_ITEM", "10")),
    )

    if not module_name:
        logger.error("Module name must be specified via CLI or environment variables.")
        sys.exit(1)

    # Load configuration and prompts
    db = TiDBSettings()
    settings = Settings()
    assistant_message_prompt = load_prompt(settings.prompt_assistant_message_file)
    intro_query_prompt = load_prompt(settings.prompt_intro_query_file)
    subsection_query_prompt = load_prompt(settings.prompt_subsection_query_file)

    # Prepare embeddings and vector store
    # Embeddings via central settings
    embedding_model = settings.create_embedding()
    embedding_dimension = settings.embedding_dimension

    tidb_connection_url = URL(
        "mysql+pymysql",
        username=db.username,
        password=db.password,
        host=db.host,
        port=int(db.port),
        database=db.db_name,
        query={"ssl_verify_cert": True, "ssl_verify_identity": True},
    )

    vector_store = TiDBVectorStore(
        connection_string=tidb_connection_url,
        table_name=vector_table_name,
        distance_strategy="cosine",
        vector_dimension=embedding_dimension,
        drop_existing_table=False,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embedding_model,
    )

    # Define base filters and retriever for this module
    # Prefer module_slug if available
    base_filters = context.module_filters()
    base_metadata_filters = MetadataFilters(filters=[MetadataFilter(**f) for f in base_filters])

    # Limit retrieval payload to reduce cost
    # Retrieval sizes depend on mode (skeleton vs per-item)
    if is_test_mode:
        top_k = int(os.getenv("D2R_TOP_K_TEST", str(context.top_k)))
        top_k_item = int(os.getenv("D2R_TOP_K_ITEM_TEST", str(context.top_k_item)))
        list_top_k = int(os.getenv("D2R_LIST_TOP_K_TEST", str(context.list_top_k_skeleton)))
    else:
        top_k = context.top_k
        top_k_item = context.top_k_item
        list_top_k = context.list_top_k_skeleton if not per_item_mode else context.list_top_k_item
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        embedding_model=embedding_model,
        filters=base_metadata_filters,
    )

    # Define the agent and its tools
    agent = Agent(
        model=settings.ai_writer_model,
        system_prompt=(
            "You are a LaTeX writer agent. Generate valid LaTeX only (no code fences). "
            "Produce one introduction section and multiple subsections."
        ),
    )

    @agent.tool
    def list_unique_topics(ctx: RunContext) -> List[str]:
        """Return unique item slugs for this module from the vector store."""
        list_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=list_top_k,
            embedding_model=embedding_model,
            filters=base_metadata_filters,
        )
        source_nodes = list_retriever.retrieve("List all the files")
        topics = sorted(list({x.metadata.get("item_slug", "") for x in source_nodes if x.metadata.get("item_slug")}))
        log.info(f"[cyan]Topics found[/] ({len(topics)}):")
        for t in topics:
            log.info(f"- {t}")
        return topics

    @agent.tool
    def get_course_and_module(ctx: RunContext) -> Dict[str, str]:
        """Return course_name and module_name for this module."""
        nodes = base_retriever.retrieve("List all the files")
        if not nodes:
            return {"course_name": "Unknown Course", "module_name": module_name}
        course_names = {n.metadata.get("course_name", "") for n in nodes}
        return {"course_name": sorted(list(course_names))[0] if course_names else "", "module_name": module_name}

    @agent.tool
    def retrieve_context_for_subsection(ctx: RunContext, title: str, content_hint: str = "") -> List[Dict]:
        """Retrieve top items relevant to a subsection (title + optional draft content).

        Note: returns only metadata (no content) to avoid logging/sending large text. Includes slugs.
        """
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=(top_k_item if per_item_mode else top_k),
            embedding_model=embedding_model,
            filters=base_metadata_filters,
        )
        query_text = f"{title}: {content_hint}" if content_hint else title
        source_nodes = retriever.retrieve(query_text)
        results = [
            {
                "course_slug": n.metadata.get("course_slug", ""),
                "module_slug": n.metadata.get("module_slug", ""),
                "lesson_slug": n.metadata.get("lesson_slug", ""),
                "item_slug": n.metadata.get("item_slug", ""),
            }
            for n in source_nodes
        ]
        # Log in INFO with pretty format
        try:
            log.info("Top nodes (metadata only) for '%s':\n%s", title, json.dumps(results, indent=2))
        except Exception:
            log.info("Top nodes (metadata only) for '%s': %s", title, results)
        return results

    # Compose the user task, embedding your existing prompt engineering as guidance
    unique_topics = list_unique_topics(None)
    unique_topics_str = ", ".join(unique_topics)
    task = (
        f"{assistant_message_prompt}\n\n=== Guidance ===\n"
        f"Use these topic cues as context seeds (not one-to-one subsections): {unique_topics_str}.\n"
        f"First, propose a comprehensive outline for the module. Then, for each transcript/topic, dissect it into detailed subsections.\n"
        f"Finally, produce an introduction and all subsections ready to be merged.\n"
        f"Intro guidance: {intro_query_prompt}\n"
        f"Subsection guidance: {subsection_query_prompt}\n"
        f"Module to process: {module_name}."
    )

    # Run the agent to obtain outline + content plan
    result = agent.run_sync(task)

    # Extract names and ensure output directories
    info = get_course_and_module(None)
    course_name = info.get("course_name", "Course") or "Course"
    module_name_out = info.get("module_name", module_name)

    out_dir = OutputPaths(base_dir=context.output_base, course_name=course_name, module_name=module_name_out).module_dir()
    if overwrite and os.path.isdir(out_dir):
        try:
            shutil.rmtree(out_dir)
            log.info(f"[yellow]Cleared writer output[/]: {out_dir}")
        except Exception as e:
            log.warning(f"Could not clear writer output {out_dir}: {e}")
    os.makedirs(out_dir, exist_ok=True)

    # Skeleton (outline) + Enhancement separation
    sections: List[SectionRef] = []
    assistant_message_path = os.path.join(out_dir, "assistant_message.tex")
    intro_title = "Introduction"
    intro_path = os.path.join(out_dir, f"{intro_title}.tex")

    data_obj = getattr(result, "data", None)
    skeleton_titles: List[str] = []
    if data_obj and hasattr(data_obj, "subsections"):
        skeleton_titles = [s.title for s in data_obj.subsections]
        # Print proposed skeleton
        log.info("[cyan]Proposed skeleton subsections[/] (%d):", len(skeleton_titles))
        for t in skeleton_titles:
            log.info(f"- {t}")
        # Persist skeleton
        try:
            with open(os.path.join(out_dir, "skeleton.json"), "w") as f:
                json.dump({"module": module_name_out, "subsections": skeleton_titles}, f, indent=2)
        except Exception:
            pass
    
    # Enhancement phase: write intro + subsections only when requested
    if phase in ("enhance", "both"):
        if data_obj and hasattr(data_obj, "intro_section_latex") and hasattr(data_obj, "subsections"):
            combined = data_obj.intro_section_latex + "\n\n" + "\n\n".join(s.enhanced_latex for s in data_obj.subsections)
            with open(assistant_message_path, "w") as f:
                f.write(combined)
            with open(intro_path, "w") as f:
                f.write(data_obj.intro_section_latex)
            sections.append(SectionRef(order=0, type="section", title=intro_title, path=intro_path))
            for i, sub in enumerate(data_obj.subsections, start=1):
                title = sub.title
                sub_path = os.path.join(out_dir, f"{title}.tex")
                with open(sub_path, "w") as f:
                    f.write(sub.enhanced_latex)
                sections.append(SectionRef(order=i, type="subsection", title=title, path=sub_path))
    else:
        # Fallback: treat result as plain LaTeX string
        raw = getattr(result, "content", None) or getattr(result, "output", None) or str(result)
        if not isinstance(raw, str):
            try:
                raw = json.dumps(raw, default=str)
            except Exception:
                raw = str(raw)
        with open(assistant_message_path, "w") as f:
            f.write(raw)
        with open(intro_path, "w") as f:
            f.write(raw)
        sections.append(SectionRef(order=0, type="section", title=intro_title, path=intro_path))

    # Only write metadata if we produced subsection files (enhancement)
    if sections:
        latex_metadata = LatexMetadata(sections=sections)
        metadata_path = os.path.join(out_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(latex_metadata.model_dump(), f, indent=4)

    # Persist diagnostics: topics and subsection-to-source mapping
    try:
        with open(os.path.join(out_dir, "unique_topics.json"), "w") as f:
            json.dump({"module_name": module_name_out, "unique_topics": unique_topics}, f, indent=2)
    except Exception as e:
        log.warning(f"Could not write unique_topics.json: {e}")

    # Build a lightweight mapping from generated subsections to top matching items
    try:
        map_top_k = int(os.getenv("D2R_MAP_TOP_K", "3"))
        mapper = VectorIndexRetriever(
            index=index,
            similarity_top_k=map_top_k,
            embedding_model=embedding_model,
            filters=base_metadata_filters,
        )
        mapping = {}
        # Prefer typed result path if available
        if skeleton_titles:
            for title in skeleton_titles:
                nodes = mapper.retrieve(sub.title)
                items = [
                    {
                        "course_slug": n.metadata.get("course_slug", ""),
                        "module_slug": n.metadata.get("module_slug", ""),
                        "lesson_slug": n.metadata.get("lesson_slug", ""),
                        "item_slug": n.metadata.get("item_slug", ""),
                    }
                    for n in nodes
                ]
                mapping[title] = items
                log.info(f"[green]Mapped[/] subsection '{title}' to items:")
                for it in items:
                    log.info("- %s", json.dumps(it))
        # Ensure coverage: all available item_slugs are represented somewhere
        try:
            all_item_slugs = set(list_unique_topics(None))
            covered = set()
            for items in mapping.values():
                covered.update(x.get("item_slug", "") for x in items)
            missing = sorted(s for s in all_item_slugs if s and s not in covered)
            if missing:
                log.info(f"[yellow]Unmapped item_slugs[/] ({len(missing)}):")
                for m in missing:
                    log.info(f"- {m}")
                # record unmapped so operators can reconcile later
                mapping["__unmapped_item_slugs__"] = missing
        except Exception:
            pass
        with open(os.path.join(out_dir, "subsection_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        log.warning(f"Could not write subsection_mapping.json: {e}")

    log.info(f"[bold green]Processing complete[/]. Enhanced content and metadata saved to '{out_dir}'.")

if __name__ == "__main__":
    # # # Hardcoded module_name for debugging
    # DEBUG_MODULE_NAME = "Topic 1 Introduction to computer security and malware"

    # # # Pass the hardcoded module_name to main
    # main(args=["--module_name", DEBUG_MODULE_NAME])

    main()


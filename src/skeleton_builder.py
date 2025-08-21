import argparse
import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import URL
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from src.logger import configure_logging, get_logger
from src.models.latex import LatexMetadata, SectionRef
from src.models.paths import OutputPaths
from models.config import TiDBSettings, OpenAISettings
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


def build_skeleton(module_name: str, module_slug: str | None, output_base_dir: str, list_top_k: int, course_name: str) -> Dict:
    load_dotenv()
    configure_logging()
    log = get_logger(__name__)

    # Setup vector access
    db = TiDBSettings()
    embedding_model = OpenAIEmbedding(model=db.embedding_model_name)
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
        table_name=os.getenv("VECTOR_TABLE_NAME", "demo_load_docs_to_llamaindex"),
        distance_strategy="cosine",
        vector_dimension=db.embedding_dimension,
        drop_existing_table=False,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embedding_model)

    base_filters = (
        [{"key": "module_slug", "value": module_slug, "operator": "=="}] if module_slug else
        [{"key": "module_name", "value": module_name, "operator": "=="}]
    )
    base_metadata_filters = MetadataFilters(filters=[MetadataFilter(**f) for f in base_filters])

    # Discover topics
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=list_top_k,
        embedding_model=embedding_model,
        filters=base_metadata_filters,
    )
    source_nodes = retriever.retrieve("List all the files")
    topics = sorted(list({x.metadata.get("item_slug", "") for x in source_nodes if x.metadata.get("item_slug")}))
    log.info("[cyan]Topics found[/] (%d):", len(topics))
    for t in topics:
        log.info("- %s", t)
    # Persist topics for transparency
    try:
        os.makedirs(os.path.join(output_base_dir, course_name, module_name_out), exist_ok=True)
        with open(os.path.join(output_base_dir, course_name, module_name_out, "unique_topics.json"), "w") as f:
            json.dump({"module": module_name_out, "unique_topics": topics}, f, indent=2)
    except Exception:
        pass

    # Course/module names come from orchestrator
    module_name_out = module_name

    paths = OutputPaths(base_dir=output_base_dir, course_name=course_name, module_name=module_name_out)
    out_dir = paths.module_dir()
    os.makedirs(out_dir, exist_ok=True)

    # Proposed skeleton: use AI to refine subsection titles from topics
    ai = OpenAISettings()
    agent = Agent(model=OpenAIModel(ai.writer_model))
    # Reuse existing, tested prompt as guidance
    def load_prompt(file_name: str) -> str:
        with open(os.path.join("prompts", file_name), "r", encoding="utf-8") as f:
            return f.read().strip()

    assistant_msg = load_prompt("assistant_message.txt")
    # Helper: build bounded RAG context string from topic slugs using retriever
    def build_context_from_topics(topic_slugs: List[str], max_chars: int, per_topic_k: int) -> str:
        chunks: List[str] = []
        budget = max_chars
        for slug in topic_slugs:
            try:
                nodes = retriever.retrieve(slug)
            except Exception:
                continue
            count = 0
            for n in nodes:
                if count >= per_topic_k:
                    break
                # extract text safely
                t = ""
                try:
                    t = n.get_text() if hasattr(n, "get_text") else getattr(n, "text", "")
                except Exception:
                    t = ""
                if not t:
                    continue
                snippet = t.strip()
                if len(snippet) > budget:
                    snippet = snippet[:budget]
                if snippet:
                    chunks.append(f"[TOPIC {slug}]\n{snippet}")
                    budget -= len(snippet)
                    count += 1
                if budget <= 0:
                    break
            if budget <= 0:
                break
        return "\n\n".join(chunks)

    # Build bounded global context for outline (RAG)
    outline_ctx_chars = int(os.getenv("D2R_SKELETON_CONTEXT_CHARS", "6000"))
    outline_ctx_k = int(os.getenv("D2R_RAG_TOP_K_SKELETON", "2"))
    outline_context = build_context_from_topics(topics, outline_ctx_chars, outline_ctx_k)
    # New contract: hierarchical JSON with intro and subsections, each optionally with subsubsections and explicit topic slugs
    outline_prompt = (
        f"{assistant_msg}\n\n"
        "Task: Propose a comprehensive outline for the module. Use subsubsections to avoid too many top-level subsections.\n"
        "Your response MUST be STRICT JSON (no code fences, no prose) with this exact shape:\n"
        "{\n"
        "  \"intro_title\": \"<short intro section title>\",\n"
        "  \"subsections\": [\n"
        "    { \"title\": \"<meaningful subsection title>\", \"topics\": [\"<item_slug>\", ...], \"subsubsections\": [\n"
        "        { \"title\": \"<meaningful subsubsection title>\", \"topics\": [\"<item_slug>\", ...] }\n"
        "    ] }\n"
        "  ]\n"
        "}\n\n"
        "Notes:\n"
        "- titles do NOT need to match topic slugs; choose what makes sense.\n"
        "- topics arrays must contain item_slugs from this module only.\n"
        "- Keep at most 6 subsections and at most 4 subsubsections per subsection.\n\n"
        f"Module: {module_name}\n"
        f"Available topic slugs: {topics}\n"
        "Context (excerpts from transcripts, do not copy verbatim; structure based on this):\n"
        f"{outline_context}"
    )
    try:
        outline = agent.run_sync(outline_prompt)
        raw = getattr(outline, "content", None) or getattr(outline, "output", None) or str(outline)
        parsed: Dict[str, Any] = json.loads(raw) if isinstance(raw, str) else raw
        # Validate contract minimally
        intro_title = parsed.get("intro_title") if isinstance(parsed, dict) else None
        subsections = parsed.get("subsections") if isinstance(parsed, dict) else None
        if not isinstance(subsections, list) or not subsections:
            raise ValueError("invalid skeleton JSON: missing subsections")
        # Enforce limits
        max_subs = int(os.getenv("D2R_MAX_SUBSECTIONS", "6"))
        max_subsubs = int(os.getenv("D2R_MAX_SUBSUBSECTIONS", "4"))
        subsections = subsections[:max_subs]
        for s in subsections:
            if isinstance(s, dict) and isinstance(s.get("subsubsections"), list):
                s["subsubsections"] = s["subsubsections"][:max_subsubs]
        skeleton_contract = {
            "intro_title": intro_title or "Introduction",
            "subsections": subsections,
        }
        # Titles for file generation (top-level only)
        skeleton_titles: List[str] = [s.get("title", "Section") for s in subsections if isinstance(s, dict)]
    except Exception as e:
        log.warning("Skeleton AI failed, falling back to topics as subsections: %s", e)
        skeleton_contract = {"intro_title": "Introduction", "subsections": [{"title": t, "topics": [t]} for t in topics]}
        skeleton_titles = topics
    with open(os.path.join(out_dir, "skeleton.json"), "w") as f:
        json.dump(skeleton_contract, f, indent=2)

    # Build mapping from contract's topics; verify against available vector topics
    available_slugs = set(topics)
    def lookup_item(slug: str) -> Dict[str, str]:
        # Try to get one representative node's metadata by querying the slug
        try:
            nodes = retriever.retrieve(slug)
            for n in nodes:
                if n.metadata.get("item_slug") == slug:
                    return {
                        "course_slug": n.metadata.get("course_slug", ""),
                        "module_slug": n.metadata.get("module_slug", ""),
                        "lesson_slug": n.metadata.get("lesson_slug", ""),
                        "item_slug": slug,
                    }
        except Exception:
            pass
        return {"course_slug": "", "module_slug": "", "lesson_slug": "", "item_slug": slug}

    mapping: Dict[str, Any] = {"subsections": [], "missing_topics": []}
    for s in skeleton_contract["subsections"]:
        title = s.get("title", "Section")
        topic_slugs = [t for t in s.get("topics", []) if t]
        missing = [t for t in topic_slugs if t not in available_slugs]
        items = [lookup_item(t) for t in topic_slugs if t in available_slugs]
        entry = {"title": title, "items": items, "missing_topics": missing}
        # Include subsubsections mapping
        subsubs_out = []
        for ss in (s.get("subsubsections", []) or []):
            ss_title = ss.get("title", "Subsection")
            ss_topics = [t for t in ss.get("topics", []) if t]
            ss_missing = [t for t in ss_topics if t not in available_slugs]
            ss_items = [lookup_item(t) for t in ss_topics if t in available_slugs]
            subsubs_out.append({"title": ss_title, "items": ss_items, "missing_topics": ss_missing})
        if subsubs_out:
            entry["subsubsections"] = subsubs_out
        mapping["subsections"].append(entry)
    # Log mapping summary
    for e in mapping["subsections"]:
        log.info("[green]Mapped[/] '%s' -> %d items", e["title"], len(e.get("items", [])))
    with open(os.path.join(out_dir, "subsection_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    # Create metadata.json with placeholder paths
    sections: List[SectionRef] = []
    intro_title = skeleton_contract.get("intro_title") or "Introduction"
    intro_path = paths.intro_path(intro_title)
    # Ensure files exist so path validation passes later and include minimal valid LaTeX
    os.makedirs(os.path.dirname(intro_path), exist_ok=True)
    if not os.path.exists(intro_path) or os.path.getsize(intro_path) == 0:
        with open(intro_path, 'w', encoding='utf-8') as f:
            f.write(f"\\section{{{intro_title}}}\n")
    sections.append(SectionRef(order=0, type="section", title=intro_title, path=intro_path))
    for i, title in enumerate(skeleton_titles, start=1):
        sub_path = paths.subsection_path(title)
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        if not os.path.exists(sub_path) or os.path.getsize(sub_path) == 0:
            with open(sub_path, 'w', encoding='utf-8') as f:
                f.write(f"\\subsection{{{title}}}\n")
        sections.append(SectionRef(order=i, type="subsection", title=title, path=sub_path))
    meta = LatexMetadata(sections=sections)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta.model_dump(), f, indent=2)

    # Optionally generate initial high-level content per subsection using prompts, then concatenate into assistant_message.tex
    if os.getenv("D2R_SKELETON_INIT_CONTENT", "1") == "1":
        try:
            # Introduction content
            intro_query = load_prompt("intro_query.txt")
            intro_prompt = (
                f"{intro_query}\n"
                f"The title is '{intro_title}'. Provide a concise introduction in LaTeX."
            )
            intro_resp = agent.run_sync(intro_prompt)
            intro_raw = getattr(intro_resp, "content", None) or getattr(intro_resp, "output", None) or str(intro_resp)
            with open(os.path.join(out_dir, "Introduction.tex"), "w", encoding="utf-8") as f:
                f.write(intro_raw)

            # Generate content for each top-level subsection using subsection_query.txt
            subsection_query = load_prompt("subsection_query.txt")
            for s in skeleton_contract.get("subsections", []):
                title = s.get("title", "Section")
                sub_path = paths.subsection_path(title)
                # Build RAG context from topic slugs
                topic_slugs = s.get("topics", []) or []
                sub_ctx_chars = int(os.getenv("D2R_SUBSECTION_CONTEXT_CHARS", "4000"))
                sub_ctx_k = int(os.getenv("D2R_RAG_TOP_K_ITEM", "2"))
                sub_context = build_context_from_topics(topic_slugs, sub_ctx_chars, sub_ctx_k)
                sub_prompt = (
                    f"{subsection_query}\n"
                    "Do not include code fences. Use subsubsections where appropriate.\n"
                    f"Subsection title: {title}\n"
                    "Context (excerpts from transcripts; summarize and structure based on this):\n"
                    f"{sub_context}\n"
                )
                sub_resp = agent.run_sync(sub_prompt)
                sub_raw = getattr(sub_resp, "content", None) or getattr(sub_resp, "output", None) or str(sub_resp)
                with open(sub_path, "w", encoding="utf-8") as f:
                    f.write(sub_raw)

            # Concatenate all subsection files into assistant_message.tex (no file listing header)
            combined = []
            for s in sections:
                if s.type == "subsection":
                    try:
                        with open(s.path, "r", encoding="utf-8") as f:
                            combined.append(f.read())
                    except Exception:
                        continue
            with open(os.path.join(out_dir, "assistant_message.tex"), "w", encoding="utf-8") as f:
                f.write("\n\n".join(combined))
        except Exception:
            # Non-fatal; skeleton artifacts already written
            pass

    return {
        "course_name": course_name,
        "module": module_name_out,
        "out_dir": out_dir,
        "topics": topics,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description="Build lecture notes skeleton: skeleton.json, metadata.json, subsection_mapping.json")
    parser.add_argument("--module_name", required=True)
    parser.add_argument("--module_slug", required=False)
    parser.add_argument("--output_base_dir", default=os.getenv("OUTPUT_BASE_DIR") or os.getenv("D2R_OUTPUT_BASE") or "assistant_latex")
    parser.add_argument("--list_top_k", type=int, default=int(os.getenv("D2R_LIST_TOP_K_SKELETON", "500")))
    parser.add_argument("--course_name", required=True)
    cli = parser.parse_args(args)

    build_skeleton(cli.module_name, cli.module_slug, cli.output_base_dir, cli.list_top_k, cli.course_name)


if __name__ == "__main__":
    main()



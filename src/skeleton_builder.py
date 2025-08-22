import argparse
import json
import os
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from sqlalchemy import URL
from src.models.settings import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from src.logger import configure_logging, get_logger
from src.models.latex import LatexMetadata, SectionRef
from src.models.paths import OutputPaths
from models.config import TiDBSettings
from pydantic_ai import Agent


def build_skeleton(module_name: str, module_slug: str | None, output_base_dir: str, list_top_k: int, course_name: str) -> Dict:
    load_dotenv()
    configure_logging()
    log = get_logger(__name__)

    # Setup vector access
    db = TiDBSettings()
    settings = Settings()
    embedding_model = settings.create_embedding()
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

    # Discover topics once; reuse nodes directly afterwards
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=list_top_k,
        embedding_model=embedding_model,
        filters=base_metadata_filters,
    )
    log.info("[lowlight]Retrieval query[/]: 'List all the files'")
    source_nodes = retriever.retrieve("List all the files")
    # Build topics list and an in-memory index from slug -> nodes
    topics = sorted(list({x.metadata.get("item_slug", "") for x in source_nodes if x.metadata.get("item_slug")}))
    slug_to_nodes: Dict[str, List[Any]] = {}
    for n in source_nodes:
        slug = n.metadata.get("item_slug")
        if not slug:
            continue
        slug_to_nodes.setdefault(slug, []).append(n)
    # Report file types per item_slug for transparency/debug
    def _guess_type(node) -> str:
        mt = node.metadata.get("type") or node.metadata.get("file_type") or node.metadata.get("mime_type") or node.metadata.get("content_type")
        if mt:
            return str(mt)
        path = node.metadata.get("file_path") or node.metadata.get("path") or node.metadata.get("source") or node.metadata.get("doc_id") or ""
        try:
            import os as _os
            ext = _os.path.splitext(str(path))[1].lower()
            return ext or "unknown"
        except Exception:
            return "unknown"
    slug_to_types: Dict[str, List[str]] = {}
    for slug, nodes in slug_to_nodes.items():
        types = sorted({ _guess_type(n) for n in nodes })
        slug_to_types[slug] = types
        try:
            log.info("[lowlight]Types for %s[/]: %s", slug, ", ".join(types))
        except Exception:
            pass
    try:
        with open(os.path.join(out_dir, "item_types.json"), "w") as f:
            json.dump(slug_to_types, f, indent=2)
    except Exception:
        pass

    # Per-topic table of counts by category (transcripts, extra notes, slides)
    def _categorize(t: str) -> str:
        tl = (t or "").lower()
        if "transcript" in tl or tl.endswith(".srt") or "text/plain" in tl:
            return "transcripts"
        if "slide" in tl or tl.endswith(".pdf") or "application/pdf" in tl:
            return "slides"
        if "note" in tl or "extra" in tl or tl.endswith(".md"):
            return "extra_notes"
        return "extra_notes"

    per_slug_counts: Dict[str, Dict[str, int]] = {}
    totals = {"transcripts": 0, "extra_notes": 0, "slides": 0}
    for slug, nodes in slug_to_nodes.items():
        c = {"transcripts": 0, "extra_notes": 0, "slides": 0}
        for n in nodes:
            cat = _categorize(_guess_type(n))
            c[cat] += 1
            totals[cat] += 1
        per_slug_counts[slug] = c

    log.info("[cyan]Topics found[/] (%d):", len(topics))
    log.info("%-44s %12s %13s %8s", "item_slug", "transcripts", "extra_notes", "slides")
    for slug in topics:
        c = per_slug_counts.get(slug, {"transcripts": 0, "extra_notes": 0, "slides": 0})
        log.info("%-44s %12d %13d %8d", slug, c["transcripts"], c["extra_notes"], c["slides"])
    log.info("%-44s %12d %13d %8d", "TOTAL", totals["transcripts"], totals["extra_notes"], totals["slides"])
    # Persist topics for transparency and potential offline usage
    try:
        topics_cache_path = os.path.join(output_base_dir, course_name, module_name_out, "unique_topics.json")
        os.makedirs(os.path.dirname(topics_cache_path), exist_ok=True)
        with open(topics_cache_path, "w") as f:
            json.dump({"module": module_name_out, "unique_topics": topics}, f, indent=2)
    except Exception:
        pass
    # (Per-topic table already logged above)
    # Course/module names come from orchestrator
    module_name_out = module_name

    paths = OutputPaths(base_dir=output_base_dir, course_name=course_name, module_name=module_name_out)
    out_dir = paths.module_dir()
    os.makedirs(out_dir, exist_ok=True)

    # One-shot high-level outline using assistant_message prompt
    agent = Agent(model=settings.ai_writer_model)
    def load_prompt(file_name: str) -> str:
        with open(os.path.join("prompts", file_name), "r", encoding="utf-8") as f:
            return f.read().strip()
    assistant_msg = load_prompt("assistant_message.txt")
    # Build lightweight RAG context from discovered topic slugs (bounded by env budgets)
    def build_context_from_topics(topic_slugs: List[str]) -> str:
        import textwrap
        ctx_chars = int(os.getenv("D2R_SKELETON_CONTEXT_CHARS", "4000"))
        per_topic_k = int(os.getenv("D2R_RAG_TOP_K_SKELETON", "1"))
        budget = ctx_chars
        chunks: List[str] = []
        cache: Dict[str, List[str]] = {}
        for slug in topic_slugs:
            if budget <= 0:
                break
            texts: List[str] = cache.get(slug, [])
            if not texts:
                nodes = slug_to_nodes.get(slug, [])
                for n in nodes[:per_topic_k]:
                    try:
                        t = n.get_text() if hasattr(n, "get_text") else getattr(n, "text", "")
                    except Exception:
                        t = ""
                    if t:
                        texts.append(t)
                cache[slug] = texts
            for t in texts:
                if budget <= 0:
                    break
                snippet = t.strip()
                if len(snippet) > budget:
                    snippet = snippet[:budget]
                if snippet:
                    preview = snippet[:160].replace("\n", " ")
                    log.info("[lowlight]Context snippet[/] %s: %s...", slug, preview)
                    chunks.append(f"[TOPIC {slug}]\n" + textwrap.shorten(snippet, width=len(snippet)))
                    budget -= len(snippet)
        return "\n\n".join(chunks)

    outline_context = build_context_from_topics(topics)
    # Strict JSON outline (structure only) + mapping prompt
    outline_prompt = (
        f"{assistant_msg}\n\n"
        "Task: Propose a comprehensive outline for the module. Use subsubsections to avoid too many top-level subsections.\n"
        "Your response MUST be STRICT JSON (no code fences, no prose) with this exact shape:\n"
        "{\n"
        "  \"intro_title\": \"<short intro section title>\",\n"
        "  \"subsections\": [\n"
        "    { \"title\": \"<meaningful subsection title>\", \"topics\": [\"<item_slug>\", ...], \"summary_latex\": \"<concise LaTeX for this subsection>\", \"subsubsections\": [\n"
        "        { \"title\": \"<meaningful subsubsection title>\", \"topics\": [\"<item_slug>\", ...] }\n"
        "    ] }\n"
        "  ]\n"
        "}\n\n"
        "Notes:\n"
        "- titles do NOT need to match topic slugs; choose what makes sense.\n"
        "- topics arrays must contain item_slugs from this module only.\n"
        "- Keep at most 6 subsections and at most 4 subsubsections per subsection.\n"
        "- Each subsection's summary_latex must be concise but include: (a) 1-2 \\subsubsection headings as an initial outline, (b) a few bullet points, (c) valid LaTeX only (no code fences).\n"
        f"Module: {module_name}\n"
        f"Available topic slugs: {topics}\n"
        "Context (excerpts from transcripts, do not copy verbatim; structure based on this):\n"
        f"{outline_context}"
    )
    # Run and parse JSON
    try:
        prompt_preview = outline_prompt[:400].replace("\n", " ")
        log.info("[highlight]Outline prompt preview[/]: %s%s", prompt_preview, "..." if len(outline_prompt) > 400 else "")
    except Exception:
        pass
    outline = agent.run_sync(outline_prompt)
    raw = getattr(outline, "content", None) or getattr(outline, "output", None) or str(outline)
    try:
        resp_preview = (raw if isinstance(raw, str) else str(raw))[:400].replace("\n", " ")
        # log.info("[highlight]Outline response preview[/]: %s%s", resp_preview, "..." if len(str(raw)) > 400 else "")
    except Exception:
        pass
    try:
        parsed: Dict[str, Any] = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        parsed = {"intro_title": "Introduction", "subsections": [{"title": t, "topics": [t]} for t in topics]}
    intro_title = parsed.get("intro_title") or "Introduction"
    subsections = parsed.get("subsections") or []
    # Enforce limits
    max_subs = int(os.getenv("D2R_MAX_SUBSECTIONS", "6"))
    max_subsubs = int(os.getenv("D2R_MAX_SUBSUBSECTIONS", "4"))
    subsections = subsections[:max_subs]
    for s in subsections:
        if isinstance(s, dict) and isinstance(s.get("subsubsections"), list):
            s["subsubsections"] = s["subsubsections"][:max_subsubs]
    # Ensure terminal Evaluation & Future Directions subsection exists (simple subsection; no topics)
    if not any(isinstance(s, dict) and s.get("title", "").strip().lower() == "evaluation and future directions" for s in subsections):
        subsections.append({
            "title": "Evaluation and Future Directions",
            "topics": [],
            "subsubsections": []
        })
    skeleton_contract = {"intro_title": intro_title, "subsections": subsections}
    with open(os.path.join(out_dir, "skeleton.json"), "w") as f:
        json.dump(skeleton_contract, f, indent=2)

    # Build mapping from topics arrays
    available_slugs = set(topics)
    def lookup_item(slug: str) -> Dict[str, str]:
        # Prefer metadata from in-memory nodes if present; fallback to minimal fields
        nodes = slug_to_nodes.get(slug, [])
        for n in nodes:
            if n.metadata.get("item_slug") == slug:
                return {
                    "course_slug": n.metadata.get("course_slug", ""),
                    "module_slug": n.metadata.get("module_slug", module_slug or ""),
                    "lesson_slug": n.metadata.get("lesson_slug", ""),
                    "item_slug": slug,
                }
        return {"course_slug": "", "module_slug": module_slug or "", "lesson_slug": "", "item_slug": slug}

    mapping: Dict[str, Any] = {"subsections": []}
    titles: List[str] = []
    eval_title = "Evaluation and Future Directions"
    for s in subsections:
        title = s.get("title", "Section")
        titles.append(title)
        if title.strip().lower() == eval_title.lower():
            # Skip mapping for evaluation subsection
            continue
        slugs = [t for t in (s.get("topics") or []) if t]
        items = [lookup_item(t) for t in slugs if t in available_slugs]
        missing = [t for t in slugs if t not in available_slugs]
        entry = {"title": title, "items": items, "missing_topics": missing}
        # Subsubsections mapping
        ss_list = []
        for ss in (s.get("subsubsections") or []):
            ss_title = ss.get("title", "Subsection")
            ss_slugs = [t for t in (ss.get("topics") or []) if t]
            ss_items = [lookup_item(t) for t in ss_slugs if t in available_slugs]
            ss_missing = [t for t in ss_slugs if t not in available_slugs]
            ss_list.append({"title": ss_title, "items": ss_items, "missing_topics": ss_missing})
        if ss_list:
            entry["subsubsections"] = ss_list
        mapping["subsections"].append(entry)
    with open(os.path.join(out_dir, "subsection_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    # Create .tex files and metadata.json (placeholders; will fill intro/eval after second query)
    sections: List[SectionRef] = []
    intro_path = paths.intro_path(intro_title)
    os.makedirs(os.path.dirname(intro_path), exist_ok=True)
    # Sanitize and write intro
    import re as _re
    def _sanitize_tex(tex: str) -> str:
        tex = _re.sub(r"```+\\s*latex|```+", "", tex)
        tex = _re.sub(r"^\\section\{Used files:.*\}\s*$", "", tex, flags=_re.MULTILINE)
        # Convert literal escaped newlines/tabs into actual characters
        tex = tex.replace("\\n", "\n").replace("\\t", "\t")
        # Remove literal tab characters and stray 'latex' lines
        tex = tex.replace("\t", " ")
        tex = _re.sub(r"(?m)^[ \t]*latex[ \t]*$", "", tex)
        # Fix common missing backslash for \textbf and common typos
        # Fix common missing backslash for \textbf
        tex = _re.sub(r"(?<!\\)extbf\{", r"\\textbf{", tex)
        tex = _re.sub(r"(?<!\\)textrightarrow", r"\\rightarrow", tex)
        tex = _re.sub(r"(?<!\\)imes\b", r"\\times", tex)
        return tex.strip()
    with open(intro_path, 'w', encoding='utf-8') as f:
        f.write(f"\\section{{{intro_title}}}\n")
    sections.append(SectionRef(order=0, type="section", title=intro_title, path=intro_path))
    eval_title = "Evaluation and Future Directions"
    eval_path = None
    for i, s in enumerate(subsections, start=1):
        title = s.get("title", "Section")
        sub_path = paths.subsection_path(title)
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        summary_latex = s.get("summary_latex") or f"\\subsection{{{title}}}\n"
        with open(sub_path, 'w', encoding='utf-8') as f:
            f.write(_sanitize_tex(summary_latex))
        sections.append(SectionRef(order=i, type="subsection", title=title, path=sub_path))
        if title.strip().lower() == eval_title.lower():
            eval_path = sub_path
    meta = LatexMetadata(sections=sections)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta.model_dump(), f, indent=2)

    # Second agent query: generate comprehensive intro and evaluation using full outline context
    try:
        eval_intro_prompt = (
            "You will draft two LaTeX parts for this module using the provided context excerpts.\n"
            "Return STRICT JSON (no code fences) with keys: intro_latex, evaluation_latex.\n"
            "Requirements:\n"
            "- intro_latex MUST be a single \\section with narrative text only; do NOT include any other \\subsection headings.\n"
            "- evaluation_latex MUST be a single \\subsection titled 'Evaluation and Future Directions' with concise bullets and short narrative, referring to the actual subsection titles (see the list below) rather than generic labels.\n"
            f"Module: {module_name}\n"
            f"Intro title: {intro_title}\n"
            f"Evaluation title: {eval_title}\n"
            f"Subsections (titles): {[s.get('title') for s in subsections]}\n"
            "Context (excerpts; summarize, do not copy verbatim):\n"
            f"{outline_context}"
        )
        prompt_preview2 = eval_intro_prompt[:400].replace("\n", " ")
        log.info("[highlight]Intro/Eval prompt preview[/]: %s%s", prompt_preview2, "..." if len(eval_intro_prompt) > 400 else "")
        resp2 = agent.run_sync(eval_intro_prompt)
        raw2 = getattr(resp2, "content", None) or getattr(resp2, "output", None) or str(resp2)
        resp_preview2 = (raw2 if isinstance(raw2, str) else str(raw2))[:400].replace("\n", " ")
        # log.info("[highlight]Intro/Eval response preview[/]: %s%s", resp_preview2, "..." if len(str(raw2)) > 400 else "")
        try:
            parsed2 = json.loads(raw2) if isinstance(raw2, str) else raw2
        except Exception:
            parsed2 = {"intro_latex": "", "evaluation_latex": ""}
        intro_latex_final = parsed2.get("intro_latex") or f"\\section{{{intro_title}}}\n"
        evaluation_latex_final = parsed2.get("evaluation_latex") or f"\\subsection{{{eval_title}}}\n"
        with open(intro_path, 'w', encoding='utf-8') as f:
            f.write(_sanitize_tex(intro_latex_final))
        if eval_path:
            with open(eval_path, 'w', encoding='utf-8') as f:
                f.write(_sanitize_tex(evaluation_latex_final))
    except Exception:
        pass

    # Collate assistant_message.tex as a convenience artifact (intro + subsections)
    try:
        parts: List[str] = []
        # Include intro content without duplicates
        try:
            with open(intro_path, 'r', encoding='utf-8') as f:
                intro_txt = _sanitize_tex(f.read())
                parts.append(intro_txt)
        except Exception:
            pass
        # Append each subsection in order
        for sref in sections:
            if sref.type != "subsection":
                continue
            try:
                with open(sref.path, 'r', encoding='utf-8') as f:
                    parts.append(_sanitize_tex(f.read()))
            except Exception:
                continue
        with open(os.path.join(out_dir, "assistant_message.tex"), 'w', encoding='utf-8') as f:
            f.write("\n\n".join([p for p in parts if p.strip()]))
        log.info("[success]Generated[/] assistant_message.tex in %s", out_dir)
    except Exception:
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
    parser.add_argument("--list_top_k", type=int, default=int(os.getenv("D2R_LIST_TOP_K_SKELETON", "5000")))
    parser.add_argument("--course_name", required=True)
    cli = parser.parse_args(args)

    build_skeleton(cli.module_name, cli.module_slug, cli.output_base_dir, cli.list_top_k, cli.course_name)


if __name__ == "__main__":
    main()



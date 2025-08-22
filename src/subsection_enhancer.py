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
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from src.logger import configure_logging, get_logger
from src.models.latex import SectionRef, SectionContent
from models.config import TiDBSettings, OpenAISettings
from src.models.paths import OutputPaths


def enhance_subsections(course_name: str, module_name: str, module_slug: str | None, output_base_dir: str, top_k_item: int) -> None:
    load_dotenv()
    configure_logging()
    log = get_logger(__name__)

    db = TiDBSettings()
    ai = OpenAISettings()
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

    # Retrieval should use slug when available; fallback to name only if slug missing
    base_filters = (
        [{"key": "module_slug", "value": module_slug, "operator": "=="}] if module_slug else
        [{"key": "module_name", "value": module_name, "operator": "=="}]
    )
    base_metadata_filters = MetadataFilters(filters=[MetadataFilter(**f) for f in base_filters])

    paths = OutputPaths(base_dir=output_base_dir, course_name=course_name, module_name=module_name)
    out_dir = paths.module_dir()
    skeleton_path = os.path.join(out_dir, "skeleton.json")
    if not os.path.exists(skeleton_path):
        log.info("skeleton.json not found at %s; nothing to enhance.", skeleton_path)
        return
    with open(skeleton_path, "r") as f:
        skeleton: Dict[str, Any] = json.load(f)
    subsections: List[Dict[str, Any]] = skeleton.get("subsections", [])
    if not isinstance(subsections, list) or not subsections:
        log.info("No subsections found in skeleton; nothing to enhance.")
        return

    # Ensure required skeleton subsection files exist before enhancing
    missing_files: List[str] = []
    for s in subsections:
        title = s.get("title", "Section")
        # Skip evaluation subsection
        if title.strip().lower() == "evaluation and future directions":
            continue
        skeleton_sub_path = paths.subsection_path(title)
        if not os.path.exists(skeleton_sub_path):
            missing_files.append(skeleton_sub_path)
    if missing_files:
        msg = (
            "Skeleton subsection files missing. Run skeleton_builder first, or ensure these files exist: "
            + ", ".join(missing_files)
        )
        log.error(msg)
        raise FileNotFoundError(msg)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=max(top_k_item, 50),
        embedding_model=embedding_model,
        filters=base_metadata_filters,
    )
    # Retrieve once and build slug->nodes map
    log.info("[lowlight]Retrieval query[/]: 'List all the files'")
    source_nodes = retriever.retrieve("List all the files")
    slug_to_nodes: Dict[str, List[Any]] = {}
    for n in source_nodes:
        slug = n.metadata.get("item_slug")
        if not slug:
            continue
        slug_to_nodes.setdefault(slug, []).append(n)

    agent = Agent(model=OpenAIModel(ai.writer_model))

    def _sanitize_tex(s: str) -> str:
        import re as _re
        # Strip code fences and fix escaped newlines/tabs
        s = _re.sub(r"```+\\s*latex|```+", "", s)
        s = s.replace("\\n", "\n").replace("\\t", "\t")
        # Remove literal tab characters that break TeX (e.g., \t prefix issues)
        s = s.replace("\t", " ")
        # Remove TeX \t macro usage (e.g., \t\textbf -> \textbf)
        s = _re.sub(r"\\t\s*", " ", s)
        # Drop stray lines that only say 'latex'
        s = _re.sub(r"(?m)^[ \t]*latex[ \t]*$", "", s)
        # Fix common missing backslash for \textbf
        s = _re.sub(r"(?<!\\)extbf\{", r"\\textbf{", s)
        # Unicode and symbol sanitization
        rep = {
            "π": "\\pi", "−": "-", "μ": "\\mu", "σ": "\\sigma", "ρ": "\\rho",
            "≈": "\\approx", "≤": "\\leq", "≥": "\\geq", "∑": "\\sum",
            "∫": "\\int", "∈": "\\in", "√": "\\sqrt{}",
        }
        for k, v in rep.items():
            s = s.replace(k, v)
        # Fix common math typos
        s = _re.sub(r"(?<!\\)textrightarrow", "\\rightarrow", s)
        s = _re.sub(r"(?<!\\)imes\b", "\\times", s)
        s = _re.sub(r"\bt\\times", r" \\times", s)
        # Fix tabular column spec 'extwidth' -> '\textwidth'
        s = _re.sub(r"p\{\s*([0-9.]+)\s*extwidth\s*\}", r"p{\1\\textwidth}", s)
        # Escape underscores outside math
        def _escape_underscores(txt: str) -> str:
            out = []
            for line in txt.splitlines():
                if ("$" not in line) and ("\\(" not in line) and ("\\[" not in line):
                    line = line.replace("_", "\\_")
                out.append(line)
            return "\n".join(out)
        s = _escape_underscores(s)
        # Escape stray & (outside alignment/table envs): simple line heuristic
        def _escape_ampersands(txt: str) -> str:
            out_lines = []
            env_depth = 0
            for line in txt.splitlines():
                if "\\begin{" in line:
                    env_depth += 1
                if env_depth == 0:
                    line = _re.sub(r"(?<!\\)&", r"\\&", line)
                if "\\end{" in line and env_depth > 0:
                    env_depth -= 1
                out_lines.append(line)
            return "\n".join(out_lines)
        s = _escape_ampersands(s)
        # Convert simple markdown bullets to LaTeX itemize
        def _md_bullets_to_itemize(txt: str) -> str:
            lines = txt.splitlines()
            out = []
            i = 0
            while i < len(lines):
                if lines[i].lstrip().startswith("- "):
                    block = []
                    while i < len(lines) and lines[i].lstrip().startswith("- "):
                        block.append("\\item " + lines[i].lstrip()[2:])
                        i += 1
                    out.append("\\begin{itemize}")
                    out.extend(block)
                    out.append("\\end{itemize}")
                    continue
                out.append(lines[i])
                i += 1
            return "\n".join(out)
        s = _md_bullets_to_itemize(s)
        return s

    # Build bounded context from topic slugs
    ctx_chars = int(os.getenv("D2R_ENH_SUBSECTION_CHARS", "4000"))
    per_slug_k = int(os.getenv("D2R_RAG_TOP_K_ITEM", "2"))

    def build_context_from_slugs(topic_slugs: List[str]) -> str:
        import textwrap
        budget = ctx_chars
        chunks: List[str] = []
        for slug in topic_slugs or []:
            nodes = slug_to_nodes.get(slug, [])
            cnt = 0
            for n in nodes:
                if cnt >= per_slug_k:
                    break
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
                    chunks.append(f"[TOPIC {slug}]\n" + textwrap.shorten(snippet, width=len(snippet)))
                    budget -= len(snippet)
                    cnt += 1
                if budget <= 0:
                    break
            if budget <= 0:
                break
        return "\n\n".join(chunks)

    eval_title = "Evaluation and Future Directions"
    subsection_prompt = None
    try:
        with open(os.path.join("prompts", "subsection_query.txt"), "r", encoding="utf-8") as f:
            subsection_prompt = f.read().strip()
    except Exception:
        subsection_prompt = ""

    # Enhanced output directory (immutable skeleton inputs)
    enhanced_dir = os.path.join(out_dir, "enhanced")
    os.makedirs(enhanced_dir, exist_ok=True)

    for s in subsections:
        title = s.get("title", "Section")
        if title.strip().lower() == eval_title.lower():
            continue
        topic_slugs = [t for t in (s.get("topics") or []) if t]
        log.info("[cyan]Enhancing[/] '%s' with %d topic slugs", title, len(topic_slugs))
        context_text = build_context_from_slugs(topic_slugs)
        prompt = (
            f"{subsection_prompt}\n"
            "Only return the final LaTeX for this subsection (no code fences).\n"
            "Strict requirements (must pass all):\n"
            "- Start with \\subsection{" + title.replace("\\", "\\\\") + "}.\n"
            "- Include 1–2 \\subsubsection headings as a mini-outline.\n"
            "- Use itemize for bullet lists; do not output raw '-' bullets.\n"
            "- Use valid LaTeX commands only; escape underscores outside math; no unescaped '&'.\n"
            "- Wrap inline math and macros (e.g., \\frac, \\sqrt) with \\( ... \\) or $ ... $.\n"
            "- Use \\textbf{...} for bold; never 'extbf'.\n"
            "- Do not include \\section, code fences, or \\end{document}.\n"
            f"Subsection title: {title}\n"
            "Context excerpts (summarize; do not copy verbatim):\n"
            f"{context_text}\n"
            "Self-check before returning: ensure no raw '&', no unescaped '_', braces balanced, and LaTeX compiles."
        )
        try:
            log.info("[highlight]Enhancer prompt preview[/]: %s%s", prompt[:400].replace("\n", " "), "..." if len(prompt) > 400 else "")
        except Exception:
            pass
        result = agent.run_sync(prompt)
        out_text = getattr(result, "content", None) or getattr(result, "output", None) or str(result)
        # Ensure heading present
        if "\\subsection" not in out_text:
            out_text = f"\\subsection{{{title}}}\n" + out_text
        try:
            out_preview = out_text[:400].replace("\n", " ")
            log.info("[highlight]Enhancer response preview[/]: %s%s", out_preview, "..." if len(out_text) > 400 else "")
        except Exception:
            pass
        # Sanitize and validate; write .err.txt on failure but keep healed output
        # Write to enhanced directory to keep skeleton immutable
        sub_path = os.path.join(enhanced_dir, f"{title}.tex")
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        healed = _sanitize_tex(out_text)
        # Write first to satisfy path existence checks, then validate and write .err.txt if needed
        with open(sub_path, "w", encoding="utf-8") as f:
            f.write(healed)
        try:
            SectionContent(ref=SectionRef(order=1, type="subsection", title=title, path=sub_path), text=healed)
        except Exception as e:
            err_path = sub_path + ".err.txt"
            try:
                with open(err_path, "w", encoding="utf-8") as ef:
                    ef.write(str(e))
                log.warning("Validation failed for %s -> wrote error: %s", title, err_path)
            except Exception:
                pass
        log.info("Wrote enhanced subsection %s -> %s", title, sub_path)


def main(args=None):
    parser = argparse.ArgumentParser(description="Enhance subsections using existing skeleton.json")
    parser.add_argument("--course_name", required=True)
    parser.add_argument("--module_name", required=True)
    parser.add_argument("--module_slug", required=False)
    parser.add_argument("--output_base_dir", default=os.getenv("OUTPUT_BASE_DIR") or os.getenv("D2R_OUTPUT_BASE") or "assistant_latex")
    parser.add_argument("--top_k_item", type=int, default=int(os.getenv("D2R_TOP_K_ITEM", "10")))
    cli = parser.parse_args(args)

    enhance_subsections(cli.course_name, cli.module_name, cli.module_slug, cli.output_base_dir, cli.top_k_item)


if __name__ == "__main__":
    main()



import os
import json
from typing import Dict, Any, List

from pydantic_ai import Agent
from src.agents.base_notes_agent import BaseNotesAgent
from src.latex_utils.writer import write_subsection_file, sanitize_latex
from src.models.latex import SectionRef, SectionContent
from src.models.paths import OutputPaths
from src.logger import get_logger


class SubsectionEnhancerAgent(BaseNotesAgent):
    def run(self, top_k_item: int) -> None:
        log = get_logger(__name__)

        # Paths and skeleton
        out_dir = self.out_dir()
        os.makedirs(out_dir, exist_ok=True)
        skeleton_path = os.path.join(out_dir, "skeleton.json")
        if not os.path.exists(skeleton_path):
            log.info("skeleton.json not found at %s; nothing to enhance.", skeleton_path)
            return
        with open(skeleton_path, "r", encoding="utf-8") as f:
            skeleton: Dict[str, Any] = json.load(f)
        subsections: List[Dict[str, Any]] = skeleton.get("subsections", [])
        if not isinstance(subsections, list) or not subsections:
            log.info("No subsections found in skeleton; nothing to enhance.")
            return

        # Build retrieval cache and slug index
        retriever = self.retriever(max(top_k_item, 50))
        log.info("[lowlight]Retrieval query[/]: 'List all the files'")
        source_nodes = retriever.retrieve("List all the files")
        slug_to_nodes: Dict[str, List[Any]] = {}
        for n in source_nodes:
            slug = n.metadata.get("item_slug")
            if not slug:
                continue
            slug_to_nodes.setdefault(slug, []).append(n)

        agent = Agent(model=self.settings.ai_writer_model)
        # Load subsection prompt
        try:
            with open(os.path.join("prompts", "subsection_query.txt"), "r", encoding="utf-8") as f:
                subsection_prompt = f.read().strip()
        except Exception:
            subsection_prompt = ""

        # Enhanced output directory
        enhanced_dir = os.path.join(out_dir, "enhanced")
        os.makedirs(enhanced_dir, exist_ok=True)

        eval_title = "Evaluation and Future Directions"

        # Helper to build bounded context from topic slugs
        def build_context_from_slugs(topic_slugs: List[str]) -> str:
            import textwrap
            ctx_chars = int(os.getenv("D2R_ENH_SUBSECTION_CHARS", "4000"))
            per_slug_k = int(os.getenv("D2R_RAG_TOP_K_ITEM", "2"))
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
            return "\n\n".join(chunks)

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
            result = agent.run_sync(prompt)
            out_text = getattr(result, "content", None) or getattr(result, "output", None) or str(result)
            if "\\subsection" not in out_text:
                out_text = f"\\subsection{{{title}}}\n" + out_text

            sub_path = os.path.join(enhanced_dir, f"{title}.tex")
            os.makedirs(os.path.dirname(sub_path), exist_ok=True)
            healed = sanitize_latex(out_text)
            write_subsection_file(title, healed, sub_path)
            try:
                SectionContent(ref=SectionRef(order=1, type="subsection", title=title, path=sub_path), text=healed)
            except Exception as e:
                err_path = sub_path + ".err.txt"
                try:
                    with open(err_path, "w", encoding='utf-8') as ef:
                        ef.write(str(e))
                    log.warning("Validation failed for %s -> wrote error: %s", title, err_path)
                except Exception:
                    pass
            log.info("Wrote enhanced subsection %s -> %s", title, sub_path)




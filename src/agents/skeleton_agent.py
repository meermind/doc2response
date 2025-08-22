import os
import json
from typing import Dict, Any, List

from pydantic_ai import Agent
from src.agents.base_notes_agent import BaseNotesAgent
from src.latex_utils.writer import write_intro_file, write_subsection_file
from src.models.latex import LatexMetadata, SectionRef
from src.models.paths import OutputPaths
from src.logger import get_logger


class SkeletonAgent(BaseNotesAgent):

    def _load_prompt(self, file_name: str) -> str:
        with open(os.path.join("prompts", file_name), "r", encoding="utf-8") as f:
            return f.read().strip()

    def run(self, list_top_k: int, course_name: str) -> dict:
        log = get_logger(__name__)

        # Discover topics and build nodes cache
        log.info("[lowlight]Retrieval query[/]: 'List all the files'")
        topics = self.list_item_slugs(list_top_k)
        source_nodes = self.retriever(list_top_k).retrieve("List all the files")
        slug_to_nodes: Dict[str, List[Any]] = {}
        for n in source_nodes:
            slug = n.metadata.get("item_slug")
            if not slug:
                continue
            slug_to_nodes.setdefault(slug, []).append(n)

        # Persist topics
        module_name_out = self.module_name
        out_dir = self.out_dir()
        os.makedirs(out_dir, exist_ok=True)
        try:
            topics_cache_path = os.path.join(self.output_base_dir, course_name, module_name_out, "unique_topics.json")
            os.makedirs(os.path.dirname(topics_cache_path), exist_ok=True)
            with open(topics_cache_path, "w") as f:
                json.dump({"module": module_name_out, "unique_topics": topics}, f, indent=2)
        except Exception:
            pass

        # Outline using assistant_message prompt
        agent = Agent(model=self.settings.ai_writer_model)
        assistant_msg = self._load_prompt("assistant_message.txt")

        # Build lightweight context from discovered topic slugs
        def build_context_from_topics(topic_slugs: List[str]) -> str:
            import textwrap
            ctx_chars = int(os.getenv("D2R_SKELETON_CONTEXT_CHARS", "4000"))
            per_topic_k = int(os.getenv("D2R_RAG_TOP_K_SKELETON", "1"))
            budget = ctx_chars
            chunks: List[str] = []
            for slug in topic_slugs:
                if budget <= 0:
                    break
                nodes = slug_to_nodes.get(slug, [])
                take = 0
                for n in nodes:
                    if take >= per_topic_k:
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
                        take += 1
            return "\n\n".join(chunks)

        outline_context = build_context_from_topics(topics)
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
            f"Module: {self.module_name}\n"
            f"Available topic slugs: {topics}\n"
            "Context (excerpts; summarize, do not copy verbatim):\n"
            f"{outline_context}"
        )
        outline = agent.run_sync(outline_prompt)
        raw = getattr(outline, "content", None) or getattr(outline, "output", None) or str(outline)
        try:
            parsed: Dict[str, Any] = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            parsed = {"intro_title": "Introduction", "subsections": [{"title": t, "topics": [t]} for t in topics]}
        intro_title = parsed.get("intro_title") or "Introduction"
        subsections = parsed.get("subsections") or []

        # Build mapping from topics arrays
        available_slugs = set(topics)
        def lookup_item(slug: str) -> Dict[str, str]:
            nodes = slug_to_nodes.get(slug, [])
            for n in nodes:
                if n.metadata.get("item_slug") == slug:
                    return {
                        "course_slug": n.metadata.get("course_slug", ""),
                        "module_slug": n.metadata.get("module_slug", self.module_slug or ""),
                        "lesson_slug": n.metadata.get("lesson_slug", ""),
                        "item_slug": slug,
                    }
            return {"course_slug": "", "module_slug": self.module_slug or "", "lesson_slug": "", "item_slug": slug}

        mapping: Dict[str, Any] = {"subsections": []}
        titles: List[str] = []
        eval_title = "Evaluation and Future Directions"
        for s in subsections:
            title = s.get("title", "Section")
            titles.append(title)
            if title.strip().lower() == eval_title.lower():
                continue
            slugs = [t for t in (s.get("topics") or []) if t]
            items = [lookup_item(t) for t in slugs if t in available_slugs]
            missing = [t for t in slugs if t not in available_slugs]
            entry = {"title": title, "items": items, "missing_topics": missing}
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

        # Create .tex files and metadata.json
        sections: List[SectionRef] = []
        paths = OutputPaths(base_dir=self.output_base_dir, course_name=course_name, module_name=module_name_out)
        intro_path = paths.intro_path(intro_title)
        os.makedirs(os.path.dirname(intro_path), exist_ok=True)
        write_intro_file(intro_title, intro_path)
        sections.append(SectionRef(order=0, type="section", title=intro_title, path=intro_path))

        eval_path = None
        for i, s in enumerate(subsections, start=1):
            title = s.get("title", "Section")
            sub_path = paths.subsection_path(title)
            os.makedirs(os.path.dirname(sub_path), exist_ok=True)
            summary_latex = s.get("summary_latex") or f"\\subsection{{{title}}}\n"
            write_subsection_file(title, summary_latex, sub_path)
            sections.append(SectionRef(order=i, type="subsection", title=title, path=sub_path))
            if title.strip().lower() == eval_title.lower():
                eval_path = sub_path

        meta = LatexMetadata(sections=sections)
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta.model_dump(), f, indent=2)

        # Second agent query: final intro/eval
        try:
            eval_intro_prompt = (
                "You will draft two LaTeX parts for this module using the provided context excerpts.\n"
                "Return STRICT JSON (no code fences) with keys: intro_latex, evaluation_latex.\n"
                "Requirements:\n"
                "- intro_latex MUST be a single \\section with narrative text only; do NOT include any other \\subsection headings.\n"
                "- evaluation_latex MUST be a single \\subsection titled 'Evaluation and Future Directions' with concise bullets and short narrative, referring to the actual subsection titles.\n"
                f"Module: {self.module_name}\n"
                f"Intro title: {intro_title}\n"
                f"Evaluation title: {eval_title}\n"
                f"Subsections (titles): {[s.get('title') for s in subsections]}\n"
                "Context (excerpts; summarize, do not copy verbatim):\n"
                f"{outline_context}"
            )
            resp2 = agent.run_sync(eval_intro_prompt)
            raw2 = getattr(resp2, "content", None) or getattr(resp2, "output", None) or str(resp2)
            try:
                parsed2 = json.loads(raw2) if isinstance(raw2, str) else raw2
            except Exception:
                parsed2 = {"intro_latex": "", "evaluation_latex": ""}
            intro_latex_final = parsed2.get("intro_latex") or f"\\section{{{intro_title}}}\n"
            evaluation_latex_final = parsed2.get("evaluation_latex") or f"\\subsection{{{eval_title}}}\n"
            with open(intro_path, 'w', encoding='utf-8') as f:
                f.write(intro_latex_final)
            if eval_path:
                with open(eval_path, 'w', encoding='utf-8') as f:
                    f.write(evaluation_latex_final)
        except Exception:
            pass

        # Collate assistant_message.tex
        try:
            self.collate([s.model_dump() for s in sections], intro_path, os.path.join(out_dir, "assistant_message.tex"))
        except Exception:
            pass

        return {
            "course_name": course_name,
            "module": module_name_out,
            "out_dir": out_dir,
            "topics": topics,
        }



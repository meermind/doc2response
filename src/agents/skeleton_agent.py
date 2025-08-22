import os
import json
from typing import Dict, Any, List

from pydantic_ai import Agent
from src.agents.base_notes_agent import BaseNotesAgent
from src.latex_utils.writer import LatexAssembler, LatexSectionWriter
from src.latex_utils.metadata_store import MetadataStore
from src.models.paths import OutputPaths


class SkeletonAgent(BaseNotesAgent):

    # Story-style steps
    def discover_topics(self, list_top_k: int) -> Dict[str, Any]:
        self.log_stage("Discover topics")
        topics = self.list_item_slugs(list_top_k)
        slug_to_nodes = self.build_slug_to_nodes(list_top_k)
        return {"topics": topics, "slug_to_nodes": slug_to_nodes}

    def outline_module(self, topics: List[str], slug_to_nodes: Dict[str, List[Any]]) -> Dict[str, Any]:
        self.log_stage("Outline module")
        use_model = self.resolve_model("skeleton")
        self.log_model("skeleton", str(use_model))
        agent = Agent(model=use_model)
        assistant_msg = self.load_prompt("assistant_message.txt")
        outline_context = self.build_outline_context(topics, slug_to_nodes)
        prompt = (
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
        outline = agent.run_sync(prompt)
        raw = getattr(outline, "content", None) or getattr(outline, "output", None) or str(outline)
        try:
            parsed: Dict[str, Any] = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            parsed = {"intro_title": "Introduction", "subsections": [{"title": t, "topics": [t]} for t in topics]}
        return parsed

    def _build_subsection_mapping(self, subsections: List[Dict[str, Any]], available_slugs: set, slug_to_nodes: Dict[str, List[Any]]) -> Dict[str, Any]:
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
        for s in subsections or []:
            title = s.get("title", "Section")
            slugs = [t for t in (s.get("topics") or []) if t]
            items = [lookup_item(t) for t in slugs if t in available_slugs]
            missing = [t for t in slugs if t not in available_slugs]
            entry: Dict[str, Any] = {"title": title, "items": items, "missing_topics": missing}
            # Optionally propagate subsubsections if present
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
        return mapping

    def build_and_write_mapping(self, subsections: List[Dict[str, Any]], topics: List[str], slug_to_nodes: Dict[str, List[Any]], out_dir: str) -> str:
        mapping = self._build_subsection_mapping(subsections, set(topics), slug_to_nodes)
        return self.metadata_store.save_mapping(out_dir, mapping)

    def write_subsections(self, course_name: str, subsections: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        assembler = LatexAssembler(self.output_base_dir, course_name, self.module_name)
        entries: List[Dict[str, str]] = []
        for i, s in enumerate(subsections, start=1):
            title = s.get("title", "Section")
            summary_latex = s.get("summary_latex") or f"\\subsection{{{title}}}\n"
            sub_path = assembler.write_subsection(title, summary_latex)
            entries.append({"order": i, "type": "subsection", "title": title, "path": sub_path})
        return entries

    def finalize_intro_and_eval(self, course_name: str, intro_title: str, subsections: List[Dict[str, Any]], outline_context: str, subsection_entries: List[Dict[str, str]], eval_title: str = "Evaluation and Future Directions") -> None:
        self.log_stage("Finalize intro and evaluation")
        use_model = self.resolve_model("skeleton")
        self.log_model("skeleton", str(use_model))
        agent = Agent(model=use_model)
        prompt = (
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
        try:
            resp2 = agent.run_sync(prompt)
            raw2 = getattr(resp2, "content", None) or getattr(resp2, "output", None) or str(resp2)
            try:
                parsed2 = json.loads(raw2) if isinstance(raw2, str) else raw2
            except Exception:
                parsed2 = {"intro_latex": "", "evaluation_latex": ""}
            writer = LatexSectionWriter()
            assembler = LatexAssembler(self.output_base_dir, course_name, self.module_name)
            # Intro (let writer handle sanitization + heading)
            intro_path = assembler.paths.intro_path(intro_title)
            writer.write_section_content(intro_title, parsed2.get("intro_latex") or f"\\section{{{intro_title}}}\n", intro_path)
            # Evaluation subsection (pass raw; writer handles sanitization)
            eval_path = assembler.write_subsection(eval_title, parsed2.get("evaluation_latex") or f"\\subsection{{{eval_title}}}\n")
            # Update metadata: section (order 0) + existing subsections + evaluation at the end
            sections: List[Dict[str, str]] = []
            sections.append({"order": 0, "type": "section", "title": intro_title, "path": intro_path})
            # Reindex orders: keep original relative order for subsections
            next_order = 1
            for e in subsection_entries:
                sections.append({"order": next_order, "type": "subsection", "title": e["title"], "path": e["path"]})
                next_order += 1
            sections.append({"order": next_order, "type": "subsection", "title": eval_title, "path": eval_path})
            assembler.update_metadata(sections)
        except Exception:
            pass

    def write_skeleton(self, out_dir: str, intro_title: str, _subsections: List[Dict[str, Any]]) -> str:
        mapping = self.metadata_store.load_mapping(out_dir)
        normalized: List[Dict[str, Any]] = []
        for s in mapping.get("subsections", []) or []:
            items = s.get("items", []) or []
            topics = [it.get("item_slug") for it in items if it.get("item_slug")]
            normalized.append({"title": s.get("title", "Section"), "topics": topics})
        return self.metadata_store.save_skeleton(out_dir, intro_title, normalized)

    def log_skeleton_table(self, skeleton_path: str) -> None:
        try:
            with open(skeleton_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rows = data.get('subsections', []) or []
            # Compute widths
            title_w = max(len('subsection'), *(len(r.get('title', '')) for r in rows)) if rows else len('subsection')
            header = f"{'subsection':<{title_w}} {'#topics':>8} topics"
            self._log.info(header)
            for r in rows:
                title = r.get('title', '')
                topics = [t for t in (r.get('topics') or []) if t]
                topics_str = ", ".join(topics)
                self._log.info(f"{title:<{title_w}} {len(topics):>8} {topics_str}")
        except Exception:
            pass

    def run(self, list_top_k: int, course_name: str) -> dict:
        out_dir = self.ensure_out_dir()
        # 1) Discover
        d = self.discover_topics(list_top_k)
        topics: List[str] = d["topics"]
        slug_to_nodes = d["slug_to_nodes"]
        # Persist topics cache
        self.metadata_store.save_unique_topics(self.output_base_dir, course_name, self.module_name, topics)
        # 2) Outline
        parsed = self.outline_module(topics, slug_to_nodes)
        intro_title = parsed.get("intro_title") or "Introduction"
        subsections = parsed.get("subsections") or []
        outline_context = self.build_outline_context(topics, slug_to_nodes)
        # 3) Mapping
        self.build_and_write_mapping(subsections, topics, slug_to_nodes, out_dir)
        # 4) Write skeleton.json
        sk_path = self.write_skeleton(out_dir, intro_title, subsections)
        self.log_skeleton_table(sk_path)
        # 5) Write subsections only (no intro/eval)
        subsection_entries = self.write_subsections(course_name, subsections)
        # 6) Finalize intro and evaluation, and update metadata
        self.finalize_intro_and_eval(course_name, intro_title, subsections, outline_context, subsection_entries)
        # 7) Collate and return assistant_message path
        assistant_message_path = os.path.join(out_dir, "assistant_message.tex")
        try:
            assembler = LatexAssembler(self.output_base_dir, course_name, self.module_name)
            assembler.collate_from_metadata(self.output_base_dir, assistant_message_path)
        except Exception:
            pass
        return {"course_name": course_name, "module": self.module_name, "out_dir": out_dir, "topics": topics, "skeleton_path": sk_path, "assistant_message": assistant_message_path}



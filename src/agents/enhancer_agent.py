import os
import json
from typing import Dict, Any, List

from pydantic_ai import Agent
from src.agents.base_notes_agent import BaseNotesAgent
from src.latex_utils.writer import collate_from_metadata, LatexSectionWriter
from src.latex_utils.metadata_store import MetadataStore
from src.models.latex import SectionRef, SectionContent
from src.logger import get_logger


class SubsectionEnhancerAgent(BaseNotesAgent):
    # Story-style steps
    def load_skeleton(self, out_dir: str) -> List[Dict[str, Any]]:
        self.log_stage("Load skeleton")
        log = get_logger(__name__)
        log.info(
            "Loading subsections from metadata at: %s",
            MetadataStore.metadata_dir(out_dir),
        )
        return self.metadata_store.load_normalized_subsections(out_dir)

    def prepare_retrieval(self, top_k_item: int) -> Dict[str, List[Any]]:
        self.log_stage("Prepare retrieval cache")
        return self.build_slug_to_nodes(max(top_k_item, 50))

    def build_context_from_slugs(self, topic_slugs: List[str], slug_to_nodes: Dict[str, List[Any]]) -> str:
        return self.build_enhance_context(topic_slugs, slug_to_nodes)

    def enhance_subsection(self, title: str, topic_slugs: List[str], subsection_prompt: str, slug_to_nodes: Dict[str, List[Any]]) -> str:
        use_model = self.resolve_model("enhancer")
        self.log_model("enhancer", str(use_model))
        agent = Agent(model=use_model)
        context_text = self.build_context_from_slugs(topic_slugs, slug_to_nodes)
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
        return getattr(result, "content", None) or getattr(result, "output", None) or str(result)

    def write_and_validate_subsection(self, enhanced_dir: str, title: str, out_text: str) -> str:
        sub_path = os.path.join(enhanced_dir, f"{title}.tex")
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        writer = LatexSectionWriter()
        # Delegate sanitization/heading to writer
        writer.write_subsection(title, out_text, sub_path)
        try:
            with open(sub_path, 'r', encoding='utf-8') as rf:
                final_text = rf.read()
            SectionContent(ref=SectionRef(order=1, type="subsection", title=title, path=sub_path), text=final_text)
        except Exception as e:
            err_path = sub_path + ".err.txt"
            try:
                with open(err_path, "w", encoding='utf-8') as ef:
                    ef.write(str(e))
            except Exception:
                pass
        return sub_path

    def run(self, top_k_item: int) -> Dict[str, str] | None:
        log = get_logger(__name__)

        # Paths and skeleton
        out_dir = self.ensure_out_dir()
        subsections = self.load_skeleton(out_dir)
        if not subsections:
            log.info("skeleton.json/subsection_mapping.json not found or no subsections; nothing to enhance.")
            return None

        # Build retrieval cache and slug index
        slug_to_nodes = self.prepare_retrieval(top_k_item)

        # Load subsection prompt
        try:
            with open(os.path.join("prompts", "subsection_query.txt"), "r", encoding="utf-8") as f:
                subsection_prompt = f.read().strip()
        except Exception:
            subsection_prompt = ""

        # Enhanced output directory
        enhanced_dir = os.path.join(out_dir, "enhanced")
        os.makedirs(enhanced_dir, exist_ok=True)

        enhanced_any = False
        mdframe_entries: List[Dict[str, str]] = []
        for s in subsections:
            title = s.get("title", "Section")
            topic_slugs = [t for t in (s.get("topics") or []) if t]
            log.info("[cyan]Enhancing[/] '%s' with %d topic slugs", title, len(topic_slugs))
            out_text = self.enhance_subsection(title, topic_slugs, subsection_prompt, slug_to_nodes)
            sub_path = self.write_and_validate_subsection(enhanced_dir, title, out_text)
            log.info("Wrote enhanced subsection %s -> %s", title, sub_path)
            enhanced_any = True
            # If the generated content references mdframe placeholders, record for mdframe stage
            try:
                with open(sub_path, 'r', encoding='utf-8') as rf:
                    txt = rf.read()
                if ("mdframe" in txt.lower()) or ("mdframed" in txt.lower()):
                    mdframe_entries.append({"title": title, "path": sub_path, "topics": topic_slugs})
            except Exception:
                pass

        assistant_message_path = None
        if enhanced_any:
            try:
                assistant_message_path = os.path.join(enhanced_dir, "assistant_message.tex")
                collate_from_metadata(self.output_base_dir, self.course_name, self.module_name, assistant_message_path)
                # Persist mdframe_skeleton.json for downstream mdframe agent
                module_dir = self.out_dir()
                if mdframe_entries:
                    MetadataStore.save_mdframe_skeleton(module_dir, enhanced_dir, mdframe_entries)
            except Exception:
                pass
        return {"assistant_message": assistant_message_path} if assistant_message_path else None




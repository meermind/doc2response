import os
import re
from typing import List, Dict, Any

from pydantic_ai import Agent
from src.agents.base_notes_agent import BaseNotesAgent
from src.latex_utils.writer import collate_from_metadata, LatexSectionWriter
from src.latex_utils.metadata_store import MetadataStore
from src.logger import get_logger


class MdframeAgent(BaseNotesAgent):
    def _find_mdframe_targets(self, text: str) -> list[dict]:
        targets: list[dict] = []
        try:
            for m in re.finditer(r"(?s)\\begin\{mdframed\}.*?\\end\{mdframed\}", text):
                targets.append({"kind": "env", "span": (m.start(), m.end()), "text": m.group(0)})
            for m in re.finditer(r"(?mi)^%.*mdframe.*$", text):
                targets.append({"kind": "placeholder", "span": (m.start(), m.end()), "text": m.group(0)})
            targets.sort(key=lambda d: d["span"][0])
        except Exception:
            return []
        return targets
    def _extract_mdframe_block(self, text: str) -> str:
        try:
            m = re.search(r"(?s)\\begin\{mdframed\}.*?\\end\{mdframed\}", text)
            if m:
                return m.group(0)
            # Fallback: a single placeholder/comment line mentioning mdframe
            m2 = re.search(r"(?mi)^%.*mdframe.*$", text)
            if m2:
                return m2.group(0)
        except Exception:
            pass
        return ""

    # Discovery/inputs
    def load_spec_and_scan(self, assistant_message_path: str) -> tuple[str, str, list[dict] | None]:
        """Return (assistant_message_text, scan_dir, entries_from_spec_or_none)."""
        log = get_logger(__name__)
        assistant_message = self._read_file(assistant_message_path)
        module_dir = self.out_dir()
        mdframe_spec = MetadataStore.load_mdframe_skeleton(module_dir)
        if mdframe_spec:
            scan_dir = mdframe_spec.get("root_path") or os.path.dirname(assistant_message_path)
            entries = mdframe_spec.get("entries") or []
            log.info("[cyan]mdframe_skeleton.json found[/]: entries=%d, root=%s", len(entries), scan_dir)
        else:
            scan_dir = os.path.dirname(assistant_message_path)
            entries = None
            log.info("[yellow]mdframe_skeleton.json missing[/]; scanning directory for candidates: %s", scan_dir)
        return assistant_message, scan_dir, entries

    def list_targets(self, scan_dir: str, entries: list[dict] | None) -> list[dict]:
        """Return list of targets: {path, title, topics?}."""
        log = get_logger(__name__)
        if entries:
            targets = []
            for e in entries:
                p = e.get("path")
                if not p:
                    continue
                targets.append({"path": p, "title": os.path.splitext(os.path.basename(p))[0], "topics": e.get("topics")})
        else:
            targets = []
            for p in self._list_candidate_tex(scan_dir):
                targets.append({"path": p, "title": os.path.splitext(os.path.basename(p))[0], "topics": None})
        log.info("Candidate files to scan: %d", len(targets))
        return targets

    # Context
    def build_item_context(self, title: str, topic_slugs: list[str], per_topic_k: int) -> str:
        """Retrieve item-level nodes per slug, log table, and build context text."""
        log = get_logger(__name__)
        # Retrieve nodes per slug using base helper; report all types but build context from extra_notes only
        slug_to_nodes = self.collect_slug_to_nodes_for_items(topic_slugs, per_topic_k)

        # Standardized reporting via base helper (report all types)
        self.log_item_slug_table(slug_to_nodes)

        # Build context chunks (only from extra_notes)
        # Build context using only extra_notes nodes
        extra_only: Dict[str, List[Any]] = {}
        extra_nodes = 0
        for slug in topic_slugs:
            extra_only[slug] = []
            for n in slug_to_nodes.get(slug, []) or []:
                if self._classify_doc_type(n) == "extra_notes":
                    extra_only[slug].append(n)
                    extra_nodes += 1
        context_text = self.build_mdframe_context(topic_slugs, extra_only)
        log.info(
            "Building mdframe for '%s' with %d topics (per_topic_k=%d, extra_nodes=%d, ctx_chars=%d)",
            title,
            len(topic_slugs),
            per_topic_k,
            extra_nodes,
            len(context_text or ""),
        )
        return context_text

    def build_prompt(self, base_prompt: str, assistant_message: str, subsection_text: str, mdframe_block: str, context_text: str) -> str:
        return (
            f"{base_prompt}\n\n"
            f"Given the context of this full topic, expand with rigorous, correct LaTeX.\n"
            f"--- Full Topic Context (assistant_message) ---\n{assistant_message}\n\n"
            f"We need to expand our knowledge on this subsection.\n"
            f"--- Subsection (current content) ---\n{subsection_text}\n\n"
            f"By enhancing this advanced topic: the mdframe below. Replace or insert a single mdframed block.\n"
            f"--- mdframe (existing/placeholder) ---\n{mdframe_block}\n\n"
            f"You may wish to refer to this extra content (topic-level retrieval).\n"
            f"--- Extra Context ---\n{context_text}\n"
        )

    def apply_mdframe_update(self, title: str, tex_path: str, content: str, mdframed: str, has_placeholder: bool) -> str:
        log = get_logger(__name__)
        # Replace placeholder comment if present; else replace first mdframed block; else append
        if re.search(r"(?mi)^%\s*mdframe.*$", content):
            new_content = re.sub(r"(?mi)^%\s*mdframe.*$", lambda m: (mdframed.strip() + "\n"), content)
        elif re.search(r"(?s)\\begin\{mdframed\}.*?\\end\{mdframed\}", content):
            new_content = re.sub(r"(?s)\\begin\{mdframed\}.*?\\end\{mdframed\}", lambda m: mdframed.strip(), content, count=1)
            log.info("Replaced existing mdframed block in %s", tex_path)
        else:
            if not has_placeholder:
                log.info("No placeholder found; appending mdframed block to end: %s", tex_path)
            new_content = content.rstrip() + "\n\n" + mdframed.strip() + "\n"
        # Redirect output path to mdframed directory
        module_dir = self.out_dir()
        md_dir = os.path.join(module_dir, "mdframed")
        os.makedirs(md_dir, exist_ok=True)
        out_path = os.path.join(md_dir, os.path.basename(tex_path))
        writer = LatexSectionWriter()
        writer.write_subsection(title, new_content, out_path)
        log.info("Inserted mdframed into %s via writer", out_path)
        return out_path
    def _load_prompt(self) -> str:
        with open(os.path.join("prompts", "mdframe.txt"), "r", encoding="utf-8") as f:
            return f.read().strip()

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def _list_candidate_tex(self, base_dir: str) -> List[str]:
        files: List[str] = []
        try:
            for name in os.listdir(base_dir):
                if not name.endswith(".tex"):
                    continue
                if name == "assistant_message.tex":
                    continue
                files.append(os.path.join(base_dir, name))
        except Exception:
            pass
        return files

    def _has_mdframe_placeholder(self, text: str) -> bool:
        t = text.lower()
        # Treat either explicit placeholder comments, mentions, or an existing mdframed env as a target
        if ("mdframe suggestions" in t) or ("% mdframe" in t):
            return True
        if "\\begin{mdframed}" in t and "\\end{mdframed}" in t:
            return True
        return ("mdframed" in t) or ("mdframe" in t)


    def run(self, assistant_message_path: str, top_k_extra: int = 10) -> Dict[str, str] | None:
        log = get_logger(__name__)
        # Determine scan dir and entries
        if not assistant_message_path or not os.path.exists(assistant_message_path):
            log.info("assistant_message not found; skipping mdframe agent.")
            return None
        assistant_message, scan_dir, entries = self.load_spec_and_scan(assistant_message_path)
        log.info("Assistant message path: %s (chars=%d)", assistant_message_path, len(assistant_message or ""))
        log.info("[lowlight]Extra notes disabled for mdframe context[/]")
        base_prompt = self._load_prompt()
        use_model = self.resolve_model("mdframe")
        self.log_model("mdframe", str(use_model))
        agent = Agent(model=use_model)

        updated_any = False
        targets = self.list_targets(scan_dir, entries)
        use_spec = entries is not None
        for t in targets:
            tex_path = t["path"]
            title = t["title"]
            content = self._read_file(tex_path)
            has_placeholder = bool(content) and self._has_mdframe_placeholder(content)
            log.info(
                "Examining: %s (chars=%d, placeholder=%s, force_process=%s)",
                tex_path, len(content or ""), has_placeholder, use_spec,
            )
            if not content:
                continue
            if (not use_spec) and (not has_placeholder):
                continue

            # Build topic-aware context
            topics = t.get("topics") if use_spec else None
            if topics:
                try:
                    per_topic_k = max(1, int(os.getenv("D2R_RAG_TOP_K_MDFRAME", "1")))
                except Exception:
                    per_topic_k = 1
                context_text = self.build_item_context(title, topics, per_topic_k)
            else:
                context_text = ""
                log.info("Building mdframe for '%s' (no topic_slugs; ctx_chars=0)", title)

            # Find all mdframe targets in the file and process each sequentially
            targets = self._find_mdframe_targets(content)
            if targets:
                log.info("Found %d mdframe target(s) in %s", len(targets), tex_path)
                # Process original targets in reverse order to avoid re-detection and span shifts
                processed = 0
                for tgt in reversed(targets[:10]):  # safety cap at 10 targets per file
                    mdframe_block = tgt.get("text") or ""
                    subsection_text = content or ""
                    prompt = self.build_prompt(base_prompt, assistant_message, subsection_text, mdframe_block, context_text)
                    try:
                        resp = agent.run_sync(prompt)
                        mdframed = getattr(resp, "content", None) or getattr(resp, "output", None) or str(resp)
                        if not mdframed or not str(mdframed).strip():
                            log.info("[yellow]Empty mdframe response[/] for %s; skipping one target", tex_path)
                            continue
                        from src.latex_utils.sanitizer import LatexSanitizer
                        mdframed = LatexSanitizer.sanitize(mdframed)
                        start, end = tgt["span"]
                        content = content[:start] + mdframed.strip() + "\n" + content[end:]
                        processed += 1
                    except Exception as e:
                        log.error("[red]Mdframe error[/] for %s: %s", tex_path, e)
                        continue
                # Write final content to mdframed path
                module_dir = self.out_dir()
                md_dir = os.path.join(module_dir, "mdframed")
                os.makedirs(md_dir, exist_ok=True)
                out_path = os.path.join(md_dir, os.path.basename(tex_path))
                writer = LatexSectionWriter()
                writer.write_subsection(title, content, out_path)
                log.info("Inserted %d mdframed block(s) into %s via writer", processed, out_path)
                updated_any = True
            else:
                # No targets; fall back to single-generation append/replace path
                subsection_text = content or ""
                mdframe_block = self._extract_mdframe_block(content)
                prompt = self.build_prompt(base_prompt, assistant_message, subsection_text, mdframe_block, context_text)
                try:
                    resp = agent.run_sync(prompt)
                    mdframed = getattr(resp, "content", None) or getattr(resp, "output", None) or str(resp)
                    if not mdframed or not str(mdframed).strip():
                        log.info("[yellow]Empty mdframe response[/] for %s; skipping write", tex_path)
                        continue
                    from src.latex_utils.sanitizer import LatexSanitizer
                    mdframed = LatexSanitizer.sanitize(mdframed)
                    self.apply_mdframe_update(title, tex_path, content, mdframed, has_placeholder)
                    updated_any = True
                except Exception as e:
                    log.error("[red]Mdframe error[/] for %s: %s", tex_path, e)
                    continue

        # Re-collate enhanced assistant_message in current scan_dir if we updated anything
        # Re-collate enhanced assistant_message in mdframed directory if we updated anything
        module_dir = self.out_dir()
        md_dir = os.path.join(module_dir, "mdframed")
        os.makedirs(md_dir, exist_ok=True)
        out_path = os.path.join(md_dir, "assistant_message.tex")
        if updated_any:
            try:
                # Collate using metadata; merger prefers mdframed over enhanced automatically
                collate_from_metadata(self.output_base_dir, self.course_name, self.module_name, out_path)
            except Exception:
                pass
            log.info("Assistant message updated: %s", out_path)
            return {"assistant_message": out_path}
        log.info("[yellow]No mdframe updates performed[/].")
        return None



import os
from collections import Counter
from typing import List, Dict, Optional, Any

from sqlalchemy import URL
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from src.models.settings import Settings
from models.config import TiDBSettings
from src.models.paths import OutputPaths
from src.latex_utils.writer import collate_assistant_message
from src.latex_utils.metadata_store import MetadataStore
from src.logger import get_logger


class BaseNotesAgent:
    def __init__(
        self,
        course_name: str,
        module_name: str,
        module_slug: Optional[str],
        output_base_dir: Optional[str] = None,
        metadata_store: MetadataStore | None = None,
    ) -> None:
        self.settings = Settings()
        self.db = TiDBSettings()

        self.course_name = course_name
        self.module_name = module_name
        self.module_slug = module_slug
        self.output_base_dir = output_base_dir or os.getenv("D2R_OUTPUT_BASE") or "assistant_latex"

        # Embeddings
        self.embedding_model = self.settings.create_embedding()
        self.embedding_dimension = self.settings.embedding_dimension

        # Vector store / index
        tidb_connection_url = URL(
            "mysql+pymysql",
            username=self.db.username,
            password=self.db.password,
            host=self.db.host,
            port=int(self.db.port),
            database=self.db.db_name,
            query={"ssl_verify_cert": True, "ssl_verify_identity": True},
        )
        self.vector_store = TiDBVectorStore(
            connection_string=tidb_connection_url,
            table_name=os.getenv("VECTOR_TABLE_NAME", "demo_load_docs_to_llamaindex"),
            distance_strategy="cosine",
            vector_dimension=self.embedding_dimension,
            drop_existing_table=False,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embedding_model,
        )

        self._log = get_logger(__name__)
        self.metadata_store = metadata_store or MetadataStore()

    def base_filters(self) -> List[Dict[str, str]]:
        if self.module_slug:
            return [{"key": "module_slug", "value": self.module_slug, "operator": "=="}]
        return [{"key": "module_name", "value": self.module_name, "operator": "=="}]

    def base_metadata_filters(self) -> MetadataFilters:
        return MetadataFilters(filters=[MetadataFilter(**f) for f in self.base_filters()])

    def retriever(self, similarity_top_k: int) -> VectorIndexRetriever:
        return VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
            embedding_model=self.embedding_model,
            filters=self.base_metadata_filters(),
        )
    
    def list_item_slugs(self, list_top_k: int) -> List[str]:
        log = get_logger(__name__)
        nodes = self.retriever(list_top_k).retrieve("List all the files")

        slugs = sorted({n.metadata.get("item_slug") for n in nodes if n.metadata.get("item_slug")})
        counts = {s: Counter() for s in slugs}
        for n in nodes:
            s = n.metadata.get("item_slug")
            if not s: 
                continue
            dt = str(n.metadata.get("doc_type") or n.metadata.get("type") or n.metadata.get("content_type") or "").lower()
            key = "transcripts" if "transcript" in dt else "extra_notes" if ("extra" in dt or "note" in dt) else "slides" if "slide" in dt else None
            if key and s in counts:
                counts[s][key] += 1

        log.info(f"[cyan]Topics found[/] ({len(slugs)}):")
        cats = ("transcripts", "extra_notes", "slides")
        slug_w = (max(len("item_slug"), *(map(len, slugs))) + 2) if slugs else len("item_slug")
        header = f"{'item_slug':<{slug_w}}" + "".join(f" {c:>12}" for c in (*cats, "total"))
        log.info(header)

        totals = Counter()
        for s in slugs:
            row_total = sum(counts[s][k] for k in cats)
            for k in cats: totals[k] += counts[s][k]
            totals["total"] += row_total
            log.info(f"{s:<{slug_w}}" + "".join(f" {counts[s][k]:>12}" for k in cats) + f" {row_total:>12}")

        log.info(f"{'TOTAL':<{slug_w}}" + "".join(f" {totals[k]:>12}" for k in cats) + f" {totals['total']:>12}")
        return slugs

    def out_dir(self) -> str:
        return OutputPaths(base_dir=self.output_base_dir, course_name=self.course_name, module_name=self.module_name).module_dir()

    def ensure_out_dir(self) -> str:
        path = self.out_dir()
        os.makedirs(path, exist_ok=True)
        return path

    def collate(self, section_refs: List[Dict[str, str]], intro_path: str, out_path: str) -> None:
        collate_assistant_message(section_refs, intro_path, out_path)

    # Common small utilities
    def load_prompt(self, file_name: str) -> str:
        with open(os.path.join("prompts", file_name), "r", encoding="utf-8") as f:
            return f.read().strip()

    def write_json(self, data: Any, path: str) -> None:
        import json as _json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2)

    def log_stage(self, title: str) -> None:
        self._log.info("\n[bold]>> %s[/]", title)

    # Shared retrieval/cache utilities
    def build_slug_to_nodes(self, similarity_top_k: int) -> Dict[str, List[Any]]:
        retriever = self.retriever(similarity_top_k)
        nodes = retriever.retrieve("List all the files")
        slug_to_nodes: Dict[str, List[Any]] = {}
        for n in nodes:
            slug = n.metadata.get("item_slug")
            if not slug:
                continue
            slug_to_nodes.setdefault(slug, []).append(n)
        return slug_to_nodes

    def build_context(self, topic_slugs: List[str], slug_to_nodes: Dict[str, List[Any]], total_chars: int, per_slug_k: int) -> str:
        import textwrap
        budget = total_chars
        chunks: List[str] = []
        for slug in topic_slugs or []:
            if budget <= 0:
                break
            nodes = slug_to_nodes.get(slug, [])
            taken = 0
            for n in nodes:
                if taken >= per_slug_k:
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
                    taken += 1
        return "\n\n".join(chunks)

    def build_outline_context(self, topic_slugs: List[str], slug_to_nodes: Dict[str, List[Any]]) -> str:
        total_chars = int(os.getenv("D2R_SKELETON_CONTEXT_CHARS", "4000"))
        per_slug_k = int(os.getenv("D2R_RAG_TOP_K_SKELETON", "1"))
        return self.build_context(topic_slugs, slug_to_nodes, total_chars, per_slug_k)

    def build_enhance_context(self, topic_slugs: List[str], slug_to_nodes: Dict[str, List[Any]]) -> str:
        total_chars = int(os.getenv("D2R_ENH_SUBSECTION_CHARS", "4000"))
        per_slug_k = int(os.getenv("D2R_RAG_TOP_K_ITEM", "2"))
        return self.build_context(topic_slugs, slug_to_nodes, total_chars, per_slug_k)



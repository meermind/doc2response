import os
from typing import List, Dict, Optional

from sqlalchemy import URL
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from src.models.settings import Settings
from models.config import TiDBSettings
from src.models.paths import OutputPaths
from src.latex_utils.writer import collate_assistant_message


class BaseNotesAgent:
    def __init__(
        self,
        course_name: str,
        module_name: str,
        module_slug: Optional[str],
        output_base_dir: Optional[str] = None,
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
        nodes = self.retriever(list_top_k).retrieve("List all the files")
        slugs = sorted(list({x.metadata.get("item_slug", "") for x in nodes if x.metadata.get("item_slug")}))
        return slugs

    def out_dir(self) -> str:
        return OutputPaths(base_dir=self.output_base_dir, course_name=self.course_name, module_name=self.module_name).module_dir()

    def collate(self, section_refs: List[Dict[str, str]], intro_path: str, out_path: str) -> None:
        collate_assistant_message(section_refs, intro_path, out_path)



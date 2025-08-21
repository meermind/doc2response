import argparse
import json
import os
from typing import List

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
    if os.path.exists(skeleton_path):
        with open(skeleton_path, "r") as f:
            skeleton = json.load(f)
            titles: List[str] = skeleton.get("subsections", [])
    else:
        log.info("skeleton.json not found at %s; nothing to enhance.", skeleton_path)
        return

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k_item,
        embedding_model=embedding_model,
        filters=base_metadata_filters,
    )
    agent = Agent(model=OpenAIModel(ai.writer_model))

    for i, title in enumerate(titles, start=1):
        nodes = retriever.retrieve(title)
        # Construct a concise context summary (metadata only) to avoid leaking content
        context = []
        for n in nodes:
            item = {
                "course_slug": n.metadata.get("course_slug", ""),
                "module_slug": n.metadata.get("module_slug", ""),
                "lesson_slug": n.metadata.get("lesson_slug", ""),
                "item_slug": n.metadata.get("item_slug", ""),
            }
            context.append(item)
        # Log retrieved documents (metadata only), one per line
        for item in context:
            log.info("Subsection '%s' doc: %s", title, json.dumps(item))
        prompt = f"Enhance the subsection titled '{title}' into detailed LaTeX (no code fences). Use the following context metadata to guide retrieval relevance (do not include this metadata in the output):\n{json.dumps(context)}"
        result = agent.run_sync(prompt)
        sub_path = paths.subsection_path(title)
        with open(sub_path, "w") as f:
            f.write(getattr(result, "content", None) or getattr(result, "output", None) or str(result))
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



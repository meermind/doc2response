import argparse
import logging
import os
import sys
from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from demo_transcripts_to_docs import transcripts_to_docs

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

import os
if os.getenv("ENABLE_CHILD_DEBUG") == "1":
    import debugpy
    debugpy.listen(("127.0.0.1", 5678))
    debugpy.wait_for_client()

def get_db_session(connection_url):
    """Creates a SQLAlchemy session for TiDB."""
    engine = create_engine(connection_url)
    Session = sessionmaker(bind=engine)
    return Session()

def check_module_exists(session, table_name, module_name):
    """
    Checks if a module_name already exists in the database using SQLAlchemy.

    :param session: SQLAlchemy session.
    :param table_name: The name of the vector table.
    :param module_name: The module name to check.
    :return: True if the module exists, False otherwise.
    """
    query = text(f"""
        SELECT COUNT(*) as count
        FROM {table_name}
        WHERE JSON_UNQUOTE(JSON_EXTRACT(CAST(`meta` AS JSON), '$.module_name')) = :module_name;
    """)
    result = session.execute(query, {"module_name": module_name}).scalar()
    return result > 0  # True if module exists


def delete_module_records(session, table_name, module_name):
    """
    Deletes all records from the database where the module_name matches using SQLAlchemy.

    :param session: SQLAlchemy session.
    :param table_name: The name of the vector table.
    :param module_name: The module name to delete.
    """
    query = text(f"""
        DELETE FROM {table_name}
        WHERE JSON_UNQUOTE(JSON_EXTRACT(CAST(`meta` AS JSON), '$.module_name')) = :module_name;
    """)
    session.execute(query, {"module_name": module_name})
    session.commit()
    logger.info(f"Deleted existing records for module: {module_name}")

def _resolve_path(path: str, base_dir: str) -> str:
    """Resolve path against base_dir if not absolute and not existing as-is."""
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(base_dir, path)
    return candidate


def main(metadata_file=None, topic_number=None, project_dir=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Load documents into TiDB VectorStore for LlamaIndex.")
    parser.add_argument("--metadata_file", required=False, help="Path to the metadata file (absolute or relative to --project_dir).")
    parser.add_argument("--topic_number", type=int, required=False, help="Module index (1-based) to process.")
    parser.add_argument("--project_dir", default="..", help="Base directory for resolving relative metadata/content paths.")
    parser.add_argument("--persist_dir", default="./storage", help="Directory to save index metadata.")

    # Parse args but allow running without CLI by using defaults/env
    args = parser.parse_args(args=None if sys.argv[0].endswith("demo_load_docs_to_llamaindex.py") else [])

    # Allow overriding with function arguments or env for debugging
    metadata_file = metadata_file or args.metadata_file or os.getenv("METADATA_FILE")
    topic_number = int(topic_number or args.topic_number or os.getenv("TOPIC_NUMBER", "0"))
    project_dir = project_dir or args.project_dir or os.getenv("PROJECT_DIR", "../course-crawler")

    if not metadata_file or topic_number < 1:
        logger.error("metadata_file and topic_number are required. Provide CLI args, function params, or set METADATA_FILE/TOPIC_NUMBER env.")
        sys.exit(1)

    # Resolve metadata path relative to project_dir when needed
    metadata_file = _resolve_path(metadata_file, project_dir)

    # Load settings from .env
    tidb_username = os.getenv("TIDB_USERNAME")
    tidb_password = os.getenv("TIDB_PASSWORD")
    tidb_host = os.getenv("TIDB_HOST")
    tidb_port = int(os.getenv("TIDB_PORT", 4000))  # Default port is 4000
    tidb_db_name = os.getenv("TIDB_DB_NAME")
    vector_table_name = os.getenv("VECTOR_TABLE_NAME", "demo_load_docs_to_llamaindex")  # Default table name

    if not (tidb_username and tidb_password and tidb_host and tidb_db_name):
        logger.error("Missing required TiDB configuration in .env file.")
        sys.exit(1)

    # Load documents for a specific module based on metadata content paths
    documents = transcripts_to_docs(None, metadata_file, topic_number=topic_number, project_dir=project_dir)

    # Extract module_name from documents
    if not documents or "module_name" not in documents[0].metadata:
        logger.error("Error: Could not extract module_name from documents.")
        sys.exit(1)

    module_name = documents[0].metadata["module_name"]

    # Define dimensions for embeddings (e.g., OpenAI ada embeddings: 1536)
    embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
    embedding_dimension = 1536

    # TiDB connection URL
    tidb_connection_url = URL(
        "mysql+pymysql",
        username=tidb_username,
        password=tidb_password,
        host=tidb_host,
        port=tidb_port,
        database=tidb_db_name,
        query={"ssl_verify_cert": True, "ssl_verify_identity": True},
    )

    # Initialize TiDB vector store
    vector_store = TiDBVectorStore(
        connection_string=tidb_connection_url,
        table_name=vector_table_name,
        distance_strategy="cosine",
        vector_dimension=embedding_dimension,
        drop_existing_table=False,
    )

    # Create SQLAlchemy session
    session = get_db_session(tidb_connection_url)

    # **Check if module exists before proceeding**
    if check_module_exists(session, vector_table_name, module_name):
        user_input = input(f"Module '{module_name}' already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
        if user_input not in ["y", "yes"]:
            print(f"Skipping processing for '{module_name}'.")
            session.close()
            sys.exit(0)  # Exit without processing

        # **Delete existing records before inserting new ones**
        delete_module_records(session, vector_table_name, module_name)

    session.close()  # Close SQLAlchemy session after database checks

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embedding=embedding_model,
    )

    # Persist index to disk
    storage_context.persist(persist_dir=args.persist_dir)
    logger.info("Index saved to disk at %s.", args.persist_dir)

if __name__ == "__main__":
    # Local debugging defaults; can be overridden by env or CLI
    DEBUG_METADATA_PATH = os.getenv(
        "METADATA_FILE",
        "/Users/matias.vizcaino/Documents/datagero_repos/meermind/course-crawler/crawled_metadata/gatech/simulation.json",
    )
    DEBUG_TOPIC_NUMBER = int(os.getenv("TOPIC_NUMBER", "1"))
    DEBUG_PROJECT_DIR = os.getenv("PROJECT_DIR", "..")

    main(metadata_file=DEBUG_METADATA_PATH, topic_number=DEBUG_TOPIC_NUMBER, project_dir=DEBUG_PROJECT_DIR)
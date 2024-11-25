import argparse
import logging
import os
import sys
from sqlalchemy import URL
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

def main(transcript_path=None, metadata_file=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Load documents into TiDB VectorStore for LlamaIndex.")
    parser.add_argument("--transcript_path", required=True, help="Path to the directory containing transcripts.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata file.")
    parser.add_argument("--persist_dir", default="./storage", help="Directory to save index metadata.")

    args = parser.parse_args()

    # Allow overriding with function arguments for debugging
    transcript_path = transcript_path or args.transcript_path
    metadata_file = metadata_file or args.metadata_file

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

    # Load documents
    documents = transcripts_to_docs(args.transcript_path, args.metadata_file)

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
    # # Mock entry point for debugging
    # DEBUG_TRANSCRIPT_PATH = "test_data/02@topic-1-malware-analysis"
    # DEBUG_METADATA_FILE = "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"

    # # Uncomment the line below for debugging
    # main(transcript_path=DEBUG_TRANSCRIPT_PATH, metadata_file=DEBUG_METADATA_FILE)

    # Uncomment the line below for production
    main()
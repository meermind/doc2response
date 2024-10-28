from llama_index.vector_stores.tidbvector import TiDBVectorStore
import os
import logging
import sys
from sqlalchemy import URL
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)

# from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
load_dotenv()  # This will load the variables from the .env file

from demo_transcripts_to_docs import transcripts_to_docs
TRANSCRIPT_PATH = os.environ['TRANSCRIPT_PATH']
METADATA_FILE = os.environ['METADATA_FILE']

# Database and vector table names
DB_NAME = os.environ['TIDB_DB_NAME']
VECTOR_TABLE_NAME = "demo_load_docs_to_llamaindex"

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Define dimensions for embeddings (e.g., OpenAI ada embeddings: 1536)
embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
embedding_dimension = 1536

from interfaces.database_interface import DatabaseInterface
vector_table_options = ["scibert_alldata","scibert_smalldata","mocked_data"]

# Load documents
documents = SimpleDirectoryReader("./test_data/").load_data()

tidb_connection_url = URL(
            "mysql+pymysql",
            username=os.environ['TIDB_USERNAME'],
            password=os.environ['TIDB_PASSWORD'],
            host=os.environ['TIDB_HOST'],
            port=4000,
            database=DB_NAME,
            query={"ssl_verify_cert": True, "ssl_verify_identity": True},
        )

vector_store = TiDBVectorStore(
    connection_string=tidb_connection_url,
    table_name= VECTOR_TABLE_NAME,
    distance_strategy="cosine",
    vector_dimension=1536, # OpenAI ada embeddings: 1536
    drop_existing_table=False,
)


# Create Faiss vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Example of loading documents with metadata
documents = transcripts_to_docs(TRANSCRIPT_PATH, METADATA_FILE)

# Build the index with metadata-aware documents
# Create the index using the embedding model
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    embedding=embedding_model
)

# Save the Faiss index and LlamaIndex metadata to disk
storage_context.persist(persist_dir="./storage")
logger.info("Index saved to disk.")

import argparse
from dotenv import load_dotenv
import os
import re
import json
import logging
import sys
from sqlalchemy import URL
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    get_response_synthesizer
)
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterCondition
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
EMBEDDING_MODEL = OpenAIEmbedding(model="text-embedding-ada-002")
EMBEDDING_DIMENSION = 1536
SECTION_PATTERN = r'\\section\{(.+?)\}(.*?)(?=\\subsection|\Z)'
SUBSECTION_PATTERN = r'\\subsection\{(.+?)\}(.*?)(?=\\subsection|\Z)'

def extract_latex_content(response):
    """
    Extracts LaTeX content from the response if available.
    Otherwise, returns the full response.
    """
    latex_match = re.search(r'``` ?latex(.*?)```', response.response, re.DOTALL)
    return latex_match.group(1) if latex_match else response.response

def load_prompt(file_name):
    """Helper: Load a prompt from a file."""
    with open(os.path.join("prompts", file_name), "r", encoding="utf-8") as f:
        return f.read().strip()

def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate enhanced LaTeX content.")
    parser.add_argument("--module_name", help="Name of the module to process.")
    parser.add_argument("--vector_table_name", default="demo_load_docs_to_llamaindex", help="Vector table name.")
    cli_args = parser.parse_args(args)

    # Override environment variables with CLI arguments if provided
    module_name = cli_args.module_name or os.getenv("MODULE_NAME")
    vector_table_name = cli_args.vector_table_name

    if not module_name:
        logger.error("Module name and database name must be specified via CLI or environment variables.")
        sys.exit(1)

    # Load prompts
    query = load_prompt("assistant_message.txt")
    intro_query = load_prompt("intro_query.txt")
    sub_query = load_prompt("subsection_query.txt")

    # Step 1: Load the index from storage
    tidb_connection_url = URL(
        "mysql+pymysql",
        username=os.getenv("TIDB_USERNAME"),
        password=os.getenv("TIDB_PASSWORD"),
        host=os.getenv("TIDB_HOST"),
        port=int(os.getenv("TIDB_PORT", 4000)),
        database=os.getenv("TIDB_DB_NAME"),
        query={"ssl_verify_cert": True, "ssl_verify_identity": True},
    )

    vector_store = TiDBVectorStore(
        connection_string=tidb_connection_url,
        table_name=vector_table_name,
        distance_strategy="cosine",
        vector_dimension=EMBEDDING_DIMENSION,
        drop_existing_table=False,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=EMBEDDING_MODEL,
    )

    # Define filters for the query
    filters = [{"key": "module_name", "value": module_name, "operator": "=="}]
    metadata_filters = MetadataFilters(filters=[MetadataFilter(**f) for f in filters])

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
        embedding_model=EMBEDDING_MODEL,
        filters=metadata_filters,
    )

    llm = OpenAI(model="gpt-4o")
    response_synthesizer = get_response_synthesizer(llm=llm)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    source_nodes = retriever.retrieve("List all the files")
    unique_topics = set([x.metadata['item_slug'] for x in source_nodes])
    unique_topics_str = ', '.join(unique_topics)

    # Query and process results
    response = query_engine.query(query + f"\n\n =============== \n Your answer should contain at least one subsection per topic as per the retrieved documents: {unique_topics_str}.")
    print(response)
    unique_topics = set([x['item_slug'] for x in response.metadata.values()])

    # Extract course and module information
    course_names = [x['course_name'] for x in response.metadata.values()]
    module_names = [x['module_name'] for x in response.metadata.values()]
    assert len(set(course_names)) == 1, "Multiple course names found."
    assert len(set(module_names)) == 1, "Multiple module names found."

    course_name = course_names[0]
    module_name = module_names[0]

    # Create output directories
    out_path = f"assistant_latex/{course_name}/{module_name}/assistant_message.tex"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save assistant message
    with open(out_path, "w") as file:
        file.write(response.response)

    tex_content = extract_latex_content(response)
    section_match = re.search(SECTION_PATTERN, tex_content, re.DOTALL)
    subsections = re.findall(SUBSECTION_PATTERN, tex_content, re.DOTALL)

    # Initialize metadata
    metadata = {"sections": []}

    # Process the main section first
    if section_match:
        section_title, section_content = section_match.groups()
        print(f"Processing section: {section_title}")

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=20,
            embedding_model=EMBEDDING_MODEL,
            filters=metadata_filters,
        )

        llm = OpenAI(model="gpt-4o-mini")
        response_synthesizer = get_response_synthesizer(llm=llm)

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        response = query_engine.query(intro_query + "The title and content to summarize are as follows:\n```latex\n\section{ " + section_title + "}\n" + section_content)

        # Step 5: Print and save the response
        print(response)

        tex_content_intro = extract_latex_content(response)

        # Create the directory if it doesn't exist
        section_out_path = f"assistant_latex/{course_name}/{module_name}/{section_title}.tex"
        os.makedirs(os.path.dirname(section_out_path), exist_ok=True)

        # Save the assistant message to a file
        with open(section_out_path, "w") as file:
            file.write(tex_content_intro)

        # Add section info to metadata
        metadata["sections"].append({
            "order": 0,
            "type": "section",
            "title": section_title,
            "path": section_out_path
        })

    # Process each subsection and save it to its respective file
    for i, (title, content) in enumerate(subsections, start=1):
        print(f"Processing subsection: {title}")
        subsection_latex_formatted = f"\subsection{{{title}}}\n{content}"

        # Step 3: Set up the query engine with the filters

        retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=10,
                    embedding_model=EMBEDDING_MODEL,
                    filters=metadata_filters,
                )

        source_nodes = retriever.retrieve(title + ': ' + content)
        item_names = set([x.metadata['item_name'] for x in source_nodes])
        print(f"Top Document Results for {title}: {item_names}")
        item_ids = [x.node_id for x in source_nodes]
        filters = [
            {
                "key": "item_name",
                "value": item_id,
                "operator": "==",
            } for item_id in item_names
        ]

        llm = OpenAI(model="gpt-4o-mini")
        metadata_filters = MetadataFilters(filters=[MetadataFilter(**f) for f in filters], condition=FilterCondition.OR)

        query_engine = index.as_query_engine(filters=metadata_filters, llm=llm)
        chat_init = '\nNow for the following LaTeX ======================= \n ```latex\n'

        response = query_engine.query(sub_query + chat_init + subsection_latex_formatted)
        tex_content_subsection = extract_latex_content(response)

        # Define the output path for each subsection
        subsection_out_path = f"assistant_latex/{course_name}/{module_name}/{title}.tex"

        # Save the enhanced content to the corresponding .tex file
        with open(subsection_out_path, 'w') as file:
            file.write(f"{tex_content_subsection}")

        # Add subsection info to metadata
        metadata["sections"].append({
            "order": i,
            "type": "subsection",
            "title": title,
            "path": subsection_out_path
        })

    # Save metadata to a JSON file
    metadata_path = f"assistant_latex/{course_name}/{module_name}/metadata.json"
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print(f"Processing complete. Enhanced content and metadata saved to 'assistant_latex/{course_name}/{module_name}'.")

if __name__ == "__main__":
    # # # Hardcoded module_name for debugging
    # DEBUG_MODULE_NAME = "Topic 1 Introduction to computer security and malware"

    # # # Pass the hardcoded module_name to main
    # main(args=["--module_name", DEBUG_MODULE_NAME])

    main()


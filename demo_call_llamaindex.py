"""
Let's refactor it for the main components:
1 -> Create Latex Assisstant Message (high-level skelethon with basic descriptions)
2 -> Create Enhanced Intro Summary
3 -> Create Enhanced Subsections
"""

from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, get_response_synthesizer
import os
import logging
import sys
from sqlalchemy import URL
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex
)
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterCondition
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

import re
import json

from dotenv import load_dotenv
load_dotenv()  # This will load the variables from the .env file

# Database and vector table names
DB_NAME = os.environ['TIDB_DB_NAME']
VECTOR_TABLE_NAME = "demo_load_docs_to_llamaindex"
MODULE_NAME = os.environ['MODULE_NAME']


# Define dimensions for embeddings (e.g., OpenAI ada embeddings: 1536)
embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
embedding_dimension = 1536

# Use regex to extract the main section and subsections
section_pattern = r'\\section\{(.+?)\}(.*?)(?=\\subsection|\Z)'  # Matches the main section
subsection_pattern = r'\\subsection\{(.+?)\}(.*?)(?=\\subsection|\Z)'  # Matches each subsection

def extract_latex_content(response):
    """
    Extracts LaTeX content from the response if available.
    Otherwise, returns the full response.
    """
    latex_match = re.search(r'``` ?latex(.*?)```', response.response, re.DOTALL)
    
    if latex_match:
        tex_content = latex_match.group(1)
        print(tex_content)
    else:
        tex_content = response.response
        print("No LaTeX portion found. Saving the entire message.")
    
    return tex_content

# Helper: Load prompts from files
def load_prompt(file_name):
    with open(os.path.join("prompts", file_name), "r", encoding="utf-8") as f:
        return f.read().strip()


# Load Prompts
query = load_prompt("assistant_message.txt")
intro_query = load_prompt("intro_query.txt")
sub_query = load_prompt("subsection_query.txt")

# Step 1: Load the index from storage
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

index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embedding_model
        )

# Step 2: Define the filters for the query
filters = [
    {
        "key": "module_name",
        "value": MODULE_NAME,
        "operator": "==",
    },
    # Add more filters as needed
]
metadata_filters = MetadataFilters(filters=[MetadataFilter(**f) for f in filters])

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=20,
    embedding_model=embedding_model,
    filters=metadata_filters,
)

llm = OpenAI(model="gpt-4o-mini")
response_synthesizer = get_response_synthesizer(llm=llm)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)


response = query_engine.query(query)

# Step 5: Print and save the response
print(response)

course_names = [x['course_name'] for x in response.metadata.values()]
module_names = [x['module_name'] for x in response.metadata.values()]

assert len(set(course_names)) == 1, "Multiple course names found."
assert len(set(module_names)) == 1, "Multiple module names found."

course_name = course_names[0]
module_name = module_names[0]

# Create the directory if it doesn't exist
out_path = f"assistant_latex/{course_name}/{module_name}/assistant_message.tex"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# Save the assistant message to a file
with open(out_path, "w") as file:
    file.write(response.response)

tex_content = extract_latex_content(response)
# Extract the main section and subsections
section_match = re.search(section_pattern, tex_content, re.DOTALL)
subsections = re.findall(subsection_pattern, tex_content, re.DOTALL)

# Initialize metadata to store the order and file paths
metadata = {"sections": []}

# Process the main section first
if section_match:
    section_title, section_content = section_match.groups()
    print(f"Processing section: {section_title}")

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
        embedding_model=embedding_model,
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
for i, (title, content) in enumerate(subsections[:3], start=1):
    print(f"Processing subsection: {title}")
    subsection_latex_formatted = f"\subsection{{{title}}}\n{content}"

    # Step 3: Set up the query engine with the filters

    retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=5,
                embedding_model=embedding_model,
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

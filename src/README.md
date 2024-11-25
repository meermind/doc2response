
## Overview

This repository contains scripts to process and manage course-related data using LlamaIndex and TiDB Vector Store. The pipeline consists of three main stages:
1. **Document Embedding**: Convert transcripts and metadata into vector embeddings stored in a vector database.
2. **Query Processing**: Generate enhanced LaTeX content based on specific module data.
3. **LaTeX Document Generation**: Compile all content into a structured LaTeX document.

These scripts are designed to work together in a pipeline or independently as needed.

---

## Scripts

### 1. `demo_load_docs_to_llamaindex.py`

#### Description
This script processes transcript files and metadata to generate document embeddings and store them in TiDB Vector Store.

#### Usage
```bash
python demo_load_docs_to_llamaindex.py --transcript_path <path_to_transcripts> --metadata_file <path_to_metadata>
```

#### Arguments
- `--transcript_path`: Path to the directory containing transcript files.
- `--metadata_file`: Path to the JSON metadata file.
- `--persist_dir` (optional): Directory to save the index metadata. Default: `./storage`.

#### Environment Variables
- `TIDB_USERNAME`: TiDB username.
- `TIDB_PASSWORD`: TiDB password.
- `TIDB_HOST`: TiDB host.
- `TIDB_PORT`: TiDB port. Default: `4000`.
- `TIDB_DB_NAME`: TiDB database name.
- `VECTOR_TABLE_NAME` (optional): Name of the vector table. Default: `demo_load_docs_to_llamaindex`.

#### Example
```bash
python demo_load_docs_to_llamaindex.py --transcript_path "test_data/02@topic-1-malware-analysis" --metadata_file "../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json"
```

---

### 2. `demo_call_llamaindex.py`

#### Description
This script queries LlamaIndex for a specific module, extracts structured content, and saves enhanced LaTeX documents.

#### Usage
```bash
python demo_call_llamaindex.py --module_name <module_name>
```

#### Arguments
- `--module_name`: Name of the module to process.
- `--vector_table_name` (optional): Vector table name. Default: `demo_load_docs_to_llamaindex`.

#### Environment Variables
- `TIDB_USERNAME`: TiDB username.
- `TIDB_PASSWORD`: TiDB password.
- `TIDB_HOST`: TiDB host.
- `TIDB_PORT`: TiDB port. Default: `4000`.
- `TIDB_DB_NAME`: TiDB database name.

#### Example
```bash
python demo_call_llamaindex.py --module_name "Topic 1 Malware analysis"
```

---

### 3. `generate_latex_doc.py`

#### Description
This script combines enhanced LaTeX sections and subsections into a single LaTeX document.

#### Usage
```bash
python generate_latex_doc.py --module <module_name>
```

#### Arguments
- `--module`: Name of the module to process.

#### Environment Variables
- `COURSE`: Name of the course.
- `MODULE_NAME`: Name of the module.

#### Example
```bash
python generate_latex_doc.py --module "Topic 1 Malware analysis"
```

---

### 4. `orchestrator.py`

#### Description
Central orchestrator to run the entire pipeline: 
`demo_load_docs_to_llamaindex.py > demo_call_llamaindex.py > generate_latex_doc.py`.

#### Usage
```bash
python orchestrator.py
```

#### Configuration
The orchestrator uses the `.env` file for configuration. Example `.env`:
```env
COURSE=CourseName
MODULE_NAME=Topic 1 Malware analysis
TRANSCRIPT_PATH=test_data/02@topic-1-malware-analysis
METADATA_FILE=../course-crawler/crawled_metadata/dl_coursera/uol-cm2025-computer-security.json
```

#### Pipeline Flags
To run specific stages, modify the `orchestrate_pipeline` function:
```python
orchestrate_pipeline(run_load=True, run_call=True, run_generate=True)
```

---

## Setup

1. **Install Dependencies**:
   Ensure you have Python and the required libraries installed. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure `.env` File**:
   Create a `.env` file in the root directory with the necessary environment variables.

---

## Example Workflow

1. Load documents into the vector database:
   ```bash
   python demo_load_docs_to_llamaindex.py --transcript_path "path/to/transcripts" --metadata_file "path/to/metadata.json"
   ```

2. Query and process module data:
   ```bash
   python demo_call_llamaindex.py --module_name "Module Name"
   ```

3. Generate LaTeX document:
   ```bash
   python generate_latex_doc.py --module "Module Name"
   ```

4. Orchestrate the entire pipeline:
   ```bash
   python orchestrator.py
   ```

---

## Troubleshooting

1. **Environment Variables Missing**:
   Ensure all required environment variables are set in `.env`.

2. **Pipeline Step Fails**:
   - Check if all dependencies are installed.
   - Ensure the paths specified for transcripts and metadata are correct.

3. **LaTeX Compilation Errors**:
   Verify that the generated LaTeX files follow the correct structure and format.

---

## Contributing

Feel free to submit issues or pull requests to improve the scripts and pipeline functionality.

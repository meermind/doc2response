## doc2response – Orchestrator Quick Start

### 1) Install
```bash
cd doc2response
pip install -e .
# or: uv sync
```

### 2) Configure env (.env or export)
```env
OPENAI_API_KEY=...
TIDB_USERNAME=...
TIDB_PASSWORD=...
TIDB_HOST=...
TIDB_PORT=4000
TIDB_DB_NAME=...

# Optional
VECTOR_TABLE_NAME=demo_load_docs_to_llamaindex
```

Minimum runtime env:
```bash
export METADATA_FILE=/abs/path/to/metadata.json
export TOPIC_NUMBER=1   # 1-based module index

# Optional: where to write/read artifacts (recommended)
export D2R_OUTPUT_BASE=./outputs
export INPUT_BASE_DIR=./outputs
```

### 3) Run the whole pipeline
```bash
python src/orchestrator.py --overwrite --skip_load
```

That will:
- Ingest the selected module into the vector store
- Generate LaTeX section files with the AI writer
- Merge them into a single `.tex` at `$D2R_OUTPUT_BASE/<course>/Lecture Notes/<module>/<module_name>/<module_name>.tex`

Notes
- To change which steps run, edit the call in `src/orchestrator.py` (run_load/run_call/run_generate) or call `orchestrate_pipeline(...)` programmatically.
- Prompts are read from `prompts/assistant_message.txt`, `prompts/intro_query.txt`, `prompts/subsection_query.txt`.

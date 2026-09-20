# Healthcare Agent

Multi-turn Agentic RAG chatbot over an healthcare strcture and unstrctured data, deployed as a **Databricks App** (Streamlit), built with **LangGraph**.


## Setup

1. Create the memory tables:
   ```
   databricks sql execute --file src/memory/schema.sql
   ```
2. Populate the secret scope (see `bundle/resources/secrets.yml`):
   ```
   databricks secrets create-scope dbx-secret-scope
   databricks secrets put-secret dbx-secret-scope DATABRICKS_TOKEN
   databricks secrets put-secret dbx-secret-scope LANGFUSE_PUBLIC_KEY
   databricks secrets put-secret dbx-secret-scope LANGFUSE_SECRET_KEY
   databricks secrets put-secret dbx-secret-scope LANGFUSE_HOST
   ```
3. Edit `config/config.yaml` - set `databricks.host`, `databricks.warehouse_id`,
   and your `data.catalog` / `data.schema` / `data.table`.

## Local run

```
cp .env.example .env   # fill in real values
pip install -r requirements.txt
streamlit run src/app/app.py
```

## Deploy (Databricks Asset Bundles)

```
cd bundle
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run ecom_sql_agent_app -t dev
```
CI (`.github/workflows/ci.yml`) lints, unit-tests and validates the bundle on
every PR. Deploy (`.github/workflows/deploy.yml`) deploys + runs the app on
push to `main`.

## Testing

```
pytest tests/unit                 # fast, no external deps
RUN_INTEGRATION=1 pytest tests/integration   # needs real warehouse + endpoint
python evals/run_evals.py         # golden-question eval suite, logged to MLflow
```

## Extending

- Add columns: update `data.columns` in `config/config.yaml` and the semantic
  notes in `src/agent/nodes/schema_retriever.py:COLUMN_NOTES`.
- Add a guardrail: add a check function + call in `src/agent/guardrails.py`.
- Add a new node: implement in `src/agent/nodes/`, wire it into
  `src/agent/graph.py`.

## Clinical agentic workflow

The clinical assistant uses an explicit plan -> execute -> review loop over
`ai_projects_catalog.agent_demo.healthcare_patient_tbl` and the clinical
guidelines PDF:

```
guardrail
   -> supervisor (creates a plan: sql, rag, or both)
   -> tool_executor
        sql: schema_retriever -> sql_generator -> sql_executor
        rag: hybrid AI Search -> reranker
   -> response_formatter (combines all selected evidence)
   -> reviewer
        pass         -> END
        retry_tools -> re-run selected tool(s)
        update_plan -> supervisor -> execute again

Maximum: 3 reviewer attempts per user turn.
```

The supervisor plans before any data retrieval. The reviewer receives the
question, current plan, tool evidence and draft answer, and controls whether
the system should accept the answer, refire a tool, or revise the plan.

Before first use, run `scripts/ingest_clinical_guidelines.py` as a Databricks
notebook/job to chunk `config.rag.pdf_volume_path`, write it to
`config.rag.chunks_table`, and create/sync the Delta Sync AI Search index at
`config.rag.index_name` on `config.rag.ai_search_endpoint`.


## Short-term memory and follow-up questions

Each chat turn is persisted in `clinical_chat_short_term_memory`. On every user turn, the Streamlit app reloads the recent persisted turns and passes them into the agent state. The current question is excluded because it is already provided separately.

The Supervisor receives this history before creating its plan, so references such as "those patients", "their kidney function", or "what about the guideline for them?" can be resolved against previous turns. Previous assistant SQL is also carried in memory and supplied as context when available. The SQL generator, answer generator, and reviewer receive the same conversational context.

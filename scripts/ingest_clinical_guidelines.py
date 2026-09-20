"""
One-time / re-runnable ingestion job for the clinical guidelines RAG index.
Run this as a Databricks notebook or job (needs `spark` and `dbutils` in
scope, same as the reference `end_to_end_rag_databricks.ipynb`). It:

  1. Loads the clinical guidelines PDF from the configured Unity Catalog volume.
  2. Recursively chunks it.
  3. Writes the chunks to a Delta table.
  4. Creates (or reuses) the Databricks AI Search endpoint + Delta Sync index
     so the rag_agent branch can query it in HYBRID (ANN + BM25 + RRF) mode.

All settings come from config/config.yaml -> rag, so this script and the
`rag_tools.hybrid_search_with_rerank` tool always point at the same index.
"""

import os
import sys
import time

try:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    _project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, _project_root)

from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from databricks.ai_search.client import AISearchClient

from src.utils.config_loader import settings

RAG = settings.rag


def create_chunks_table():
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {RAG.chunks_full_table_name} (
        id STRING,
        text STRING,
        source STRING,
        page INT,
        chunk_id INT
    )
    USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)


def load_and_chunk_pdf() -> list[Document]:
    if not os.path.exists(RAG.pdf_volume_path):
        raise FileNotFoundError(
            f"PDF not found at {RAG.pdf_volume_path}. Upload it to the Unity "
            "Catalog volume first."
        )

    pages = PyMuPDFLoader(RAG.pdf_volume_path).load()
    print(f"Loaded {len(pages)} PDF pages from {RAG.pdf_volume_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG.chunk_size,
        chunk_overlap=RAG.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    normalized = []
    for i, doc in enumerate(chunks):
        page_number = int(doc.metadata.get("page", 0)) + 1
        normalized.append(
            Document(
                page_content=doc.page_content.strip(),
                metadata={
                    "source": os.path.basename(RAG.pdf_volume_path),
                    "page": page_number,
                    "chunk_id": i,
                },
            )
        )
    chunks = [d for d in normalized if d.page_content]
    print(f"Created {len(chunks)} chunks.")
    return chunks


def write_chunks_to_delta(chunks: list[Document]):
    rows = [
        (
            f"{d.metadata['source']}::{d.metadata['chunk_id']}",
            d.page_content,
            d.metadata["source"],
            int(d.metadata["page"]),
            int(d.metadata["chunk_id"]),
        )
        for d in chunks
    ]
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("text", StringType(), False),
        StructField("source", StringType(), True),
        StructField("page", IntegerType(), True),
        StructField("chunk_id", IntegerType(), True),
    ])
    df = spark.createDataFrame(rows, schema=schema)
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(RAG.chunks_full_table_name)
    )
    print(f"Saved {df.count()} chunks to {RAG.chunks_full_table_name}")


def create_or_reuse_endpoint(ai_client: AISearchClient):
    try:
        ai_client.get_endpoint(name=RAG.ai_search_endpoint)
        print(f"Reusing existing AI Search endpoint: {RAG.ai_search_endpoint}")
    except Exception:
        print(f"Creating AI Search endpoint: {RAG.ai_search_endpoint}")
        ai_client.create_endpoint(name=RAG.ai_search_endpoint, endpoint_type="STANDARD")


def create_or_reuse_index(ai_client: AISearchClient):
    try:
        index = ai_client.get_index(endpoint_name=RAG.ai_search_endpoint, index_name=RAG.full_index_name)
        print(f"Reusing existing index: {RAG.full_index_name}. Triggering sync...")
        index.sync()
        return index
    except Exception:
        print(f"Creating Delta Sync index: {RAG.full_index_name}")
        index = ai_client.create_delta_sync_index(
            endpoint_name=RAG.ai_search_endpoint,
            source_table_name=RAG.chunks_full_table_name,
            index_name=RAG.full_index_name,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_source_column="text",
            embedding_model_endpoint_name=RAG.embedding_model,
        )

    max_wait_seconds = 900
    start_time = time.time()
    while True:
        description = index.describe()
        status = description.get("status", {})
        state = status.get("detailed_state", "")
        pipeline_state = status.get("delta_sync_index_spec", {}).get("pipeline_status", {}).get("state", "UNKNOWN")
        print(f"Index state: {state}, pipeline state: {pipeline_state}")

        if state.startswith("ONLINE") and pipeline_state in ["COMPLETED", "FAILED", "CANCELED"]:
            break
        if "FAILED" in state or "ERROR" in state:
            raise RuntimeError(f"Index failed: {description}")
        if time.time() - start_time > max_wait_seconds:
            raise TimeoutError(f"Index did not come ONLINE within {max_wait_seconds}s.")
        time.sleep(10)

    index.sync()
    return index


def main():
    create_chunks_table()
    chunks = load_and_chunk_pdf()
    write_chunks_to_delta(chunks)

    ai_client = AISearchClient()
    create_or_reuse_endpoint(ai_client)
    create_or_reuse_index(ai_client)
    print("Ingestion complete. The rag_agent branch can now query the index.")


if __name__ == "__main__":
    main()

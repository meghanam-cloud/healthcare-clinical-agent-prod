"""
Loads config.yaml once and resolves all secret references through
databricks_secrets.get_secret(). Everything else in the app should
import `settings` from here instead of touching os.environ or yaml directly.
"""

import os
import yaml
from dataclasses import dataclass, field
from functools import lru_cache

from src.utils.databricks_secrets import get_secret

DEFAULT_CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")


@dataclass
class DataConfig:
    catalog: str
    schema: str
    table: str
    columns: dict

    @property
    def full_table_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.table}"


@dataclass
class RagConfig:
    catalog: str
    schema: str
    chunks_table: str
    index_name: str
    ai_search_endpoint: str
    embedding_model: str
    rerank_model: str
    pdf_volume_path: str
    chunk_size: int
    chunk_overlap: int
    ann_k: int
    final_context_k: int

    @property
    def chunks_full_table_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.chunks_table}"

    @property
    def full_index_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.index_name}"


@dataclass
class MemoryConfig:
    catalog: str
    schema: str
    short_term_table: str
    long_term_table: str
    short_term_turns_limit: int

    @property
    def short_term_full_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.short_term_table}"

    @property
    def long_term_full_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.long_term_table}"


@dataclass
class LLMConfig:
    endpoint_name: str
    temperature: float
    max_tokens: int


@dataclass
class GuardrailsConfig:
    pii_check: bool
    prompt_injection_check: bool
    topical_scope_check: bool
    allowed_topics_keywords: list = field(default_factory=list)


@dataclass
class Secrets:
    databricks_token: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str


@dataclass
class Settings:
    app_name: str
    environment: str
    databricks_host: str
    warehouse_id: str
    data: DataConfig
    rag: RagConfig
    memory: MemoryConfig
    llm: LLMConfig
    guardrails: GuardrailsConfig
    mlflow_experiment_path: str
    mlflow_prompt_registry_name: str
    langfuse_enabled: bool
    secrets: Secrets
    raw: dict


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_secrets(raw: dict) -> Secrets:
    scope = raw["secrets"]["scope"]
    keys = raw["secrets"]["keys"]
    return Secrets(
        databricks_token=get_secret(scope, keys["databricks_token"], "DATABRICKS_TOKEN"),
        langfuse_public_key=get_secret(scope, keys["langfuse_public_key"], "LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=get_secret(scope, keys["langfuse_secret_key"], "LANGFUSE_SECRET_KEY"),
        langfuse_host=get_secret(scope, keys["langfuse_host"], "LANGFUSE_HOST"),
    )


@lru_cache(maxsize=1)
def load_settings(config_path: str = DEFAULT_CONFIG_PATH) -> Settings:
    raw = _load_yaml(config_path)

    return Settings(
        app_name=raw["app"]["name"],
        environment=raw["app"]["environment"],
        databricks_host=os.environ.get("DATABRICKS_HOST", raw["databricks"]["host"]),
        warehouse_id=os.environ.get("DATABRICKS_WAREHOUSE_ID", raw["databricks"]["warehouse_id"]),
        data=DataConfig(**raw["data"]),
        rag=RagConfig(**raw["rag"]),
        memory=MemoryConfig(**raw["memory"]),
        llm=LLMConfig(**raw["llm"]),
        guardrails=GuardrailsConfig(**raw["guardrails"]),
        mlflow_experiment_path=raw["mlflow"]["experiment_path"],
        mlflow_prompt_registry_name=raw["mlflow"]["prompt_registry_name"],
        langfuse_enabled=raw["langfuse"]["enabled"],
        secrets=_resolve_secrets(raw),
        raw=raw,
    )


# Module-level singleton for convenient `from src.utils.config_loader import settings`
settings = load_settings()

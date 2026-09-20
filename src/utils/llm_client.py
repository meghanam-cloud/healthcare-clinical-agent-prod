"""
Single shared LangChain chat model pointed at the Databricks Model Serving
endpoint configured in config.yaml -> llm. Used by both sql_generator and
response_formatter so temperature/endpoint changes happen in one place.
"""

from functools import lru_cache
from databricks_langchain import ChatDatabricks

from src.utils.config_loader import settings


@lru_cache(maxsize=1)
def get_llm():
    return ChatDatabricks(
        endpoint=settings.llm.endpoint_name,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )

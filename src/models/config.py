from pydantic_settings import BaseSettings
from pydantic import Field


class TiDBSettings(BaseSettings):
    """Configuration for TiDB connection and vector table settings.

    Values are read from environment variables using the provided aliases.
    """

    username: str = Field(alias="TIDB_USERNAME")
    password: str = Field(alias="TIDB_PASSWORD")
    host: str = Field(alias="TIDB_HOST")
    port: int = Field(default=4000, alias="TIDB_PORT")
    db_name: str = Field(alias="TIDB_DB_NAME")

    # Application-level defaults
    vector_table_name: str = Field(default="demo_load_docs_to_llamaindex")
    embedding_model_name: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)

    # Prompt files
    prompt_assistant_message_file: str = Field(default="assistant_message.txt")
    prompt_intro_query_file: str = Field(default="intro_query.txt")
    prompt_subsection_query_file: str = Field(default="subsection_query.txt")

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


class OpenAISettings(BaseSettings):
    """Models used by the agent for reasoning and generation."""

    # Reasoning model for main agent output
    writer_model: str = Field(default="gpt-5-nano")
    # More cost-effective model for follow-up retrieval queries
    summarizer_model: str = Field(default="gpt-5-nano")

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }



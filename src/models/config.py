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

    # Application-level defaults (DB-side only)
    vector_table_name: str = Field(default="demo_load_docs_to_llamaindex")

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


    # Note: prompts and model settings are centrally managed in src/models/settings.py



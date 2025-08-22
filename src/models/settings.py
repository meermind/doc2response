from pydantic_settings import BaseSettings
from pydantic import Field
from pydantic_ai.models import KnownModelName


class Settings(BaseSettings):
    """Central application settings (env-driven).

    - AI_WRITER_MODEL: any pydantic-ai KnownModelName (e.g., "openai:gpt-4o", "google-gla:gemini-2.5-pro").
    - EMBEDDING_MODEL: embedding model name (optionally provider-prefixed) for embeddings, e.g., "openai:text-embedding-3-small".
    - EMBEDDING_DIMENSION: integer dimension of the embedding vector.
    - PROMPT_ASSISTANT_MESSAGE_FILE: prompt file name under prompts/
    - PROMPT_INTRO_QUERY_FILE: prompt file name under prompts/
    - PROMPT_SUBSECTION_QUERY_FILE: prompt file name under prompts/
    """

    # anthropic:claude-sonnet-4-20250514
    # openai:gpt-4o"
    # openai:gpt-5-mini
    # openai:gpt-5-nano
    # openai:gpt-5
    ai_writer_model: KnownModelName = Field(default="openai:gpt-5-nano", alias="AI_WRITER_MODEL")

    # Optional per-stage overrides
    ai_skeleton_model: KnownModelName | None = Field(default="openai:gpt-5-mini", alias="AI_SKELETON_MODEL")
    ai_enhancer_model: KnownModelName | None = Field(default="openai:gpt-5-nano", alias="AI_ENHANCER_MODEL")
    ai_mdframe_model: KnownModelName | None = Field(default="openai:gpt-5-nano", alias="AI_MDFRAME_MODEL")

    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1536, alias="EMBEDDING_DIMENSION")

    prompt_assistant_message_file: str = Field(default="assistant_message.txt", alias="PROMPT_ASSISTANT_MESSAGE_FILE")
    prompt_intro_query_file: str = Field(default="intro_query.txt", alias="PROMPT_INTRO_QUERY_FILE")
    prompt_subsection_query_file: str = Field(default="subsection_query.txt", alias="PROMPT_SUBSECTION_QUERY_FILE")

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    def _parse_embedding(self) -> tuple[str, str]:
        """Return (provider, name) from embedding_model setting.

        Accepts values with optional provider prefix, e.g., "openai:text-embedding-3-small".
        Defaults provider to "openai" if omitted.
        """
        model = (self.embedding_model or "").strip()
        if ":" in model:
            provider, name = model.split(":", 1)
            return provider.strip().lower(), name.strip()
        return "openai", model

    def create_embedding(self):
        """Create and return a LlamaIndex embedding instance based on settings.

        Currently supports provider: "openai".
        """
        provider, name = self._parse_embedding()
        if provider == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding
            return OpenAIEmbedding(model=name)
        raise ValueError(f"Unsupported embedding provider: {provider}")



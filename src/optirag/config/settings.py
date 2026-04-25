from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPTIRAG_",
        env_file=(".env",),
        extra="ignore",
    )

    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPTIRAG_GEMINI_API_KEY"),
    )
    pinecone_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("PINECONE_API_KEY", "OPTIRAG_PINECONE_API_KEY"),
    )
    pinecone_index_host: str = Field(
        default="",
        validation_alias=AliasChoices("PINECONE_INDEX_HOST", "OPTIRAG_PINECONE_INDEX_HOST"),
    )
    pinecone_auto_create: bool = Field(
        default=False,
        description="If true, create serverless indexes via API for each (dim, metric) when missing from registry.",
    )
    pinecone_cloud: str = Field(
        default="aws",
        validation_alias=AliasChoices("OPTIRAG_PINECONE_CLOUD", "PINECONE_CLOUD"),
    )
    pinecone_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices("OPTIRAG_PINECONE_REGION", "PINECONE_REGION"),
    )
    pinecone_index_prefix: str = Field(
        default="optirag",
        validation_alias=AliasChoices("OPTIRAG_PINECONE_INDEX_PREFIX", "PINECONE_INDEX_PREFIX"),
    )
    pinecone_index_registry_path: Path | None = Field(
        default=None,
        description="JSON map 'dim:metric' -> host; default artifacts/pinecone_index_registry.json",
    )
    index_force_fresh: bool = Field(
        default=False,
        description="If true, clear namespace and rebuild index data from scratch.",
    )
    index_upsert_max_retries: int = Field(
        default=3,
        description="Maximum retries for transient Pinecone upsert failures per batch.",
    )
    index_upsert_backoff_base_seconds: float = Field(
        default=0.5,
        description="Base seconds for exponential backoff on upsert retry.",
    )
    data_dir: Path = Field(default=Path("data"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    log_level: str = "INFO"
    # Generation model for answers + RAGAS (plan: google_genai:gemma-4-31b-it)
    chat_model: str = Field(
        default="google_genai:gemma-4-31b-it",
        description="LangChain model id for chat",
    )
    # Alternative plain model name for Google Genai SDK
    genai_model_name: str = Field(
        default="gemma-3-4b-it",  # fallback if 31b not available; user overrides
        description="Google Genai native model name for generate_content",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def pinecone_registry_path(s: Settings | None = None) -> Path:
    st = s or get_settings()
    return st.pinecone_index_registry_path or (st.artifacts_dir / "pinecone_index_registry.json")

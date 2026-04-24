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

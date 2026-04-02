from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    app_name: str = "E-Commerce Support Multi-Agent System"
    app_env: str = "development"
    google_cloud_project: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT_ID"),
    )
    google_cloud_location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices("GOOGLE_CLOUD_LOCATION", "GCP_LOCATION"),
    )
    gemini_model: str = "gemini-1.5-flash"
    embedding_model: str = "text-embedding-004"
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "ecommerce-support"
    pinecone_namespace: str = "default"
    max_chunk_size: int = 700
    chunk_overlap: int = 120
    retrieval_top_k: int = 5
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"
    policies_dir: Path = BASE_DIR / "data" / "mock_policies"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    print("Pinecone index name=====>>>",settings.pinecone_index_name)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    settings.policies_dir.mkdir(parents=True, exist_ok=True)
    return settings

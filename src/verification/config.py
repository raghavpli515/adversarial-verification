"""Central configuration, loaded from environment / .env.

Every tunable that affects agent behavior or eval results lives here so a
config change is a one-line diff, not a hunt through the codebase — this
matters for the eval harness, which logs these values as MLflow run params.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    chroma_persist_dir: str = ".chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 12

    max_revision_cycles: int = 1

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "adversarial-verification"


settings = Settings()

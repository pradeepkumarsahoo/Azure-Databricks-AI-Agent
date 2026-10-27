# Author: Pradeep Sahoo
"""Environment-driven configuration. Picks local vs. Databricks backends via PP_ENV."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"

# A diversified set of large, well-known filers across sectors — enough real filings
# (one 10-K each) to build and test retrieval at meaningful scale without pulling the
# whole EDGAR corpus or adding new metadata dimensions (fiscal years, filing types).
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "JPM", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JNJ", "V",
    "WMT", "CVX", "PG", "HD", "BAC", "KO", "PEP", "CSCO", "INTC", "DIS",
]

# Upgraded from all-MiniLM-L6-v2 (384-dim) to bge-base-en-v1.5 (768-dim).
# Re-ingest is required when changing this — existing Chroma stores are dimension-locked.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

# Unity Catalog / Vector Search resource names — must match whatever's actually
# provisioned in the workspace (see README "Next steps").
UC_CATALOG = "topsecretcatalog"
UC_SCHEMA = "filings"
UC_CHUNKS_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.chunks"
VECTOR_SEARCH_ENDPOINT = "policypilot-vs-endpoint"
VECTOR_SEARCH_INDEX = f"{UC_CATALOG}.{UC_SCHEMA}.chunks_index"

# Unity Gateway model service routing to Groq (resources/ai_gateway.yml) — the deployed LLM
# path, with guardrails/rate limits Groq itself doesn't provide. A UC object, so it's
# addressed by its three-part name.
AI_GATEWAY_MODEL_SERVICE = f"{UC_CATALOG}.{UC_SCHEMA}.gpt-oss"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    pp_env: str = "local"
    groq_api_key: str | None = None
    sec_edgar_user_agent: str = "PolicyPilot research prototype (no contact set)"
    databricks_host: str | None = None
    databricks_token: str | None = None

    # RAG quality knobs — tune via env vars without code changes
    top_k_retrieval: int = Field(default=20, gt=0)
    top_k_rerank: int = Field(default=5, gt=0)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    multi_query: bool = True

    @property
    def is_local(self) -> bool:
        return self.pp_env == "local"

    @property
    def env(self) -> str:
        return self.pp_env

    @property
    def sec_user_agent(self) -> str:
        return self.sec_edgar_user_agent


def get_settings() -> Settings:
    return Settings()

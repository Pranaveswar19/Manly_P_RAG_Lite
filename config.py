import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    default_persist_dir: str = os.getenv("RAG_PERSIST_DIR", "data/index")
    default_llm_model: str = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
    default_k: int = int(os.getenv("RAG_TOP_K", "12"))
    default_top_n: int = int(os.getenv("RAG_TOP_N", "6"))
    enable_rerank: bool = os.getenv("RAG_RERANK", "true").lower() == "true"

CONFIG = AppConfig()

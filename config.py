import os
import logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration with validation."""
    
    openai_api_key: Optional[str] = None
    default_persist_dir: str = "data/index"
    default_llm_model: str = "gpt-4o-mini"
    default_k: int = 12
    default_top_n: int = 6
    enable_rerank: bool = True
    
    def __post_init__(self):
        """Validate and load configuration from environment."""
        # Load from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.default_persist_dir = os.getenv("RAG_PERSIST_DIR", self.default_persist_dir)
        self.default_llm_model = os.getenv("RAG_LLM_MODEL", self.default_llm_model)
        
        # Parse integers with validation
        try:
            self.default_k = int(os.getenv("RAG_TOP_K", str(self.default_k)))
            if self.default_k < 1 or self.default_k > 100:
                logger.warning(f"RAG_TOP_K={self.default_k} out of range [1,100], using 12")
                self.default_k = 12
        except ValueError:
            logger.warning(f"Invalid RAG_TOP_K value, using default: {self.default_k}")
        
        try:
            self.default_top_n = int(os.getenv("RAG_TOP_N", str(self.default_top_n)))
            if self.default_top_n < 1 or self.default_top_n > 50:
                logger.warning(f"RAG_TOP_N={self.default_top_n} out of range [1,50], using 6")
                self.default_top_n = 6
        except ValueError:
            logger.warning(f"Invalid RAG_TOP_N value, using default: {self.default_top_n}")
        
        # Parse boolean
        rerank_val = os.getenv("RAG_RERANK", "true").lower()
        self.enable_rerank = rerank_val in ["true", "1", "yes", "on"]
        
        # Validate top_n <= top_k
        if self.default_top_n > self.default_k:
            logger.warning(f"RAG_TOP_N ({self.default_top_n}) > RAG_TOP_K ({self.default_k}), adjusting")
            self.default_top_n = self.default_k
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is not set")
        
        if self.default_k < 1:
            errors.append(f"Invalid RAG_TOP_K: {self.default_k}")
        
        if self.default_top_n < 1:
            errors.append(f"Invalid RAG_TOP_N: {self.default_top_n}")
        
        if self.default_top_n > self.default_k:
            errors.append(f"RAG_TOP_N ({self.default_top_n}) cannot exceed RAG_TOP_K ({self.default_k})")
        
        return len(errors) == 0, errors
    
    def __repr__(self) -> str:
        """String representation (masks API key)."""
        key_display = f"{self.openai_api_key[:8]}..." if self.openai_api_key else "None"
        return (
            f"AppConfig(\n"
            f"  api_key={key_display}\n"
            f"  persist_dir={self.default_persist_dir}\n"
            f"  llm_model={self.default_llm_model}\n"
            f"  top_k={self.default_k}\n"
            f"  top_n={self.default_top_n}\n"
            f"  enable_rerank={self.enable_rerank}\n"
            f")"
        )


# Global configuration instance
CONFIG = AppConfig()

# Log configuration on import (useful for debugging)
if __name__ != "__main__":
    logger.info("Configuration loaded")
    is_valid, errors = CONFIG.validate()
    if not is_valid:
        for error in errors:
            logger.error(f"Config validation error: {error}")

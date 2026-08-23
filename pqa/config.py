"""
A file used to configure the QA system from the .env file.
It reads the model name and embedding model name from the .env file and sets them as environment variables.
The .env file should contain the following variables:
- LLM_MODEL: The model to use for question answering.
- EMBEDDING_MODEL: The embedding model to use for vector search.
The .env file should be located in the root directory of the project, if not present, the default values will be used.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    base_url: str; api_key: str; llm_model: str
    embedding_model: str; index_dir: Path; top_k: int; hybrid: bool

def load_config() -> Config:
    load_dotenv()
    return Config(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "gemma3:4b-it-qat"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
        index_dir=Path(os.getenv("INDEX_DIR", ".index")),
        top_k=int(os.getenv("TOP_K", "5")),
        # Dense-only by default: BM25 underperforms on Czech inflection (see README).
        hybrid=os.getenv("HYBRID", "false").strip().lower() in ("1", "true", "yes"),
    )

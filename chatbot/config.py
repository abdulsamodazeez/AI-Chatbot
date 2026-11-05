import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str | None
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def load_settings() -> Settings:
    return Settings(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        chat_model=os.environ.get("CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "150")),
        top_k=int(os.environ.get("TOP_K", "4")),
    )

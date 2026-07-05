"""
Configuration for RAG indexing.
"""

import os
from dataclasses import dataclass


@dataclass
class IndexConfig:
    """Settings for text chunking and embedding."""

    # Embedding provider: "dashscope" (default) or "local"
    provider: str = "dashscope"

    # DashScope settings (used when provider="dashscope")
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "text-embedding-v3"

    # Local model settings (used when provider="local")
    local_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Embedding dimension (must match model)
    # DashScope text-embedding-v3: 1024
    # local paraphrase-multilingual: 384
    dimension: int = 1024

    # Chunking settings
    chunk_size: int = 500       # words per chunk
    overlap: int = 50           # overlapping words between chunks

    # Search settings
    top_k: int = 5              # default number of results to return

    def __post_init__(self):
        # Auto-load API key from environment if not set
        if not self.dashscope_api_key:
            self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")

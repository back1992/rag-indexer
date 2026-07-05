"""
RAG Indexer - embedding abstraction layer for local and DashScope providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from .config import IndexConfig

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        pass


class DashScopeEmbedding(EmbeddingProvider):
    """DashScope (Alibaba Cloud) embedding API."""

    def __init__(self, config: IndexConfig):
        if not config.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY not set. "
                "Set environment variable or pass in IndexConfig."
            )

        from openai import OpenAI

        self._client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
        )
        self._model = config.dashscope_model
        self._dimension = config.dimension
        logger.info(f"Using DashScope embedding: {self._model}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using DashScope API."""
        # DashScope has a batch limit of 10
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimension,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)


class LocalEmbedding(EmbeddingProvider):
    """Local sentence-transformers embedding."""

    def __init__(self, config: IndexConfig):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.local_model_name)
        logger.info(f"Using local embedding: {config.local_model_name}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using local model."""
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)


def create_embedding_provider(config: IndexConfig) -> EmbeddingProvider:
    """Factory function to create the appropriate embedding provider."""
    if config.provider == "dashscope":
        return DashScopeEmbedding(config)
    elif config.provider == "local":
        return LocalEmbedding(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

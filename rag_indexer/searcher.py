"""
RAG Searcher - semantic search over indexed documents.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    text: str
    score: float
    chunk_index: int
    metadata: dict


class RAGSearcher:
    """Search over a FAISS index with semantic similarity."""

    def __init__(self, indexer):
        """
        Args:
            indexer: A RAGIndexer instance (loaded or freshly created).
        """
        self.indexer = indexer

    @classmethod
    def from_directory(cls, index_dir: Union[str, Path], config=None) -> "RAGSearcher":
        """
        Create a searcher from a saved index directory.

        Args:
            index_dir: Directory containing index.faiss and chunks.json.
            config: Optional IndexConfig override.

        Returns:
            RAGSearcher instance.
        """
        from .indexer import RAGIndexer
        indexer = RAGIndexer.load(index_dir, config)
        return cls(indexer)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search the index for chunks semantically similar to the query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects, sorted by relevance (best first).
        """
        if not self.indexer.chunks:
            return []

        # Encode query
        query_embedding = self.indexer.embedding.encode([query])

        # Search FAISS
        k = min(top_k, len(self.indexer.chunks))
        distances, indices = self.indexer.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.indexer.chunks):
                continue
            chunk = self.indexer.chunks[idx]
            # Convert L2 distance to similarity score (0-1, higher is better)
            score = 1.0 / (1.0 + dist)
            results.append(SearchResult(
                text=chunk.get("text", ""),
                score=score,
                chunk_index=int(idx),
                metadata={k: v for k, v in chunk.items() if k not in ("text", "chunk_index")},
            ))

        logger.info(f"Search '{query[:50]}...' → {len(results)} results")
        return results

    def search_context(self, query: str, top_k: int = 3, max_chars: int = 3000) -> str:
        """
        Search and return results as a formatted context string.

        Useful for passing to LLM prompts in RAG pipelines.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            max_chars: Maximum total characters in the context string.

        Returns:
            Formatted string with search results.
        """
        results = self.search(query, top_k=top_k)

        context_parts = []
        total_chars = 0
        for i, result in enumerate(results):
            source = result.metadata.get("source", "unknown")
            chunk_text = result.text

            if total_chars + len(chunk_text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    chunk_text = chunk_text[:remaining] + "..."
                else:
                    break

            context_parts.append(f"[Source: {source} | Relevance: {result.score:.2f}]\n{chunk_text}")
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

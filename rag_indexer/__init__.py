"""
rag_indexer - Index PDF documents for RAG with semantic search.

Quick start:
    from rag_indexer import RAGIndexer, RAGSearcher

    # Index a PDF
    indexer = RAGIndexer()
    indexer.index_pdf("chapter.pdf", output_dir="./index/")

    # Search
    searcher = RAGSearcher(indexer)
    results = searcher.search("What is communication theory?", top_k=3)
    for r in results:
        print(f"[{r.score:.2f}] {r.text[:100]}...")
"""

__version__ = "1.0.0"

from .config import IndexConfig
from .indexer import RAGIndexer
from .searcher import RAGSearcher, SearchResult

__all__ = [
    "IndexConfig",
    "RAGIndexer",
    "RAGSearcher",
    "SearchResult",
]

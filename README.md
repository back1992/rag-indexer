# rag-indexer

Index PDF documents for RAG (Retrieval-Augmented Generation) with semantic search using FAISS and sentence-transformers.

## Features

- **PDF text extraction** — extract text from PDF pages
- **Smart chunking** — overlapping word chunks for context preservation
- **Semantic embeddings** — sentence-transformers (all-MiniLM-L6-v2 by default)
- **FAISS indexing** — fast approximate nearest neighbor search
- **Semantic search** — query with natural language, get relevant chunks
- **LLM context builder** — format search results as context for RAG prompts

## Installation

```bash
pip install -e .
```

## Quick Start

### Python API

```python
from rag_indexer import RAGIndexer, RAGSearcher

# 1. Index a PDF
indexer = RAGIndexer()
result = indexer.index_pdf("chapter.pdf", output_dir="./index/")
# {"chunk_count": 42, "index_path": "./index/index.faiss", ...}

# 2. Search
searcher = RAGSearcher(indexer)
results = searcher.search("What is communication theory?", top_k=3)
for r in results:
    print(f"[{r.score:.2f}] {r.text[:100]}...")

# 3. Get formatted context for LLM
context = searcher.search_context("What is communication theory?")
# "[Source: chapter.pdf | Relevance: 0.85]\nCommunication theory is..."
```

### Load saved index

```python
from rag_indexer import RAGIndexer, RAGSearcher

# Load previously saved index
indexer = RAGIndexer.load("./index/")
searcher = RAGSearcher(indexer)
results = searcher.search("key concepts")
```

### CLI

```bash
# Index a PDF
rag-index index chapter.pdf --output ./index/

# Search
rag-index search ./index/ "What is communication theory?"

# Search with LLM-ready context output
rag-index search ./index/ "key concepts" --context
```

## Configuration

```python
from rag_indexer import IndexConfig, RAGIndexer

config = IndexConfig(
    model_name="all-MiniLM-L6-v2",  # embedding model
    chunk_size=500,                   # words per chunk
    overlap=50,                       # overlapping words
)

indexer = RAGIndexer(config)
```

## Package Structure

```
rag_indexer/
├── __init__.py     # Public API exports
├── config.py       # IndexConfig dataclass
├── indexer.py      # RAGIndexer (chunk, embed, index, save/load)
├── searcher.py     # RAGSearcher (semantic search, context builder)
└── cli.py          # CLI: rag-index
```

## Dependencies

| Core |
|------|
| `PyMuPDF>=1.24.0` |
| `sentence-transformers>=2.2.0` |
| `faiss-cpu>=1.7.4` |
| `numpy>=1.24.0` |

## License

MIT

#!/usr/bin/env python3
"""
RAG Indexer - standalone demo script.

Usage:
    python main.py                          # Index and search chapter04.pdf (DashScope)
    python main.py --provider local         # Use local sentence-transformers
    python main.py --pdf /path/to/file.pdf  # Index a different PDF
    python main.py --query "数组是什么"      # Search with a query
    python main.py --index-only             # Index without searching
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Default PDF path
DEFAULT_PDF = Path(__file__).parent.parent / "data" / "chapter04.pdf"
DEFAULT_OUTPUT = Path(__file__).parent / "output"


def _load_env_file():
    """Load environment variables from backend/.env if it exists."""
    env_path = Path(__file__).parent.parent.parent / "backend" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if key and not os.environ.get(key):
                        os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(
        description="RAG Indexer demo - index PDF and search with semantic similarity",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Path to PDF file (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for index files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Search query (if not provided, uses default queries)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return (default: 3)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only index the PDF, don't perform search",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["dashscope", "local"],
        default="dashscope",
        help="Embedding provider (default: dashscope)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="DashScope API key (or set DASHSCOPE_API_KEY env var)",
    )

    args = parser.parse_args()

    # Check PDF exists
    if not args.pdf.exists():
        print(f"Error: PDF file not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    # Load API key from env file or argument
    _load_env_file()
    if args.api_key:
        os.environ["DASHSCOPE_API_KEY"] = args.api_key

    # Import here to avoid slow startup when just checking args
    from rag_indexer import RAGIndexer, RAGSearcher, IndexConfig

    print(f"=" * 70)
    print(f"RAG Indexer Demo")
    print(f"=" * 70)
    print(f"PDF: {args.pdf}")
    print(f"Provider: {args.provider}")
    print(f"Output: {args.output}")
    print()

    # Step 1: Index the PDF
    print("Step 1: Indexing PDF...")
    print("-" * 70)

    config = IndexConfig(provider=args.provider)
    indexer = RAGIndexer(config=config)

    start_time = time.time()
    result = indexer.index_pdf(args.pdf, args.output)
    elapsed = time.time() - start_time

    print(f"✓ Indexed in {elapsed:.2f}s")
    print(f"  Chunks: {result['chunk_count']}")
    print(f"  Index path: {result['index_path']}")
    print()

    if args.index_only:
        print("Done (index-only mode)")
        return

    # Step 2: Load and search
    print("Step 2: Searching...")
    print("-" * 70)

    searcher = RAGSearcher.from_directory(args.output, config)

    # Determine queries
    if args.query:
        queries = [args.query]
    else:
        # Default queries based on language detection
        queries = [
            "数组的定义和特点",
            "链表是什么",
            "栈和队列的区别",
        ]

    # Perform searches
    for query in queries:
        print(f"\nQuery: {query}")
        print("─" * 70)

        results = searcher.search(query, top_k=args.top_k)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.4f}")
            print(f"   Chunk: {result.text[:100]}...")
            if result.metadata:
                print(f"   Metadata: {result.metadata}")

    print()
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()

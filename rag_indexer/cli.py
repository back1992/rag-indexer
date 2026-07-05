"""
CLI for rag-index: index PDFs and search them.

Usage:
    rag-index index chapter.pdf --output ./index/
    rag-index search ./index/ "What is communication theory?"
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Index PDFs for RAG semantic search.")
    subparsers = parser.add_subparsers(dest="command")

    # index command
    idx_parser = subparsers.add_parser("index", help="Index a PDF document")
    idx_parser.add_argument("pdf_path", help="Path to PDF file")
    idx_parser.add_argument("-o", "--output", default="./rag_index", help="Output directory")
    idx_parser.add_argument("--chunk-size", type=int, default=500, help="Words per chunk")
    idx_parser.add_argument("--overlap", type=int, default=50, help="Overlapping words")
    idx_parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")

    # search command
    search_parser = subparsers.add_parser("search", help="Search an index")
    search_parser.add_argument("index_dir", help="Index directory")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--context", action="store_true", help="Output as context string for LLM")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "index":
        _cmd_index(args)
    elif args.command == "search":
        _cmd_search(args)


def _cmd_index(args):
    from .config import IndexConfig
    from .indexer import RAGIndexer

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    config = IndexConfig(
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(f"Indexing {pdf_path}...")
    indexer = RAGIndexer(config)
    result = indexer.index_pdf(pdf_path, args.output)

    print(f"\n✅ Indexed {result['chunk_count']} chunks")
    print(f"   Index: {result['index_path']}")
    print(f"   Metadata: {result['metadata_path']}")


def _cmd_search(args):
    from .searcher import RAGSearcher

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        print(f"Error: Index not found: {index_dir}", file=sys.stderr)
        sys.exit(1)

    searcher = RAGSearcher.from_directory(index_dir)

    if args.context:
        context = searcher.search_context(args.query, top_k=args.top_k)
        print(context)
    else:
        results = searcher.search(args.query, top_k=args.top_k)
        print(f"\nResults for: \"{args.query}\"\n")
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            print(f"  {i}. [{r.score:.3f}] {source}")
            print(f"     {r.text[:200]}...")
            print()


if __name__ == "__main__":
    main()

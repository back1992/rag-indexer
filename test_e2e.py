"""
End-to-end test for RAG Indexer with DashScope.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_indexer.indexer import RAGIndexer
from rag_indexer.searcher import RAGSearcher
from rag_indexer.config import IndexConfig


def test_full_indexing_flow():
    """Test complete indexing flow with DashScope embeddings."""
    # Load API key from backend/.env
    env_path = Path(__file__).parent.parent.parent / "backend" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("DASHSCOPE_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    os.environ["DASHSCOPE_API_KEY"] = key
                    break
    
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("⚠ Skipping DashScope test: DASHSCOPE_API_KEY not set")
        return
    
    print("Testing DashScope integration...")
    print("-" * 60)
    
    # Use a small sample text
    sample_text = (
        "Python是一种高级编程语言。"
        "Python支持面向对象编程。"
        "Python有丰富的标准库。"
        "Python适合数据科学和机器学习。"
        "Python语法简洁易读。"
    )
    
    config = IndexConfig(provider="dashscope")
    indexer = RAGIndexer(config)
    
    # Test chunking
    chunks = indexer.chunk_text(sample_text)
    print(f"Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars")
    
    # Test embedding
    print("\nGenerating embeddings...")
    embeddings = indexer.embedding.encode(chunks)
    print(f"Embeddings shape: {embeddings.shape}")
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == config.dimension
    
    # Test indexing
    print("\nBuilding FAISS index...")
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    metadata = {"source": "test_sample"}
    indexer.add_chunks(chunks, metadata)
    indexer.save(output_dir)
    
    print(f"Index saved to: {output_dir}")
    
    # Test search
    print("\nTesting search...")
    searcher = RAGSearcher.from_directory(output_dir, config)
    results = searcher.search("Python编程", top_k=2)
    
    print(f"Search results: {len(results)}")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result.score:.4f}")
        print(f"     Text: {result.text[:50]}...")
    
    # Cleanup
    import shutil
    shutil.rmtree(output_dir)
    
    print("\n✓ DashScope integration test passed")


def test_chunk_character_limit():
    """Test that chunks respect DashScope's 8192 character limit."""
    print("Testing character limit compliance...")
    print("-" * 60)
    
    config = IndexConfig(provider="local")  # Use local to avoid API calls
    indexer = RAGIndexer(config)
    
    # Create text that would exceed 8192 chars if not chunked properly
    long_sentence = "这是一个测试句子。" * 200  # ~1800 chars per sentence
    text = "。".join([long_sentence] * 10)  # ~18000 chars total
    
    chunks = indexer.chunk_text(text, chunk_size=800)
    
    print(f"Total text: {len(text)} chars")
    print(f"Chunks created: {len(chunks)}")
    
    max_chunk_len = 0
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        max_chunk_len = max(max_chunk_len, chunk_len)
        if i < 5 or i >= len(chunks) - 2:
            print(f"  Chunk {i+1}: {chunk_len} chars")
        elif i == 5:
            print(f"  ... ({len(chunks) - 7} more) ...")
        assert chunk_len <= 8000, f"Chunk {i+1} exceeds 8000 chars: {chunk_len}"
    
    print(f"\nMax chunk length: {max_chunk_len} chars")
    print(f"DashScope limit: 8192 chars")
    print(f"Safety margin: {8192 - max_chunk_len} chars")
    
    print("\n✓ Character limit test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Indexer End-to-End Tests")
    print("=" * 60)
    
    try:
        test_chunk_character_limit()
        print()
        test_full_indexing_flow()
        print()
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

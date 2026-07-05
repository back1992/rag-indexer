"""
Unit tests for RAG Indexer chunking logic.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_indexer.indexer import RAGIndexer
from rag_indexer.config import IndexConfig


def test_chunk_text_respects_char_limit():
    """Test that chunk_text respects the max_chars limit."""
    config = IndexConfig(provider="local")  # Use local to avoid API calls
    indexer = RAGIndexer(config)
    
    # Create a long Chinese text (simulating PDF content)
    long_sentence = "这是一个很长的句子" * 100  # ~1200 chars
    text = "。".join([long_sentence] * 20)  # ~24000 chars total
    
    chunks = indexer.chunk_text(text, chunk_size=800)
    
    print(f"Total text length: {len(text)}")
    print(f"Number of chunks: {len(chunks)}")
    
    # Verify each chunk is under the hard cap (8000)
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        print(f"Chunk {i+1}: {chunk_len} chars")
        assert chunk_len <= 8000, f"Chunk {i+1} exceeds 8000 chars: {chunk_len}"
    
    # Should have more than 4 chunks with chunk_size=800
    assert len(chunks) > 4, f"Expected >4 chunks with chunk_size=800, got {len(chunks)}"
    
    print("✓ All chunks respect the character limit")


def test_chunk_text_handles_short_text():
    """Test that short text is not split unnecessarily."""
    config = IndexConfig(provider="local")
    indexer = RAGIndexer(config)
    
    short_text = "这是一个短句。这是另一个短句。"
    chunks = indexer.chunk_text(short_text)
    
    print(f"Short text: {len(short_text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    
    # Should be 1 chunk (may have trailing period removed)
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert "这是一个短句" in chunks[0]
    assert "这是另一个短句" in chunks[0]
    
    print("✓ Short text handled correctly")


def test_chunk_text_handles_empty_text():
    """Test that empty text returns empty list."""
    config = IndexConfig(provider="local")
    indexer = RAGIndexer(config)
    
    chunks = indexer.chunk_text("")
    assert len(chunks) == 0
    
    print("✓ Empty text handled correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Running RAG Indexer chunk tests...")
    print("=" * 60)
    
    try:
        test_chunk_text_respects_char_limit()
        print()
        test_chunk_text_handles_short_text()
        print()
        test_chunk_text_handles_empty_text()
        print()
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
RAG Indexer - chunk text, embed with embeddings, index with FAISS.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Union

import fitz
import numpy as np

from .config import IndexConfig
from .embedding import create_embedding_provider, EmbeddingProvider

logger = logging.getLogger(__name__)


def _is_cjk_pair(a: str, b: str) -> bool:
    """Check if two adjacent characters are both CJK (should join without space)."""
    def is_cjk(c: str) -> bool:
        return '\u4e00' <= c <= '\u9fff' or c in '，。！？、；：""''（）《》'
    return is_cjk(a) or is_cjk(b)


# Section header patterns that always start a new paragraph
_HEADER_RE = re.compile(
    r'^(第[一二三四五六七八九十\d]+[章节篇部]|'
    r'\d+\.\d+|'
    r'[A-Z]\.\s|'
    r'Chapter\s|Part\s|Section\s)',
    re.I
)

# Garbage lines (short, non-text characters from PDF artifacts)
_GARBAGE_RE = re.compile(
    r'^[\s\x08\u200b\u2003\u0c5c\u00a0]*$|^[^\u4e00-\u9fff\w]{1,3}$'
)


class RAGIndexer:
    """Index PDF documents for semantic search using FAISS."""

    def __init__(self, config: Optional[IndexConfig] = None):
        """
        Args:
            config: Indexing configuration. Uses defaults if None.
        """
        import faiss

        self.config = config or IndexConfig()
        self.embedding: EmbeddingProvider = create_embedding_provider(self.config)
        self.dimension = self.config.dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks: list[dict] = []

    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract all text from a PDF file, merging line breaks within paragraphs.

        PDF text extraction produces artificial line breaks (e.g. "符号互\\n动")
        that split sentences mid-word. This method merges lines that belong to
        the same paragraph while preserving actual paragraph boundaries.
        """
        doc = fitz.open(str(pdf_path))
        all_text = []
        for page in doc:
            lines = page.get_text().split('\n')
            merged = self._merge_lines(lines)
            all_text.append(merged)
        doc.close()
        return '\n'.join(all_text)

    @staticmethod
    def _merge_lines(lines: list[str]) -> str:
        """
        Merge consecutive lines that belong to the same paragraph.

        A new paragraph starts when:
        - The line is empty or garbage
        - The line starts with a section header pattern
        - The previous line ended with a sentence-ending punctuation mark

        CJK lines are joined without space; Latin lines with a space.
        """
        if not lines:
            return ""

        paragraphs: list[str] = []
        current = ""

        for line in lines:
            line = line.replace('\x08', '').replace('\u200b', '').strip()

            # Skip garbage lines
            if not line or _GARBAGE_RE.match(line):
                if current:
                    paragraphs.append(current)
                    current = ""
                continue

            # Section header always starts a new paragraph
            if _HEADER_RE.match(line):
                if current:
                    paragraphs.append(current)
                paragraphs.append(line)
                current = ""
                continue

            # If no current paragraph, start one
            if not current:
                current = line
                continue

            # Decide whether to merge with current paragraph
            prev_ends_sentence = current[-1] in '。！？.!?"")）》'

            if prev_ends_sentence:
                # Previous line ended a sentence — start new paragraph
                paragraphs.append(current)
                current = line
            else:
                # Merge: CJK-to-CJK without space, otherwise with space
                if _is_cjk_pair(current[-1], line[0]):
                    current += line
                else:
                    current += ' ' + line

        if current:
            paragraphs.append(current)

        return '\n'.join(paragraphs)

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
        """
        Split text into overlapping chunks by sentences.

        Groups sentences together until chunk_size is reached, then starts
        a new chunk with the last complete sentence(s) from the previous chunk
        as overlap context (sentence-aware, never cuts mid-word).

        Args:
            text: Input text to chunk
            chunk_size: Target characters per chunk (default 800)
            overlap: Approximate chars of overlap (last complete sentences, default 100)

        Returns:
            List of text chunks (each guaranteed < 8192 chars for DashScope)
        """
        MAX_CHARS = 8000  # hard cap for DashScope's 8192 limit

        # Split into sentences (handle both CJK and Latin punctuation)
        sentences = re.split(r'(?<=[。！？.!?])\s*', text.replace('\n', ' '))
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = (current_chunk + sentence) if current_chunk else sentence

            if len(test_chunk) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk)
                # Sentence-aware overlap: take last sentence(s) from previous chunk
                tail = self._get_overlap_tail(current_chunk, overlap)
                current_chunk = (tail + sentence) if tail else sentence
            else:
                current_chunk = test_chunk

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk)

        # Safety: split any chunk that exceeds the hard cap
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= MAX_CHARS:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(chunk), MAX_CHARS):
                    final_chunks.append(chunk[i:i + MAX_CHARS])

        return final_chunks

    @staticmethod
    def _get_overlap_tail(chunk: str, max_chars: int) -> str:
        """
        Extract the last complete sentence(s) from a chunk for overlap.

        Never cuts mid-word or mid-sentence. Returns empty string if
        no sentence fits within max_chars.
        """
        # Split chunk into sentences, take last ones that fit
        chunk_sentences = re.split(r'(?<=[。！？.!?])\s*', chunk)
        chunk_sentences = [s for s in chunk_sentences if s.strip()]

        tail_parts: list[str] = []
        tail_len = 0
        for s in reversed(chunk_sentences):
            s = s.strip()
            if not s:
                continue
            if tail_len + len(s) > max_chars:
                break
            tail_parts.insert(0, s)
            tail_len += len(s)

        return "".join(tail_parts)

    def add_chunks(self, chunks: list[str], metadata: dict):
        """Add text chunks to the FAISS index with metadata."""
        for i, chunk in enumerate(chunks):
            self.chunks.append({
                **metadata,
                "chunk_index": i,
                "text": chunk,
            })

        embeddings = self.embedding.encode(chunks)
        self.index.add(embeddings)

    def index_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Index a PDF file: extract text → chunk → embed → save index.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory where index files will be saved.
            metadata: Additional metadata to attach to each chunk.

        Returns:
            dict with chunk_count, index_path, metadata_path.
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)

        if not chunks:
            raise ValueError(f"No text extracted from {pdf_path}")

        meta = metadata or {}
        meta.setdefault("source", str(pdf_path.name))
        self.add_chunks(chunks, meta)
        self.save(output_dir)

        result = {
            "chunk_count": len(chunks),
            "index_path": str(output_dir / "index.faiss"),
            "metadata_path": str(output_dir / "chunks.json"),
        }

        logger.info(f"Indexed {pdf_path.name}: {len(chunks)} chunks → {output_dir}")
        return result

    def save(self, output_dir: Union[str, Path]):
        """Save the FAISS index and chunk metadata to disk."""
        import faiss

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(output_dir / "index.faiss"))

        with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved index ({len(self.chunks)} chunks) → {output_dir}")

    @classmethod
    def load(cls, index_dir: Union[str, Path], config: Optional[IndexConfig] = None) -> "RAGIndexer":
        """
        Load a previously saved index from disk.

        Args:
            index_dir: Directory containing index.faiss and chunks.json.
            config: Optional config override.

        Returns:
            RAGIndexer instance with loaded index.
        """
        import faiss

        index_dir = Path(index_dir)
        config = config or IndexConfig()

        instance = cls.__new__(cls)
        instance.config = config
        instance.dimension = config.dimension

        # Load FAISS index
        instance.index = faiss.read_index(str(index_dir / "index.faiss"))

        # Load chunk metadata
        with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
            instance.chunks = json.load(f)

        # Create embedding provider for search queries
        instance.embedding = create_embedding_provider(config)

        logger.info(f"Loaded index ({len(instance.chunks)} chunks) from {index_dir}")
        return instance

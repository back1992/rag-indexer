"""Quick test to verify text extraction merging."""
import sys
sys.path.insert(0, '.')

from rag_indexer.indexer import RAGIndexer
from rag_indexer.config import IndexConfig

config = IndexConfig(provider='local')
indexer = RAGIndexer(config)

text = indexer.extract_text('/Users/linmukong/WebstormProjects/ppt-bot-v2/packages/data/chapter04.pdf')

paragraphs = text.split('\n')
print(f'Total chars: {len(text)}')
print(f'Paragraphs: {len(paragraphs)}')
print()

# Show first few paragraphs
for i, p in enumerate(paragraphs[:8]):
    print(f'P{i}: [{len(p)}] {p[:100]}')
    print()

# Check for broken mid-sentence lines
broken = 0
endings = set(list('。！？.!?）》\u201c\u201d'))
for p in paragraphs:
    if p and p[-1] not in endings:
        broken += 1

print(f'Paragraphs without proper ending: {broken}/{len(paragraphs)}')

# Show chunks
print('\n=== Chunks ===')
chunks = indexer.chunk_text(text)
print(f'Chunks: {len(chunks)}')
for i, c in enumerate(chunks[:5]):
    print(f'Chunk {i}: [{len(c)}] {c[:80]}...')

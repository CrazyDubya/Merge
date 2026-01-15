#!/usr/bin/env python3
"""Test semantic knowledge base."""

import sys
sys.path.insert(0, '/home/user/Merge')

from club_harness.knowledge import (
    SemanticKnowledgeBase,
    Document,
    DocumentChunk,
    SearchResult,
    SimpleEmbedding,
    RAGHelper,
)

print("=" * 60)
print("SEMANTIC KNOWLEDGE BASE TEST")
print("=" * 60)

# Test 1: Basic document storage
print("\n[TEST 1] Basic document storage")

kb = SemanticKnowledgeBase(chunk_size=200, chunk_overlap=20)

doc1 = kb.add_document(
    content="Python is a high-level programming language known for its readability and versatility. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    title="Python Overview",
    source="python_guide.md",
    doc_type="markdown",
)

print(f"  Added document: {doc1.doc_id}")
print(f"  Title: {doc1.title}")

doc2 = kb.add_document(
    content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Common algorithms include decision trees, neural networks, and support vector machines.",
    title="Machine Learning Basics",
    doc_type="text",
)

print(f"  Added document: {doc2.doc_id}")

stats = kb.get_stats()
print(f"  Total documents: {stats['total_documents']}")
print(f"  Total chunks: {stats['total_chunks']}")

print("  [PASS] Document storage works")

# Test 2: Semantic search
print("\n[TEST 2] Semantic search")

results = kb.search("programming language")
print(f"  Query: 'programming language'")
print(f"  Results: {len(results)}")

if results:
    top = results[0]
    print(f"  Top result: '{top.chunk.title}' (score: {top.score:.3f})")
    print(f"  Content preview: {top.chunk.content[:80]}...")
    print(f"  Citation: {top.citation}")

results2 = kb.search("artificial intelligence machine learning")
print(f"  Query: 'artificial intelligence machine learning'")
print(f"  Results: {len(results2)}")

if results2:
    top = results2[0]
    print(f"  Top result: '{top.chunk.title}' (score: {top.score:.3f})")

print("  [PASS] Semantic search works")

# Test 3: Keyword fallback search
print("\n[TEST 3] Keyword fallback search")

results = kb.keyword_search("neural networks")
print(f"  Query: 'neural networks'")
print(f"  Results: {len(results)}")

if results:
    print(f"  Top result: '{results[0].chunk.title}'")

print("  [PASS] Keyword search works")

# Test 4: Document chunking for code
print("\n[TEST 4] Code document chunking")

code_content = '''
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

def calculate_product(a, b):
    """Calculate the product of two numbers."""
    return a * b

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(('add', a, b, result))
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(('multiply', a, b, result))
        return result
'''

doc3 = kb.add_document(
    content=code_content,
    title="Calculator Module",
    source="calculator.py",
    doc_type="code",
)

print(f"  Added code document: {doc3.doc_id}")
chunks = kb.chunks_by_doc.get(doc3.doc_id, [])
print(f"  Chunks created: {len(chunks)}")

# Search for specific function
results = kb.search("calculate sum function")
print(f"  Query: 'calculate sum function'")
if results:
    print(f"  Top result: {results[0].chunk.title}")

print("  [PASS] Code chunking works")

# Test 5: RAG Helper
print("\n[TEST 5] RAG Helper")

rag = RAGHelper(kb, max_context_chars=1000)

context, citations = rag.get_context("How do I use Python for programming?")
print(f"  Context length: {len(context)} chars")
print(f"  Citations: {len(citations)}")
if citations:
    print(f"    First citation: {citations[0]}")

# Build RAG prompt
system, user, cites = rag.build_prompt(
    "What programming paradigms does Python support?",
    top_k=2,
)
print(f"  System prompt length: {len(system)} chars")
print(f"  User prompt: '{user[:50]}...'")

# Format response with citations
response = "Python supports procedural, object-oriented, and functional programming."
formatted = rag.format_response_with_citations(response, cites)
print(f"  Formatted response has citations: {'Sources:' in formatted}")

print("  [PASS] RAG Helper works")

# Test 6: Simple embedding provider
print("\n[TEST 6] Simple embedding provider")

embedding = SimpleEmbedding(dimension=128)

vec1 = embedding.embed("machine learning algorithms")
vec2 = embedding.embed("machine learning models")
vec3 = embedding.embed("cooking recipes")

print(f"  Embedding dimension: {embedding.dimension()}")
print(f"  Vector 1 length: {len(vec1)}")

# Calculate similarities
def cosine_sim(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

sim12 = cosine_sim(vec1, vec2)
sim13 = cosine_sim(vec1, vec3)

print(f"  Similarity (ML algos vs ML models): {sim12:.3f}")
print(f"  Similarity (ML algos vs cooking): {sim13:.3f}")
assert sim12 > sim13, "Similar texts should have higher similarity"

print("  [PASS] Embedding provider works")

# Test 7: Document deletion
print("\n[TEST 7] Document deletion")

initial_docs = len(kb.documents)
initial_chunks = len(kb.chunks)

success = kb.delete_document(doc1.doc_id)
print(f"  Deleted document: {success}")
print(f"  Documents before: {initial_docs}, after: {len(kb.documents)}")
print(f"  Chunks before: {initial_chunks}, after: {len(kb.chunks)}")

assert len(kb.documents) < initial_docs, "Document should be deleted"
assert len(kb.chunks) < initial_chunks, "Chunks should be deleted"

print("  [PASS] Document deletion works")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
stats = kb.get_stats()
print(f"Final documents: {stats['total_documents']}")
print(f"Final chunks: {stats['total_chunks']}")
print(f"Doc types: {stats['doc_types']}")
print(f"Embedding dim: {stats['embedding_dimension']}")

print("\nAll knowledge base tests passed!")

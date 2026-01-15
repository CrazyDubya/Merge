"""
Knowledge base module for Club Harness.

Provides semantic search capabilities for RAG (Retrieval Augmented Generation).
"""

from .semantic_kb import (
    Document,
    DocumentChunk,
    SearchResult,
    SemanticKnowledgeBase,
    EmbeddingProvider,
    SimpleEmbedding,
    RAGHelper,
)

__all__ = [
    "Document",
    "DocumentChunk",
    "SearchResult",
    "SemanticKnowledgeBase",
    "EmbeddingProvider",
    "SimpleEmbedding",
    "RAGHelper",
]

"""
Semantic Knowledge Base for Club Harness.

Provides a semantic knowledge base that agents can query using natural language,
with vector similarity search.

Inspired by repos/hivey/hivey/knowledge/semantic_kb.py

Features:
- Document chunking and indexing
- Pluggable embedding backends
- Similarity search with configurable thresholds
- Source attribution and citation
- RAG (Retrieval Augmented Generation) helper
"""

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid


@dataclass
class Document:
    """A document in the knowledge base."""
    doc_id: str
    title: str
    content: str
    source: str = ""  # File path, URL, etc.
    doc_type: str = "text"  # text, code, markdown, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            content=data["content"],
            source=data.get("source", ""),
            doc_type=data.get("doc_type", "text"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


@dataclass
class DocumentChunk:
    """A chunk of a document for indexing."""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    embedding: Optional[List[float]] = None

    # Context
    title: str = ""
    source: str = ""
    doc_type: str = "text"

    # Position info
    start_char: int = 0
    end_char: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "embedding": self.embedding,
            "title": self.title,
            "source": self.source,
            "doc_type": self.doc_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Result of a semantic search."""
    chunk: DocumentChunk
    score: float
    rank: int

    # Citation info
    citation: str = ""

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.3f}, title='{self.chunk.title[:30]}...', chunk={self.chunk.chunk_index})"


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SimpleEmbedding(EmbeddingProvider):
    """
    Simple TF-IDF-like embedding without external dependencies.

    Good for testing and basic use cases. For production, use
    OpenAI embeddings or a local model.
    """

    def __init__(self, dimension: int = 256):
        self._dimension = dimension
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens

    def _build_vocab(self, tokens: List[str]):
        """Build vocabulary from tokens."""
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) % self._dimension

    def embed(self, text: str) -> List[float]:
        """Generate a simple embedding based on word frequencies."""
        tokens = self._tokenize(text)
        self._build_vocab(tokens)

        # Create term frequency vector
        embedding = [0.0] * self._dimension

        # Count token frequencies
        token_counts: Dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Build embedding
        for token, count in token_counts.items():
            idx = self._vocab.get(token, hash(token) % self._dimension)
            # TF-IDF-like weighting
            tf = 1 + math.log(count) if count > 0 else 0
            idf = self._idf.get(token, 1.0)
            embedding[idx] += tf * idf

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    def dimension(self) -> int:
        return self._dimension

    def update_idf(self, documents: List[str]):
        """Update IDF values from document corpus."""
        self._doc_count = len(documents)
        doc_freqs: Dict[str, int] = {}

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freqs[token] = doc_freqs.get(token, 0) + 1

        for token, df in doc_freqs.items():
            self._idf[token] = math.log(self._doc_count / (1 + df))


class SemanticKnowledgeBase:
    """
    Semantic knowledge base with vector similarity search.

    Supports:
    - Document ingestion and chunking
    - Similarity search using embeddings
    - Keyword fallback search
    - Source attribution
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.3,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize knowledge base.

        Args:
            embedding_provider: Provider for generating embeddings
            chunk_size: Target size for document chunks (in characters)
            chunk_overlap: Overlap between chunks
            similarity_threshold: Minimum similarity for results
            storage_path: Path for persistence (optional)
        """
        self.embedding_provider = embedding_provider or SimpleEmbedding()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.storage_path = Path(storage_path) if storage_path else None

        # Storage
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, DocumentChunk] = {}
        self.chunks_by_doc: Dict[str, List[str]] = {}

        # Load from storage if available
        if self.storage_path:
            self._load()

    def add_document(
        self,
        content: str,
        title: str = "",
        source: str = "",
        doc_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Add a document to the knowledge base.

        Args:
            content: Document content
            title: Document title
            source: Source reference (path, URL, etc.)
            doc_type: Type of document
            metadata: Additional metadata

        Returns:
            The created Document
        """
        doc_id = str(uuid.uuid4())[:12]

        doc = Document(
            doc_id=doc_id,
            title=title or f"Document {doc_id}",
            content=content,
            source=source,
            doc_type=doc_type,
            metadata=metadata or {},
        )

        self.documents[doc_id] = doc

        # Chunk and index
        self._chunk_document(doc)

        # Persist
        if self.storage_path:
            self._save()

        return doc

    def add_from_file(
        self,
        file_path: str,
        title: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Document:
        """Add a document from a file."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text()

        # Auto-detect doc type
        if doc_type is None:
            ext = path.suffix.lower()
            type_map = {
                '.py': 'code',
                '.js': 'code',
                '.ts': 'code',
                '.md': 'markdown',
                '.txt': 'text',
                '.json': 'json',
                '.yaml': 'yaml',
                '.yml': 'yaml',
            }
            doc_type = type_map.get(ext, 'text')

        return self.add_document(
            content=content,
            title=title or path.name,
            source=str(path),
            doc_type=doc_type,
        )

    def _chunk_document(self, doc: Document):
        """Split document into chunks and generate embeddings."""
        chunks = self._split_into_chunks(doc.content, doc.doc_type)

        self.chunks_by_doc[doc.doc_id] = []

        for i, (chunk_text, start, end) in enumerate(chunks):
            chunk_id = f"{doc.doc_id}_chunk_{i}"

            # Generate embedding
            embedding = self.embedding_provider.embed(chunk_text)

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                content=chunk_text,
                chunk_index=i,
                embedding=embedding,
                title=doc.title,
                source=doc.source,
                doc_type=doc.doc_type,
                start_char=start,
                end_char=end,
                metadata=doc.metadata,
            )

            self.chunks[chunk_id] = chunk
            self.chunks_by_doc[doc.doc_id].append(chunk_id)

    def _split_into_chunks(
        self,
        content: str,
        doc_type: str = "text",
    ) -> List[Tuple[str, int, int]]:
        """
        Split content into overlapping chunks.

        Returns list of (chunk_text, start_char, end_char).
        """
        chunks = []

        if doc_type == "code":
            # Split code by function/class boundaries
            chunks = self._split_code(content)
        elif doc_type == "markdown":
            # Split markdown by sections
            chunks = self._split_markdown(content)
        else:
            # Default: split by paragraphs or fixed size
            chunks = self._split_text(content)

        return chunks

    def _split_text(self, content: str) -> List[Tuple[str, int, int]]:
        """Split plain text into chunks."""
        chunks = []

        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', content)

        current_chunk = ""
        current_start = 0
        pos = 0

        for para in paragraphs:
            para_start = content.find(para, pos)
            para_end = para_start + len(para)

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append((current_chunk, current_start, pos))

                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + para if overlap_text else para
                current_start = para_start - len(overlap_text)

            pos = para_end

        # Add final chunk
        if current_chunk:
            chunks.append((current_chunk, current_start, len(content)))

        return chunks

    def _split_code(self, content: str) -> List[Tuple[str, int, int]]:
        """Split code by functions/classes."""
        chunks = []

        # Pattern to match function/class definitions
        patterns = [
            r'((?:async\s+)?def\s+\w+[^:]*:.*?)(?=\n(?:async\s+)?def\s+|\nclass\s+|\Z)',
            r'(class\s+\w+[^:]*:.*?)(?=\nclass\s+|\n(?:async\s+)?def\s+(?!\s)|\Z)',
        ]

        # Find all matches
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                matches.append((match.start(), match.end(), match.group(1)))

        # Sort by position
        matches.sort(key=lambda x: x[0])

        # Create chunks from matches
        if matches:
            for start, end, text in matches:
                if len(text) <= self.chunk_size:
                    chunks.append((text.strip(), start, end))
                else:
                    # Large function/class - split further
                    sub_chunks = self._split_text(text)
                    for sub_text, sub_start, sub_end in sub_chunks:
                        chunks.append((sub_text, start + sub_start, start + sub_end))
        else:
            # No matches - fall back to text splitting
            chunks = self._split_text(content)

        return chunks if chunks else self._split_text(content)

    def _split_markdown(self, content: str) -> List[Tuple[str, int, int]]:
        """Split markdown by sections."""
        chunks = []

        # Split by headers
        sections = re.split(r'(^#{1,3}\s+.+$)', content, flags=re.MULTILINE)

        current_chunk = ""
        current_start = 0
        pos = 0

        for section in sections:
            section_start = content.find(section, pos)
            section_end = section_start + len(section)

            if section.startswith('#'):
                # New section header
                if current_chunk:
                    chunks.append((current_chunk.strip(), current_start, pos))
                current_chunk = section
                current_start = section_start
            else:
                # Section content
                if len(current_chunk) + len(section) <= self.chunk_size:
                    current_chunk += section
                else:
                    if current_chunk:
                        chunks.append((current_chunk.strip(), current_start, pos))
                    current_chunk = section
                    current_start = section_start

            pos = section_end

        if current_chunk:
            chunks.append((current_chunk.strip(), current_start, len(content)))

        return chunks if chunks else self._split_text(content)

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_types: Optional[List[str]] = None,
        include_metadata: bool = True,
    ) -> List[SearchResult]:
        """
        Search the knowledge base using semantic similarity.

        Args:
            query: Search query
            top_k: Maximum results to return
            doc_types: Filter by document types
            include_metadata: Include document metadata in results

        Returns:
            List of SearchResult sorted by relevance
        """
        if not self.chunks:
            return []

        # Generate query embedding
        query_embedding = self.embedding_provider.embed(query)

        # Calculate similarities
        results = []

        for chunk_id, chunk in self.chunks.items():
            # Filter by doc type
            if doc_types and chunk.doc_type not in doc_types:
                continue

            if chunk.embedding is None:
                continue

            # Cosine similarity
            score = self._cosine_similarity(query_embedding, chunk.embedding)

            if score >= self.similarity_threshold:
                results.append((chunk, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Build search results
        search_results = []
        for rank, (chunk, score) in enumerate(results[:top_k]):
            citation = f"[{chunk.title}]"
            if chunk.source:
                citation += f" ({chunk.source})"

            result = SearchResult(
                chunk=chunk,
                score=score,
                rank=rank + 1,
                citation=citation,
            )
            search_results.append(result)

        return search_results

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Fallback keyword-based search."""
        query_words = set(query.lower().split())
        results = []

        for chunk_id, chunk in self.chunks.items():
            content_words = set(chunk.content.lower().split())
            overlap = len(query_words & content_words)

            if overlap > 0:
                # Score based on overlap ratio
                score = overlap / len(query_words)
                results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)

        search_results = []
        for rank, (chunk, score) in enumerate(results[:top_k]):
            result = SearchResult(
                chunk=chunk,
                score=score,
                rank=rank + 1,
                citation=f"[{chunk.title}]",
            )
            search_results.append(result)

        return search_results

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a chunk by ID."""
        return self.chunks.get(chunk_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        if doc_id not in self.documents:
            return False

        # Delete chunks
        for chunk_id in self.chunks_by_doc.get(doc_id, []):
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]

        # Delete document
        del self.documents[doc_id]
        if doc_id in self.chunks_by_doc:
            del self.chunks_by_doc[doc_id]

        if self.storage_path:
            self._save()

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "doc_types": list(set(d.doc_type for d in self.documents.values())),
            "embedding_dimension": self.embedding_provider.dimension(),
            "chunk_size": self.chunk_size,
            "similarity_threshold": self.similarity_threshold,
        }

    def _save(self):
        """Save knowledge base to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Save documents
        docs_file = self.storage_path / "documents.json"
        with open(docs_file, 'w') as f:
            json.dump([d.to_dict() for d in self.documents.values()], f)

        # Save chunks (without embeddings for now - could be large)
        chunks_file = self.storage_path / "chunks.json"
        with open(chunks_file, 'w') as f:
            json.dump([c.to_dict() for c in self.chunks.values()], f)

    def _load(self):
        """Load knowledge base from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        docs_file = self.storage_path / "documents.json"
        if docs_file.exists():
            with open(docs_file) as f:
                for doc_data in json.load(f):
                    doc = Document.from_dict(doc_data)
                    self.documents[doc.doc_id] = doc

        chunks_file = self.storage_path / "chunks.json"
        if chunks_file.exists():
            with open(chunks_file) as f:
                for chunk_data in json.load(f):
                    chunk = DocumentChunk(**{k: v for k, v in chunk_data.items() if k != 'embedding'})
                    # Regenerate embedding if needed
                    if chunk.content:
                        chunk.embedding = self.embedding_provider.embed(chunk.content)
                    self.chunks[chunk.chunk_id] = chunk

                    if chunk.doc_id not in self.chunks_by_doc:
                        self.chunks_by_doc[chunk.doc_id] = []
                    self.chunks_by_doc[chunk.doc_id].append(chunk.chunk_id)


class RAGHelper:
    """
    Helper for Retrieval Augmented Generation (RAG).

    Combines knowledge base search with prompt construction.
    """

    def __init__(
        self,
        knowledge_base: SemanticKnowledgeBase,
        max_context_chars: int = 4000,
        include_citations: bool = True,
    ):
        """
        Initialize RAG helper.

        Args:
            knowledge_base: The knowledge base to use
            max_context_chars: Maximum characters for context
            include_citations: Include source citations
        """
        self.kb = knowledge_base
        self.max_context_chars = max_context_chars
        self.include_citations = include_citations

    def get_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> Tuple[str, List[str]]:
        """
        Get relevant context for a query.

        Args:
            query: The query to search for
            top_k: Number of results to include

        Returns:
            Tuple of (context_text, citations)
        """
        results = self.kb.search(query, top_k=top_k)

        if not results:
            # Fallback to keyword search
            results = self.kb.keyword_search(query, top_k=top_k)

        context_parts = []
        citations = []
        total_chars = 0

        for result in results:
            chunk_text = result.chunk.content

            # Check size limit
            if total_chars + len(chunk_text) > self.max_context_chars:
                # Truncate if needed
                remaining = self.max_context_chars - total_chars
                if remaining > 100:
                    chunk_text = chunk_text[:remaining] + "..."
                else:
                    break

            context_parts.append(f"[Source: {result.chunk.title}]\n{chunk_text}")
            total_chars += len(chunk_text)

            if self.include_citations:
                citations.append(result.citation)

        context = "\n\n---\n\n".join(context_parts)
        return context, citations

    def build_prompt(
        self,
        query: str,
        system_template: Optional[str] = None,
        top_k: int = 3,
    ) -> Tuple[str, str, List[str]]:
        """
        Build a RAG prompt with context.

        Args:
            query: User query
            system_template: System prompt template (use {context} placeholder)
            top_k: Number of context chunks

        Returns:
            Tuple of (system_prompt, user_prompt, citations)
        """
        context, citations = self.get_context(query, top_k=top_k)

        default_template = """You are a helpful assistant. Answer the user's question based on the following context.

Context:
{context}

If the context doesn't contain relevant information, say so and provide your best general knowledge answer."""

        template = system_template or default_template
        system_prompt = template.format(context=context) if context else template.replace("{context}", "No relevant context found.")

        return system_prompt, query, citations

    def format_response_with_citations(
        self,
        response: str,
        citations: List[str],
    ) -> str:
        """Format response with citations at the end."""
        if not citations:
            return response

        citation_text = "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)
        return response + citation_text

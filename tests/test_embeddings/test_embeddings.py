"""
Tests for RAGX Embeddings — chunking strategies and Document utility methods.
All offline (no embedding model API calls).
"""

from __future__ import annotations

import pytest

from ragx.ingestion.loaders.base import Document
from ragx.embeddings.chunking.recursive_splitter import RecursiveChunker


# ── Document utility methods ──────────────────────────────────────────────────

class TestDocumentUtilities:
    def test_doc_id_auto_generated(self):
        doc = Document(content="hello")
        assert doc.doc_id
        assert len(doc.doc_id) == 36  # UUID v4 format

    def test_content_hash_computed(self):
        doc = Document(content="hello")
        assert len(doc.content_hash) == 64  # SHA-256 hex

    def test_same_content_same_hash(self):
        doc1 = Document(content="same text")
        doc2 = Document(content="same text")
        assert doc1.content_hash == doc2.content_hash

    def test_different_content_different_hash(self):
        doc1 = Document(content="text A")
        doc2 = Document(content="text B")
        assert doc1.content_hash != doc2.content_hash

    def test_to_langchain_conversion(self):
        doc = Document(
            content="RAGX content",
            metadata={"source": "test.txt", "page": 1},
            source_path="test.txt",
        )
        lc_doc = doc.to_langchain()
        assert lc_doc.page_content == "RAGX content"
        assert lc_doc.metadata["source"] == "test.txt"
        assert lc_doc.metadata["doc_id"] == doc.doc_id

    def test_from_langchain_roundtrip(self):
        from langchain_core.documents import Document as LCDocument

        lc_doc = LCDocument(
            page_content="roundtrip content",
            metadata={"source": "file.txt", "doc_id": "abc-123"},
        )
        doc = Document.from_langchain(lc_doc)
        assert doc.content == "roundtrip content"
        assert doc.doc_id == "abc-123"
        assert doc.metadata["source"] == "file.txt"

    def test_source_property(self):
        doc = Document(content="text", metadata={"source": "myfile.pdf"})
        assert doc.source == "myfile.pdf"

    def test_source_property_none_when_missing(self):
        doc = Document(content="text")
        assert doc.source is None

    def test_created_at_property(self):
        doc = Document(content="text", metadata={"created_at": "2026-01-01"})
        assert doc.created_at == "2026-01-01"

    def test_created_at_none_when_missing(self):
        doc = Document(content="text")
        assert doc.created_at is None


# ── RecursiveChunker ──────────────────────────────────────────────────────────

class TestRecursiveChunker:
    def setup_method(self):
        self.chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)

    def test_split_single_short_document(self):
        doc = Document(content="Short doc.", metadata={"source": "a.txt"})
        chunks = self.chunker.split([doc])
        assert len(chunks) >= 1
        assert all(isinstance(c, Document) for c in chunks)

    def test_split_long_document_produces_multiple_chunks(self):
        long_text = "Word " * 200  # ~1000 chars, chunk_size=100 → many chunks
        doc = Document(content=long_text, metadata={"source": "long.txt"})
        chunks = self.chunker.split([doc])
        assert len(chunks) > 1

    def test_chunks_inherit_metadata(self):
        doc = Document(content="A" * 300, metadata={"source": "src.txt", "page": 5})
        chunks = self.chunker.split([doc])
        for chunk in chunks:
            assert chunk.metadata.get("source") == "src.txt"

    def test_chunks_have_chunk_index(self):
        doc = Document(content="Word " * 200)
        chunks = self.chunker.split([doc])
        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk.metadata

    def test_chunks_have_parent_doc_id(self):
        doc = Document(content="Text " * 100)
        chunks = self.chunker.split([doc])
        for chunk in chunks:
            assert chunk.metadata.get("parent_doc_id") == doc.doc_id

    def test_chunks_have_unique_doc_ids(self):
        doc = Document(content="Word " * 200)
        chunks = self.chunker.split([doc])
        ids = [c.doc_id for c in chunks]
        assert len(ids) == len(set(ids))  # All unique

    def test_split_empty_document_list(self):
        chunks = self.chunker.split([])
        assert chunks == []

    def test_split_multiple_documents(self):
        docs = [
            Document(content="Doc A content. " * 10, metadata={"source": "a.txt"}),
            Document(content="Doc B content. " * 10, metadata={"source": "b.txt"}),
        ]
        chunks = self.chunker.split(docs)
        assert len(chunks) >= 2

    def test_split_text_raw(self):
        text = "Sentence one. Sentence two. Sentence three. " * 10
        chunks = self.chunker.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_split_text_empty_string(self):
        chunks = self.chunker.split_text("")
        assert chunks == []

    def test_chunk_size_respected(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
        long_text = "A " * 200
        chunks = chunker.split_text(long_text)
        for chunk in chunks:
            # Each chunk should be at most chunk_size chars (approximately)
            assert len(chunk) <= 60  # Some tolerance for overlap

"""
Tests for RAGX Ingestion Pipeline — loaders, preprocessor, metadata, pipeline.
All tests are offline (no API keys required).
"""

from __future__ import annotations

import tempfile
import os
from pathlib import Path

import pytest

from ragx.ingestion.loaders.base import Document
from ragx.ingestion.loaders.txt_loader import TxtLoader
from ragx.ingestion.loaders.markdown_loader import MarkdownLoader
from ragx.ingestion.loaders.csv_loader import CsvLoader
from ragx.ingestion.loaders import get_loader, get_supported_extensions
from ragx.ingestion.preprocessor import TextPreprocessor
from ragx.ingestion.metadata import MetadataGenerator
from ragx.ingestion.pipeline import IngestionPipeline


# ── Document dataclass ────────────────────────────────────────────────────────

class TestDocument:
    def test_document_creation(self):
        doc = Document(content="Hello world", source_path="/tmp/test.txt")
        assert doc.content == "Hello world"
        assert doc.source_path == "/tmp/test.txt"
        assert isinstance(doc.metadata, dict)

    def test_document_default_metadata(self):
        doc = Document(content="text")
        assert doc.metadata == {}

    def test_document_with_metadata(self):
        meta = {"page": 1, "source": "file.pdf"}
        doc = Document(content="text", metadata=meta)
        assert doc.metadata["page"] == 1


# ── Loader registry ───────────────────────────────────────────────────────────

class TestLoaderRegistry:
    def test_supported_extensions_not_empty(self):
        exts = get_supported_extensions()
        assert len(exts) > 0

    def test_known_extensions_registered(self):
        exts = get_supported_extensions()
        for ext in [".pdf", ".txt", ".md", ".csv", ".docx"]:
            assert ext in exts, f"Extension {ext} not registered"

    def test_get_loader_txt(self):
        loader = get_loader("document.txt")
        assert isinstance(loader, TxtLoader)

    def test_get_loader_markdown(self):
        loader = get_loader("readme.md")
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_csv(self):
        loader = get_loader("data.csv")
        assert isinstance(loader, CsvLoader)

    def test_get_loader_unsupported_raises(self):
        with pytest.raises(ValueError, match="No loader registered"):
            get_loader("file.xyz123")

    def test_get_loader_case_insensitive(self):
        loader = get_loader("README.MD")
        assert isinstance(loader, MarkdownLoader)


# ── TxtLoader ────────────────────────────────────────────────────────────────

class TestTxtLoader:
    def test_load_basic_text(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello from RAGX!", encoding="utf-8")
        loader = TxtLoader()
        docs = loader.load(str(f))
        assert len(docs) == 1
        assert "Hello from RAGX!" in docs[0].content

    def test_load_nonexistent_raises(self):
        loader = TxtLoader()
        with pytest.raises(Exception):
            loader.load("/nonexistent/path/file.txt")

    def test_load_sets_source_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")
        loader = TxtLoader()
        docs = loader.load(str(f))
        assert docs[0].source_path is not None or docs[0].metadata.get("source") is not None


# ── MarkdownLoader ────────────────────────────────────────────────────────────

class TestMarkdownLoader:
    def test_load_markdown_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nSome paragraph text here.\n", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1
        combined = " ".join(d.content for d in docs)
        assert "Title" in combined or "paragraph" in combined

    def test_load_empty_markdown(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")
        loader = MarkdownLoader()
        # Should not raise, may return empty list
        docs = loader.load(str(f))
        assert isinstance(docs, list)


# ── CsvLoader ────────────────────────────────────────────────────────────────

class TestCsvLoader:
    def test_load_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,value\nfoo,1\nbar,2\n", encoding="utf-8")
        loader = CsvLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1
        combined = " ".join(d.content for d in docs)
        assert "foo" in combined or "name" in combined


# ── TextPreprocessor ──────────────────────────────────────────────────────────

class TestTextPreprocessor:
    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_basic_processing(self):
        result = self.preprocessor.process("Hello   world")
        assert "Hello" in result
        assert "world" in result

    def test_strips_extra_whitespace(self):
        result = self.preprocessor.process("  leading and trailing  ")
        assert result == result.strip()

    def test_handles_empty_string(self):
        result = self.preprocessor.process("")
        assert isinstance(result, str)

    def test_handles_newlines(self):
        result = self.preprocessor.process("line1\n\n\nline2")
        assert "line1" in result
        assert "line2" in result

    def test_does_not_destroy_content(self):
        text = "RAGX is a RAG engine using ChromaDB and sentence-transformers."
        result = self.preprocessor.process(text)
        assert "RAGX" in result
        assert "ChromaDB" in result


# ── MetadataGenerator ─────────────────────────────────────────────────────────

class TestMetadataGenerator:
    def setup_method(self):
        self.gen = MetadataGenerator()

    def test_generates_document_id(self):
        meta = self.gen.generate(source="/tmp/test.txt", content="some content")
        assert "document_id" in meta
        assert len(meta["document_id"]) > 0

    def test_includes_source(self):
        meta = self.gen.generate(source="/tmp/test.txt", content="content")
        assert "source" in meta or "document_id" in meta

    def test_different_docs_different_ids(self):
        meta1 = self.gen.generate(source="/tmp/a.txt", content="content a")
        meta2 = self.gen.generate(source="/tmp/b.txt", content="content b")
        assert meta1["document_id"] != meta2["document_id"]

    def test_returns_dict(self):
        meta = self.gen.generate(source="/tmp/x.txt", content="text")
        assert isinstance(meta, dict)


# ── IngestionPipeline ─────────────────────────────────────────────────────────

class TestIngestionPipeline:
    def test_ingest_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("RAGX is an advanced RAG engine for Python.", encoding="utf-8")

        pipeline = IngestionPipeline()
        docs = pipeline.ingest_file(str(f))

        assert len(docs) >= 1
        combined = " ".join(d.content for d in docs)
        assert "RAGX" in combined

    def test_ingest_markdown_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# RAGX\n\nA retrieval and generation engine.\n", encoding="utf-8")

        pipeline = IngestionPipeline()
        docs = pipeline.ingest_file(str(f))

        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_ingest_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "test.unknown"
        f.write_text("some content")
        pipeline = IngestionPipeline()
        with pytest.raises(ValueError):
            pipeline.ingest_file(str(f))

    def test_ingest_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("Document A content.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Document B content.", encoding="utf-8")
        (tmp_path / "ignore.xyz").write_text("ignored")

        pipeline = IngestionPipeline()
        docs = pipeline.ingest_directory(str(tmp_path))

        assert len(docs) >= 2

    def test_ingest_nonexistent_dir_raises(self):
        pipeline = IngestionPipeline()
        with pytest.raises(NotADirectoryError):
            pipeline.ingest_directory("/nonexistent/path/12345")

    def test_ingest_empty_directory(self, tmp_path):
        pipeline = IngestionPipeline()
        docs = pipeline.ingest_directory(str(tmp_path))
        assert docs == []

    def test_documents_have_metadata(self, tmp_path):
        f = tmp_path / "meta_test.txt"
        f.write_text("Check metadata generation.", encoding="utf-8")
        pipeline = IngestionPipeline()
        docs = pipeline.ingest_file(str(f))
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc.metadata, dict)
            assert "document_id" in doc.metadata

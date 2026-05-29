"""
RAGX Ingestion Package — Data loading, preprocessing, and ingestion pipeline.

This package provides:

* **Loaders** — Format-specific readers (PDF, DOCX, TXT, CSV, Markdown, Web).
* **TextPreprocessor** — A chainable text-cleaning pipeline.
* **MetadataGenerator** — Rich, validated metadata for every document.
* **IngestionPipeline** — End-to-end orchestration of ingestion.

Quick-start::

    from ragx.ingestion import IngestionPipeline

    pipeline = IngestionPipeline()
    docs = pipeline.ingest_file("report.pdf")
"""

from __future__ import annotations

from ragx.ingestion.loaders import (
    BaseLoader,
    CsvLoader,
    DocxLoader,
    Document,
    MarkdownLoader,
    PdfLoader,
    TxtLoader,
    WebLoader,
    get_loader,
    get_supported_extensions,
)
from ragx.ingestion.metadata import DocumentMetadata, MetadataGenerator
from ragx.ingestion.pipeline import IngestionPipeline
from ragx.ingestion.preprocessor import TextPreprocessor

__all__ = [
    # Loaders
    "BaseLoader",
    "CsvLoader",
    "DocxLoader",
    "Document",
    "MarkdownLoader",
    "PdfLoader",
    "TxtLoader",
    "WebLoader",
    "get_loader",
    "get_supported_extensions",
    # Preprocessor
    "TextPreprocessor",
    # Metadata
    "DocumentMetadata",
    "MetadataGenerator",
    # Pipeline
    "IngestionPipeline",
]

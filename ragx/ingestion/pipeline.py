"""
RAGX Ingestion Pipeline — Orchestrates file/URL ingestion end-to-end.

Coordinates loaders, the text preprocessor, and the metadata generator
to produce clean, enriched :class:`Document` objects ready for chunking
and embedding.  Supports single-file, directory-batch, and URL ingestion
with parallel processing, progress bars, and per-document error handling.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from ragx.config.logging_config import get_logger
from ragx.config.settings import Settings, get_settings
from ragx.ingestion.loaders import WebLoader, get_loader, get_supported_extensions
from ragx.ingestion.loaders.base import Document
from ragx.ingestion.metadata import MetadataGenerator
from ragx.ingestion.preprocessor import TextPreprocessor

logger = get_logger(__name__)

_MAX_WORKERS = min(8, (os.cpu_count() or 4) + 2)


class IngestionPipeline:
    """End-to-end document ingestion pipeline.

    Loads raw files or web pages, cleans the text, enriches metadata,
    and persists the processed documents as JSON.

    Args:
        settings: Optional explicit :class:`Settings` instance.  If
            ``None`` the global singleton from :func:`get_settings` is
            used.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._preprocessor = TextPreprocessor()
        self._metadata_gen = MetadataGenerator()
        self._web_loader = WebLoader()
        logger.info(
            "IngestionPipeline initialised",
            processed_path=str(self._settings.data_processed_path),
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def ingest_file(self, file_path: str) -> list[Document]:
        """Ingest a single file and return processed documents.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            A list of processed :class:`Document` instances.
        """
        path = Path(file_path).resolve()
        logger.info("Ingesting file", path=str(path))

        loader = get_loader(str(path))
        raw_docs = loader.load(str(path))

        processed = self._process_documents(raw_docs, source=str(path))
        self._save_processed(processed)
        return processed

    def ingest_directory(
        self,
        dir_path: str,
        glob: str = "**/*",
    ) -> list[Document]:
        """Ingest all supported files in a directory tree.

        Files whose extension is not recognised are silently skipped.
        Processing is parallelised via a thread pool.

        Args:
            dir_path: Root directory to scan.
            glob: Glob pattern for file discovery (default ``**/*``).

        Returns:
            A flat list of all processed :class:`Document` instances.
        """
        root = Path(dir_path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        supported = set(get_supported_extensions())
        file_paths = [
            str(p)
            for p in root.glob(glob)
            if p.is_file() and p.suffix.lower() in supported
        ]

        if not file_paths:
            logger.warning("No supported files found", dir=str(root), glob=glob)
            return []

        logger.info(
            "Ingesting directory",
            dir=str(root),
            files=len(file_paths),
        )

        all_docs: list[Document] = []
        errors: int = 0

        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            progress = tqdm(total=len(file_paths), desc="Ingesting files", unit="file")
        except ImportError:
            progress = None

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._safe_ingest_file, fp): fp
                for fp in file_paths
            }

            for future in as_completed(futures):
                fp = futures[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                except Exception as exc:
                    errors += 1
                    logger.error(
                        "File ingestion failed",
                        file=fp,
                        error=str(exc),
                    )
                finally:
                    if progress is not None:
                        progress.update(1)

        if progress is not None:
            progress.close()

        logger.info(
            "Directory ingestion complete",
            dir=str(root),
            total_docs=len(all_docs),
            errors=errors,
        )
        return all_docs

    def ingest_url(self, url: str) -> list[Document]:
        """Ingest a single web page and return processed documents.

        Args:
            url: The URL to fetch and process.

        Returns:
            A list of processed :class:`Document` instances.
        """
        logger.info("Ingesting URL", url=url)
        raw_docs = self._web_loader.load(url)
        processed = self._process_documents(raw_docs, source=url)
        self._save_processed(processed)
        return processed

    def ingest_urls(self, urls: list[str]) -> list[Document]:
        """Ingest multiple web pages in parallel.

        Args:
            urls: A list of URLs to fetch and process.

        Returns:
            A flat list of all processed :class:`Document` instances.
        """
        if not urls:
            return []

        logger.info("Ingesting URLs", count=len(urls))
        all_docs: list[Document] = []
        errors: int = 0

        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            progress = tqdm(total=len(urls), desc="Ingesting URLs", unit="url")
        except ImportError:
            progress = None

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._safe_ingest_url, u): u
                for u in urls
            }

            for future in as_completed(futures):
                u = futures[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                except Exception as exc:
                    errors += 1
                    logger.error("URL ingestion failed", url=u, error=str(exc))
                finally:
                    if progress is not None:
                        progress.update(1)

        if progress is not None:
            progress.close()

        logger.info(
            "URL batch ingestion complete",
            total_docs=len(all_docs),
            errors=errors,
        )
        return all_docs

    # ── Internal helpers ────────────────────────────────────────────────────

    def _safe_ingest_file(self, file_path: str) -> list[Document]:
        """Ingest a single file, catching and logging any errors.

        Designed to be submitted to a thread pool.  If the file fails
        the error is logged and an empty list is returned so that the
        batch can continue.
        """
        try:
            return self.ingest_file(file_path)
        except Exception as exc:
            logger.error(
                "Safe ingest failed for file",
                file=file_path,
                error=str(exc),
            )
            return []

    def _safe_ingest_url(self, url: str) -> list[Document]:
        """Ingest a single URL with error isolation."""
        try:
            return self.ingest_url(url)
        except Exception as exc:
            logger.error(
                "Safe ingest failed for URL",
                url=url,
                error=str(exc),
            )
            return []

    def _process_documents(
        self,
        documents: list[Document],
        source: str,
    ) -> list[Document]:
        """Clean text and enrich metadata for a batch of documents.

        Args:
            documents: Raw documents from a loader.
            source: Original source path/URL (for metadata).

        Returns:
            The same documents, mutated in-place with cleaned content
            and enriched metadata.
        """
        processed: list[Document] = []
        for doc in documents:
            try:
                # Clean text
                doc.content = self._preprocessor.process(doc.content)

                # Skip documents that are empty after cleaning
                if not doc.content.strip():
                    logger.debug("Document empty after preprocessing — skipped", source=source)
                    continue

                # Enrich metadata
                extra_meta = dict(doc.metadata)
                enriched = self._metadata_gen.generate(
                    source=doc.source_path or source,
                    content=doc.content,
                    **extra_meta,
                )
                doc.metadata = enriched
                processed.append(doc)
            except Exception as exc:
                logger.warning(
                    "Failed to process document — skipping",
                    source=source,
                    error=str(exc),
                )

        return processed

    def _save_processed(self, documents: list[Document]) -> None:
        """Persist processed documents as JSON files.

        Each document is saved as ``<document_id>.json`` under the
        configured ``data_processed_path``.

        Args:
            documents: Processed documents to persist.
        """
        output_dir = Path(self._settings.data_processed_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        for doc in documents:
            doc_id = doc.metadata.get("document_id", "unknown")
            out_path = output_dir / f"{doc_id}.json"

            payload = {
                "content": doc.content,
                "metadata": doc.metadata,
                "source_path": doc.source_path,
            }

            try:
                out_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.debug("Saved processed document", path=str(out_path))
            except Exception as exc:
                logger.error(
                    "Failed to save processed document",
                    path=str(out_path),
                    error=str(exc),
                )

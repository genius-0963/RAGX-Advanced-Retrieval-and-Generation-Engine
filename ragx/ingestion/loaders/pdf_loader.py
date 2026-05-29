"""
RAGX PDF Loader — Extract text from PDF documents using pypdf.

Reads PDFs page-by-page, storing per-page metadata including page number.
Encrypted or protected PDFs are handled gracefully with error logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import BaseLoader, Document

logger = get_logger(__name__)


class PdfLoader(BaseLoader):
    """Loader that extracts text from PDF files using *pypdf*.

    Each page of the PDF becomes a separate :class:`Document` so that
    downstream chunking can respect page boundaries.
    """

    supported_extensions: ClassVar[list[str]] = [".pdf"]

    def load(self, source: str) -> list[Document]:
        """Load a PDF file and return one document per page.

        Args:
            source: File-system path to a ``.pdf`` file.

        Returns:
            A list of :class:`Document` instances, one per page.

        Raises:
            FileNotFoundError: If *source* does not exist.
        """
        from pypdf import PdfReader  # type: ignore[import-untyped]

        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        logger.info("Loading PDF", path=str(path))
        documents: list[Document] = []

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            logger.error("Failed to open PDF", path=str(path), error=str(exc))
            raise

        # Handle encrypted PDFs
        if reader.is_encrypted:
            try:
                reader.decrypt("")
                logger.warning("PDF was encrypted; decrypted with empty password", path=str(path))
            except Exception as exc:
                logger.error(
                    "Cannot decrypt protected PDF — skipping",
                    path=str(path),
                    error=str(exc),
                )
                return documents

        total_pages = len(reader.pages)
        logger.debug("PDF page count", path=str(path), pages=total_pages)

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "Failed to extract text from page",
                    path=str(path),
                    page=page_num,
                    error=str(exc),
                )
                text = ""

            if not text.strip():
                logger.debug("Empty page skipped", path=str(path), page=page_num)
                continue

            doc = Document(
                content=text,
                metadata={
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "file_name": path.name,
                },
                source_path=str(path),
            )
            documents.append(doc)

        logger.info(
            "PDF loaded",
            path=str(path),
            pages_extracted=len(documents),
            total_pages=total_pages,
        )
        return documents

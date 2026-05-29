"""
RAGX Loaders Package — Registry and factory for document loaders.

Provides a ``get_loader`` factory that automatically selects the correct
loader class based on a file's extension, and exposes all loader classes
for direct import.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import BaseLoader, Document
from ragx.ingestion.loaders.csv_loader import CsvLoader
from ragx.ingestion.loaders.docx_loader import DocxLoader
from ragx.ingestion.loaders.markdown_loader import MarkdownLoader
from ragx.ingestion.loaders.pdf_loader import PdfLoader
from ragx.ingestion.loaders.txt_loader import TxtLoader
from ragx.ingestion.loaders.web_loader import WebLoader

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ── Loader registry ────────────────────────────────────────────────────────
# Maps file extensions (lowercased, with leading dot) to loader classes.
_LOADER_REGISTRY: dict[str, type[BaseLoader]] = {}


def _register_loader(loader_cls: type[BaseLoader]) -> None:
    """Register a loader class for each of its supported extensions."""
    for ext in loader_cls.supported_extensions:
        ext_lower = ext.lower()
        _LOADER_REGISTRY[ext_lower] = loader_cls
        logger.debug("Registered loader", extension=ext_lower, loader=loader_cls.__name__)


# Automatically register all built-in loaders
for _cls in (PdfLoader, DocxLoader, TxtLoader, CsvLoader, MarkdownLoader):
    _register_loader(_cls)


def get_loader(file_path: str) -> BaseLoader:
    """Return an appropriate loader instance for *file_path*.

    The loader is selected by matching the file's extension against the
    global registry.

    Args:
        file_path: Path to the file to be loaded.

    Returns:
        An instantiated :class:`BaseLoader` subclass.

    Raises:
        ValueError: If no loader is registered for the file extension.
    """
    ext = Path(file_path).suffix.lower()
    loader_cls = _LOADER_REGISTRY.get(ext)

    if loader_cls is None:
        supported = ", ".join(sorted(_LOADER_REGISTRY.keys()))
        raise ValueError(
            f"No loader registered for extension '{ext}'. "
            f"Supported extensions: {supported}"
        )

    logger.debug("Resolved loader", extension=ext, loader=loader_cls.__name__)
    return loader_cls()


def get_supported_extensions() -> list[str]:
    """Return a sorted list of all file extensions with registered loaders."""
    return sorted(_LOADER_REGISTRY.keys())


__all__ = [
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
]

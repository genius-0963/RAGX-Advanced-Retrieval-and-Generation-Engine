"""
RAGX Text Loader — Load plain-text files with automatic encoding detection.

Uses *chardet* to detect file encoding before reading, so that files in
various encodings (UTF-8, Latin-1, Windows-1252, etc.) are handled
transparently.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import BaseLoader, Document

logger = get_logger(__name__)

_DEFAULT_ENCODING = "utf-8"


class TxtLoader(BaseLoader):
    """Loader for plain-text files (``.txt``, ``.text``).

    Encoding is auto-detected via *chardet*.  If detection fails the
    loader falls back to UTF-8 with replacement characters.
    """

    supported_extensions: ClassVar[list[str]] = [".txt", ".text"]

    def _detect_encoding(self, raw_bytes: bytes) -> str:
        """Detect the encoding of *raw_bytes* using chardet.

        Args:
            raw_bytes: Raw file content.

        Returns:
            The detected encoding string, or ``utf-8`` as fallback.
        """
        try:
            import chardet  # type: ignore[import-untyped]

            result = chardet.detect(raw_bytes)
            encoding = result.get("encoding") or _DEFAULT_ENCODING
            confidence = result.get("confidence", 0.0)
            logger.debug(
                "Encoding detected",
                encoding=encoding,
                confidence=confidence,
            )
            return encoding
        except ImportError:
            logger.warning("chardet not installed — falling back to utf-8")
            return _DEFAULT_ENCODING
        except Exception as exc:
            logger.warning(
                "Encoding detection failed — falling back to utf-8",
                error=str(exc),
            )
            return _DEFAULT_ENCODING

    def load(self, source: str) -> list[Document]:
        """Load a text file and return a single-element document list.

        Args:
            source: File-system path to a ``.txt`` file.

        Returns:
            A list containing one :class:`Document`.

        Raises:
            FileNotFoundError: If *source* does not exist.
        """
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")

        logger.info("Loading text file", path=str(path))

        raw_bytes = path.read_bytes()
        encoding = self._detect_encoding(raw_bytes)

        try:
            content = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            logger.warning(
                "Decode with detected encoding failed — using utf-8 with replace",
                encoding=encoding,
                error=str(exc),
            )
            content = raw_bytes.decode(_DEFAULT_ENCODING, errors="replace")

        doc = Document(
            content=content,
            metadata={
                "encoding": encoding,
                "file_name": path.name,
                "file_size_bytes": len(raw_bytes),
            },
            source_path=str(path),
        )

        logger.info("Text file loaded", path=str(path), chars=len(content))
        return [doc]

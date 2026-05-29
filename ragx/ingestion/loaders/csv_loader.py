"""
RAGX CSV Loader — Load CSV / TSV files into documents using pandas.

Each row of the file is converted into a :class:`Document` whose content
is a key-value representation of the row.  Optional column selection and
row filtering are supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import BaseLoader, Document

logger = get_logger(__name__)


class CsvLoader(BaseLoader):
    """Loader for CSV and TSV files.

    Each row in the file is emitted as a separate :class:`Document`.

    Args:
        columns: Optional list of column names to include.  When *None*
            all columns are used.
        filter_column: Optional column name to filter on.
        filter_value: Value the *filter_column* must equal for a row to
            be included.
    """

    supported_extensions: ClassVar[list[str]] = [".csv", ".tsv"]

    def __init__(
        self,
        columns: Optional[list[str]] = None,
        filter_column: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> None:
        self.columns = columns
        self.filter_column = filter_column
        self.filter_value = filter_value

    def load(self, source: str) -> list[Document]:
        """Load a CSV/TSV file and return one document per row.

        Args:
            source: File-system path to a ``.csv`` or ``.tsv`` file.

        Returns:
            A list of :class:`Document` instances.

        Raises:
            FileNotFoundError: If *source* does not exist.
        """
        import pandas as pd  # type: ignore[import-untyped]

        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        logger.info("Loading CSV/TSV", path=str(path))

        separator = "\t" if path.suffix.lower() == ".tsv" else ","

        try:
            df = pd.read_csv(str(path), sep=separator, dtype=str, keep_default_na=False)
        except Exception as exc:
            logger.error("Failed to read CSV/TSV", path=str(path), error=str(exc))
            raise

        # ── Column selection ────────────────────────────────────────────
        if self.columns:
            missing = [c for c in self.columns if c not in df.columns]
            if missing:
                logger.warning("Requested columns not found — ignoring", missing=missing)
            valid_cols = [c for c in self.columns if c in df.columns]
            if valid_cols:
                df = df[valid_cols]

        # ── Row filtering ───────────────────────────────────────────────
        if self.filter_column and self.filter_value is not None:
            if self.filter_column in df.columns:
                df = df[df[self.filter_column] == self.filter_value]
                logger.debug(
                    "Rows after filter",
                    column=self.filter_column,
                    value=self.filter_value,
                    rows=len(df),
                )
            else:
                logger.warning(
                    "Filter column not found — skipping filter",
                    column=self.filter_column,
                )

        documents: list[Document] = []
        column_names = list(df.columns)

        for row_idx, row in df.iterrows():
            # Build a human-readable key: value representation
            parts = [f"{col}: {row[col]}" for col in column_names]
            content = "\n".join(parts)

            doc = Document(
                content=content,
                metadata={
                    "row_index": int(row_idx),  # type: ignore[arg-type]
                    "columns": column_names,
                    "file_name": path.name,
                },
                source_path=str(path),
            )
            documents.append(doc)

        logger.info("CSV/TSV loaded", path=str(path), rows=len(documents))
        return documents

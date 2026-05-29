"""
RAGX Text Preprocessor — Chainable text-cleaning pipeline.

Provides a set of composable cleaning methods that normalise raw
document text before it is chunked and embedded.  Each method is
idempotent and can be applied independently or as part of the full
:meth:`TextPreprocessor.process` pipeline.
"""

from __future__ import annotations

import re
import unicodedata

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)

# ── Pre-compiled patterns ──────────────────────────────────────────────────
# Common header / footer markers found in PDFs and scanned docs
_HEADER_FOOTER_PATTERNS: list[re.Pattern[str]] = [
    # Page numbers: "Page 1 of 10", "- 1 -", "1/10"
    re.compile(r"^[-–—]*\s*page\s+\d+\s*(of\s+\d+)?\s*[-–—]*$", re.IGNORECASE),
    re.compile(r"^[-–—]\s*\d+\s*[-–—]$"),
    re.compile(r"^\d+\s*/\s*\d+$"),
    # Confidential / proprietary notices
    re.compile(r"^\s*(confidential|proprietary|internal\s+use\s+only)\s*$", re.IGNORECASE),
    # "All rights reserved" lines
    re.compile(r"^.*all\s+rights\s+reserved.*$", re.IGNORECASE),
    # Copyright lines
    re.compile(r"^©\s*\d{4}.*$"),
    re.compile(r"^copyright\s+\d{4}.*$", re.IGNORECASE),
    # Draft watermarks
    re.compile(r"^\s*draft\s*$", re.IGNORECASE),
]

_MULTI_SPACES = re.compile(r"[ \t]+")
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_SPECIAL_CHARS = re.compile(r"[^\w\s.,;:!?'\"\-()/&@#%+=$\[\]{}|<>~`^\\]")

# Common Unicode replacement artifacts
_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u00a0": " ",       # non-breaking space
    "\u200b": "",        # zero-width space
    "\u200c": "",        # zero-width non-joiner
    "\u200d": "",        # zero-width joiner
    "\ufeff": "",        # BOM / zero-width no-break space
    "\u2018": "'",       # left single quote
    "\u2019": "'",       # right single quote
    "\u201c": '"',       # left double quote
    "\u201d": '"',       # right double quote
    "\u2013": "-",       # en-dash
    "\u2014": "-",       # em-dash
    "\u2026": "...",     # ellipsis
    "\u00ad": "",        # soft hyphen
    "\ufffd": "",        # replacement character
}


class TextPreprocessor:
    """Chainable text-cleaning pipeline for ingested documents.

    Each public method accepts a string and returns a cleaned string.
    Call :meth:`process` to run the full default pipeline in order.
    """

    def remove_headers_footers(self, text: str) -> str:
        """Remove common header and footer lines such as page numbers.

        Uses a set of regex patterns targeting frequent header/footer
        content found in PDFs and scanned documents.

        Args:
            text: Raw input text.

        Returns:
            Text with matched header/footer lines removed.
        """
        lines = text.splitlines()
        cleaned: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned.append(line)
                continue
            is_header_footer = any(pat.match(stripped) for pat in _HEADER_FOOTER_PATTERNS)
            if not is_header_footer:
                cleaned.append(line)

        result = "\n".join(cleaned)
        if len(result) != len(text):
            logger.debug(
                "Removed header/footer lines",
                removed_chars=len(text) - len(result),
            )
        return result

    def remove_duplicates(self, text: str) -> str:
        """Remove duplicate consecutive lines.

        Blank lines are preserved (only one instance of consecutive
        blanks), but non-blank lines that repeat adjacently are
        collapsed to a single occurrence.

        Args:
            text: Input text.

        Returns:
            Text with adjacent duplicate lines removed.
        """
        lines = text.splitlines()
        if not lines:
            return text

        deduped: list[str] = [lines[0]]
        for line in lines[1:]:
            if line.strip() == "" and deduped[-1].strip() == "":
                continue
            if line.strip() and line.strip() == deduped[-1].strip():
                continue
            deduped.append(line)

        result = "\n".join(deduped)
        if len(result) != len(text):
            logger.debug(
                "Removed duplicate lines",
                removed_chars=len(text) - len(result),
            )
        return result

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces, tabs, and excessive newlines.

        Runs of horizontal whitespace are reduced to a single space.
        Three or more consecutive newlines are reduced to two (one blank
        line).

        Args:
            text: Input text.

        Returns:
            Text with normalised whitespace.
        """
        text = _MULTI_SPACES.sub(" ", text)
        text = _MULTI_NEWLINES.sub("\n\n", text)
        return text.strip()

    def clean_unicode(self, text: str) -> str:
        """Fix common Unicode encoding artifacts.

        Replaces smart quotes, zero-width characters, non-breaking
        spaces, and other frequently problematic codepoints with their
        ASCII equivalents (or removes them entirely).  Also applies NFC
        normalisation.

        Args:
            text: Input text.

        Returns:
            Cleaned text with normalised Unicode.
        """
        for char, replacement in _UNICODE_REPLACEMENTS.items():
            text = text.replace(char, replacement)

        # NFC normalisation for composed characters
        text = unicodedata.normalize("NFC", text)
        return text

    def remove_special_characters(self, text: str) -> str:
        """Remove non-standard special characters.

        Keeps alphanumeric characters, common punctuation, and
        whitespace.  Less-common symbols are stripped.

        Args:
            text: Input text.

        Returns:
            Text with special characters removed.
        """
        return _SPECIAL_CHARS.sub("", text)

    def process(self, text: str) -> str:
        """Run the full cleaning pipeline in the recommended order.

        Pipeline order:
            1. clean_unicode
            2. remove_headers_footers
            3. remove_duplicates
            4. normalize_whitespace
            5. remove_special_characters

        Args:
            text: Raw input text.

        Returns:
            Fully cleaned text.
        """
        if not text:
            return text

        logger.debug("Starting text preprocessing", input_chars=len(text))

        text = self.clean_unicode(text)
        text = self.remove_headers_footers(text)
        text = self.remove_duplicates(text)
        text = self.normalize_whitespace(text)
        text = self.remove_special_characters(text)

        logger.debug("Text preprocessing complete", output_chars=len(text))
        return text

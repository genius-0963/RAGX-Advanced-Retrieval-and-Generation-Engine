"""
RAGX Web Loader — Fetch and parse web pages into documents.

Uses *requests* for HTTP and *BeautifulSoup* for HTML parsing.  Noise
elements (``<nav>``, ``<footer>``, ``<script>``, ``<style>``, etc.) are
stripped before text extraction.  Rate limiting is applied when fetching
multiple pages.
"""

from __future__ import annotations

import time
from typing import ClassVar, Optional
from urllib.parse import urlparse

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import BaseLoader, Document

logger = get_logger(__name__)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; RAGXBot/1.0; +https://github.com/ragx)"
)
_DEFAULT_TIMEOUT = 30  # seconds
_DEFAULT_DELAY = 1.0  # seconds between requests (rate-limiting)

# HTML elements that are generally noise for RAG content
_NOISE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"}


class WebLoader(BaseLoader):
    """Loader that fetches and parses web pages.

    Args:
        user_agent: Custom ``User-Agent`` header string.
        timeout: HTTP request timeout in seconds.
        delay: Minimum seconds to wait between consecutive requests
            (used in batch scenarios).
    """

    supported_extensions: ClassVar[list[str]] = []  # URL-based, not file-extension-based

    def __init__(
        self,
        user_agent: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        delay: float = _DEFAULT_DELAY,
    ) -> None:
        self.user_agent = user_agent or _DEFAULT_USER_AGENT
        self.timeout = timeout
        self.delay = delay
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Ensure at least ``self.delay`` seconds between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.delay:
            sleep_for = self.delay - elapsed
            logger.debug("Rate-limiting", sleep_seconds=round(sleep_for, 2))
            time.sleep(sleep_for)
        self._last_request_time = time.monotonic()

    def _extract_text(self, html: str) -> tuple[str, str]:
        """Parse *html* and return ``(title, body_text)``.

        Args:
            html: Raw HTML string.

        Returns:
            A tuple of the page title and the cleaned body text.
        """
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Remove noise elements
        for tag_name in _NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Try to find main content containers first
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"role": "main"})
            or soup.find("div", {"id": "content"})
            or soup.find("div", {"class": "content"})
        )

        target = main_content if main_content else soup.body if soup.body else soup

        # Get text with newline separators
        text = target.get_text(separator="\n", strip=True)

        return title, text

    def load(self, source: str) -> list[Document]:
        """Fetch a web page and return its content as a document.

        Args:
            source: A fully-qualified URL (``https://…``).

        Returns:
            A single-element list containing the page :class:`Document`.
        """
        import requests  # type: ignore[import-untyped]

        parsed = urlparse(source)
        if not parsed.scheme:
            source = f"https://{source}"
            parsed = urlparse(source)

        logger.info("Fetching web page", url=source)
        self._rate_limit()

        headers = {"User-Agent": self.user_agent}

        try:
            response = requests.get(source, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("Request timed out", url=source, timeout=self.timeout)
            raise
        except requests.exceptions.HTTPError as exc:
            logger.error(
                "HTTP error",
                url=source,
                status_code=getattr(exc.response, "status_code", None),
                error=str(exc),
            )
            raise
        except requests.exceptions.RequestException as exc:
            logger.error("Request failed", url=source, error=str(exc))
            raise

        title, text = self._extract_text(response.text)

        if not text.strip():
            logger.warning("No text extracted from page", url=source)

        doc = Document(
            content=text,
            metadata={
                "title": title,
                "url": source,
                "domain": parsed.netloc,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
            },
            source_path=source,
        )

        logger.info("Web page loaded", url=source, chars=len(text))
        return [doc]

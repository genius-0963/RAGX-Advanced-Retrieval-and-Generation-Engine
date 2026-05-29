"""
RAGX Conversation Memory — Session-based conversation history management.

Provides windowed conversation history with session isolation and
LangChain message format conversion.
"""

from __future__ import annotations

import uuid
from typing import Any

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    Manages conversation history with session-based isolation.

    Stores messages in-memory with a configurable sliding window,
    supporting multiple concurrent sessions.
    """

    # Class-level storage for cross-instance session sharing
    _sessions: dict[str, list[dict[str, str]]] = {}

    def __init__(
        self,
        window_size: int = 10,
        session_id: str | None = None,
    ) -> None:
        """
        Initialize conversation memory.

        Args:
            window_size: Maximum number of messages to retain (most recent).
            session_id: Session identifier. Auto-generated if None.
        """
        self.window_size = window_size
        self.session_id = session_id or str(uuid.uuid4())

        if self.session_id not in self._sessions:
            self._sessions[self.session_id] = []

    @property
    def _messages(self) -> list[dict[str, str]]:
        """Get the message list for this session."""
        return self._sessions[self.session_id]

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role ('user', 'assistant', 'system').
            content: Message content.
        """
        self._messages.append({"role": role, "content": content})

        # Trim to window size
        if len(self._messages) > self.window_size * 2:
            # Keep system messages + last window_size messages
            system_msgs = [m for m in self._messages if m["role"] == "system"]
            non_system = [m for m in self._messages if m["role"] != "system"]
            trimmed = non_system[-self.window_size * 2 :]
            self._sessions[self.session_id] = system_msgs + trimmed

        logger.debug(
            "message_added",
            session_id=self.session_id,
            role=role,
            history_length=len(self._messages),
        )

    def get_history(self) -> list[dict[str, str]]:
        """
        Get conversation history within the window.

        Returns:
            List of message dicts with 'role' and 'content'.
        """
        messages = self._messages
        if len(messages) <= self.window_size * 2:
            return list(messages)

        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        return system_msgs + non_system[-self.window_size * 2 :]

    def get_context_string(self) -> str:
        """
        Format conversation history as a string for prompt injection.

        Returns:
            Formatted conversation history string.
        """
        history = self.get_history()
        if not history:
            return ""

        lines: list[str] = []
        for msg in history:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    def to_langchain_messages(self) -> list[Any]:
        """
        Convert conversation history to LangChain message objects.

        Returns:
            List of LangChain message objects.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        history = self.get_history()
        lc_messages: list[Any] = []

        role_map = {
            "user": HumanMessage,
            "human": HumanMessage,
            "assistant": AIMessage,
            "ai": AIMessage,
            "system": SystemMessage,
        }

        for msg in history:
            message_class = role_map.get(msg["role"], HumanMessage)
            lc_messages.append(message_class(content=msg["content"]))

        return lc_messages

    def clear(self) -> None:
        """Clear all messages for this session."""
        self._sessions[self.session_id] = []
        logger.info("memory_cleared", session_id=self.session_id)

    @classmethod
    def clear_all_sessions(cls) -> None:
        """Clear all session data."""
        cls._sessions.clear()
        logger.info("all_sessions_cleared")

    @classmethod
    def list_sessions(cls) -> list[str]:
        """List all active session IDs."""
        return list(cls._sessions.keys())

    def __len__(self) -> int:
        """Return number of messages in current session."""
        return len(self._messages)

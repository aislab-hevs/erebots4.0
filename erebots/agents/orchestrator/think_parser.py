"""Streaming parser that splits ``<think>...</think>`` reasoning blocks
from the visible response text.

Robust to chunk boundaries: a tag split across chunks
(e.g. ``"<thi"`` + ``"nk>"``) is reassembled from an internal buffer.
"""

from __future__ import annotations


class ThinkSplitter:
    """Stateful splitter for streamed chunks.

    Yields a list of ``(kind, text)`` events for each chunk, where
    ``kind`` is either ``"think"`` (content inside
    ``<think>...</think>``) or ``"response"`` (everything else).
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        """Start outside a think block, with an empty carry-over buffer."""
        self._in_think = False
        self._buffer = ""

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        """Consume ``chunk``, emit any complete ``(kind, text)`` events."""
        events: list[tuple[str, str]] = []
        text = self._buffer + chunk
        self._buffer = ""
        i = 0
        while i < len(text):
            if self._in_think:
                end = text.find(self._CLOSE, i)
                if end == -1:
                    # Hold back the tail so a split close tag can still match.
                    safe = max(i, len(text) - len(self._CLOSE) + 1)
                    if safe > i:
                        events.append(("think", text[i:safe]))
                    self._buffer = text[safe:]
                    return events
                if end > i:
                    events.append(("think", text[i:end]))
                i = end + len(self._CLOSE)
                self._in_think = False
            else:
                start = text.find(self._OPEN, i)
                if start == -1:
                    safe = max(i, len(text) - len(self._OPEN) + 1)
                    if safe > i:
                        events.append(("response", text[i:safe]))
                    self._buffer = text[safe:]
                    return events
                if start > i:
                    events.append(("response", text[i:start]))
                i = start + len(self._OPEN)
                self._in_think = True
        return events

    def flush(self) -> list[tuple[str, str]]:
        """Emit any leftover buffered content at end of stream."""
        if not self._buffer:
            return []
        kind = "think" if self._in_think else "response"
        out = [(kind, self._buffer)]
        self._buffer = ""
        return out

"""Tests for ``chat_stream`` reasoning wrapping and reasoning splitting."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from erebots.agents.generative import GenerativeLayer
from examples.gradio_ui.chat import _split_reasoning


def _make_delta(content: str | None = None, reasoning: str | None = None):
    return SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=None)


def _make_chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _FakeStream:
    """Async iterable of OpenAI-like ``chunk.choices[0].delta`` events.

    ``block_after`` makes the stream hang on that many chunks, like a model
    that has gone quiet, so a test can cancel the consumer mid-stream.
    """

    def __init__(self, deltas, block_after: int | None = None):
        self._deltas = list(deltas)
        self._block_after = block_after
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._block_after is not None and self._block_after <= 0:
            await asyncio.Event().wait()
        if self._block_after is not None:
            self._block_after -= 1
        if not self._deltas:
            raise StopAsyncIteration
        return _make_chunk(self._deltas.pop(0))

    async def close(self):
        self.closed = True


def _generative() -> GenerativeLayer:
    return GenerativeLayer(
        api_key="x",
        api_base_url="http://example.invalid",
        model="m",
        system_prompt="",
    )


async def _collect(gen) -> str:
    out = []
    async for piece in gen:
        out.append(piece)
    return "".join(out)


async def test_chat_stream_passes_content_through():
    gl = _generative()
    gl.client.chat.completions.create = AsyncMock(
        return_value=_FakeStream([
            _make_delta(content="Hello "),
            _make_delta(content="world"),
        ])
    )
    out = await _collect(gl.chat_stream([{"role": "user", "content": "hi"}]))
    assert out == "Hello world"


async def test_chat_stream_wraps_reasoning_content_as_think_tags():
    gl = _generative()
    gl.client.chat.completions.create = AsyncMock(
        return_value=_FakeStream([
            _make_delta(reasoning="step1 "),
            _make_delta(reasoning="step2"),
            _make_delta(content="The answer."),
        ])
    )
    out = await _collect(gl.chat_stream([{"role": "user", "content": "hi"}]))
    assert out == "<think>step1 step2</think>The answer."


async def test_chat_stream_closes_think_at_stream_end_without_content():
    """If the stream ends inside reasoning, the close tag is still emitted."""
    gl = _generative()
    gl.client.chat.completions.create = AsyncMock(
        return_value=_FakeStream([
            _make_delta(reasoning="thinking..."),
        ])
    )
    out = await _collect(gl.chat_stream([{"role": "user", "content": "hi"}]))
    assert out == "<think>thinking...</think>"


def test_split_reasoning_separates_think_block():
    think, resp = _split_reasoning("<think>internal monologue</think>Final answer.")
    assert think == "internal monologue"
    assert resp == "Final answer."


def test_split_reasoning_no_think_returns_full_response():
    think, resp = _split_reasoning("Plain response with no tags.")
    assert think == ""
    assert resp == "Plain response with no tags."


def test_split_reasoning_handles_multiple_blocks():
    think, resp = _split_reasoning("<think>a</think>part1<think>b</think>part2")
    assert think == "ab"
    assert resp == "part1part2"


async def test_chat_stream_closes_the_http_stream_on_completion():
    stream = _FakeStream([_make_delta("a")])
    gen = _generative()
    gen.client.chat.completions.create = AsyncMock(return_value=stream)
    assert await _collect(gen.chat_stream([{"role": "user", "content": "x"}])) == "a"
    assert stream.closed


async def test_chat_stream_closes_the_http_stream_when_the_consumer_is_cancelled():
    """Cancelling the turn's task (what a per-user stop does) releases the connection."""
    stream = _FakeStream([_make_delta("a"), _make_delta("b")], block_after=1)
    gen = _generative()
    gen.client.chat.completions.create = AsyncMock(return_value=stream)
    got: list[str] = []

    async def consume():
        async for piece in gen.chat_stream([{"role": "user", "content": "x"}]):
            got.append(piece)

    task = asyncio.create_task(consume())
    while not got and not task.done():
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert got == ["a"]
    assert stream.closed


async def test_stream_with_tools_closes_the_http_stream_when_cancelled():
    stream = _FakeStream([_make_delta("a"), _make_delta("b")], block_after=1)
    gen = _generative()
    gen.client.chat.completions.create = AsyncMock(return_value=stream)
    got = []

    async def consume():
        async for item in gen.stream_with_tools([{"role": "user", "content": "x"}], tools=[]):
            got.append(item)

    task = asyncio.create_task(consume())
    while not got and not task.done():
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed

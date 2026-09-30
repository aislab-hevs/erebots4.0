"""Tests for the streaming ``ThinkSplitter``."""

from __future__ import annotations

from erebots.agents.orchestrator.think_parser import ThinkSplitter


def _collect(splitter: ThinkSplitter, chunks: list[str]) -> tuple[str, str]:
    """Drive ``splitter`` with the chunks, return ``(think, response)``."""
    think_parts: list[str] = []
    resp_parts: list[str] = []
    for c in chunks:
        for kind, text in splitter.feed(c):
            (think_parts if kind == "think" else resp_parts).append(text)
    for kind, text in splitter.flush():
        (think_parts if kind == "think" else resp_parts).append(text)
    return "".join(think_parts), "".join(resp_parts)


def test_pure_response_no_thinking():
    think, resp = _collect(ThinkSplitter(), ["Hello ", "world!"])
    assert think == ""
    assert resp == "Hello world!"


def test_single_thinking_block():
    chunks = ["<think>reasoning here</think>", "Final answer."]
    think, resp = _collect(ThinkSplitter(), chunks)
    assert think == "reasoning here"
    assert resp == "Final answer."


def test_open_tag_split_across_chunks():
    chunks = ["<thi", "nk>reasoning ", "split</thi", "nk>", "Answer."]
    think, resp = _collect(ThinkSplitter(), chunks)
    assert think == "reasoning split"
    assert resp == "Answer."


def test_thinking_content_split_across_chunks():
    chunks = ["<think>first ", "second ", "third</think>", "OK."]
    think, resp = _collect(ThinkSplitter(), chunks)
    assert think == "first second third"
    assert resp == "OK."


def test_multiple_thinking_blocks():
    chunks = ["<think>a</think>part1<think>b</think>part2"]
    think, resp = _collect(ThinkSplitter(), chunks)
    assert think == "ab"
    assert resp == "part1part2"


def test_unterminated_think_flushes_as_think():
    """A stream that ends inside ``<think>`` should keep the leftover."""
    chunks = ["<think>unfinished"]
    think, resp = _collect(ThinkSplitter(), chunks)
    assert think == "unfinished"
    assert resp == ""

"""Tests for the headless ``OrchestratorAgent.run()`` event stream.

``run()`` is UI-neutral: it yields :class:`AgentEvent` and is fully async, so it
can be driven without Gradio. These tests mock routing, the A2A client and the
generative stream to assert the event sequence for each path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from erebots.agents.orchestrator.agent import OrchestratorAgent


def _orch() -> OrchestratorAgent:
    return OrchestratorAgent(api_key="x", api_base_url="http://invalid", model="m")


async def _collect(agen) -> list:
    return [event async for event in agen]


async def test_run_routes_to_specialist():
    orch = _orch()
    orch.register_agent("nutrition", "http://x/a2a/nutrition", "Nutrition specialist.")
    orch.route_request = AsyncMock(return_value=["nutrition"])

    async def _stream(_target, _message, _context=None, task_id=None):
        for piece in ["eat ", "oats"]:
            yield piece

    orch.a2a_client.send_message_stream = _stream

    events = await _collect(orch.run("breakfast?", context={}))
    types = [e.type for e in events]

    assert types[0] == "routing"
    assert events[0].data["target"] == "nutrition"
    # the specialist answer is streamed token-by-token
    assert "".join(e.text for e in events if e.type == "token") == "eat oats"
    final = next(e for e in events if e.type == "final")
    assert final.text == "eat oats"
    assert final.agent == "nutrition"


async def test_run_fallback_streams_tokens():
    orch = _orch()
    orch.route_request = AsyncMock(return_value=[])

    async def _stream(_messages):
        for piece in ["Hel", "lo"]:
            yield piece

    orch.generative.chat_stream = _stream

    events = await _collect(orch.run("hi", context={}))
    tokens = [e.text for e in events if e.type == "token"]
    assert "".join(tokens) == "Hello"
    final = next(e for e in events if e.type == "final")
    assert final.text == "Hello"
    assert final.agent == "orchestrator"


async def test_run_separates_reasoning_from_visible_text():
    orch = _orch()
    orch.route_request = AsyncMock(return_value=[])

    async def _stream(_messages):
        for piece in ["<think>", "secret", "</think>", "answer"]:
            yield piece

    orch.generative.chat_stream = _stream

    events = await _collect(orch.run("hi", context={}))
    visible = "".join(e.text for e in events if e.type == "token")
    reasoning = "".join(e.text for e in events if e.type == "reasoning")
    assert visible == "answer"
    assert "secret" in reasoning


async def test_run_specialist_error_emits_error_event():
    orch = _orch()
    orch.route_request = AsyncMock(return_value=["nutrition"])

    async def _stream(_target, _message, _context=None, task_id=None):
        raise RuntimeError("boom")
        yield  # pragma: no cover — makes this an async generator

    orch.a2a_client.send_message_stream = _stream

    events = await _collect(orch.run("x", context={}))
    assert events[-1].type == "error"
    assert "boom" in events[-1].text


async def test_run_routing_failure_emits_error_event():
    """The routing call is the first LLM round-trip of every turn: if it dies,
    the stream must end with an error event, not propagate out of the
    generator."""
    orch = _orch()
    orch.route_request = AsyncMock(side_effect=RuntimeError("router down"))

    events = await _collect(orch.run("hi", context={}))

    assert [e.type for e in events] == ["error"]
    assert "router down" in events[0].text


async def test_process_skips_side_channel_tokens():
    """Fan-out contributions stream on the side channel and are then composed
    on the answer channel; collecting both would duplicate the content."""
    from erebots.runtime.events import AgentEvent

    orch = _orch()

    async def _run(_message, _context=None):
        yield AgentEvent("token", text="raw contribution ", agent="a", channel="side")
        yield AgentEvent("token", text="merged answer", agent="orchestrator")
        yield AgentEvent("final", text="merged answer", agent="orchestrator")

    orch.run = _run

    response = await orch.process("hi", context={})
    assert response.content == "merged answer"


class _FakeA2AClient:
    """Stands in for ``A2AClient``, recording whether anyone closed it."""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_an_injected_a2a_client_is_the_one_used():
    """The seam that makes a custom agent testable without a listening socket."""
    fake = _FakeA2AClient()

    orch = OrchestratorAgent(
        api_key="x", api_base_url="http://invalid", model="m", a2a_client=fake
    )

    assert orch.a2a_client is fake


async def test_an_injected_a2a_client_is_not_closed_by_the_orchestrator():
    """It belongs to whoever built it; they may still be using it."""
    fake = _FakeA2AClient()
    orch = OrchestratorAgent(
        api_key="x", api_base_url="http://invalid", model="m", a2a_client=fake
    )

    await orch.aclose()

    assert fake.closed is False


async def test_the_orchestrator_closes_the_client_it_built_itself():
    """The default path still has to release its connection pool on shutdown."""
    orch = _orch()
    orch.a2a_client = _FakeA2AClient()

    await orch.aclose()

    assert orch.a2a_client.closed is True

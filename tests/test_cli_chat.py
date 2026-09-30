"""Tests for the CLI frontend's event rendering (no LLM needed).

Proves the headless ``run()`` event stream drives a non-Gradio frontend: a fake
orchestrator yields ``AgentEvent``s and we assert the terminal rendering.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from erebots.runtime.conversation import ConversationService
from erebots.runtime.events import AgentEvent
from examples.cli_chat import chat_once


class _FakeOrchestrator:
    """Yields scripted events, with the real (store-less) conversation service."""

    name = "orchestrator"

    def __init__(self, events, store=None):
        self._events = events
        self.conversation = ConversationService(store)
        self.contexts: list[dict | None] = []

    async def run(self, message, context=None):
        self.contexts.append(context)
        for event in self._events:
            yield event


async def test_cli_renders_routing_and_specialist_final(capsys):
    orch = _FakeOrchestrator(
        [
            AgentEvent("routing", data={"target": "nutrition"}),
            AgentEvent("final", text="eat oats", agent="nutrition"),
        ]
    )
    await chat_once(orch, "breakfast?")
    out = capsys.readouterr().out
    assert "[routing → nutrition]" in out
    assert "eat oats" in out


async def test_cli_streams_fallback_tokens_without_double_printing(capsys):
    orch = _FakeOrchestrator(
        [
            AgentEvent("routing", data={"target": "none"}),
            AgentEvent("token", text="Hel", agent="orchestrator"),
            AgentEvent("token", text="lo", agent="orchestrator"),
            AgentEvent("final", text="Hello", agent="orchestrator"),
        ]
    )
    await chat_once(orch, "hi")
    out = capsys.readouterr().out
    # tokens stream to "Hello" exactly once; the orchestrator's own final
    # event must not print it again.
    assert out.count("Hello") == 1
    assert "[routing" not in out  # "none" → no routing banner


async def test_cli_reports_errors(capsys):
    orch = _FakeOrchestrator([AgentEvent("error", text="boom", agent="orchestrator")])
    await chat_once(orch, "x")
    assert "[error] boom" in capsys.readouterr().out


@pytest.mark.parametrize("agent", ["personal", "orchestrator"])
async def test_cli_renders_a_streamed_answer_once(agent, capsys):
    orch = _FakeOrchestrator([
        AgentEvent("token", text="Hel", agent=agent),
        AgentEvent("token", text="lo", agent=agent),
        AgentEvent("final", text="Hello", agent=agent),
    ])
    await chat_once(orch, "hi")
    assert capsys.readouterr().out == "bot> Hello\n"


async def test_cli_side_tokens_do_not_hide_a_final_only_answer(capsys):
    orch = _FakeOrchestrator([
        AgentEvent("token", text="contribution", agent="personal", channel="side"),
        AgentEvent("token", text="", agent="orchestrator"),
        AgentEvent("final", text="Combined answer", agent="orchestrator"),
    ])
    await chat_once(orch, "hi")
    assert capsys.readouterr().out == "bot> Combined answer\n"


async def test_cli_turn_carries_identity_and_the_users_stored_history(capsys):
    """The CLI goes through the same conversation service as the HTTP adapter."""
    store = SimpleNamespace(
        get_or_create=AsyncMock(
            return_value=SimpleNamespace(
                chat_history=[{"role": "user", "content": "earlier"}],
                username="alice",
                profile={},
                consents={},
            )
        ),
    )
    orch = _FakeOrchestrator(
        [AgentEvent("final", text="hello alice", agent="orchestrator")], store=store
    )

    pending, answer = await chat_once(orch, "hi", session={"user_id": "alice"})

    context = orch.contexts[0]
    assert context["user_id"] == "alice"
    assert context["user_context"] == "User: alice"
    assert context["chat_history"] == [{"role": "user", "content": "earlier"}]
    # The answer comes back so the caller can persist the turn.
    assert (pending, answer) == (None, "hello alice")
    assert "hello alice" in capsys.readouterr().out


async def test_cli_anonymous_turn_sends_no_identity():
    orch = _FakeOrchestrator([AgentEvent("final", text="hi", agent="orchestrator")])
    await chat_once(orch, "hi")
    assert orch.contexts == [{}]


async def test_cli_pending_handle_rides_along_with_the_context():
    """Resuming a parked task must not drop the turn's identity and history."""
    store = SimpleNamespace(
        get_or_create=AsyncMock(
            return_value=SimpleNamespace(
                chat_history=[], username="alice", profile={}, consents={}
            )
        ),
    )
    orch = _FakeOrchestrator([AgentEvent("final", text="ok", agent="personal")], store=store)

    await chat_once(
        orch, "Italian", pending={"agent": "personal", "task_id": "t-1"},
        session={"user_id": "alice"},
    )

    context = orch.contexts[0]
    assert context["pending"] == {"agent": "personal", "task_id": "t-1"}
    assert context["user_id"] == "alice"

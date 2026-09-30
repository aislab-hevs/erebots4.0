"""Tests for the Gradio adapter's per-turn stop.

The stop button used to bump a process-wide counter on the orchestrator's
generative layer, which cut every user's stream at once and left delegated
A2A work running. It now cancels this session's turn task (Gradio ``cancels``)
and settles the chat afterwards; these tests cover the adapter's two halves of
that: the turn unwinds cleanly when cancelled, and the settle handler leaves a
coherent, persisted history behind.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from erebots.runtime.conversation import ConversationService
from erebots.runtime.events import AgentEvent
from examples.gradio_ui.chat import GradioChat


class _Orchestrator:
    """Fake orchestrator whose run() streams one token then hangs until cancelled."""

    name = "orchestrator"

    def __init__(self):
        self.conversation = ConversationService(None)
        self.conversation.persist_history = AsyncMock()
        self.generative = MagicMock()
        self.unwound = False

    async def run(self, message, context=None):
        try:
            yield AgentEvent("token", text="partial", agent="orchestrator")
            await asyncio.Event().wait()
        finally:
            self.unwound = True


async def test_cancelling_the_turn_unwinds_the_orchestrator_stream_without_a_global_stop():
    orch = _Orchestrator()
    chat = GradioChat(orch)
    frames = []

    async def consume():
        async for messages in chat.chat([{"role": "user", "content": "hi"}], {"user_id": "u1"}):
            frames.append([dict(m) for m in messages])

    task = asyncio.create_task(consume())
    while not task.done() and not any(m.get("content") == "partial" for f in frames for m in f):
        await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert orch.unwound
    orch.generative.stop.assert_not_called()
    orch.conversation.persist_history.assert_not_called()


async def test_stopped_settles_pending_bubbles_and_persists_what_arrived():
    orch = _Orchestrator()
    chat = GradioChat(orch)
    interrupted = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "metadata": {"title": "Routed to `x`", "status": "done"},
        },
        {
            "role": "assistant",
            "content": "",
            "metadata": {"title": "Streaming response...", "status": "pending"},
        },
        {"role": "assistant", "content": "partial answ"},
    ]

    settled = await chat.stopped(interrupted, {"user_id": "u1"})

    statuses = [m["metadata"]["status"] for m in settled if "metadata" in m]
    assert "pending" not in statuses
    assert settled[2]["metadata"]["title"].startswith("Stopped")
    assert settled[3] == {"role": "assistant", "content": "partial answ"}
    assert settled[-1]["metadata"]["title"].startswith("Stopped")
    orch.conversation.persist_history.assert_awaited_once()
    persisted = orch.conversation.persist_history.await_args.args[1]
    assert persisted is settled


async def test_stopped_on_an_empty_chat_is_harmless():
    orch = _Orchestrator()
    settled = await GradioChat(orch).stopped(None, None)
    assert len(settled) == 1


# --- the transcript this frontend owns has to start out whole ----------------


async def test_the_chatbot_is_seeded_from_the_stored_conversation_on_load():
    """This frontend owns its transcript: a turn posts the chatbot's messages
    and an edit, a retry or a clear *replaces* what is stored. That is only
    safe if the chatbot opens holding the whole conversation — otherwise a
    reload starts from an empty box and the first turn replaces the user's
    history with just that turn.
    """
    from types import SimpleNamespace

    stored = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "reply"},
    ]
    store = SimpleNamespace(
        get_or_create=AsyncMock(
            return_value=SimpleNamespace(
                chat_history=list(stored), username="alice", profile={}, consents={}
            )
        ),
        set_chat_history=AsyncMock(),
    )
    orch = _Orchestrator()
    orch.conversation = ConversationService(store)
    chat = GradioChat(orch)

    update = await chat.restore_history({"user_id": "alice"})

    assert update["value"] == stored
    assert update["height"] == "65vh", "an empty-height chatbot would hide the conversation"


async def test_an_anonymous_load_leaves_the_chatbot_untouched():
    orch = _Orchestrator()
    orch.conversation = ConversationService(None)
    update = await GradioChat(orch).restore_history({})
    assert "value" not in update or update["value"] is None


def test_the_profile_menu_is_absent_without_authentication(monkeypatch):
    """No Keycloak, no identity menu.

    With ``enable_auth`` off there is no account to open and no session to end,
    and both menu entries pointed at a Keycloak that is not running: clicking
    "User profile" gave the browser ERR_CONNECTION_REFUSED. A control that
    cannot work is worse than no control, because the reader has to find out by
    trying it.
    """
    from dataclasses import replace

    from erebots.config import settings
    from examples.gradio_ui import profile_menu as menu

    # Settings is frozen, so the object is replaced rather than a field set.
    monkeypatch.setattr(menu, "settings", replace(settings, enable_auth=False))
    assert menu.profile_menu() is None


def test_the_account_link_uses_the_browser_facing_keycloak_url(monkeypatch):
    """The link a browser follows has to be the URL a browser can reach.

    ``keycloak_url`` is what the *server* uses; behind a proxy it is a container
    address that resolves to nothing on the user's machine. Every other
    browser-facing URL in the framework goes through ``get_keycloak_public_url``,
    and this one did not.
    """
    from pathlib import Path

    source = Path("examples/gradio_ui/profile_menu.py").read_text(encoding="utf-8")
    account_line = next(
        line for line in source.splitlines() if "/account" in line and "f\"" in line
    )
    assert "get_keycloak_public_url()" in account_line, account_line
    assert "settings.keycloak_url" not in account_line, account_line

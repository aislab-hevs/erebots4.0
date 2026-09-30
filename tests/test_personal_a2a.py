"""Tests for the PersonalAgent A2A integration.

PersonalAgent mirrors NutritionAgent: skills are MCP tools advertised
on the A2A agent-card; ``user_id`` is always overridden server-side
before the MCP call so the LLM cannot mutate other users' records.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.applications import Starlette

from erebots.agents.a2a.server import _build_card, create_a2a_app
from specialists.personal import PersonalAgent


@pytest.fixture
def personal() -> PersonalAgent:
    return PersonalAgent(
        api_key="x",
        api_base_url="http://example.invalid",
        model="m",
        mcp_server_urls=[],
    )


def test_personal_registers_three_skills(personal: PersonalAgent):
    names = {s.name for s in personal.skills.list_skills()}
    assert names == {"get_user_context", "update_profile", "manage_consent"}


def test_personal_a2a_card_advertises_skills(personal: PersonalAgent):
    card = _build_card(personal, "http://example.invalid/a2a/personal", "1.0.0")
    advertised = {s.name for s in card.skills}
    assert advertised == {"get_user_context", "update_profile", "manage_consent"}


def test_personal_a2a_card_name_and_description(personal: PersonalAgent):
    card = _build_card(personal, "http://example.invalid/a2a/personal", "1.0.0")
    assert card.name == "personal"
    assert "personal" in card.description
    # The card advertises a JSON-RPC interface at the given external URL.
    assert card.supported_interfaces[0].url == "http://example.invalid/a2a/personal"


def test_create_a2a_app_returns_mountable_starlette_app(personal: PersonalAgent):
    app = create_a2a_app(personal, url="http://example.invalid/a2a/personal")
    assert isinstance(app, Starlette)


def _completion(content: str = "", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {"role": "assistant", "content": content}
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


async def test_personal_process_returns_llm_response_without_tools(personal: PersonalAgent):
    """When the LLM emits no tool_calls, ``process`` returns the message text."""
    personal.mcp_client.get_openai_tools = lambda: []
    personal.generative.chat = AsyncMock(return_value="hello user")

    response = await personal.process(
        "What's my favorite food?",
        context={"user_id": "u-1"},
    )

    assert response.content == "hello user"
    assert response.agent_name == "personal"
    personal.generative.chat.assert_awaited()


async def test_personal_process_overrides_user_id_in_tool_call(personal: PersonalAgent):
    """The LLM-provided ``user_id`` must be overwritten by the trusted one."""
    # Force tools to be present so the loop dispatches to MCP.
    personal.mcp_client.get_openai_tools = lambda: [{"type": "function", "function": {"name": "update_profile"}}]
    personal.mcp_client.call_tool = AsyncMock(return_value="Profile updated for elia: {'name': 'Elia'}")

    tc = MagicMock()
    tc.id = "call-1"
    tc.function.name = "update_profile"
    # LLM tries to set a DIFFERENT user_id; the agent must override it.
    tc.function.arguments = '{"user_id": "attacker", "updates": {"name": "Elia"}}'
    first = _completion(tool_calls=[tc])
    second = _completion("OK, ricorderò")

    personal.generative.chat = AsyncMock(side_effect=[first, second])

    response = await personal.process(
        "Mi chiamo Elia",
        context={"user_id": "trusted-id"},
    )

    assert response.content == "OK, ricorderò"
    personal.mcp_client.call_tool.assert_awaited_once()
    called_args = personal.mcp_client.call_tool.await_args.args
    # call_tool(name, args)
    assert called_args[0] == "update_profile"
    # user_id must be the TRUSTED one, not the LLM-provided "attacker".
    assert called_args[1]["user_id"] == "trusted-id"
    assert called_args[1]["updates"] == {"name": "Elia"}


async def test_the_profile_store_owns_exactly_two_columns(tmp_path):
    """The personal agent's store names the columns it owns, and only those.

    The framework offers a column-merge primitive and does not know what a
    profile is; this class is where "profile" and "consents" are decided to be
    this agent's. If a second agent ever needs its own column, it adds its own
    store beside this one rather than a method to the framework's.
    """
    from erebots.persistence import Base, UserContextStore, make_engine, make_session_factory
    from specialists.personal import ProfileStore

    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'p.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        store = ProfileStore(UserContextStore(make_session_factory(engine)))

        await store.update_profile("u1", {"allergies": ["peanuts"]})
        await store.update_consents("u1", {"analytics": True})
        ctx = await store.read("u1")

        assert ctx.profile == {"allergies": ["peanuts"]}
        assert ctx.consents == {"analytics": True}
        # Untouched by either write: the conversation is not this agent's.
        assert ctx.chat_history == []
    finally:
        await engine.dispose()

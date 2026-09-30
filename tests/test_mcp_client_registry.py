"""Tests for MCPClient tool-registry isolation.

Regression guard for the shared-global-registry bug: each MCPClient must own
its registry so one agent's discovered tools never leak into another's.
"""

from __future__ import annotations

import pytest

from erebots.agents.mcp.client import MCPClient, MCPToolError


def test_each_client_has_an_independent_registry():
    a = MCPClient()
    b = MCPClient()

    a.register_tool(name="search_recipes", url="http://x/tools/recipes")

    assert [t["name"] for t in a.list_tools()] == ["search_recipes"]
    assert b.list_tools() == []  # b must not see a's tools


def test_openai_tools_are_not_shared_between_clients():
    a = MCPClient()
    b = MCPClient()
    a.register_tool(name="update_profile", url="http://x/tools/personal")

    assert {t["function"]["name"] for t in a.get_openai_tools()} == {"update_profile"}
    assert b.get_openai_tools() == []


def test_registering_a_tool_makes_it_visible_to_the_model():
    """`register_tool` and `get_openai_tools` are both public; the path between must be too.

    Found by building an agent against the installed wheel using only the
    documented API: a registered tool left `get_openai_tools` empty, because
    the list was rebuilt by a private method that `BaseSpecialist` happened to
    call and an outside caller had no reason to know existed. An empty tool
    list is also silent — the agent simply never calls a tool.
    """
    client = MCPClient()

    client.register_tool(
        name="forecast",
        url="http://x/tools/weather/mcp",
        description="Forecast for a city.",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )

    exposed = client.get_openai_tools()

    assert [t["function"]["name"] for t in exposed] == ["forecast"]
    assert exposed[0]["function"]["description"] == "Forecast for a city."
    assert exposed[0]["function"]["parameters"]["properties"] == {"city": {"type": "string"}}



class _FakeSession:
    """Stands in for ``mcp.ClientSession`` — only what ``call_tool`` touches.

    It opens a real anyio task group, because the real ``ClientSession`` does.
    That detail is not decoration: an exception raised *inside* the ``async
    with`` escapes as an ``ExceptionGroup`` wrapping it, so a caller catching
    ``MCPToolError`` never sees one. A fake without the task group passes
    happily against a client that is broken in production.
    """

    def __init__(self, result):
        self._result = result
        self._task_group = None

    async def __aenter__(self):
        import anyio

        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._task_group.__aexit__(*exc)

    async def initialize(self):
        return None

    async def call_tool(self, name, arguments):
        return self._result


def _fake_transport(monkeypatch, result):
    """Point the client at a canned CallToolResult instead of a socket.

    The suite is offline; ``streamablehttp_client`` opens a real connection.
    The result itself is a genuine ``mcp.types.CallToolResult``, so what is
    under test is our branch on it, not a mock of ourselves.
    """
    from contextlib import asynccontextmanager

    from erebots.agents.mcp import client as mod

    @asynccontextmanager
    async def fake_streamablehttp_client(url, headers=None):
        yield (None, None, None)

    monkeypatch.setattr(mod, "streamablehttp_client", fake_streamablehttp_client)
    monkeypatch.setattr(mod, "ClientSession", lambda read, write: _FakeSession(result))


def _result(text: str | None, *, is_error: bool):
    from mcp.types import CallToolResult, TextContent

    content = [] if text is None else [TextContent(type="text", text=text)]
    return CallToolResult(content=content, isError=is_error)


async def test_a_failing_tool_raises_instead_of_returning_its_error_text(monkeypatch):
    """An error and an answer used to be the same thing: a string.

    ``call_tool`` concatenated the text parts and returned them whatever
    ``isError`` said, so a tool that refused its arguments came back looking
    exactly like one that had answered. Nothing was logged and the trace
    recorded it as a result.
    """
    client = MCPClient()
    client.register_tool(name="forecast", url="http://x/tools/weather/mcp")
    _fake_transport(monkeypatch, _result("days must be between 1 and 7, got 99", is_error=True))

    with pytest.raises(MCPToolError) as caught:
        await client.call_tool("forecast", {"city": "Sion", "days": 99})

    # The server's own message, unchanged: it goes to the model as the
    # observation for this tool call.
    assert str(caught.value) == "days must be between 1 and 7, got 99"
    assert caught.value.tool == "forecast"


async def test_a_successful_call_still_returns_its_text(monkeypatch):
    """Guards the other direction: the check must not raise on every call."""
    client = MCPClient()
    client.register_tool(name="forecast", url="http://x/tools/weather/mcp")
    _fake_transport(monkeypatch, _result("Sion: 21C, clear (3d)", is_error=False))

    assert await client.call_tool("forecast", {"city": "Sion"}) == "Sion: 21C, clear (3d)"


async def test_a_failure_with_no_message_still_raises(monkeypatch):
    """``content`` is not guaranteed to be populated on an error result."""
    client = MCPClient()
    client.register_tool(name="forecast", url="http://x/tools/weather/mcp")
    _fake_transport(monkeypatch, _result(None, is_error=True))

    with pytest.raises(MCPToolError) as caught:
        await client.call_tool("forecast", {})

    assert "forecast" in str(caught.value)

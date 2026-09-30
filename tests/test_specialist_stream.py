"""BaseSpecialist.process_stream: tool loop, trace markers and exhaustion.

Uses stub generative/MCP layers — no network, no LLM.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from erebots.agents.mcp.client import MCPToolError
from erebots.agents.models import TraceStep
from erebots.agents.specialist import BaseSpecialist
from erebots.config import settings

_TOOL_CALL = {"id": "c1", "name": "calc", "args": "{}"}


class _StubGenerative:
    """Replays one scripted (kind, value) sequence per LLM round."""

    def __init__(self, rounds: list[list[tuple]]):
        self._rounds = list(rounds)
        # Mirrors the real GenerativeLayer: the trace reads these after each call.
        self.last_completion_tokens = 0
        self.last_prompt_tokens = 0

    async def stream_with_tools(self, messages, tools=None, system_prompt=None):
        for item in self._rounds.pop(0):
            yield item


class _StubMCP:
    def get_openai_tools(self):
        return [{"type": "function", "function": {"name": "calc"}}]

    async def call_tool(self, name, args):
        return "42"


class _StubSpecialist(BaseSpecialist):
    def setup_skills(self):
        pass


def _specialist(rounds: list[list[tuple]]) -> BaseSpecialist:
    sp = _StubSpecialist(
        agent_name="stub",
        api_key="EMPTY",
        api_base_url="http://x/v1",
        model="local",
        system_prompt="s",
        description="stub specialist",
    )
    sp.generative = _StubGenerative(rounds)
    sp.mcp_client = _StubMCP()

    async def _noop():
        return None

    sp._ensure_tools_discovered = _noop
    return sp


async def _collect(sp):
    text, steps = "", []
    async for item in sp.process_stream("hi", {}):
        if isinstance(item, TraceStep):
            steps.append(item)
        else:
            text += item
    return text, steps


async def test_stream_tool_round_then_answer_emits_full_trace():
    sp = _specialist([
        [("tools", [_TOOL_CALL])],            # round 1: model requests a tool
        [("text", "answer"), ("tools", None)],  # round 2: model answers
    ])
    text, steps = await _collect(sp)

    assert text == "answer"
    assert [s.action for s in steps] == [
        "processing", "tool_call", "tool_result", "response",
    ]


async def test_stream_tool_loop_exhaustion_yields_fallback_not_silence():
    """If the model requests tools every round, the stream must not end empty."""
    rounds = [[("tools", [_TOOL_CALL])] for _ in range(settings.tool_loop_max_iterations)]
    sp = _specialist(rounds)

    text, steps = await _collect(sp)

    assert "Unable to complete" in text
    assert steps[-1].action == "response"
    assert "exhausted" in steps[-1].detail


# --------------------------------------------------------------------------
# Tool-call arguments and the ask_user path
# --------------------------------------------------------------------------


class _RecordingGenerative(_StubGenerative):
    """Remembers the ``tools`` argument of each round."""

    def __init__(self, rounds):
        super().__init__(rounds)
        self.seen_tools: list = []

    async def stream_with_tools(self, messages, tools=None, system_prompt=None):
        self.seen_tools.append(tools)
        for item in self._rounds.pop(0):
            yield item


class _NoToolsMCP:
    def get_openai_tools(self):
        return []

    async def call_tool(self, name, args):  # pragma: no cover - never reached
        raise AssertionError("no tools to call")


async def test_an_agent_without_tools_sends_no_tools_field():
    """ask_user must not conjure a tools array where there was none.

    Some backends reject ``tools`` outright; an agent exposing no tool used to
    work on those and has to keep working.
    """
    sp = _specialist([[("text", "ciao")]])
    sp.generative = _RecordingGenerative([[("text", "ciao")]])
    sp.mcp_client = _NoToolsMCP()

    await _collect(sp)

    assert sp.generative.seen_tools == [None]


async def test_an_agent_with_tools_is_also_offered_ask_user():
    sp = _specialist([[("text", "ciao")]])
    sp.generative = _RecordingGenerative([[("text", "ciao")]])

    await _collect(sp)

    names = [t["function"]["name"] for t in sp.generative.seen_tools[0]]
    assert names == ["calc", "ask_user"]


async def test_ask_user_with_non_object_arguments_still_parks():
    """``"null"`` is valid JSON but not an object.

    Parsing it and calling ``.get`` raises, and the executor turns that into a
    failed task — the opposite of the pending question the model asked for.
    """
    from erebots.agents.models import InputRequired

    sp = _specialist([[("tools", [{"id": "c1", "name": "ask_user", "args": "null"}])]])

    parked = [i async for i in sp.process_stream("hi", {}) if isinstance(i, InputRequired)]

    assert len(parked) == 1
    assert parked[0].question


async def test_a_tool_call_with_non_object_arguments_still_runs():
    """Same degradation for an ordinary tool: no arguments, not a crash."""
    seen: list = []

    class _Recording(_StubMCP):
        async def call_tool(self, name, args):
            seen.append(args)
            return "42"

    sp = _specialist(
        [
            [("tools", [{"id": "c1", "name": "calc", "args": '"nope"'}])],
            [("text", "fatto")],
        ]
    )
    sp.mcp_client = _Recording()

    text, _ = await _collect(sp)

    assert seen == [{}]
    assert "fatto" in text


async def test_a_failing_tool_becomes_an_observation_and_the_turn_continues(caplog):
    """A tool that refuses must not end the turn, and must be visible.

    The model needs the failure as an observation so it can correct itself or
    explain, and an operator needs it in the log and the trace. Before
    ``MCPToolError`` existed the text arrived at the model but nothing else
    happened: no warning, and a trace step marked ``tool_result``.
    """
    sp = _specialist([
        [("tools", [_TOOL_CALL])],
        [("text", "sorry, 99 days is too far ahead"), ("tools", None)],
    ])

    class _FailingMCP(_StubMCP):
        async def call_tool(self, name, args):
            raise MCPToolError(name, "days must be between 1 and 7, got 99")

    sp.mcp_client = _FailingMCP()

    with caplog.at_level(logging.WARNING):
        text, steps = await _collect(sp)

    assert text == "sorry, 99 days is too far ahead"
    assert [s.action for s in steps] == ["processing", "tool_call", "tool_error", "response"]
    # The tool's own message, not a re-wrapped one: the model sees it unchanged.
    assert "days must be between 1 and 7, got 99" in next(
        s.detail for s in steps if s.action == "tool_error"
    )
    assert "reported failure" in caplog.text


async def test_an_unreachable_tool_is_reported_differently_from_a_refusing_one(caplog):
    """Transport failures keep their own wording, and say which tool."""
    sp = _specialist([
        [("tools", [_TOOL_CALL])],
        [("text", "the tool is down"), ("tools", None)],
    ])

    class _UnreachableMCP(_StubMCP):
        async def call_tool(self, name, args):
            raise ConnectionError("connection refused")

    sp.mcp_client = _UnreachableMCP()

    with caplog.at_level(logging.WARNING):
        _, steps = await _collect(sp)

    failure = next(s for s in steps if s.action == "tool_error")
    assert "Error calling tool calc: connection refused" in failure.detail
    assert "MCP tool call failed" in caplog.text


# --- prepare_tool_args, on both tool loops ----------------------------------
#
# The hook is documented as the place to enforce values the LLM must not
# control, `user_id` being the example in its own docstring. Until now the only
# thing asserting it was applied lived in the example app's tests, tied to
# agents that override it -- so a deployment shipping other agents would have
# lost the evidence that the framework calls the hook at all.
#
# `process` and `process_stream` each call it at their own site, and a test of
# one says nothing about the other: a mutation of the non-streaming line
# survived the streaming test untouched, which is why there are two below.


class _RecordingMCP:
    """Publishes one tool and records the arguments it is actually called with."""

    def __init__(self, seen: dict):
        self._seen = seen

    def get_openai_tools(self):
        return [{"type": "function", "function": {"name": "calc"}}]

    async def call_tool(self, name, args):
        self._seen.update(args)
        return "42"


class _PinningSpecialist(_StubSpecialist):
    """Overrides the hook the way an agent owning a user record has to."""

    def prepare_tool_args(self, tool_args: dict, context: dict | None) -> dict:
        return {**tool_args, "user_id": (context or {}).get("user_id", "anonymous")}


def _pinning_specialist(generative, seen: dict) -> BaseSpecialist:
    sp = _PinningSpecialist(
        agent_name="pinning",
        api_key="EMPTY",
        api_base_url="http://x/v1",
        model="local",
        system_prompt="s",
        description="stub specialist",
    )
    sp.generative = generative
    sp.mcp_client = _RecordingMCP(seen)

    async def _noop():
        return None

    sp._ensure_tools_discovered = _noop
    return sp


async def test_the_streaming_tool_loop_applies_prepare_tool_args():
    """The value the model supplied never reaches the tool."""
    seen: dict = {}
    sp = _pinning_specialist(
        _StubGenerative([
            [("tools", [{"id": "c1", "name": "calc", "args": '{"user_id": "attacker"}'}])],
            [("text", "done"), ("tools", None)],
        ]),
        seen,
    )

    async for _ in sp.process_stream("hi", {"user_id": "u-real"}):
        pass

    assert seen["user_id"] == "u-real", "the model's user_id reached the tool"


async def test_the_non_streaming_tool_loop_applies_prepare_tool_args_too():
    """The same guarantee on the path a ``message/send`` turn takes."""
    seen: dict = {}

    class _ToolCall:
        def __init__(self):
            self.function = SimpleNamespace(name="calc", arguments='{"user_id": "attacker"}')
            self.id = "c1"

    class _Message:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls
            self.content = "" if tool_calls else "done"

        def model_dump(self):
            return {"role": "assistant", "content": self.content}

    class _ChatGenerative:
        def __init__(self):
            self._rounds = [[_ToolCall()], None]
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0

        async def chat(self, messages, tools=None):
            return SimpleNamespace(choices=[SimpleNamespace(message=_Message(self._rounds.pop(0)))])

    sp = _pinning_specialist(_ChatGenerative(), seen)

    await sp.process("hi", {"user_id": "u-real"})

    assert seen["user_id"] == "u-real", "the model's user_id reached the tool"

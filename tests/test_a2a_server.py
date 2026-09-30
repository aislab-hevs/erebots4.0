"""Round-trip tests for the a2a-sdk-based A2A server (mounted on a sub-path).

Uses an in-process httpx ASGITransport so no network/server is needed, and the
official a2a client to exercise the real JSON-RPC + agent-card path exactly as a
remote caller (and the orchestrator) would.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
from a2a import types as T
from a2a.client import ClientConfig, ClientFactory
from starlette.applications import Starlette
from starlette.routing import Mount

from erebots.agents.a2a.server import create_a2a_app
from erebots.agents.models import AgentResponse, TraceStep
from erebots.agents.skill import Skill, SkillRegistry

MOUNT = "/a2a/foo"
BASE = "http://test"


class _DummyAgent:
    """Minimal agent: records the input/context and returns a fixed response."""

    name = "foo"
    description = "foo specialist"

    def __init__(self):
        self.skills = SkillRegistry()
        self.skills.register(
            Skill(name="do_foo", description="does foo")
        )
        self.seen: list[tuple] = []

    async def process(self, message, context=None) -> AgentResponse:
        self.seen.append((message, context))
        return AgentResponse(
            content=f"handled: {message}",
            agent_name=self.name,
            trace=[TraceStep(agent=self.name, action="processing", detail="d")],
        )


def _mounted_app(agent) -> Starlette:
    a2a_app = create_a2a_app(agent, url=f"{BASE}{MOUNT}")
    return Starlette(routes=[Mount(MOUNT, app=a2a_app)])


async def _client(http: httpx.AsyncClient):
    factory = ClientFactory(ClientConfig(httpx_client=http, streaming=False))
    return await factory.create_from_url(f"{BASE}{MOUNT}")


def _collect(resp, texts, traces):
    task = getattr(resp, "task", None)
    if task and task.artifacts:
        from google.protobuf.json_format import MessageToDict

        for art in task.artifacts:
            for p in art.parts:
                if p.text:
                    texts.append(p.text)
            meta = MessageToDict(art, preserving_proto_field_name=True).get("metadata", {})
            if "trace" in meta:
                traces.append(meta["trace"])


async def test_round_trip_returns_response_and_trace():
    agent = _DummyAgent()
    app = _mounted_app(agent)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        client = await _client(http)
        msg = T.Message(message_id="m1", role=T.Role.values()[1], parts=[T.Part(text="ping")])
        texts: list[str] = []
        traces: list = []
        async for resp in client.send_message(T.SendMessageRequest(message=msg)):
            _collect(resp, texts, traces)

    assert any("handled: ping" in t for t in texts)
    assert traces and traces[0][0]["agent"] == "foo"
    assert agent.seen and agent.seen[0][0] == "ping"


async def test_context_is_forwarded_to_the_agent():
    """Context stashed in message metadata reaches agent.process (e.g. user_id)."""
    agent = _DummyAgent()
    app = _mounted_app(agent)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        client = await _client(http)
        msg = T.Message(
            message_id="m2",
            role=T.Role.values()[1],
            parts=[T.Part(text="hi")],
            metadata={"context": {"user_id": "u-42"}},
        )
        async for _ in client.send_message(T.SendMessageRequest(message=msg)):
            pass

    assert agent.seen[0][1].get("user_id") == "u-42"


async def test_well_known_card_exposes_fields_the_loader_reads():
    """The agent-card endpoint must expose name/description/skills (loader contract)."""
    agent = _DummyAgent()
    app = _mounted_app(agent)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        r = await http.get(f"{MOUNT}/.well-known/agent-card.json")
    card = r.json()
    assert r.status_code == 200
    assert card["name"] == "foo"
    assert card["description"] == "foo specialist"
    assert [s["name"] for s in card["skills"]] == ["do_foo"]
    urls = {i["protocolBinding"]: i["url"] for i in card["supportedInterfaces"]}
    # JSON-RPC is advertised with a trailing slash: its route lives at "/" inside
    # the mount and clients POST the card URL verbatim (a bare "<mount>" would
    # need a slash-redirect that a root-mounted UI swallows). REST appends its
    # own path to the URL, so that one is the bare mount.
    assert urls["JSONRPC"].endswith("/")
    assert urls["HTTP+JSON"] == urls["JSONRPC"].rstrip("/")


class _StreamingAgent(_DummyAgent):
    """Agent that streams its answer in chunks via ``process_stream``."""

    async def process_stream(self, message, context=None):
        self.seen.append((message, context))
        for chunk in ["Hel", "lo ", "wor", "ld"]:
            yield chunk


async def test_a2a_client_streams_specialist_chunks():
    """A2AClient.send_message_stream yields the answer incrementally over SSE."""
    from erebots.agents.a2a.client import A2AClient

    agent = _StreamingAgent()
    app = _mounted_app(agent)
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    chunks = [c async for c in client.send_message_stream("foo", "hi", context={"user_id": "u"})]

    assert len(chunks) >= 2  # arrived incrementally, not as one blob
    assert "".join(chunks) == "Hello world"
    assert agent.seen[0][1].get("user_id") == "u"


class _TracingStreamingAgent(_DummyAgent):
    """Streams text interleaved with TraceStep markers (like BaseSpecialist)."""

    async def process_stream(self, message, context=None):
        yield TraceStep(agent=self.name, action="tool_call", detail="Calling MCP tool: bmi")
        yield "Hel"
        yield TraceStep(agent=self.name, action="tool_result", detail="Tool bmi: 24.7")
        yield "lo"


async def test_stream_carries_specialist_trace_steps():
    """TraceStep markers survive the A2A stream, interleaved with the text."""
    from erebots.agents.a2a.client import A2AClient

    agent = _TracingStreamingAgent()
    app = _mounted_app(agent)
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    chunks = [c async for c in client.send_message_stream("foo", "hi", context={})]

    steps = [c for c in chunks if isinstance(c, TraceStep)]
    text = "".join(c for c in chunks if isinstance(c, str))
    assert text == "Hello"
    assert [(s.action, s.agent) for s in steps] == [("tool_call", "foo"), ("tool_result", "foo")]
    assert steps[0].detail == "Calling MCP tool: bmi"


async def test_send_message_reassembles_a_streamed_answer():
    """The non-streaming path returns the text whole, not one token per line.

    A streaming specialist emits one artifact chunk per token, so the task the
    client reads back holds "Hel", "lo ", "wor", "ld" as four parts. Joining
    those with a newline handed every caller that did not stream a reply broken
    into one fragment per line.
    """
    from erebots.agents.a2a.client import A2AClient

    app = _mounted_app(_StreamingAgent())
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    resp = await client.send_message("foo", "hi", context={})

    assert resp.content == "Hello world"


async def test_send_message_keeps_trace_markers_out_of_the_text():
    """A streaming specialist's trace reaches a non-streaming caller too.

    Two things are asserted because two things were broken. The markers travel
    as chunks whose only part is empty text, so the answer's parts here are
    "", "Hel", "", "lo" and still have to join into one word. And the steps
    themselves used to vanish: each chunk wrote them under the same metadata
    key, which the SDK merges per key, so only the last one was left in the
    stored task. The trace was absent here while the streaming path showed it
    in full - same call, same return type, a difference the caller could not
    see, because whether the remote streams is invisible over A2A.
    """
    from erebots.agents.a2a.client import A2AClient

    app = _mounted_app(_TracingStreamingAgent())
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    resp = await client.send_message("foo", "hi", context={})

    assert resp.content == "Hello"
    assert [(s.agent, s.action) for s in resp.trace] == [
        ("foo", "tool_call"),
        ("foo", "tool_result"),
    ]
    assert resp.trace[1].detail == "Tool bmi: 24.7"


async def test_client_pools_connection_and_fetches_card_once():
    """Two turns to the same agent → one agent-card fetch (client is pooled)."""
    from erebots.agents.a2a.client import A2AClient

    agent = _StreamingAgent()
    inner = _mounted_app(agent)
    card_fetches = 0

    async def counting_app(scope, receive, send):
        nonlocal card_fetches
        if scope["type"] == "http" and scope["path"].endswith("agent-card.json"):
            card_fetches += 1
        await inner(scope, receive, send)

    client = A2AClient(transport=httpx.ASGITransport(app=counting_app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    first = "".join([c async for c in client.send_message_stream("foo", "a", context={}) if isinstance(c, str)])
    second = "".join([c async for c in client.send_message_stream("foo", "b", context={}) if isinstance(c, str)])
    await client.aclose()

    assert first == second == "Hello world"
    assert card_fetches == 1


async def test_database_task_store_backs_a2a(tmp_path):
    """create_a2a_app works with a persistent DatabaseTaskStore (sqlite)."""
    from a2a.server.tasks import DatabaseTaskStore
    from sqlalchemy.ext.asyncio import create_async_engine

    from erebots.agents.a2a.client import A2AClient

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}")
    store = DatabaseTaskStore(engine)
    await store.initialize()

    agent = _DummyAgent()
    a2a_app = create_a2a_app(agent, url=f"{BASE}{MOUNT}", task_store=store)
    app = Starlette(routes=[Mount(MOUNT, app=a2a_app)])
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    resp = await client.send_message("foo", "ping", context={})
    assert resp.content == "handled: ping"
    await engine.dispose()


async def test_a2a_client_wrapper_round_trip():
    """The orchestrator-facing A2AClient wrapper round-trips through the server."""
    from erebots.agents.a2a.client import A2AClient

    agent = _DummyAgent()
    app = _mounted_app(agent)
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    resp = await client.send_message("foo", "ping", context={"user_id": "u-9"})

    assert resp.content == "handled: ping"
    assert resp.agent_name == "foo"
    assert resp.trace and resp.trace[0].agent == "foo"
    assert agent.seen[0][1].get("user_id") == "u-9"


# --- wire format: A2A mandates camelCase and RFC 3339 (DM-SERIAL-001/003) ----


def test_trace_is_serialized_in_camel_case_with_an_rfc3339_timestamp():
    from erebots.agents.a2a.server import _serialize_trace
    from erebots.agents.models import TraceStep

    step = TraceStep(
        agent="nutrition", action="response", detail="d",
        parent_id="p1", elapsed_ms=12.5, tokens=7, prompt_tokens=1200,
    )
    item = _serialize_trace([step])[0]

    assert not [k for k in item if "_" in k], f"snake_case leaked: {sorted(item)}"
    assert {"stepId", "parentId", "elapsedMs", "promptTokens"} <= item.keys()
    assert isinstance(item["timestamp"], str)
    assert item["timestamp"].endswith("Z")


def test_client_reads_back_what_the_server_writes():
    """Round-trip: the two sides must agree on the wire names."""
    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.a2a.server import _serialize_trace
    from erebots.agents.models import TraceStep

    original = TraceStep(
        agent="nutrition", action="a2a_call", detail="d",
        parent_id="p1", elapsed_ms=12.5, tokens=7, prompt_tokens=1200,
    )
    parsed = A2AClient()._parse_trace(_serialize_trace([original]))[0]

    assert (parsed.agent, parsed.action, parsed.detail) == ("nutrition", "a2a_call", "d")
    assert (parsed.step_id, parsed.parent_id) == (original.step_id, "p1")
    assert (parsed.elapsed_ms, parsed.tokens, parsed.prompt_tokens) == (12.5, 7, 1200)
    assert abs(parsed.timestamp - original.timestamp) < 1.0


def test_client_still_reads_a_legacy_snake_case_peer():
    """An older build on the other side must not become unreadable."""
    from erebots.agents.a2a.client import A2AClient

    legacy = [{
        "agent": "nutrition", "action": "response", "detail": "d",
        "timestamp": 1_760_000_000.0, "step_id": "s1", "parent_id": "p1",
        "elapsed_ms": 12.5, "tokens": 7, "prompt_tokens": 1200,
    }]
    parsed = A2AClient()._parse_trace(legacy)[0]

    assert (parsed.step_id, parsed.parent_id) == ("s1", "p1")
    assert (parsed.elapsed_ms, parsed.prompt_tokens) == (12.5, 1200)
    assert parsed.timestamp == 1_760_000_000.0


# --- card caching: CARD-CACHE-001/002/003 ------------------------------------


async def test_the_card_is_served_with_caching_headers():
    """Cache-Control with max-age, ETag and Last-Modified, as the spec asks."""
    app = _mounted_app(_DummyAgent())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        resp = await http.get(f"{MOUNT}/.well-known/agent-card.json")

    assert resp.status_code == 200
    assert "max-age" in resp.headers.get("cache-control", "")
    assert resp.headers.get("etag")
    assert resp.headers.get("last-modified")
    assert resp.json()["name"] == "foo", "the body must be the card, untouched"


async def test_a_matching_etag_revalidates_to_304():
    """The ETag exists to be sent back: a match must answer 304 with no body."""
    app = _mounted_app(_DummyAgent())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        first = await http.get(f"{MOUNT}/.well-known/agent-card.json")
        second = await http.get(
            f"{MOUNT}/.well-known/agent-card.json",
            headers={"If-None-Match": first.headers["etag"]},
        )
        stale = await http.get(
            f"{MOUNT}/.well-known/agent-card.json",
            headers={"If-None-Match": '"something-else"'},
        )

    assert second.status_code == 304
    assert not second.content
    assert second.headers.get("etag") == first.headers["etag"]
    assert stale.status_code == 200, "a non-matching ETag must get the full card"


async def test_other_routes_are_not_touched_by_card_caching():
    """The middleware must buffer only the card: a POST turn works as before
    and carries none of the caching headers."""
    agent = _DummyAgent()
    app = _mounted_app(agent)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        client = await _client(http)
        msg = T.Message(message_id="m1", role=T.Role.values()[1], parts=[T.Part(text="ping")])
        responses = [r async for r in client.send_message(T.SendMessageRequest(message=msg))]
        rest = await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "hi"}],
                    "messageId": "m2",
                }
            },
            headers={"A2A-Version": "1.0"},
        )

    assert responses, "the JSON-RPC turn must still answer"
    assert rest.status_code == 200
    assert "etag" not in rest.headers


# --- input-required: a question parks the task instead of ending it ---------


class _AskingAgent(_DummyAgent):
    """A specialist that asks instead of answering."""

    def __init__(self, question="Quanti anni hai?"):
        super().__init__()
        self._question = question

    async def process_stream(self, message, context=None):
        from erebots.agents.models import InputRequired

        yield TraceStep(agent=self.name, action="processing", detail="thinking")
        yield InputRequired(question=self._question)


async def _task_states(agent) -> list[str]:
    """Drive one turn through the real A2A stack and return the task states seen."""
    app = _mounted_app(agent)
    states: list[str] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        client = await _client(http)
        msg = T.Message(message_id="m1", role=T.Role.values()[1], parts=[T.Part(text="ciao")])
        async for resp in client.send_message(T.SendMessageRequest(message=msg)):
            task = getattr(resp, "task", None)
            if task is not None:
                # protobuf enums arrive as ints; name them so failures read well
                states.append(T.TaskState.Name(task.status.state))
    return states


async def test_a_parked_question_carries_its_trace_to_a_non_streaming_caller():
    """The steps that led to a question survive the park.

    A task in input-required is read back like any other by ``send_message``,
    so the trace has to be flushed before the executor returns the question -
    not only on the path that completes.
    """
    from erebots.agents.a2a.client import A2AClient

    app = _mounted_app(_AskingAgent())
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    resp = await client.send_message("foo", "ciao", context={})

    assert resp.content == "Quanti anni hai?"
    assert [s.action for s in resp.trace] == ["processing"]


async def test_a_question_parks_the_task_in_input_required():
    """The task must stay open: that is what makes the answer resumable."""
    states = await _task_states(_AskingAgent())
    assert states, "no task in the response"
    assert "INPUT_REQUIRED" in states[-1], f"expected input-required, got {states[-1]}"


async def test_a_normal_answer_still_completes_the_task():
    """The ordinary path must be untouched: no question, no parking."""
    states = await _task_states(_StreamingAgent())
    assert "COMPLETED" in states[-1], f"expected completed, got {states[-1]}"


# --- resume: the user's answer returns to the same task, same agent ----------


class _ResumableAgent(_DummyAgent):
    """Asks on its first turn, answers on the second — the ask_user shape."""

    async def process_stream(self, message, context=None):
        from erebots.agents.models import InputRequired

        self.seen.append((message, context))
        if len(self.seen) == 1:
            yield InputRequired(question="Quanti anni hai?")
        else:
            yield f"Fabbisogno per {message} anni: 2000 kcal"


async def test_the_answer_resumes_the_parked_task():
    """The full loop: question parks the task, the answer reopens *that* task.

    This is the round-trip that could not be tested before there was an API to
    express it: same client, same task id, two turns, one task.
    """
    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.models import InputRequired

    agent = _ResumableAgent()
    app = _mounted_app(agent)
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    first = [c async for c in client.send_message_stream("foo", "quante calorie?")]
    parked = [c for c in first if isinstance(c, InputRequired)]
    assert parked and parked[0].task_id, "the question must carry a task handle"
    task_id = parked[0].task_id
    assert await client.can_resume("foo", task_id)

    second = [
        c async for c in client.send_message_stream("foo", "38", task_id=task_id)
    ]
    text = "".join(c for c in second if isinstance(c, str))

    assert "38" in text, "the resumed turn must answer with the user's datum"
    assert not any(isinstance(c, InputRequired) for c in second)
    assert len(agent.seen) == 2, "both turns must reach the same agent"
    # The task left input-required, so the handle is now spent.
    assert not await client.can_resume("foo", task_id)


async def test_can_resume_is_false_for_an_unknown_task_or_agent():
    """A stale handle must mean "route normally", never an exception."""
    from erebots.agents.a2a.client import A2AClient

    app = _mounted_app(_ResumableAgent())
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    assert not await client.can_resume("foo", "no-such-task")
    assert not await client.can_resume("never-registered", "t-1")
    assert not await client.can_resume("foo", "")


async def test_the_answer_resumes_the_parked_task_without_streaming():
    """``message/send`` must speak the same resume dialect as ``message/stream``:
    a caller that cannot stream still needs to answer a parked question."""
    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.models import InputRequired

    agent = _ResumableAgent()
    app = _mounted_app(agent)
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    first = [c async for c in client.send_message_stream("foo", "quante calorie?")]
    task_id = next(c for c in first if isinstance(c, InputRequired)).task_id

    resp = await client.send_message("foo", "38", task_id=task_id)

    assert "38" in resp.content
    assert len(agent.seen) == 2, "the answer must reach the same agent"
    assert not await client.can_resume("foo", task_id)


# --- resume ownership: an answer only counts from who was asked --------------


async def test_a_parked_task_only_resumes_for_its_owner():
    """The resume handle travels through the frontend, so a forged one could
    answer — and thereby act on — someone else's question. The task remembers
    who opened it; anyone else is told the task does not exist, and the real
    owner can still answer afterwards.
    """
    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.models import InputRequired

    agent = _ResumableAgent()
    app = _mounted_app(agent)
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    first = [
        c
        async for c in client.send_message_stream(
            "foo", "quante calorie?", context={"user_id": "alice"}
        )
    ]
    task_id = next(c for c in first if isinstance(c, InputRequired)).task_id

    # The pre-flight already says no to anyone but alice.
    assert not await client.can_resume("foo", task_id, user_id="mallory")
    assert not await client.can_resume("foo", task_id, user_id=None)
    assert await client.can_resume("foo", task_id, user_id="alice")

    # A caller that skips the pre-flight is stopped by the server itself.
    try:
        async for _ in client.send_message_stream(
            "foo", "48", context={"user_id": "mallory"}, task_id=task_id
        ):
            pass
    except Exception:
        pass  # the refusal may surface as a raised protocol error — fine
    assert len(agent.seen) == 1, "the forged answer must never reach the agent"
    assert await client.can_resume("foo", task_id, user_id="alice"), (
        "the probe must leave the task intact for its owner"
    )

    second = [
        c
        async for c in client.send_message_stream(
            "foo", "38", context={"user_id": "alice"}, task_id=task_id
        )
    ]
    text = "".join(c for c in second if isinstance(c, str))
    assert "38" in text, "the real owner's answer must still work"


# --- expiry across restarts: the timestamp outlives the timer ----------------


async def test_a_parked_task_expires_on_resume_even_without_its_timer(monkeypatch):
    """Restart scenario: the in-memory timer is gone, the parked task is not.

    Both sides must judge by the park timestamp instead — ``can_resume`` says
    no, and a resume that arrives anyway cancels the task rather than letting a
    long-dead question be answered.
    """
    import asyncio
    from dataclasses import replace

    from erebots.agents.a2a import client as client_mod
    from erebots.agents.a2a import server as server_mod
    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.a2a.server import _AgentExecutor
    from erebots.agents.models import InputRequired

    # A restart in two touches: no timer is ever armed, and the TTL is tiny.
    monkeypatch.setattr(
        _AgentExecutor, "_schedule_expiry", lambda self, task_id, ctx: None
    )
    tiny = replace(server_mod.settings, a2a_parked_task_ttl=0.05)
    monkeypatch.setattr(server_mod, "settings", tiny)
    monkeypatch.setattr(client_mod, "settings", tiny)

    agent = _ResumableAgent()
    app = _mounted_app(agent)
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    first = [c async for c in client.send_message_stream("foo", "quante calorie?")]
    task_id = next(c for c in first if isinstance(c, InputRequired)).task_id
    await asyncio.sleep(0.1)  # past the TTL, with no timer left to enforce it

    assert not await client.can_resume("foo", task_id)

    # Straight to the task, skipping the pre-flight: the task is canceled and
    # the send is refused with the SDK's own terminal-task error — the same
    # outcome as if the lost timer had fired.
    answered: list = []
    try:
        async for c in client.send_message_stream("foo", "38", task_id=task_id):
            answered.append(c)
    except Exception:
        pass  # the refusal surfaces as a protocol error — fine
    assert not any(isinstance(c, str) and c for c in answered)
    assert len(agent.seen) == 1, "an expired question must not be answerable"

    # Regression: the store now says canceled, but the SDK's in-process task
    # machinery still remembers the task as parked — a further resume used to
    # be answered from that memory. It must stay refused.
    try:
        async for c in client.send_message_stream("foo", "39", task_id=task_id):
            answered.append(c)
    except Exception:
        pass
    assert len(agent.seen) == 1, "a canceled task must stay dead in-process too"


# --- expiry: a question nobody answers must not leave a task open forever ----


def _parked_task(state=None) -> T.Task:
    return T.Task(
        id="t-parked",
        context_id="c1",
        status=T.TaskStatus(
            state=state if state is not None else T.TaskState.TASK_STATE_INPUT_REQUIRED
        ),
    )


async def test_an_unanswered_parked_task_expires_to_canceled():
    from a2a.server.context import ServerCallContext
    from a2a.server.tasks import InMemoryTaskStore

    from erebots.agents.a2a.server import _AgentExecutor

    store, ctx = InMemoryTaskStore(), ServerCallContext()
    await store.save(_parked_task(), ctx)
    executor = _AgentExecutor(_DummyAgent(), task_store=store)

    await executor._expire("t-parked", ctx, ttl=0)

    saved = await store.get("t-parked", ctx)
    assert saved.status.state == T.TaskState.TASK_STATE_CANCELED


async def test_expiry_leaves_a_task_that_moved_on_alone():
    """Answered (or resumed) before the deadline: the timer must be a no-op."""
    from a2a.server.context import ServerCallContext
    from a2a.server.tasks import InMemoryTaskStore

    from erebots.agents.a2a.server import _AgentExecutor

    store, ctx = InMemoryTaskStore(), ServerCallContext()
    await store.save(_parked_task(T.TaskState.TASK_STATE_COMPLETED), ctx)
    executor = _AgentExecutor(_DummyAgent(), task_store=store)

    await executor._expire("t-parked", ctx, ttl=0)

    saved = await store.get("t-parked", ctx)
    assert saved.status.state == T.TaskState.TASK_STATE_COMPLETED


async def test_a_second_question_survives_the_first_questions_timer():
    """ask → answer → ask again on the same task: the first timer fires while
    the *second* question is parked, and must leave it to its own timer.

    Regression: ``_expire`` used to check only the state, so any re-parked
    question inherited the previous question's (shorter) deadline.
    """
    import asyncio

    from a2a.server.context import ServerCallContext
    from a2a.server.tasks import InMemoryTaskStore

    from erebots.agents.a2a.server import _AgentExecutor

    store, ctx = InMemoryTaskStore(), ServerCallContext()
    first = _parked_task()
    first.status.timestamp.GetCurrentTime()  # first question parked "now"
    await store.save(first, ctx)
    executor = _AgentExecutor(_DummyAgent(), task_store=store)
    timer1 = asyncio.create_task(executor._expire("t-parked", ctx, ttl=0.3))

    await asyncio.sleep(0.05)  # the user answers; the agent asks again:
    second = await store.get("t-parked", ctx)
    second.status.state = T.TaskState.TASK_STATE_INPUT_REQUIRED
    second.status.timestamp.GetCurrentTime()  # re-parked, fresh deadline
    await store.save(second, ctx)

    await timer1

    saved = await store.get("t-parked", ctx)
    assert saved.status.state == T.TaskState.TASK_STATE_INPUT_REQUIRED, (
        "the second question was cancelled by the first question's timer"
    )


async def test_no_expiry_timer_when_disabled(monkeypatch):
    from dataclasses import replace

    from a2a.server.context import ServerCallContext
    from a2a.server.tasks import InMemoryTaskStore

    from erebots.agents.a2a import server as server_module
    from erebots.agents.a2a.server import _AgentExecutor

    monkeypatch.setattr(
        server_module, "settings", replace(server_module.settings, a2a_parked_task_ttl=0)
    )
    executor = _AgentExecutor(_DummyAgent(), task_store=InMemoryTaskStore())

    executor._schedule_expiry("t-parked", ServerCallContext())

    assert not executor._expiries


async def test_parking_arms_an_expiry_timer(monkeypatch):
    from dataclasses import replace

    from a2a.server.context import ServerCallContext
    from a2a.server.tasks import InMemoryTaskStore

    from erebots.agents.a2a import server as server_module
    from erebots.agents.a2a.server import _AgentExecutor

    monkeypatch.setattr(
        server_module, "settings", replace(server_module.settings, a2a_parked_task_ttl=60)
    )
    executor = _AgentExecutor(_DummyAgent(), task_store=InMemoryTaskStore())

    executor._schedule_expiry("t-parked", ServerCallContext())

    assert len(executor._expiries) == 1
    next(iter(executor._expiries)).cancel()


# --------------------------------------------------------------------------
# Content-type guard (JSONRPC-SSE-002)
# --------------------------------------------------------------------------


async def _post_raw(headers: dict[str, str]) -> dict:
    """POST a valid JSON-RPC body with caller-chosen headers; return the payload."""
    app = _mounted_app(_DummyAgent())
    body = (
        '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":'
        '{"role":"ROLE_USER","parts":[{"text":"hi"}],"messageId":"m1"}}}'
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        resp = await http.post(f"{MOUNT}/", content=body.encode(), headers=headers)
    return resp.json()


async def test_non_json_content_type_is_reported_as_such():
    """A wrong content type must not surface as a parse error.

    -32700 tells the caller its JSON is malformed; the actual problem is the
    type it declared, which is -32005.
    """
    body = await _post_raw({"Content-Type": "text/plain", "A2A-Version": "1.0"})

    assert body["error"]["code"] == -32005
    reasons = [d.get("reason") for d in body["error"]["data"]]
    assert "CONTENT_TYPE_NOT_SUPPORTED" in reasons


async def test_json_content_type_with_parameters_is_accepted():
    """``application/json; charset=utf-8`` is still JSON — the guard must not bite."""
    body = await _post_raw(
        {"Content-Type": "application/json; charset=utf-8", "A2A-Version": "1.0"}
    )

    assert "error" not in body, body


async def test_a_request_declaring_no_content_type_is_left_alone():
    """Nothing was declared, so there is no unsupported type to report."""
    body = await _post_raw({"A2A-Version": "1.0"})

    assert body.get("error", {}).get("code") != -32005


# --------------------------------------------------------------------------
# HTTP+JSON (REST) binding
# --------------------------------------------------------------------------

_REST_BODY = {
    "message": {
        "role": "ROLE_USER",
        "parts": [{"text": "hi"}],
        "messageId": "m1",
    }
}


@asynccontextmanager
async def _rest_session(app=None):
    """A client bound to one mounted app, so successive calls share its task store."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app or _mounted_app(_DummyAgent())),
        base_url=BASE,
        follow_redirects=True,
        headers={"A2A-Version": "1.0"},
    ) as http:
        yield http


async def _rest(method: str, path: str, **kwargs) -> httpx.Response:
    """One-shot call to a REST route on a fresh app."""
    async with _rest_session() as http:
        return await http.request(method, f"{MOUNT}{path}", **kwargs)


async def test_the_card_declares_both_bindings_it_serves():
    """BIND-FIELD-001: what the card declares is what the server answers.

    Declaring a binding it does not serve is worse than serving one it does not
    declare — a client picks the transport from this list.
    """
    app = _mounted_app(_DummyAgent())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        card = (await http.get(f"{MOUNT}/.well-known/agent-card.json")).json()

    bindings = [i["protocolBinding"] for i in card["supportedInterfaces"]]
    assert bindings == ["JSONRPC", "HTTP+JSON"]
    assert "GRPC" not in bindings  # no gRPC server exists; declaring it would lie


async def test_rest_send_message_answers_like_jsonrpc():
    """Both bindings run the same handler, so they must return the same answer."""
    r = await _rest("POST", "/message:send", json=_REST_BODY)

    assert r.status_code == 200
    artifacts = r.json()["task"]["artifacts"]
    assert artifacts[0]["parts"][0]["text"] == "handled: hi"


async def test_rest_get_task_returns_the_task_just_created():
    """The GET route resolves too, not only the POST ones."""
    async with _rest_session() as http:
        created = await http.post(f"{MOUNT}/message:send", json=_REST_BODY)
        task_id = created.json()["task"]["id"]

        r = await http.get(f"{MOUNT}/tasks/{task_id}")

    assert r.status_code == 200
    assert r.json()["id"] == task_id


async def test_rest_reports_a_wrong_content_type_as_415():
    """REST states the error in its HTTP status, where JSON-RPC uses a body code.

    The spec's error table maps ContentTypeNotSupportedError to 415; the SDK's
    own mapping says 400, so this pins the spec's value.
    """
    r = await _rest(
        "POST",
        "/message:send",
        content=json.dumps(_REST_BODY).encode(),
        headers={"Content-Type": "text/plain"},
    )

    assert r.status_code == 415
    error = r.json()["error"]
    assert error["code"] == 415
    reasons = [d.get("reason") for d in error["details"]]
    assert "CONTENT_TYPE_NOT_SUPPORTED" in reasons


async def test_an_extended_card_is_served_and_declared_when_configured():
    """The capability follows the configuration: given a card for authenticated
    callers, the public card declares it and ``GET /extendedAgentCard`` serves
    exactly that card, not the public one."""
    extended = T.AgentCard(
        name="foo",
        description="what authenticated callers may also see",
        version="1.0.0",
        skills=[T.AgentSkill(id="secret", name="secret", description="s")],
    )
    app = Starlette(
        routes=[
            Mount(
                MOUNT,
                app=create_a2a_app(
                    _DummyAgent(), url=f"{BASE}{MOUNT}", extended_card=extended
                ),
            )
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        public = (await http.get(f"{MOUNT}/.well-known/agent-card.json")).json()
        served = (
            await http.get(
                f"{MOUNT}/extendedAgentCard", headers={"A2A-Version": "1.0"}
            )
        ).json()

    assert public["capabilities"]["extendedAgentCard"] is True
    assert [s["id"] for s in served["skills"]] == ["secret"]


async def test_without_an_extended_card_the_capability_stays_undeclared():
    """Declaring a capability the server cannot honor is worse than omitting
    it — a caller would get errors from an endpoint the card promised."""
    async with _rest_session() as http:
        public = (await http.get(f"{MOUNT}/.well-known/agent-card.json")).json()
        refused = await http.get(
            f"{MOUNT}/extendedAgentCard", headers={"A2A-Version": "1.0"}
        )

    assert not public["capabilities"].get("extendedAgentCard")
    assert refused.status_code >= 400
    assert "error" in refused.json()


async def test_the_agent_card_survives_the_rest_tenant_mount():
    """Regression: the REST routes end with a ``/{tenant}`` mount that matches any
    single leading segment. Registered before the card route it swallows
    ``/.well-known/...`` and answers 404 from inside — the card must win."""
    r = await _rest("GET", "/.well-known/agent-card.json")

    assert r.status_code == 200
    assert r.json()["name"] == "foo"


# --------------------------------------------------------------------------
# Artifact metadata: wire names and placement
# --------------------------------------------------------------------------


class _AskingFirstAgent(_DummyAgent):
    """Asks as its very first chunk — the typical case when the model calls
    ``ask_user`` straight away, and the one that exposes first/append handling."""

    async def process_stream(self, message, context=None):
        from erebots.agents.models import InputRequired

        yield InputRequired(question="Quanto pesi?")


async def _first_artifact_metadata(agent) -> dict:
    """Metadata of the first artifact update seen on the wire."""
    from google.protobuf.json_format import MessageToDict

    app = _mounted_app(agent)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        factory = ClientFactory(ClientConfig(httpx_client=http, streaming=False))
        client = await factory.create_from_url(f"{BASE}{MOUNT}")
        msg = T.Message(message_id="m1", role=T.Role.ROLE_USER, parts=[T.Part(text="hi")])
        async for resp in client.send_message(T.SendMessageRequest(message=msg)):
            task = getattr(resp, "task", None)
            if task and task.artifacts:
                return MessageToDict(task.artifacts[0]).get("metadata", {})
    return {}


async def test_a_question_asked_first_still_names_its_agent():
    """The agent name rides on the *first* artifact, questions included.

    The two branches had opposite conditions, so a stream that opened with a
    question shipped an unnamed artifact and re-attached the name later.
    """
    meta = await _first_artifact_metadata(_AskingFirstAgent())

    assert meta.get("agentName") == "foo"


async def test_a_trace_step_first_still_names_its_agent():
    """Sibling of the test above, and the case that actually ships.

    ``BaseSpecialist.process_stream`` always opens with a trace step, so this
    branch decides the name for every real specialist — and it was the one
    branch that did not attach it. The caller then fell back to whatever name
    it had registered the agent under, which is its own alias, not the
    responder's name.
    """
    meta = await _first_artifact_metadata(_TracingStreamingAgent())

    assert meta.get("agentName") == "foo"


async def test_non_streaming_metadata_uses_wire_names():
    """DM-SERIAL-001 covers the keys we add under ``metadata`` too.

    ``_DummyAgent`` has no ``process_stream``, so this is the non-streaming
    branch — which kept emitting snake_case while the streaming one did not.
    """
    meta = await _first_artifact_metadata(_DummyAgent())

    assert "agentName" in meta and "agent_name" not in meta
    assert "trace" in meta


# --- resume context: the specialist gets the exchange back from the task -----


async def test_a_resumed_turn_gets_the_question_back_without_client_history():
    """The thin-client case: answer + task id, no transcript echoed back.

    Nothing in the request carries what was asked, so the specialist must read
    the exchange off the task or it answers its own question blind.
    """
    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.models import InputRequired

    agent = _ResumableAgent()
    client = A2AClient(transport=httpx.ASGITransport(app=_mounted_app(agent)))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    first = [
        c
        async for c in client.send_message_stream(
            "foo", "quante calorie?", context={"user_id": "alice"}
        )
    ]
    task_id = next(c for c in first if isinstance(c, InputRequired)).task_id

    # Identity only — deliberately no chat_history in the context.
    async for _ in client.send_message_stream(
        "foo", "38", context={"user_id": "alice"}, task_id=task_id
    ):
        pass
    await client.aclose()

    _, resumed_context = agent.seen[1]
    assert resumed_context["chat_history"] == [
        {"role": "user", "content": "quante calorie?"},
        {"role": "assistant", "content": "Quanti anni hai?"},
    ]
    # The identity of the turn is untouched by the reconstruction.
    assert resumed_context["user_id"] == "alice"


async def test_a_resumed_turn_never_sees_the_question_twice():
    """A client that *does* echo the exchange must not double it.

    The task is the single source for the parked exchange; the caller's copy of
    it is replaced, not appended to.
    """
    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.models import InputRequired

    agent = _ResumableAgent()
    client = A2AClient(transport=httpx.ASGITransport(app=_mounted_app(agent)))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    first = [c async for c in client.send_message_stream("foo", "quante calorie?")]
    task_id = next(c for c in first if isinstance(c, InputRequired)).task_id

    # A client whose own copy of the exchange is duplicated and out of date:
    # if it were merged in — or simply trusted — the question would repeat.
    echoed = [
        {"role": "user", "content": "quante calorie?"},
        {"role": "assistant", "content": "Quanti anni hai?"},
        {"role": "user", "content": "quante calorie?"},
        {"role": "assistant", "content": "Quanti anni hai?"},
    ]
    async for _ in client.send_message_stream(
        "foo", "38", context={"chat_history": echoed}, task_id=task_id
    ):
        pass
    await client.aclose()

    history = agent.seen[1][1]["chat_history"]
    assert [m["content"] for m in history].count("Quanti anni hai?") == 1
    # The task's version, not the caller's.
    assert history == [
        {"role": "user", "content": "quante calorie?"},
        {"role": "assistant", "content": "Quanti anni hai?"},
    ]


async def test_a_first_turn_keeps_the_history_the_caller_sent():
    """Only a resume rebuilds the history; an ordinary turn is untouched."""
    from erebots.agents.a2a.client import A2AClient

    agent = _ResumableAgent()
    client = A2AClient(transport=httpx.ASGITransport(app=_mounted_app(agent)))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    sent = [{"role": "user", "content": "earlier"}]
    async for _ in client.send_message_stream(
        "foo", "quante calorie?", context={"chat_history": sent}
    ):
        pass
    await client.aclose()

    assert agent.seen[0][1]["chat_history"] == sent


async def test_a_resumed_turn_keeps_the_context_that_preceded_the_question():
    """The antecedent case: "update it" only means something together with the
    turns before it. A clarifying question must not cost the specialist that
    context — after the answer it still has to know what "it" was."""
    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.models import InputRequired

    agent = _ResumableAgent()
    client = A2AClient(transport=httpx.ASGITransport(app=_mounted_app(agent)))
    client.register_agent("foo", f"{BASE}{MOUNT}")

    prior = [
        {"role": "user", "content": "il mio peso è 70kg"},
        {"role": "assistant", "content": "annotato"},
    ]
    first = [
        c
        async for c in client.send_message_stream(
            "foo", "aggiornalo", context={"user_id": "alice", "chat_history": prior}
        )
    ]
    task_id = next(c for c in first if isinstance(c, InputRequired)).task_id

    # The answer carries the task id and identity only — the frontend that
    # asked the question is not required to echo the conversation back.
    async for _ in client.send_message_stream(
        "foo", "38", context={"user_id": "alice"}, task_id=task_id
    ):
        pass
    await client.aclose()

    _, resumed_context = agent.seen[1]
    assert resumed_context["chat_history"] == prior + [
        {"role": "user", "content": "aggiornalo"},
        {"role": "assistant", "content": "Quanti anni hai?"},
    ], "the turns before the request must survive the detour, in order and once"


# --- resume across a restart: the task store is the only thing that survives -


async def test_a_parked_task_resumes_after_the_server_and_store_are_recreated(tmp_path):
    """Point 15 end to end, on a persistent store across a restart.

    The in-process tests keep the same app object alive, so they cannot tell a
    task store from a Python dict. Here the app, the executor, the engine and
    the client are all destroyed after the question is asked, and rebuilt from
    the database file alone. What comes back has to be enough: the exchange,
    the conversation that preceded it, and who the task belongs to.
    """
    from a2a.server.tasks import DatabaseTaskStore

    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.models import InputRequired
    from erebots.persistence.db import make_engine

    db = f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}"
    prior = [
        {"role": "user", "content": "il mio peso è 70kg"},
        {"role": "assistant", "content": "annotato"},
    ]

    async def instance():
        """One server process: its own engine, store, app and client."""
        engine = make_engine(db)
        store = DatabaseTaskStore(engine)
        await store.initialize()
        agent = _ResumableAgent()
        app = Starlette(
            routes=[Mount(MOUNT, app=create_a2a_app(agent, url=f"{BASE}{MOUNT}", task_store=store))]
        )
        client = A2AClient(transport=httpx.ASGITransport(app=app))
        client.register_agent("foo", f"{BASE}{MOUNT}")
        return engine, agent, client

    # --- instance A: ask the question, then go away entirely.
    engine, agent_a, client_a = await instance()
    try:
        first = [
            c
            async for c in client_a.send_message_stream(
                "foo", "aggiornalo", context={"user_id": "alice", "chat_history": prior}
            )
        ]
        task_id = next(c for c in first if isinstance(c, InputRequired)).task_id
    finally:
        await client_a.aclose()
        await engine.dispose()

    # --- instance B: a different app, executor and engine on the same file.
    engine, agent_b, client_b = await instance()
    try:
        assert not agent_b.seen, "instance B starts with no memory of the question"
        assert await client_b.can_resume("foo", task_id, user_id="alice")
        # An answer from anyone else must not reopen alice's task.
        assert not await client_b.can_resume("foo", task_id, user_id="mallory")
        assert not await client_b.can_resume("foo", task_id, user_id=None)

        # Identity and task id only — no transcript echoed back.
        async for _ in client_b.send_message_stream(
            "foo", "38", context={"user_id": "alice"}, task_id=task_id
        ):
            pass
    finally:
        await client_b.aclose()
        await engine.dispose()

    (message, resumed_context), = agent_b.seen
    assert message == "38"
    assert resumed_context["user_id"] == "alice"
    assert resumed_context["chat_history"] == prior + [
        {"role": "user", "content": "aggiornalo"},
        {"role": "assistant", "content": "Quanti anni hai?"},
    ], "the whole conversation has to come back off the store, not off the client"

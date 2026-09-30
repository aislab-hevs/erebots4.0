"""Round-trip tests for the A2A conformance endpoint.

Drives it exactly as ``a2a-tck`` does — over real JSON-RPC, selecting the
scenario through the ``messageId`` — so these tests fail for the same reasons
the suite would, without needing the suite installed.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import pytest
from a2a import types as T
from a2a.client import ClientConfig, ClientFactory
from a2a.server.tasks import InMemoryTaskStore
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette
from starlette.routing import Mount

from erebots.agents.a2a.conformance import create_conformance_app, scenario_of

MOUNT = "/a2a/tck"
BASE = "http://test"

# What tck.requirements.base.tck_id() builds: the session suffix is fresh per
# run, so only the prefix carries meaning.
SESSION = "0123456789abcdef"


def tck_id(scenario: str) -> str:
    return f"tck-{scenario}-{SESSION}"


def _mounted_app() -> Starlette:
    return Starlette(routes=[Mount(MOUNT, app=create_conformance_app(url=f"{BASE}{MOUNT}"))])


@asynccontextmanager
async def _session():
    """One endpoint instance, so several turns share a task store."""
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        factory = ClientFactory(ClientConfig(httpx_client=http, streaming=False))
        client = await factory.create_from_url(f"{BASE}{MOUNT}")

        async def send(scenario: str, *, task_id: str = "", context_id: str = ""):
            msg = T.Message(
                message_id=tck_id(scenario),
                role=T.Role.ROLE_USER,
                parts=[T.Part(text="TCK artifact test")],
                task_id=task_id,
                context_id=context_id,
            )
            return [r async for r in client.send_message(T.SendMessageRequest(message=msg))]

        yield send


async def _send(scenario: str):
    """Run one scenario on a fresh endpoint and return the responses."""
    async with _session() as send:
        return await send(scenario)


def _first_part(responses):
    """The first part of the first artifact — what the TCK validators read."""
    for resp in responses:
        task = getattr(resp, "task", None)
        if task and task.artifacts:
            return task.artifacts[0].parts[0]
    return None


def _task(responses):
    for resp in responses:
        task = getattr(resp, "task", None)
        if task and task.id:
            return task
    return None


# --------------------------------------------------------------------------
# Scenario selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        "artifact-text",
        "artifact-file",
        "artifact-file-url",
        "artifact-data",
        "message-response",
        "input-required",
        "complete-task",
    ],
)
def test_every_scenario_is_recognized(scenario):
    assert scenario_of(tck_id(scenario)) == scenario


def test_longer_scenario_name_wins_over_its_prefix():
    """``artifact-file`` is a prefix of ``artifact-file-url``.

    Matching in declaration order would classify every file-url request as a
    plain file one, and the two scenarios assert different part types — so the
    file-url requirement would fail while looking correctly implemented.
    """
    assert scenario_of(tck_id("artifact-file-url")) == "artifact-file-url"


def test_non_tck_message_ids_select_no_scenario():
    assert scenario_of("m1") is None
    assert scenario_of("tck-unknown-scenario-abc") is None


# --------------------------------------------------------------------------
# Agent card
# --------------------------------------------------------------------------


async def test_card_declares_skills_on_the_wire():
    """CARD-STRUCT-001 requires a ``skills`` key in the served JSON.

    Asserting on the served document rather than on the card object: protobuf
    omits an empty repeated field from its JSON, so a card built with no skills
    looks fine in Python and arrives missing the field.
    """
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        card = (await http.get(f"{MOUNT}/.well-known/agent-card.json")).json()

    assert "skills" in card
    assert card["skills"]


async def test_the_conformance_card_carries_caching_headers():
    """CARD-CACHE-001/002/003 are measured on this endpoint's card."""
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        resp = await http.get(f"{MOUNT}/.well-known/agent-card.json")

    assert "max-age" in resp.headers.get("cache-control", "")
    assert resp.headers.get("etag")
    assert resp.headers.get("last-modified")


async def test_conformance_endpoint_shares_the_content_type_guard():
    """The suite checks error mapping here, so it must behave like the real ones."""
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        resp = await http.post(
            f"{MOUNT}/", content=b"not json", headers={"Content-Type": "text/plain"}
        )

    assert resp.json()["error"]["code"] == -32005


async def test_card_declares_the_same_bindings_as_a_real_agent():
    """The suite builds one client per declared binding and runs its tests through
    each. A binding missing here is not tested at all; one declared but unserved
    fails every test it gates — so this card must track the real ones exactly."""
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        card = (await http.get(f"{MOUNT}/.well-known/agent-card.json")).json()
        rest = await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "hi"}],
                    "messageId": "tck-complete-task-1",
                }
            },
            headers={"A2A-Version": "1.0"},
        )

    assert [i["protocolBinding"] for i in card["supportedInterfaces"]] == [
        "JSONRPC",
        "HTTP+JSON",
    ]
    assert rest.status_code == 200
    assert rest.json()["task"]["status"]["state"] == "TASK_STATE_COMPLETED"


async def test_the_extended_card_is_served_and_differs_from_the_public_one():
    """CARD-EXT-001: the declared capability is honored, and what comes back is
    the *extended* card — the marker skill is the observable difference."""
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        public = (await http.get(f"{MOUNT}/.well-known/agent-card.json")).json()
        extended = (
            await http.get(f"{MOUNT}/extendedAgentCard", headers={"A2A-Version": "1.0"})
        ).json()

    assert public["capabilities"]["extendedAgentCard"] is True
    public_skills = {s["id"] for s in public["skills"]}
    extended_skills = {s["id"] for s in extended["skills"]}
    assert "conformance-extended" in extended_skills - public_skills


# --------------------------------------------------------------------------
# Required extension (CORE-CAP-004)
# --------------------------------------------------------------------------

_EXT_URI = "urn:a2a:tck:required-extension"


def _jsonrpc_send(message_id: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "extension test"}],
                "messageId": message_id,
            }
        },
    }


async def test_the_card_declares_the_tck_extension_as_required():
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        card = (await http.get(f"{MOUNT}/.well-known/agent-card.json")).json()

    extensions = card["capabilities"].get("extensions", [])
    assert [e["uri"] for e in extensions if e.get("required")] == [_EXT_URI]


async def test_the_cap_ext_scenario_without_the_extension_errors_on_jsonrpc():
    """CORE-CAP-004 over JSON-RPC: the suite validates ``error.code == -32008``."""
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        resp = await http.post(
            f"{MOUNT}/",
            json=_jsonrpc_send(tck_id("cap-ext-004-jsonrpc")),
            headers={"A2A-Version": "1.0"},
        )

    assert resp.json()["error"]["code"] == -32008


async def test_the_cap_ext_scenario_without_the_extension_errors_on_rest():
    """CORE-CAP-004 over REST: 400, AIP-193 body, reason EXTENSION_SUPPORT_REQUIRED."""
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        resp = await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "extension test"}],
                    "messageId": tck_id("cap-ext-004-http-json"),
                }
            },
            headers={"A2A-Version": "1.0"},
        )

    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == 400
    reasons = [
        d.get("reason")
        for d in error.get("details", [])
        if d.get("@type", "").endswith("google.rpc.ErrorInfo")
    ]
    assert "EXTENSION_SUPPORT_REQUIRED" in reasons


async def test_activating_the_extension_lets_the_scenario_complete():
    """The mirror case: a client that does activate it must not be rejected."""
    app = _mounted_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        resp = await http.post(
            f"{MOUNT}/",
            json=_jsonrpc_send(tck_id("cap-ext-004-jsonrpc")),
            headers={"A2A-Version": "1.0", "A2A-Extensions": _EXT_URI},
        )

    body = resp.json()
    assert "error" not in body
    assert body["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"


async def test_other_scenarios_ignore_the_required_extension():
    """Enforcement must stay scoped to cap-ext-004: the suite activates the
    extension in no other test, so a card-wide gate would fail all of them."""
    task = _task(await _send("complete-task"))
    assert task is not None
    assert task.status.state == T.TaskState.TASK_STATE_COMPLETED


# --------------------------------------------------------------------------
# Task states
# --------------------------------------------------------------------------


async def test_input_required_parks_the_task():
    """The state on the returned task is what the suite asserts on."""
    task = _task(await _send("input-required"))
    assert task is not None
    assert task.status.state == T.TaskState.TASK_STATE_INPUT_REQUIRED


async def test_complete_task_completes():
    task = _task(await _send("complete-task"))
    assert task is not None
    assert task.status.state == T.TaskState.TASK_STATE_COMPLETED


async def test_a_parked_task_is_resumed_by_a_follow_up_on_the_same_id():
    """The multi-turn flow several requirements build on.

    The suite parks a task, sends follow-ups carrying its ``taskId``, and
    expects the last one to complete *that* task rather than open a new one.
    """
    async with _session() as send:
        parked = _task(await send("input-required"))
        assert parked is not None
        assert parked.status.state == T.TaskState.TASK_STATE_INPUT_REQUIRED

        resumed = _task(
            await send("complete-task", task_id=parked.id, context_id=parked.context_id)
        )
        assert resumed is not None
        assert resumed.id == parked.id
        assert resumed.status.state == T.TaskState.TASK_STATE_COMPLETED


async def test_unknown_scenario_still_completes():
    """An id the suite does not use must not wedge the endpoint."""
    responses = await _send("not-a-scenario")
    part = _first_part(responses)
    assert part is not None and part.text


# --------------------------------------------------------------------------
# Artifact shapes
# --------------------------------------------------------------------------


async def test_text_artifact_carries_the_expected_text():
    part = _first_part(await _send("artifact-text"))
    assert part is not None
    assert part.text == "Generated text content"


async def test_file_artifact_carries_bytes_and_metadata():
    part = _first_part(await _send("artifact-file"))
    assert part is not None
    assert part.raw
    assert part.filename == "output.txt"
    assert part.media_type == "text/plain"


async def test_file_url_artifact_carries_a_url_not_bytes():
    part = _first_part(await _send("artifact-file-url"))
    assert part is not None
    assert part.url
    assert not part.raw
    assert part.filename == "output.txt"
    assert part.media_type == "text/plain"


async def test_data_artifact_carries_the_expected_mapping():
    part = _first_part(await _send("artifact-data"))
    assert part is not None
    data = MessageToDict(part, preserving_proto_field_name=True)["data"]
    assert data == {"key": "value", "count": 42}


async def test_each_artifact_scenario_returns_exactly_one_part():
    """The suite reads ``artifacts[0].parts[0]`` and ignores the rest.

    A second artifact (or part) would make the assertion depend on emission
    order rather than on the payload.
    """
    for scenario in ("artifact-text", "artifact-file", "artifact-file-url", "artifact-data"):
        task = _task(await _send(scenario))
        assert task is not None, scenario
        assert len(task.artifacts) == 1, scenario
        assert len(task.artifacts[0].parts) == 1, scenario


async def test_artifact_has_an_id():
    task = _task(await _send("artifact-text"))
    assert task is not None
    assert task.artifacts[0].artifact_id


# --------------------------------------------------------------------------
# Message response
# --------------------------------------------------------------------------


async def test_message_response_returns_a_message_and_no_task():
    responses = await _send("message-response")
    messages = [
        r.message for r in responses if getattr(r, "message", None) and r.message.message_id
    ]
    assert messages, "expected a Message payload"
    assert messages[0].parts[0].text == "Direct message response"
    assert messages[0].role == T.Role.ROLE_AGENT
    # A Task in the same exchange is what makes DM-MSG-001 fail: the suite
    # reports "got a Task or no payload".
    assert _task(responses) is None


# --------------------------------------------------------------------------
# Lenient bodies (DM-SERIAL-005)
# --------------------------------------------------------------------------


@asynccontextmanager
async def _raw_http():
    """A raw client, for payloads the SDK client would never produce."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_mounted_app()),
        base_url=BASE,
        headers={"A2A-Version": "1.0"},
        follow_redirects=True,
    ) as http:
        yield http


async def test_jsonrpc_ignores_unrecognized_fields():
    """Extra fields in the params and inside the message must not fail the
    request (DM-SERIAL-005); the strict SDK parser would answer -32602."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "unrecognized field test"}],
                "messageId": tck_id("complete-task"),
                "tckUnknownField": "should-be-ignored",
            },
            "tckExtraParam": 42,
        },
    }
    async with _raw_http() as http:
        r = await http.post(f"{MOUNT}/", json=payload)
    body = r.json()
    assert "error" not in body, body.get("error")
    assert body["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"


async def test_rest_ignores_unrecognized_fields():
    payload = {
        "message": {
            "role": "ROLE_USER",
            "parts": [{"text": "unrecognized field test"}],
            "messageId": tck_id("complete-task"),
            "tckUnknownField": "should-be-ignored",
        },
        "tckExtraParam": 42,
    }
    async with _raw_http() as http:
        r = await http.post(f"{MOUNT}/message:send", json=payload)
    assert r.status_code < 400, r.text


async def test_malformed_requests_are_still_rejected():
    """Lenient parsing must not widen what the server accepts: a body that is
    not valid JSON, and a known field with the wrong type, still fail
    (CORE-ERR-002)."""
    async with _raw_http() as http:
        not_json = await http.post(
            f"{MOUNT}/", content=b"{not json", headers={"Content-Type": "application/json"}
        )
        wrong_type = await http.post(
            f"{MOUNT}/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "SendMessage",
                "params": {"message": {"parts": "not-a-list"}},
            },
        )
    assert "error" in not_json.json()
    assert "error" in wrong_type.json()


# --------------------------------------------------------------------------
# History with reused message ids (CORE-HIST-005/006)
# --------------------------------------------------------------------------


async def test_history_keeps_messages_sent_with_reused_ids():
    """The TCK reuses one message id per scenario for a whole run; the SDK's
    id-only dedup would drop every follow-up from history. The handler
    disambiguates colliding ids, so history keeps what was exchanged, in
    order."""

    def jsonrpc(text: str, message_id: str, task_id: str = ""):
        message = {"role": "ROLE_USER", "parts": [{"text": text}], "messageId": message_id}
        if task_id:
            message["taskId"] = task_id
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {"message": message},
        }

    reused = tck_id("input-required")
    async with _raw_http() as http:
        r = await http.post(f"{MOUNT}/", json=jsonrpc("opening question", reused))
        task_id = r.json()["result"]["task"]["id"]
        await http.post(f"{MOUNT}/", json=jsonrpc("first answer", reused, task_id))
        await http.post(f"{MOUNT}/", json=jsonrpc("second answer", reused, task_id))
        await http.post(f"{MOUNT}/", json=jsonrpc("wrap up", tck_id("complete-task"), task_id))
        r = await http.post(
            f"{MOUNT}/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "GetTask",
                "params": {"id": task_id, "historyLength": 20},
            },
        )

    history = r.json()["result"].get("history", [])
    texts = [p.get("text", "") for m in history for p in m.get("parts", [])]
    user_texts = [t for t in texts if t in ("first answer", "second answer")]
    assert user_texts == ["first answer", "second answer"], texts


async def test_history_deduplicates_identical_retransmissions():
    """Same id *and* same content is a retry, not a new message: it must not
    grow the history."""

    def jsonrpc(text: str, message_id: str, task_id: str = ""):
        message = {"role": "ROLE_USER", "parts": [{"text": text}], "messageId": message_id}
        if task_id:
            message["taskId"] = task_id
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {"message": message},
        }

    reused = tck_id("input-required")
    async with _raw_http() as http:
        r = await http.post(f"{MOUNT}/", json=jsonrpc("opening question", reused))
        task_id = r.json()["result"]["task"]["id"]
        retry = jsonrpc("same answer", f"{reused}-retry", task_id)
        await http.post(f"{MOUNT}/", json=retry)
        await http.post(f"{MOUNT}/", json=retry)  # byte-identical retransmission
        r = await http.post(
            f"{MOUNT}/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "GetTask",
                "params": {"id": task_id, "historyLength": 20},
            },
        )

    history = r.json()["result"].get("history", [])
    texts = [p.get("text", "") for m in history for p in m.get("parts", [])]
    assert texts.count("same answer") == 1, texts


# --------------------------------------------------------------------------
# Terminal-task errors
#
# Both are cases where the SDK answers a *different* error than the spec names
# for a task that has already finished. Driven over the wire because what is
# wrong is the code on the response, not anything visible from Python.
# --------------------------------------------------------------------------


@asynccontextmanager
async def _completed_task():
    """A client on an endpoint holding a finished task it is done running.

    Two endpoints over one store rather than one endpoint twice: right after a
    send the SDK still has the task's machinery in memory and answers from it,
    so the same request gives a different answer depending on how quickly it
    arrives. Coming back on a second endpoint pins the case these tests are
    about — a client asking about a task the process has already retired — and
    is what the suite hits, having done a GetTask in between.
    """
    store = InMemoryTaskStore()
    creator = Starlette(
        routes=[Mount(MOUNT, app=create_conformance_app(url=f"{BASE}{MOUNT}", task_store=store))]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=creator), base_url=BASE, follow_redirects=True
    ) as http:
        created = await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "TCK prerequisite task creation"}],
                    "messageId": tck_id("complete-task"),
                }
            },
            headers={"A2A-Version": "1.0"},
        )
        task = created.json()["task"]
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"

    returning = Starlette(
        routes=[Mount(MOUNT, app=create_conformance_app(url=f"{BASE}{MOUNT}", task_store=store))]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=returning), base_url=BASE, follow_redirects=True
    ) as http:
        yield http, task["id"]


async def test_canceling_a_finished_task_answers_409_over_rest():
    """CORE-CANCEL-002: the spec's error table maps TaskNotCancelableError to
    409 Conflict; the SDK's own table says 400, which reads as a malformed
    request when the id is good and the task has simply finished."""
    async with _completed_task() as (http, task_id):
        resp = await http.post(f"{MOUNT}/tasks/{task_id}:cancel", headers={"A2A-Version": "1.0"})

    assert resp.status_code == 409
    assert resp.json()["error"]["details"][0]["reason"] == "TASK_NOT_CANCELABLE"


async def test_canceling_a_finished_task_answers_32002_over_jsonrpc():
    """The same condition on the other binding, which the SDK already gets
    right — pinned so correcting the REST status cannot disturb it."""
    async with _completed_task() as (http, task_id):
        resp = await http.post(
            f"{MOUNT}/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "CancelTask",
                "params": {"id": task_id},
            },
            headers={"A2A-Version": "1.0"},
        )

    assert resp.json()["error"]["code"] == -32002


async def test_subscribing_to_a_finished_task_answers_unsupported_operation():
    """STREAM-SUB-003: spec §3.1.6 names UnsupportedOperationError (-32004) for
    a terminal task. The SDK's active-task setup raises InvalidParamsError
    (-32602) instead, blaming parameters that are in fact valid."""
    async with _completed_task() as (http, task_id):
        resp = await http.post(
            f"{MOUNT}/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "SubscribeToTask",
                "params": {"id": task_id},
            },
            headers={"A2A-Version": "1.0"},
        )

    assert resp.json()["error"]["code"] == -32004


async def test_subscribing_to_an_unknown_task_still_answers_task_not_found():
    """The terminal-state guard must not swallow the id check it runs before:
    an unknown id is TaskNotFound (-32001), not an unsupported operation."""
    async with _completed_task() as (http, _):
        resp = await http.post(
            f"{MOUNT}/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "SubscribeToTask",
                "params": {"id": "no-such-task"},
            },
            headers={"A2A-Version": "1.0"},
        )

    assert resp.json()["error"]["code"] == -32001


# --------------------------------------------------------------------------
# The capability-less twin
#
# CORE-CAP-001/002/003 pin the error owed to a caller asking for a capability
# the server does not have, so they only run against a card that declares none
# — which is what /a2a/tck-minimal serves.
# --------------------------------------------------------------------------

MINIMAL = "/a2a/tck-minimal"


@asynccontextmanager
async def _minimal_client():
    app = Starlette(
        routes=[
            Mount(
                MINIMAL,
                app=create_conformance_app(url=f"{BASE}{MINIMAL}", declare_capabilities=False),
            )
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, follow_redirects=True
    ) as http:
        yield http


async def _minimal_rpc(http, method, params):
    resp = await http.post(
        f"{MINIMAL}/",
        json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params},
        headers={"A2A-Version": "1.0"},
    )
    return resp.json()


async def test_the_minimal_card_declares_none_of_the_optional_capabilities():
    """What makes the suite run the absence tests instead of skipping them."""
    async with _minimal_client() as http:
        card = (await http.get(f"{MINIMAL}/.well-known/agent-card.json")).json()

    caps = card["capabilities"]
    assert caps["streaming"] is False
    assert caps["pushNotifications"] is False
    assert caps["extendedAgentCard"] is False


async def test_minimal_endpoint_refuses_push_configuration():
    """CORE-CAP-001: -32003, PushNotificationNotSupportedError."""
    async with _minimal_client() as http:
        body = await _minimal_rpc(
            http,
            "CreateTaskPushNotificationConfig",
            {"task_id": "t", "url": "https://example.com"},
        )

    assert body["error"]["code"] == -32003


async def test_minimal_endpoint_refuses_streaming():
    """CORE-CAP-002: -32004, UnsupportedOperationError."""
    async with _minimal_client() as http:
        body = await _minimal_rpc(
            http,
            "SendStreamingMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "stream test"}],
                    "messageId": tck_id("cap-stream"),
                }
            },
        )

    assert body["error"]["code"] == -32004


async def test_minimal_endpoint_refuses_the_extended_card():
    """CORE-CAP-003: -32004, and *not* the not-configured error — the card
    denies having the capability at all, which is a different requirement
    (CARD-EXT-002) measuring a misconfiguration this server cannot be in."""
    async with _minimal_client() as http:
        body = await _minimal_rpc(http, "GetExtendedAgentCard", {})

    assert body["error"]["code"] == -32004
    assert body["error"]["data"][0]["reason"] == "UNSUPPORTED_OPERATION"


async def test_minimal_endpoint_still_replays_the_scenarios():
    """The absence tests need tasks to exist, so the fixtures must still run
    on this endpoint — only the three capabilities are meant to differ."""
    async with _minimal_client() as http:
        resp = await http.post(
            f"{MINIMAL}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "TCK prerequisite task creation"}],
                    "messageId": tck_id("complete-task"),
                }
            },
            headers={"A2A-Version": "1.0"},
        )

    assert resp.json()["task"]["status"]["state"] == "TASK_STATE_COMPLETED"

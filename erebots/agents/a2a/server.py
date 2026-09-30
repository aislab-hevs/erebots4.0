"""Expose a :class:`BaseAgent` as an A2A v1.0 service via the official ``a2a-sdk``.

The agent is wrapped in an :class:`a2a.server.agent_execution.AgentExecutor`
that runs ``agent.process`` and publishes the result (plus the execution trace,
carried in the artifact ``metadata``) as a completed task. ``create_a2a_app``
returns a Starlette app meant to be mounted at the path part of ``url`` (e.g.
``/a2a/conversion``); the agent card advertises the full external ``url``.

Two of the spec's three bindings are served, both off the same request handler:
JSON-RPC at the mount root and HTTP+JSON (REST) on the paths below it. gRPC is
deliberately absent — it cannot be mounted in this ASGI app, so it would need a
server and a port of its own, outside the middleware that authenticates every
other route.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from email.utils import formatdate
from typing import TYPE_CHECKING

from a2a import types as T
from a2a.helpers.proto_helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.default_request_handler import TERMINAL_TASK_STATES
from a2a.server.request_handlers.response_helpers import build_error_response
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.routes.jsonrpc_dispatcher import JsonRpcDispatcher
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    TaskStore,
    TaskUpdater,
)
from a2a.utils.error_handlers import build_rest_error_payload
from a2a.utils.errors import (
    A2A_ERROR_MAPPING,
    ContentTypeNotSupportedError,
    ErrorMapping,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from google.protobuf.json_format import MessageToDict, ParseDict
from pydantic.alias_generators import to_camel
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.responses import JSONResponse

from erebots.agents.a2a.push import push_support
from erebots.agents.models import InputRequired
from erebots.config import settings

if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext

    from erebots.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# Path of the JSON-RPC endpoint *inside* the mounted sub-app. The route ends up
# at ``<mount>/`` (e.g. ``/a2a/conversion/``); the card advertises that URL.
_RPC_URL = "/"
_CARD_URL = "/.well-known/agent-card.json"


_JSON_MEDIA_TYPE = "application/json"

#: HTTP status for ``ContentTypeNotSupportedError`` on the REST binding. The
#: spec's error table (§5.4) maps it to 415; the SDK's own mapping says 400, so
#: the status is set here rather than taken from ``build_rest_error_payload``.
_UNSUPPORTED_MEDIA_TYPE = 415

# ``TaskNotCancelableError`` on the REST binding, for the same reason: the
# spec's error table maps it to 409 Conflict, the SDK's says 400. Unlike the
# content-type case there is no response of ours to correct — the SDK's REST
# dispatcher builds it and reads this table directly — so the entry itself is
# fixed, which also aligns every other REST answer carrying this error. The
# JSON-RPC binding is untouched: it answers -32002 from a separate table.
# Verified still wrong in a2a-sdk 1.1.2 and at the (unpublished) v1.1.3 tag.
A2A_ERROR_MAPPING[TaskNotCancelableError] = ErrorMapping(
    409, "FAILED_PRECONDITION", "TASK_NOT_CANCELABLE"
)


def _declares_json(content_type: str) -> bool:
    """Is this ``Content-Type`` JSON? Parameters (``; charset=utf-8``) are fine."""
    return content_type.split(";", 1)[0].strip().lower() == _JSON_MEDIA_TYPE


def _path_within_app(scope) -> str:
    """The request path relative to where this app is mounted.

    Starlette leaves ``scope["path"]`` absolute and records the mount in
    ``root_path``, so the JSON-RPC endpoint arrives as ``/a2a/conversion/``
    rather than the ``/`` it is registered under.
    """
    return scope["path"][len(scope.get("root_path") or "") :] or _RPC_URL


def _content_type_error_response(path: str) -> JSONResponse:
    """The ``ContentTypeNotSupportedError`` shaped for the binding that was hit.

    Both bindings report the same error, in their own dialect: JSON-RPC as a
    ``-32005`` envelope carried by HTTP 200, REST as an AIP-193 payload whose
    HTTP status *is* the error. Answering one in the other's shape would leave
    the caller parsing a body it does not expect.
    """
    if path in (_RPC_URL, ""):
        return JSONResponse(build_error_response(None, ContentTypeNotSupportedError()))
    payload = build_rest_error_payload(ContentTypeNotSupportedError())
    payload["error"]["code"] = _UNSUPPORTED_MEDIA_TYPE
    return JSONResponse(payload, status_code=_UNSUPPORTED_MEDIA_TYPE)


class JsonContentTypeGuard:
    """Reject a POST declaring a non-JSON body before anything tries to parse it.

    Both bindings carry ``application/json``; a request declaring anything else
    must be answered with ``ContentTypeNotSupportedError``. Without this the
    body reaches the JSON parser first, so a wrong content type is reported as a
    parse error — telling the caller its JSON is malformed when the actual
    problem is the type it declared.

    The response body comes from the SDK's own builders, so it carries the same
    envelope (including the ``ErrorInfo`` details) as every other A2A error.

    Plain ASGI rather than ``BaseHTTPMiddleware``: the latter wraps the response
    body, which does not sit well in front of the SSE streaming endpoint.

    A request declaring *no* content type passes through: there is no
    unsupported type to report, and the parser's own error still applies.
    """

    def __init__(self, app):
        """Wrap the ASGI ``app`` whose POSTs are guarded."""
        self.app = app

    async def __call__(self, scope, receive, send):
        """Answer the error for a non-JSON POST, or hand the request on.

        Only POSTs are inspected: no other method carries a body either
        binding parses.
        """
        if scope["type"] == "http" and scope["method"] == "POST":
            declared = Headers(scope=scope).get("content-type")
            if declared and not _declares_json(declared):
                response = _content_type_error_response(_path_within_app(scope))
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


#: Request protos for the REST endpoints whose POST body the SDK parses, keyed
#: by the path suffix that selects them. Only these three routes read a body;
#: the rest take their parameters from the path.
_REST_BODY_PROTOS = {
    "message:send": T.SendMessageRequest,
    "message:stream": T.SendMessageRequest,
    "pushNotificationConfigs": T.TaskPushNotificationConfig,
}


class LenientJsonBodies:
    """Drop unrecognized JSON fields from request bodies (DM-SERIAL-005).

    A2A says implementations *should* ignore fields they do not recognize, but
    the SDK's dispatchers parse bodies strictly (``json_format.Parse`` with its
    default) and answer ``InvalidParamsError`` / HTTP 400 instead. This
    middleware re-canonicalizes the body before the dispatcher sees it: the
    JSON is parsed into the request's own proto with ``ignore_unknown_fields``
    and serialized back, so unknown fields — at any nesting depth — are gone
    and the strict parser downstream has nothing left to reject.

    A body this middleware cannot make sense of (not JSON, unknown method, a
    known field with the wrong type) is forwarded untouched: the dispatcher's
    own error is the correct answer for a malformed request (CORE-ERR-002),
    and rewriting must never widen what the server accepts.

    Plain ASGI for the same reason as :class:`JsonContentTypeGuard`.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not (scope["type"] == "http" and scope["method"] == "POST"):
            await self.app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if not message.get("more_body"):
                break

        cleaned = self._clean(_path_within_app(scope), bytes(body))
        if cleaned != bytes(body):
            scope = dict(scope)
            scope["headers"] = [
                (name, value)
                for name, value in scope["headers"]
                if name.lower() != b"content-length"
            ] + [(b"content-length", str(len(cleaned)).encode())]

        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": cleaned, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    def _clean(path: str, body: bytes) -> bytes:
        try:
            payload = json.loads(body)
        except Exception:
            return body
        if not isinstance(payload, dict):
            return body

        if path in (_RPC_URL, ""):
            model = JsonRpcDispatcher.METHOD_TO_MODEL.get(payload.get("method"))
            params = payload.get("params", {})
            if model is None or not isinstance(params, dict):
                return body
            try:
                cleaned_params = MessageToDict(
                    ParseDict(params, model(), ignore_unknown_fields=True)
                )
            except Exception:
                return body
            envelope = {key: payload[key] for key in ("jsonrpc", "id", "method") if key in payload}
            envelope["params"] = cleaned_params
            return json.dumps(envelope).encode()

        model = next(
            (
                proto
                for suffix, proto in _REST_BODY_PROTOS.items()
                if path.rstrip("/").endswith(suffix)
            ),
            None,
        )
        if model is None:
            return body
        try:
            return json.dumps(
                MessageToDict(ParseDict(payload, model(), ignore_unknown_fields=True))
            ).encode()
        except Exception:
            return body


#: How long a client may reuse the card without asking again. Short on purpose:
#: a redeploy can change capabilities, and the ETag makes revalidation cheap.
_CARD_MAX_AGE_SECONDS = 300


class AgentCardCaching:
    """Serve the agent card with HTTP caching headers (CARD-CACHE-001/002/003).

    The spec asks the card endpoint for ``Cache-Control`` and ``ETag`` (SHOULD)
    and ``Last-Modified`` (MAY); the SDK's card route sends none of them. The
    card only changes when the process restarts, so ``Last-Modified`` is the
    app's start time and the ETag is a hash of the served body — computed per
    response rather than precomputed, so this wrapper never has to agree with
    the SDK on how the card serializes. A matching ``If-None-Match`` gets 304,
    which is the whole point of advertising an ETag.

    Plain ASGI for the same reason as :class:`JsonContentTypeGuard`; the body
    is buffered only for the card path, and a card is small.
    """

    def __init__(self, app):
        """Wrap the ASGI ``app`` serving the card, stamping the process start time.

        ``Last-Modified`` is fixed here because the card is built once at
        startup: within a process it genuinely never changes again.
        """
        self.app = app
        self._last_modified = formatdate(usegmt=True)

    async def __call__(self, scope, receive, send):
        """Buffer the card response and re-send it with caching headers.

        Any request that is not a GET on the card path is forwarded without
        buffering, so the streaming endpoints are untouched.
        """
        if not (
            scope["type"] == "http"
            and scope["method"] == "GET"
            and _path_within_app(scope) == _CARD_URL
        ):
            await self.app(scope, receive, send)
            return

        start: dict | None = None
        body = bytearray()

        async def buffer(message) -> None:
            """Collect the response, replying once the last body chunk arrives."""
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
            elif message["type"] == "http.response.body":
                body.extend(message.get("body", b""))
                if not message.get("more_body"):
                    await self._reply(scope, send, start, bytes(body))

        await self.app(scope, receive, buffer)

    async def _reply(self, scope, send, start: dict | None, body: bytes) -> None:
        """Send the buffered card, as 304 or as 200 with the caching headers.

        Any pre-existing caching header is dropped rather than appended to:
        duplicates of ``ETag`` or ``Cache-Control`` are not merged by caches.

        Args:
            scope: The ASGI scope, read for ``If-None-Match``.
            send: The ASGI send channel.
            start: The wrapped app's response-start message, or ``None`` if it
                never sent one.
            body: The complete buffered response body, hashed into the ETag.
        """
        if start is None or start["status"] != 200:
            # Not the card (404, error): forward untouched, nothing to cache.
            await send(start or {"type": "http.response.start", "status": 500})
            await send({"type": "http.response.body", "body": body})
            return
        etag = f'"{hashlib.sha256(body).hexdigest()[:32]}"'
        caching = [
            (b"cache-control", f"public, max-age={_CARD_MAX_AGE_SECONDS}".encode()),
            (b"etag", etag.encode()),
            (b"last-modified", self._last_modified.encode()),
        ]
        if Headers(scope=scope).get("if-none-match") == etag:
            await send({"type": "http.response.start", "status": 304, "headers": caching})
            await send({"type": "http.response.body", "body": b""})
            return
        headers = [
            (name, value)
            for name, value in start.get("headers", [])
            if name.lower() not in {b"cache-control", b"etag", b"last-modified"}
        ] + caching
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body})


#: Metadata keys carried on artifacts. Derived with the same converter the
#: client reads them back with, rather than typed out: two hand-written copies
#: are exactly how ``agent_name`` and ``agentName`` came to disagree.
_AGENT_NAME_KEY = to_camel("agent_name")
_TRACE_STEP_KEY = to_camel("trace_step")
_TRACE_KEY = to_camel("trace")


def _serialize_trace(trace) -> list[dict]:
    """Serialize TraceStep objects into plain dicts for artifact metadata.

    Derived from the dataclass rather than an enumerated field list: a hand-kept
    list silently drops any field added to :class:`TraceStep` later, which is
    exactly how ``prompt_tokens`` went missing across the A2A boundary.

    Names are camelCased and the timestamp rendered as an RFC 3339 UTC string,
    because A2A mandates both for every JSON field on the wire (DM-SERIAL-001,
    DM-SERIAL-003) — including the ones we add under ``metadata``. Converting
    here keeps the internal dataclass idiomatic Python.

    Args:
        trace: The :class:`TraceStep` objects to put on the wire.

    Returns:
        One dict per step, ready to serialize into artifact metadata.
    """
    out: list[dict] = []
    for step in trace:
        item = {to_camel(key): value for key, value in asdict(step).items()}
        item["timestamp"] = (
            datetime.fromtimestamp(step.timestamp, tz=UTC).isoformat().replace("+00:00", "Z")
        )
        out.append(item)
    return out


def _agent_input_from_message(message: T.Message) -> str | list[dict]:
    """Convert an incoming A2A message into the agent's input format.

    Args:
        message: The inbound message.

    Returns:
        A plain string for the common single-text case, so agents that never
        handle images need not unpack a list; otherwise OpenAI-style content
        parts. A message carrying no usable part becomes ``""``.
    """
    parts: list[dict] = []
    for part in message.parts:
        if part.text:
            parts.append({"type": "text", "text": part.text})
        elif part.url:
            parts.append({"type": "image_url", "image_url": {"url": part.url}})
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts or ""


def _context_from_message(message: T.Message) -> dict:
    """Extract the ``context`` dict the caller stashed in message metadata.

    Args:
        message: The inbound message.

    Returns:
        The caller's context, or ``{}`` when the message carries none or
        carries something that is not a mapping.
    """
    if not message.metadata:
        return {}
    meta = MessageToDict(message, preserving_proto_field_name=True).get("metadata", {})
    ctx = meta.get("context") or {}
    return ctx if isinstance(ctx, dict) else {}


def _opening_context(task: T.Task) -> dict:
    """The ``context`` the message that opened ``task`` carried.

    A resumed turn gets nothing but an answer and a task id, so anything the
    original caller sent — who the turn belongs to, the conversation it came
    from — has to be read back off the message that started the task. That
    message is in ``history``, which is what the task store persists.
    """
    for message in task.history:
        if message.role == T.Role.ROLE_USER:
            return _context_from_message(message)
    return {}


def _resumed_history(task: T.Task) -> list[dict]:
    """The conversation a parked task is the continuation of, as chat messages.

    Three parts, in order: the turns that preceded the request (from the
    opening message's context — without them a request like "update it" loses
    what "it" referred to), then the exchange held in ``task.history``, then
    the question itself, which is the parked *status* message and which the SDK
    does not keep in ``history``.
    """
    history: list[dict] = []
    prior = _opening_context(task).get("chat_history")
    if isinstance(prior, list):
        history += [
            {"role": message["role"], "content": message["content"]}
            for message in prior
            if isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ]
    for message in task.history:
        text = "\n".join(part.text for part in message.parts if part.text)
        if text:
            role = "assistant" if message.role == T.Role.ROLE_AGENT else "user"
            history.append({"role": role, "content": text})
    question = getattr(task.status, "message", None)
    if question is not None:
        text = "\n".join(part.text for part in question.parts if part.text)
        if text:
            history.append({"role": "assistant", "content": text})
    return history


def _parked_owner(task: T.Task) -> str:
    """The ``user_id`` the task was opened for, or ``""`` for an anonymous turn."""
    user_id = _opening_context(task).get("user_id")
    return str(user_id) if user_id else ""


def _parked_longer_than(task: T.Task, ttl: float) -> bool:
    """Return whether ``task`` has been in ``input-required`` for ``ttl``.

    A task with no park timestamp cannot be judged and is left alone — expiring
    it on a missing field would cancel tasks written by an older build.

    Args:
        task: The parked task.
        ttl: How long a task may stay parked, in seconds.

    Returns:
        True if the task is past its TTL and may be expired.
    """
    if not task.status.HasField("timestamp"):
        return False
    parked_at = task.status.timestamp.ToDatetime(tzinfo=UTC).timestamp()
    return time.time() - parked_at >= ttl


class _AgentExecutor(AgentExecutor):
    """Bridge an A2A task to ``BaseAgent.process``."""

    def __init__(self, agent: BaseAgent, task_store: TaskStore | None = None):
        """Bind the executor to the agent it runs.

        Args:
            agent: The agent whose ``process``/``process_stream`` serves tasks.
            task_store: The store the request handler persists tasks in. Needed
                to expire parked tasks once the turn's event queue is gone;
                without it a parked task simply never expires.
        """
        self.agent = agent
        # Same store the request handler persists tasks in; needed to expire
        # parked tasks after the turn's event queue is gone.
        self._task_store = task_store
        self._expiries: set[asyncio.Task] = set()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run one turn of the agent and publish it as task events.

        Streaming agents go through :meth:`_stream`; the rest are awaited and
        published as a single artifact. The turn ends in one of three states:
        completed, parked in ``input-required`` when the agent asked the user
        something, or failed.

        Exceptions raised while running the agent or publishing its output are
        caught and reported as a failed task: an agent that raises must produce
        a task the client can observe, not a dropped connection.

        Args:
            context: The inbound request, carrying the message and the task
                being continued (if any).
            event_queue: Where task events are published.
        """
        task = context.current_task
        # Read before start_work(): moving the task to "working" replaces the
        # status message, and that message is the question being answered.
        resuming = task is not None and (
            task.status.state == T.TaskState.TASK_STATE_INPUT_REQUIRED
        )
        exchange = _resumed_history(task) if resuming else []
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        agent_input = _agent_input_from_message(context.message)
        agent_context = _context_from_message(context.message)
        if resuming:
            # Without this the specialist would read its own answer with no
            # memory of what it asked, unless the caller happened to echo the
            # exchange back. It *replaces* the caller's history rather than
            # extending it: appending would put the question in the prompt
            # twice. What the caller sent when the task was opened is not lost
            # either — ``_resumed_history`` reads it back off the opening
            # message, so a request whose subject was in the earlier
            # conversation still has it after the detour.
            agent_context["chat_history"] = exchange
        try:
            if hasattr(self.agent, "process_stream"):
                question = await self._stream(updater, agent_input, agent_context)
                if question is not None:
                    # Park instead of completing: the task stays open and the
                    # user's answer resumes it on the same task id, so a pending
                    # question is protocol state rather than a guess the next
                    # turn has to make. Expiry covers the user who never
                    # answers — without it every abandoned question would
                    # leave a task open forever.
                    await updater.requires_input(
                        message=updater.new_agent_message([T.Part(text=question)])
                    )
                    self._schedule_expiry(task.id, context.call_context)
                    return
                await updater.complete()
                return
            else:
                response = await self.agent.process(agent_input, agent_context)
                await updater.add_artifact(
                    parts=[T.Part(text=response.content)],
                    name="response",
                    metadata={
                        _AGENT_NAME_KEY: response.agent_name,
                        _TRACE_KEY: _serialize_trace(response.trace),
                    },
                )
            await updater.complete()
        except Exception:
            logger.exception("A2A executor [%s] failed", self.agent.name)
            await updater.failed()

    async def _stream(self, updater: TaskUpdater, agent_input, agent_context) -> str | None:
        """Emit the agent's answer as incremental artifact chunks (SSE streaming);
        the same chunks are aggregated for non-streaming ``message/send``.

        Text chunks carry the answer; ``TraceStep`` markers yielded by the agent
        travel as empty-text chunks whose metadata holds the serialized step, so
        streaming callers can rebuild the full trace.

        The steps are *also* collected and written once, at the end, under the
        plural trace key. A streaming caller reads each artifact-update event as
        it arrives, but a non-streaming one reads the accumulated task - and the
        SDK merges an appended artifact's metadata per key
        (``append_artifact_to_task``), so every chunk writing ``trace_step``
        overwrote the one before it and exactly one step survived. Nothing the
        client could parse its way out of: the steps were gone before it looked.
        The final write uses the same key ``process`` uses, which the client
        already reads, and costs one empty-text part that both paths discard.
        """
        artifact_id = str(uuid.uuid4())
        first = True
        steps: list = []

        async def flush_trace() -> None:
            """Write the whole trace where a non-streaming reader will find it."""
            if steps:
                await updater.add_artifact(
                    parts=[T.Part(text="")],
                    artifact_id=artifact_id,
                    name="response",
                    append=True,
                    metadata={_TRACE_KEY: steps},
                )

        async for chunk in self.agent.process_stream(agent_input, agent_context):
            if not chunk:
                continue
            if isinstance(chunk, InputRequired):
                # The question is the agent's reply, so it streams like any
                # other text; the task state is what marks it as pending.
                await updater.add_artifact(
                    parts=[T.Part(text=chunk.question)],
                    artifact_id=artifact_id,
                    name="response",
                    append=not first,
                    metadata={_AGENT_NAME_KEY: self.agent.name} if first else None,
                )
                # A parked task is read back by a non-streaming caller too, so
                # the steps that led to the question belong in it.
                await flush_trace()
                return chunk.question
            if isinstance(chunk, str):
                parts = [T.Part(text=chunk)]
                metadata: dict = {}
            else:  # TraceStep marker
                step = _serialize_trace([chunk])[0]
                steps.append(step)
                parts = [T.Part(text="")]
                metadata = {_TRACE_STEP_KEY: step}
            if first:
                # The name rides on the first artifact whatever kind of chunk
                # opens the stream. Said once here rather than inside each
                # branch, because saying it per branch is what let a question,
                # and then a trace step, ship an unnamed artifact — and every
                # BaseSpecialist opens with a trace step.
                metadata[_AGENT_NAME_KEY] = self.agent.name
            await updater.add_artifact(
                parts=parts,
                artifact_id=artifact_id,
                name="response",
                append=not first,
                metadata=metadata or None,
            )
            first = False
        if first:  # nothing streamed → still produce an (empty) artifact
            await updater.add_artifact(
                parts=[T.Part(text="")],
                artifact_id=artifact_id,
                name="response",
                metadata={_AGENT_NAME_KEY: self.agent.name},
            )
            return None
        await flush_trace()
        return None

    def _schedule_expiry(self, task_id: str, call_context: ServerCallContext) -> None:
        """Arrange for a parked task to be canceled if nobody ever answers.

        A per-task timer rather than a sweeper: the store interface is
        owner-scoped, and here we still hold the request's ``call_context``
        that scopes it. The timer keeps ticking across a resume — by then the
        task has left ``input-required``, so firing is a no-op.

        Does nothing when no task store was provided or the TTL is disabled —
        a parked task then stays open until the client abandons it.

        Args:
            task_id: The task just parked.
            call_context: The request's context, which scopes store access.
        """
        ttl = settings.a2a_parked_task_ttl
        if self._task_store is None or ttl <= 0:
            return
        timer = asyncio.create_task(self._expire(task_id, call_context, ttl))
        self._expiries.add(timer)
        timer.add_done_callback(self._expiries.discard)

    async def _expire(self, task_id: str, call_context: ServerCallContext, ttl: float) -> None:
        """Wait out the TTL, then cancel the task if it is still unanswered.

        Armed by :meth:`_schedule_expiry`. Returns without doing anything when
        the task was meanwhile answered, resumed, removed, or re-parked by a
        newer question that carries its own timer.

        Args:
            task_id: The task to expire.
            call_context: The request's context, which scopes store access.
            ttl: How long to wait, and the age the park must exceed to count.
        """
        await asyncio.sleep(ttl)
        try:
            task = await self._task_store.get(task_id, call_context)
            if task is None or task.status.state != T.TaskState.TASK_STATE_INPUT_REQUIRED:
                return  # answered, resumed, or already gone; nothing to expire
            # Still input-required is not enough: an answer may have resumed
            # the task and the agent asked *again*, re-parking it after this
            # timer was armed. The park timestamp tells the two apart — a
            # recent one belongs to a newer question, covered by its own timer.
            # (ToDatetime keeps the nanoseconds; .seconds would truncate and
            # misjudge sub-second re-parks.)
            parked_at = task.status.timestamp.ToDatetime(tzinfo=UTC).timestamp()
            if time.time() - parked_at < ttl:
                return
            task.status.state = T.TaskState.TASK_STATE_CANCELED
            task.status.timestamp.GetCurrentTime()
            await self._task_store.save(task, call_context)
            logger.info(
                "A2A [%s] canceled task %s: parked in input-required for %ss with no answer",
                self.agent.name,
                task_id,
                ttl,
            )
        except Exception:  # a failed expiry must never take the server down
            logger.exception("A2A [%s] failed to expire task %s", self.agent.name, task_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Mark the task canceled.

        The agent turn itself is not interrupted: the state transition is what
        the protocol requires, and an in-flight ``process`` finishes into a task
        nobody is reading any more.

        Args:
            context: The request naming the task to cancel.
            event_queue: Where the cancellation is published.
        """
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


class HistoryPreservingHandler(DefaultRequestHandler):
    """The request handler both endpoints build on, correcting two SDK answers.

    Keeping every exchanged message in task history (CORE-HIST-005/006) is the
    main one; refusing a subscription to a finished task with the spec's error
    rather than the SDK's is the other (see :meth:`on_subscribe_to_task`).

    The SDK deduplicates incoming messages by ``message_id`` alone
    (``ActiveTask._handle_task_modification_event``): a message whose id is
    already in history is silently dropped, even when its content is new. That
    is the right call for an at-least-once retransmission, but it also drops
    legitimate follow-ups from clients that reuse ids — the a2a-tck does
    exactly that, and its history requirements measure the result.

    Writing the message to the store here does not survive: the SDK caches an
    ``ActiveTask`` per task id across requests, and its task manager prefers
    the in-memory copy over the store on the next save. So instead, an id that
    collides with a *different* message already in history is disambiguated
    with a fresh suffix before the SDK sees it — the dedup check then fails
    and the message travels the normal append path. A byte-identical
    retransmission keeps its id and stays deduplicated, so history does not
    grow on network retries.
    """

    async def on_message_send(self, params, context):
        await self._disambiguate_reused_id(params, context)
        return await super().on_message_send(params, context)

    async def on_message_send_stream(self, params, context):
        await self._disambiguate_reused_id(params, context)
        async for event in super().on_message_send_stream(params, context):
            yield event

    async def on_subscribe_to_task(self, params, context):
        """Refuse a subscription to a terminal task with the error the spec names.

        Spec §3.1.6 lists ``UnsupportedOperationError`` for this case. The SDK
        reaches it through its active-task machinery, whose setup answers
        ``InvalidParamsError`` instead — telling the caller its parameters are
        invalid when the id is perfectly good and the task simply finished.
        Judging the stored state first keeps that path from being entered.

        Verified in a2a-sdk 1.1.2 (and at the unpublished v1.1.3 tag); the v1
        handler this one replaced got it right.
        """
        task = await self.task_store.get(params.id, context) if params.id else None
        if task is not None and task.status.state in TERMINAL_TASK_STATES:
            raise UnsupportedOperationError(
                message=f"Task {task.id} is in terminal state: {task.status.state}"
            )
        async for event in super().on_subscribe_to_task(params, context):
            yield event

    @staticmethod
    def _payload_of(message: T.Message) -> tuple:
        """What makes two same-id messages the *same* message: role and parts.

        Not the whole proto — the stored copy is normalized on save (e.g. the
        server fills in ``context_id``), so a byte comparison would call every
        retransmission a new message.
        """
        return (
            message.role,
            tuple(part.SerializeToString(deterministic=True) for part in message.parts),
        )

    async def _disambiguate_reused_id(self, params, context) -> None:
        message = params.message
        if not message.task_id or not message.message_id:
            return  # a new task's opening message cannot collide yet
        task = await self.task_store.get(message.task_id, context)
        if task is None or task.status.state in TERMINAL_TASK_STATES:
            return  # the SDK rejects the send on its own; nothing to preserve
        incoming = self._payload_of(message)
        for previous in task.history:
            if previous.message_id != message.message_id:
                continue
            if self._payload_of(previous) == incoming:
                return  # identical retransmission: keep the id, dedup applies
            message.message_id = f"{message.message_id}-{uuid.uuid4().hex[:8]}"
            return


class _GuardedRequestHandler(HistoryPreservingHandler):
    """A request handler that vets a resume before the task machinery starts.

    Two checks the SDK does not make, and that must not live in the executor:
    an exception raised there is persisted as ``TASK_STATE_FAILED``, so a
    stranger's probe would kill the owner's parked question. Here the stored
    task is still untouched, and an ``A2AError`` maps to a plain protocol
    error on both bindings.

    Ownership — the resume handle travels through the frontend, so a forged
    one could answer (and thereby act on) someone else's question. The task
    remembers who opened it; anyone else gets the same answer as for a task
    that does not exist.

    Expiry — the in-memory timer does not survive a restart, so a long-dead
    question could otherwise still be answered. The park timestamp is judged
    again here; an expired task is canceled in the store and the send refused.
    Refusing (rather than delegating) matters: the SDK judges terminal states
    from its in-process task machinery, which still remembers the task as
    parked — the same reason a send to any store-canceled task is refused here.
    """

    async def on_message_send(self, params, context):
        """Vet a resume, then serve ``message/send`` as usual.

        Returns:
            The completed task, exactly as the SDK builds it.
        """
        await self._check_resume(params, context)
        return await super().on_message_send(params, context)

    async def on_message_send_stream(self, params, context):
        """Vet a resume, then serve ``message/stream`` as usual.

        The check runs before the first event, so a refused resume fails the
        request outright instead of opening a stream that then errors.

        Yields:
            The SDK's task events, unchanged.
        """
        await self._check_resume(params, context)
        async for event in super().on_message_send_stream(params, context):
            yield event

    async def _check_resume(self, params, context) -> None:
        """Refuse a send that must not reach the task machinery.

        Returns without objection for anything that is not a resume: a message
        with no task id, an unknown id (the SDK reports that itself), or a task
        in a state the SDK already handles.

        Args:
            params: The inbound send parameters.
            context: The call context, which scopes store access.

        Raises:
            TaskNotFoundError: The task belongs to a different user. Deliberately
                indistinguishable from a task that does not exist, so a forged
                handle learns nothing about whose it was.
            UnsupportedOperationError: The task is canceled, or was parked past
                its TTL — in which case it is canceled in the store first.
        """
        task_id = params.message.task_id
        if not task_id:
            return
        task = await self.task_store.get(task_id, context)
        if task is None:
            return  # unknown id; the SDK answers TaskNotFound on its own
        if task.status.state == T.TaskState.TASK_STATE_CANCELED:
            # The SDK rejects sends to terminal tasks by *store* state, but a
            # task canceled while its in-process machinery is still alive — an
            # expired park, a cancel-while-parked — would be answered from
            # memory. Refuse it here, where the store is the truth.
            raise UnsupportedOperationError("task is canceled")
        if task.status.state != T.TaskState.TASK_STATE_INPUT_REQUIRED:
            return  # not a resume of a parked task; the SDK handles the rest
        incoming = _context_from_message(params.message).get("user_id")
        owner = _parked_owner(task)
        if owner and owner != str(incoming or ""):
            raise TaskNotFoundError()
        ttl = settings.a2a_parked_task_ttl
        if ttl > 0 and _parked_longer_than(task, ttl):
            task.status.state = T.TaskState.TASK_STATE_CANCELED
            task.status.timestamp.GetCurrentTime()
            await self.task_store.save(task, context)
            logger.info("A2A canceled task %s on resume: parked past its %ss ttl", task_id, ttl)
            raise UnsupportedOperationError("task expired awaiting input")


def supported_interfaces(url: str) -> list[T.AgentInterface]:
    """The bindings this server speaks, as the card must declare them.

    Both are served by the same request handler, so they answer the same
    operations with the same semantics — which is what a card declaring two
    bindings promises. gRPC is deliberately absent: it needs a server of its
    own, outside the ASGI stack that carries this app's authentication.

    The JSON-RPC endpoint sits at the mount root and is advertised with a
    trailing slash (see :func:`create_a2a_app`); the REST paths hang off the
    mount itself (``<mount>/message:send``), so that one drops the slash.
    """
    return [
        T.AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version="1.0"),
        T.AgentInterface(url=url.rstrip("/"), protocol_binding="HTTP+JSON", protocol_version="1.0"),
    ]


def _build_card(
    agent: BaseAgent, url: str, version: str, has_extended_card: bool = False
) -> T.AgentCard:
    """Build the protobuf AgentCard advertised at the well-known endpoint.

    ``has_extended_card`` follows the configuration rather than being a
    constant: the SDK's handler answers ``GetExtendedAgentCard`` only when the
    public card declares the capability, so declaring it without a card to
    serve would turn every such call into a "not configured" error.
    """
    skills = [
        T.AgentSkill(
            id=s.get("id") or s["name"],
            name=s["name"],
            description=s.get("description", ""),
            tags=list(s.get("tags") or []),
        )
        for s in agent.skills.to_a2a_skills()
    ]
    return T.AgentCard(
        name=agent.name,
        description=agent.description or f"A2A server for {agent.name}",
        version=version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=T.AgentCapabilities(
            streaming=hasattr(agent, "process_stream"),
            push_notifications=True,
            extended_agent_card=has_extended_card,
        ),
        supported_interfaces=supported_interfaces(url),
        skills=skills,
    )


def build_routes(handler: DefaultRequestHandler, card: T.AgentCard) -> list:
    """Routes for both bindings plus the agent card, in an order that works.

    The card and JSON-RPC routes come first on purpose: the REST route list
    ends with a ``/{tenant}`` mount that matches any single leading segment,
    and a mount that matches wins outright — registered earlier it would
    swallow ``/.well-known/agent-card.json`` and answer 404 from inside.
    """
    routes = create_agent_card_routes(card, card_url=_CARD_URL)
    routes += create_jsonrpc_routes(handler, rpc_url=_RPC_URL)
    return [*routes, *create_rest_routes(handler)]


def create_a2a_app(
    agent: BaseAgent,
    url: str = "http://localhost:7860/a2a",
    version: str = "1.0.0",
    task_store: TaskStore | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
    extended_card: T.AgentCard | None = None,
) -> Starlette:
    """Create an A2A Starlette app exposing ``agent``.

    Args:
        agent: The :class:`BaseAgent` to expose.
        url: Public base URL of this A2A endpoint (advertised in the card);
            the app is meant to be mounted at this URL's path.
        version: Version string for the agent card.
        task_store: Optional task store (defaults to in-memory; pass a
            ``DatabaseTaskStore`` for multi-instance deployments).
        push_config_store: Where push-notification configurations live
            (defaults to in-memory). Whatever is passed is wrapped in the
            webhook validation, so it never sees an unchecked URL.
        extended_card: Card served to *authenticated* callers via
            ``GetExtendedAgentCard`` — the place for skills or detail the
            public card should not carry. When given, the public card declares
            the ``extendedAgentCard`` capability; when omitted, callers get
            the capability-not-supported error instead.

    Returns:
        A Starlette app serving the agent card and both bindings (JSON-RPC and
        HTTP+JSON), which share one request handler.
    """
    # Advertise the endpoint with a trailing slash: the JSON-RPC route lives at
    # "/" *inside* this mounted app, so clients that POST the card's URL verbatim
    # must hit "<mount>/" — the bare "<mount>" would need a slash-redirect, which
    # a catch-all UI mount at the app root would swallow (observed 404).
    url = url.rstrip("/") + "/"
    card = _build_card(agent, url, version, has_extended_card=extended_card is not None)
    # Executor and handler share the store: the executor needs it to expire
    # tasks parked in input-required after the request is over.
    store = task_store or InMemoryTaskStore()
    push_store, push_sender = push_support(push_config_store)
    handler = _GuardedRequestHandler(
        agent_executor=_AgentExecutor(agent, task_store=store),
        task_store=store,
        agent_card=card,
        push_config_store=push_store,
        push_sender=push_sender,
        extended_agent_card=extended_card,
    )
    return Starlette(
        routes=build_routes(handler, card),
        middleware=[
            Middleware(JsonContentTypeGuard),
            Middleware(AgentCardCaching),
            Middleware(LenientJsonBodies),
        ],
    )

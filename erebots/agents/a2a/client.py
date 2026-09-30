"""A2A client for talking to specialist agents via the official ``a2a-sdk``.

Thin wrapper over ``a2a.client`` that preserves the orchestrator-facing API
(``register_agent`` / ``send_message`` / ``token_provider``). Replaces the
previous hand-rolled JSON-RPC client and its task-polling loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime

import httpx
from a2a import types as T
from a2a.client import ClientConfig, ClientFactory
from google.protobuf.json_format import MessageToDict
from pydantic.alias_generators import to_camel

from erebots.agents.models import AgentResponse, InputRequired, TraceStep
from erebots.config import settings

TokenProvider = Callable[[], Awaitable[str]]

# Protobuf Role enum: index 0 is ROLE_UNSPECIFIED, index 1 is the user role.
_USER_ROLE = T.Role.values()[1]


class AgentTaskFailed(RuntimeError):
    """The remote agent moved the task to the terminal ``failed`` state."""


class _BearerAuth(httpx.Auth):
    """Injects a fresh bearer per request, read from the owning A2AClient.

    Token resolution happens at request time (not client-creation time) so a
    pooled connection never sends an expired token — the provider itself
    caches until expiry.
    """

    def __init__(self, owner: A2AClient):
        """Bind the flow to the client whose provider it reads.

        Args:
            owner: The client holding the ``token_provider``. Kept as a
                reference, not a token, so a provider set after the pooled
                client was built still applies.
        """
        self._owner = owner

    async def async_auth_flow(self, request: httpx.Request):
        """Add the current bearer to ``request``, when a provider is configured.

        Yields:
            The request, authorized in place. Unauthenticated deployments
            (no provider) pass it through untouched.
        """
        if self._owner.token_provider is not None:
            token = await self._owner.token_provider()
            request.headers["Authorization"] = f"Bearer {token}"
        yield request


def _field(item: dict, name: str):
    """Read a trace field by its wire name, falling back to the legacy one.

    A2A mandates camelCase on the wire, but a peer running an older build still
    sends snake_case; reading both keeps discovery working across versions.
    """
    value = item.get(to_camel(name))
    return item.get(name) if value is None else value


def _status_text(status) -> str:
    """The text of a status message, when the server attached one."""
    message = getattr(status.status, "message", None)
    if message is None:
        return ""
    return "".join(part.text for part in message.parts if part.text).strip()


def _parse_timestamp(value) -> float:
    """Accept an RFC 3339 string (current) or a epoch number (legacy)."""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return time.time()
    try:
        return float(value)
    except (TypeError, ValueError):
        return time.time()


class A2AClient:
    """Client for communicating with specialist agents over A2A v1.0.

    Per-agent SDK clients (and their HTTP connections) are created once and
    pooled: the agent card is fetched on first use, not on every message.
    """

    def __init__(
        self,
        token_provider: TokenProvider | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        """Initialize the client.

        Args:
            token_provider: Optional async callable returning a bearer token,
                sent as ``Authorization`` on every request.
            transport: Optional httpx transport (e.g. ``ASGITransport``) for
                in-process testing; defaults to real network transport.
        """
        self._agents: dict[str, str] = {}  # name → base URL
        self.token_provider = token_provider
        self._transport = transport
        # (name, streaming) → (httpx client, a2a sdk client); card fetched once.
        self._pool: dict[tuple[str, bool], tuple[httpx.AsyncClient, object]] = {}
        self._pool_lock = asyncio.Lock()

    def register_agent(self, name: str, base_url: str):
        """Associate an agent name with its A2A base URL.

        Re-registering with a different URL drops the pooled clients so the
        next message re-discovers the agent at its new address.

        Args:
            name: The name the orchestrator routes by.
            base_url: The agent's A2A base URL; a trailing slash is ignored.
        """
        base_url = base_url.rstrip("/")
        if self._agents.get(name) != base_url:
            for key in [(name, False), (name, True)]:
                cached = self._pool.pop(key, None)
                if cached is not None:
                    with contextlib.suppress(RuntimeError):
                        asyncio.get_running_loop().create_task(cached[0].aclose())
        self._agents[name] = base_url

    async def _get_client(self, agent_name: str, streaming: bool):
        """Return the pooled SDK client for ``agent_name`` (create on first use).

        Streaming and non-streaming clients are pooled separately: the SDK
        binds the choice at construction. Creation happens under a lock and is
        re-checked inside it, so concurrent first turns build one client, and a
        card fetch that fails leaves nothing pooled to retry against.

        Args:
            agent_name: A registered agent name.
            streaming: Which of the agent's two clients to return.

        Returns:
            The SDK client, with its agent card already resolved.

        Raises:
            ValueError: if the agent is not registered.
        """
        key = (agent_name, streaming)
        cached = self._pool.get(key)
        if cached is not None:
            return cached[1]
        base_url = self._agents.get(agent_name)
        if not base_url:
            raise ValueError(f"Agent '{agent_name}' not registered")
        async with self._pool_lock:
            cached = self._pool.get(key)
            if cached is not None:
                return cached[1]
            http_kwargs: dict = {
                "auth": _BearerAuth(self),
                "follow_redirects": True,
                "timeout": 120.0,
            }
            if self._transport is not None:
                http_kwargs["transport"] = self._transport
            http = httpx.AsyncClient(**http_kwargs)
            try:
                # Empty would mean JSONRPC only: a peer whose card publishes
                # just HTTP+JSON would be unreachable. Server preference wins.
                factory = ClientFactory(
                    ClientConfig(
                        httpx_client=http,
                        streaming=streaming,
                        supported_protocol_bindings=["JSONRPC", "HTTP+JSON"],
                    )
                )
                client = await factory.create_from_url(base_url)
            except BaseException:
                await http.aclose()
                raise
            self._pool[key] = (http, client)
            return client

    async def aclose(self):
        """Close all pooled HTTP clients (call on application shutdown)."""
        pool, self._pool = self._pool, {}
        for http, _ in pool.values():
            with contextlib.suppress(Exception):
                await http.aclose()

    def list_agents(self) -> list[str]:
        """Return the registered agent names."""
        return list(self._agents.keys())

    def _content_to_parts(self, content: str | list[dict]) -> list[T.Part]:
        """Convert message content to A2A protobuf parts.

        Args:
            content: Plain text, or OpenAI-style ``text``/``image_url`` parts.

        Returns:
            The protobuf parts. Entries in an unknown shape are dropped, and
            an empty result becomes a single empty text part — A2A requires a
            message to carry at least one.
        """
        if isinstance(content, str):
            return [T.Part(text=content)]
        parts: list[T.Part] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append(T.Part(text=item["text"]))
            elif item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url")
                if url:
                    parts.append(T.Part(url=url))
        return parts or [T.Part(text="")]

    def _parse_trace(self, value) -> list[TraceStep]:
        """Parse raw trace dicts (from artifact metadata) into TraceStep objects.

        Tolerant by design: a trace is observability, so a peer sending a
        malformed step must not fail the turn that carries it. Entries without
        both ``agent`` and ``action`` are skipped, and unparseable optional
        fields are left at their defaults.

        Args:
            value: The ``trace`` metadata value, of whatever shape it arrived in.

        Returns:
            The steps that could be parsed, in wire order; empty when none could.
        """
        if not isinstance(value, list):
            return []
        trace: list[TraceStep] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            agent = str(item.get("agent", "")).strip()
            action = str(item.get("action", "")).strip()
            if not agent or not action:
                continue
            step_kwargs: dict = {
                "agent": agent,
                "action": action,
                "detail": str(item.get("detail", "")).strip(),
                "timestamp": _parse_timestamp(item.get("timestamp")),
            }
            for field_name in ("step_id", "parent_id"):
                value = _field(item, field_name)
                if value:
                    step_kwargs[field_name] = str(value)
            for opt in ("elapsed_ms", "tokens", "prompt_tokens"):
                v = _field(item, opt)
                if v is not None:
                    try:
                        step_kwargs[opt] = float(v) if opt == "elapsed_ms" else int(v)
                    except (TypeError, ValueError):
                        pass
            trace.append(TraceStep(**step_kwargs))
        return trace

    def _response_from_artifacts(self, artifacts, fallback_agent_name: str) -> AgentResponse:
        """Extract content + trace from a task's artifacts.

        The parts are joined with nothing between them, and that is not a
        stylistic choice. A streaming specialist emits one appended artifact
        chunk per token (see ``_stream``), so these are consecutive pieces of a
        single answer - "Hel", "lo ", "wor", "ld" - and not separate
        statements. Joining them with a newline returned every multi-token
        reply as one fragment per line, split mid-word.

        It went unnoticed because the orchestrator reads the streaming path;
        only a caller using ``send_message`` ever saw it.

        Args:
            artifacts: The finished task's artifacts.
            fallback_agent_name: Attributed as the author when no artifact
                declares one — the registered name is the best guess available.

        Returns:
            The assembled response: the artifact texts concatenated, plus
            whatever trace the agent attached.
        """
        text_parts: list[str] = []
        trace: list[TraceStep] = []
        agent_name = fallback_agent_name
        for art in artifacts:
            data = MessageToDict(art, preserving_proto_field_name=True)
            for p in data.get("parts", []):
                if p.get("text"):
                    text_parts.append(p["text"])
            meta = data.get("metadata") or {}
            # camelCase on the wire, snake_case from an older peer.
            if _field(meta, "agent_name"):
                agent_name = _field(meta, "agent_name")
            if meta.get("trace"):
                trace = self._parse_trace(meta["trace"])
        return AgentResponse(
            content="".join(text_parts),
            agent_name=agent_name,
            trace=trace,
        )

    def _response_from_message(self, message: T.Message, agent_name: str) -> AgentResponse:
        """Read a direct reply, including optional EREBOTS trace metadata.

        Keeps the newline that ``_response_from_artifacts`` drops, because the
        two paths carry different things. An artifact's parts are token chunks
        of one answer from our own executor. A direct message comes from a
        foreign peer, where a part is a content block and text parts can sit
        on either side of a file or data part - run together they would read
        as one sentence nobody wrote.
        """
        meta = MessageToDict(message, preserving_proto_field_name=True).get("metadata", {})
        return AgentResponse(
            content="\n".join(part.text for part in message.parts if part.text),
            agent_name=_field(meta, "agent_name") or agent_name,
            trace=self._parse_trace(meta.get("trace")),
        )

    async def _message_for(
        self,
        agent_name: str,
        content: str | list[dict],
        context: dict | None,
        task_id: str | None,
    ) -> T.Message:
        """Build the outgoing message, resolving a resume's ``context_id``.

        A message that addresses an existing task has to carry that task's
        context id: the server refuses an event whose context does not match
        the task's (``InvalidParams``). The SDK client fills it in from the
        previous turn it saw, which is why an in-process resume works — but a
        client that only holds an answer and a task id (a new process, another
        instance, a frontend that just reconnected) has no such memory, and the
        resume used to fail *after* ``can_resume`` had already said yes. So it
        is read back off the task, which is the only thing that survives a
        restart. A task we cannot read is left to the server to reject.
        """
        message = T.Message(
            message_id=str(uuid.uuid4()),
            role=_USER_ROLE,
            parts=self._content_to_parts(content),
            metadata={"context": context or {}},
        )
        if task_id:
            message.task_id = task_id
            # A task we cannot read is left to the server to reject, the same
            # way an unknown task id already was.
            with contextlib.suppress(Exception):
                client = await self._get_client(agent_name, streaming=False)
                task = await client.get_task(T.GetTaskRequest(id=task_id))
                if task.context_id:
                    message.context_id = task.context_id
        return message

    async def send_message(
        self,
        agent_name: str,
        content: str | list[dict],
        context: dict | None = None,
        task_id: str | None = None,
    ) -> AgentResponse:
        """Send a message to a registered agent and return its response.

        ``task_id`` addresses an existing task instead of opening a new one —
        the non-streaming counterpart of the same parameter on
        :meth:`send_message_stream`, so a parked question can be answered from
        either path.

        Args:
            agent_name: A registered agent name.
            content: Text, or OpenAI-style content parts for multi-modal input.
            context: Request context forwarded as message metadata
                (``user_id``, ``user_context``, ...).
            task_id: Address an existing task instead of opening a new one.

        Returns:
            The agent's answer with its execution trace. A task that produced
            no artifact yields a response with empty content rather than an
            error.

        Raises:
            ValueError: if the agent is not registered.
        """
        message = await self._message_for(agent_name, content, context, task_id)
        client = await self._get_client(agent_name, streaming=False)

        artifacts = []
        failure = ""
        async for resp in client.send_message(T.SendMessageRequest(message=message)):
            # Protobuf exposes empty submessages even when a different payload
            # is selected, so check field presence instead of truthiness.
            if isinstance(resp, T.StreamResponse) and resp.HasField("message"):
                return self._response_from_message(resp.message, agent_name)
            task = getattr(resp, "task", None)
            if not task:
                continue
            if task.status.state == T.TaskState.TASK_STATE_FAILED:
                failure = _status_text(task) or f"{agent_name} failed the task"
            if task.artifacts:
                artifacts = list(task.artifacts)
        if failure:
            raise AgentTaskFailed(failure)

        return self._response_from_artifacts(artifacts, fallback_agent_name=agent_name)

    async def send_message_stream(
        self,
        agent_name: str,
        content: str | list[dict],
        context: dict | None = None,
        task_id: str | None = None,
    ) -> AsyncIterator[str | TraceStep | InputRequired]:
        """Send a message and stream the agent's answer back incrementally.

        Uses A2A ``message/stream`` over SSE. Text chunks arrive interleaved
        with the :class:`TraceStep` markers the agent emits, carried in
        per-chunk artifact metadata.

        Args:
            agent_name: A registered agent name.
            content: Text, or OpenAI-style content parts for multi-modal input.
            context: Request context forwarded as message metadata
                (``user_id``, ``user_context``, ...).
            task_id: Address an existing task instead of opening a new one —
                that is how an answer reaches a task parked in
                ``input-required``. The server rejects an unknown or
                already-finished task; check :meth:`can_resume` first to fall
                back gracefully.

        Yields:
            :class:`str` for answer text as it is produced,
            :class:`TraceStep` for observability steps, and at most one
            :class:`InputRequired` when the agent parks the task to ask the
            user something.

        Raises:
            ValueError: if the agent is not registered.
        """
        message = await self._message_for(agent_name, content, context, task_id)
        client = await self._get_client(agent_name, streaming=True)

        async for resp in client.send_message(T.SendMessageRequest(message=message)):
            if isinstance(resp, T.StreamResponse) and resp.HasField("message"):
                response = self._response_from_message(resp.message, agent_name)
                for step in response.trace:
                    yield step
                if response.content:
                    yield response.content
                continue
            # A parked task is signalled by its state, not by the artifact: the
            # question itself already streamed as ordinary text above.
            status = getattr(resp, "status_update", None)
            if status is not None and status.task_id:
                if status.status.state == T.TaskState.TASK_STATE_INPUT_REQUIRED:
                    yield InputRequired(question=_status_text(status), task_id=status.task_id)
                elif status.status.state == T.TaskState.TASK_STATE_FAILED:
                    # A failed task ends the stream with no artifacts; swallowing
                    # it here would surface as an empty answer instead of an error.
                    raise AgentTaskFailed(_status_text(status) or f"{agent_name} failed the task")
                continue

            update = getattr(resp, "artifact_update", None)
            if not update:
                continue
            meta = MessageToDict(update.artifact, preserving_proto_field_name=True).get(
                "metadata", {}
            )
            if _field(meta, "trace_step"):
                steps = self._parse_trace([_field(meta, "trace_step")])
                if steps:
                    yield steps[0]
                continue
            for part in update.artifact.parts:
                if part.text:
                    yield part.text

    async def can_resume(self, agent_name: str, task_id: str, user_id: str | None = None) -> bool:
        """Is ``task_id`` still parked in ``input-required`` on ``agent_name``?

        The pre-flight for resuming: the handle a frontend echoes back may be
        stale — the task expired, the server restarted, the agent was dropped —
        and sending to a dead task fails the whole turn. Any failure here just
        means "route normally instead", so errors deliberately map to False.

        Two more reasons to say no, mirroring what the serving agent enforces:
        a task parked longer than the TTL (the server timer may be gone after a
        restart, so the timestamp is judged here too), and a task opened for a
        different ``user_id`` than this turn's — a handle from another session
        degrades to normal routing instead of failing the turn downstream.

        Args:
            agent_name: The agent the handle claims to belong to.
            task_id: The parked task id echoed back by the frontend.
            user_id: The current turn's user. A task whose owner cannot be
                determined is accepted; one owned by somebody else is not.

        Returns:
            True only if the task can safely be resumed. False means "route
            this turn normally", never "the turn failed".
        """
        if not agent_name or not task_id:
            return False
        try:
            client = await self._get_client(agent_name, streaming=False)
            task = await client.get_task(T.GetTaskRequest(id=task_id))
        except Exception:
            return False
        if task.status.state != T.TaskState.TASK_STATE_INPUT_REQUIRED:
            return False
        ttl = settings.a2a_parked_task_ttl
        if ttl > 0 and task.status.HasField("timestamp"):
            parked_at = task.status.timestamp.ToDatetime(tzinfo=UTC).timestamp()
            if time.time() - parked_at >= ttl:
                return False
        owner = self._task_owner(task)
        return not owner or owner == str(user_id or "")

    @staticmethod
    def _task_owner(task) -> str:
        """Return the owner from the task's first user message.

        Args:
            task: The task as returned by the serving agent.

        Returns:
            The owner's id, or ``""`` when the task carries no user message or
            none of them declared one — an unknown owner, not a public task.
        """
        for message in task.history:
            if message.role == _USER_ROLE:
                meta = MessageToDict(message, preserving_proto_field_name=True).get("metadata", {})
                ctx = meta.get("context") or {}
                user_id = ctx.get("user_id") if isinstance(ctx, dict) else None
                return str(user_id) if user_id else ""
        return ""

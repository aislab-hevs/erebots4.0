"""Server-Sent Events (SSE) adapter for the headless orchestrator.

Exposes ``OrchestratorAgent.run()`` — the framework's neutral
:class:`~erebots.runtime.events.AgentEvent` stream — as a ``text/event-stream``
HTTP endpoint. This is the canonical, transport-standard integration point: the
same stream the CLI consumes in-process is served to any web/mobile frontend.
SSE is the streaming mechanism the rest of the stack already speaks (A2A
``message/stream``, MCP, the LLM completion API).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.responses import JSONResponse, Response, StreamingResponse

from erebots.config import settings
from erebots.runtime.events import AgentEvent

logger = logging.getLogger(__name__)

SessionResolver = Callable[[Request], Awaitable[dict]]


# --- Request schema -----------------------------------------------------------
#
# The chat body is the one place untrusted JSON turns into LLM prompt material,
# so it is validated here, before any context is built. Only the two
# conversational roles and the two content part types the stack understands are
# accepted; anything else (a ``system`` turn, an unknown key, an unbounded
# blob) is rejected with 422 rather than dropped silently, so a misbehaving
# client hears about it.


class _TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str


class _ImageUrl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str


class _ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"]
    image_url: _ImageUrl


Content = str | list[_TextPart | _ImagePart]


def _check_content_limits(content: Content) -> Content:
    """Enforce the per-message size limits from settings."""
    max_chars = settings.chat_message_max_chars
    if isinstance(content, str):
        if len(content) > max_chars:
            raise ValueError(f"message exceeds {max_chars} characters")
        return content
    text_chars = sum(len(part.text) for part in content if isinstance(part, _TextPart))
    if text_chars > max_chars:
        raise ValueError(f"message text exceeds {max_chars} characters")
    images = [part for part in content if isinstance(part, _ImagePart)]
    if len(images) > settings.chat_images_max_per_message:
        raise ValueError(f"message carries more than {settings.chat_images_max_per_message} images")
    for image in images:
        if len(image.image_url.url) > settings.chat_image_max_bytes:
            raise ValueError(f"image exceeds {settings.chat_image_max_bytes} bytes")
    return content


class HistoryMessage(BaseModel):
    """One prior conversational turn, as the client remembers it."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: Content

    @field_validator("content")
    @classmethod
    def _limits(cls, value: Content) -> Content:
        return _check_content_limits(value)


class PendingRef(BaseModel):
    """The handle echoed back from an earlier ``input_required`` event."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1)
    task_id: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat/stream``."""

    model_config = ConfigDict(extra="forbid")

    message: Content = ""
    #: Omitted (``None``) means "I keep no transcript — use the stored one and
    #: persist this turn for me". An explicit ``[]`` means "I own the
    #: transcript and it is empty", so nothing stored is pulled in. The two are
    #: different requests and must stay distinguishable on the wire.
    history: list[HistoryMessage] | None = None
    pending: PendingRef | None = None

    @field_validator("message")
    @classmethod
    def _limits(cls, value: Content) -> Content:
        return _check_content_limits(value)


async def _read_body(request: Request, max_bytes: int) -> bytes | None:
    """Read the body, giving up (``None``) as soon as it exceeds ``max_bytes``.

    Checks ``Content-Length`` first so an honest oversized upload is refused
    before a byte is buffered, then counts the stream for the chunked case.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


# --- Streaming ----------------------------------------------------------------


def format_sse_event(event: AgentEvent) -> str:
    """Render one :class:`AgentEvent` as an SSE wire frame.

    The ``event:`` line carries the type (so ``EventSource.addEventListener`` can
    dispatch by name); ``data:`` carries the full JSON payload.

    Args:
        event: The event to serialize.

    Returns:
        The frame, terminated by the blank line SSE requires.
    """
    payload: dict = {
        "type": event.type,
        "text": event.text,
        "agent": event.agent,
        "channel": event.channel,
        "data": event.data,
    }
    if event.trace_step is not None:
        payload["trace_step"] = asdict(event.trace_step)
    return f"event: {event.type}\ndata: {json.dumps(payload)}\n\n"


def _rendered_answer(event: AgentEvent, answer: str, streamed: bool) -> tuple[str, bool]:
    """Accumulate the answer text a client renders, from one event.

    The same rule the CLI renders by: streamed ``token`` chunks are the answer,
    and ``final`` only stands in for it when nothing streamed (a final-only
    peer). ``side`` belongs to a fan-out contribution, not to the reply.
    """
    if event.channel == "side":
        return answer, streamed
    if event.type == "token":
        return answer + event.text, streamed or bool(event.text)
    if event.type == "final" and not streamed:
        return answer + event.text, streamed
    return answer, streamed


def sse_response(
    events: AsyncIterator[AgentEvent],
    on_complete: Callable[[str], Awaitable[None]] | None = None,
) -> StreamingResponse:
    """Stream an AgentEvent iterator as a ``text/event-stream`` HTTP response.

    A final ``done`` frame lets clients close cleanly instead of waiting.

    ``on_complete`` is awaited with the answer as the client rendered it, once
    the turn has run to the end — the hook the endpoint persists the turn from.
    It is skipped when the turn failed, so a broken turn is not written to the
    conversation as if it had been answered.

    Args:
        events: The turn's event stream, consumed lazily as the client reads.
        on_complete: Awaited with the rendered answer once the turn has run to
            the end, and skipped when the turn failed.

    Returns:
        A streaming response with proxy buffering disabled, so frames reach the
        client as they are produced rather than in one block.
    """

    async def stream() -> AsyncIterator[str]:
        """Render the event stream as wire frames.

        Yields:
            One frame per event, an ``error`` frame if the turn raises, then
            the terminating ``done`` frame.
        """
        answer = ""
        streamed = False
        failed = False
        # The generator is the last error boundary: past this point an exception
        # can only drop the connection mid-stream, which a client cannot tell
        # apart from a network failure.
        try:
            async for event in events:
                if event.type == "error":
                    failed = True
                answer, streamed = _rendered_answer(event, answer, streamed)
                yield format_sse_event(event)
        except Exception as exc:
            logger.exception("Event stream failed mid-turn")
            failed = True
            yield format_sse_event(AgentEvent("error", text=f"Internal error: {exc}"))
        if on_complete is not None and not failed and answer:
            # After the events, before ``done``: a client that closes on
            # ``done`` has no way to observe a write that happens after it.
            try:
                await on_complete(answer)
            except Exception:
                logger.exception("Failed to persist the turn after streaming")
        yield 'event: done\ndata: {"type": "done"}\n\n'

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_chat_router(
    orchestrator,
    session_resolver: SessionResolver | None = None,
    path: str = "/api/chat/stream",
    before_turn: Callable[[], Awaitable[None]] | None = None,
) -> APIRouter:
    """Build a router exposing the orchestrator's turn as an SSE endpoint.

    The request body is a :class:`ChatRequest`: ``{"message", "history"}`` plus
    an optional ``"pending": {"agent", "task_id"}`` — the handle from an earlier
    ``input_required`` event, echoed back to mark this message as the answer to
    that agent's question.

    ``history`` decides who owns the conversation. Send it (``[]`` included)
    and the client owns the transcript: it is used for this turn and nothing is
    persisted. Omit it and the service owns it: the identified user's stored
    history is loaded, and the turn is appended to it once the stream ends — so
    a frontend that keeps no state still holds a conversation across requests.
    Such a frontend reads that conversation back from ``GET <path>/history``,
    registered alongside this endpoint. A body over ``chat_request_max_bytes`` gets 413, non-JSON 400, and
    anything the schema rejects 422. The optional ``session_resolver`` turns the
    request (cookie/bearer) into a session dict (e.g. ``{"user_id",
    "jwt_claims"}``); without it, turns run anonymously. ``before_turn`` runs
    before each turn (e.g. lazy agent discovery).

    Args:
        orchestrator: The agent whose ``run()`` serves the endpoint.
        session_resolver: Turns a request into a session dict; omit for
            anonymous turns.
        path: Where the streaming endpoint is mounted.
        before_turn: Awaited before each turn.

    Returns:
        A router carrying the POST endpoint and the ``/history`` GET beside it,
        to include in the app.
    """
    router = APIRouter()

    @router.get(path.rsplit("/", 1)[0] + "/history")
    async def chat_history(request: Request) -> Response:
        """The conversation the service is keeping for this session.

        The other half of the append path above: a frontend that keeps no
        transcript reads its conversation back here on load, so the user sees
        the chat the model already remembers instead of an empty box.

        ``persisted`` tells the client which side owns the conversation. It
        takes both halves: an identity to store a conversation under *and* a
        store to put it in. Either one missing and every append is a no-op, so
        a client that stopped sending its transcript would be handing it to
        nobody. A client that gets false has to send its own ``history`` with
        each turn.
        """
        session = await session_resolver(request) if session_resolver else {}
        conversation = orchestrator.conversation
        history = await conversation.load_history(session)
        persisted = bool(session.get("user_id")) and conversation.persists_history
        return JSONResponse({"history": history, "persisted": persisted})

    @router.post(path)
    async def chat_stream(request: Request) -> Response:
        """Run one turn for the posted message and stream it back as SSE."""
        if before_turn is not None:
            await before_turn()
        raw = await _read_body(request, settings.chat_request_max_bytes)
        if raw is None:
            return JSONResponse(
                {"detail": f"Request body exceeds {settings.chat_request_max_bytes} bytes"},
                status_code=413,
            )
        try:
            body = json.loads(raw)
        except ValueError:
            return JSONResponse({"detail": "Request body must be JSON"}, status_code=400)
        try:
            req = ChatRequest.model_validate(body)
        except ValidationError as exc:
            return JSONResponse(
                {"detail": jsonable_encoder(exc.errors(include_url=False))}, status_code=422
            )

        message = req.message if isinstance(req.message, str) else _dump(req.message)
        client_owns_history = req.history is not None
        history = (
            [{"role": m.role, "content": _dump(m.content)} for m in req.history]
            if client_owns_history
            else None
        )
        session = await session_resolver(request) if session_resolver else {}
        conversation = orchestrator.conversation
        context = await conversation.build_request_context(session, history)
        if req.pending is not None:
            context = dict(context or {})
            context["pending"] = req.pending.model_dump()

        # A client that sent no transcript is keeping none, so this turn is
        # only in the conversation if the service appends it — that is what
        # lets a plain web client hold a conversation across requests. One that
        # sent its own transcript owns it, and POSTs it back next turn; writing
        # its (possibly trimmed) view into the store instead would let it
        # truncate the conversation it does not fully hold.
        on_complete = None
        if not client_owns_history:

            async def on_complete(answer: str) -> None:
                await conversation.append_turn(
                    session,
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": answer},
                    ],
                )

        return sse_response(orchestrator.run(message, context), on_complete)

    return router


def _dump(content: Content) -> str | list[dict]:
    """Back to the plain ``str | list[dict]`` shape the orchestrator speaks."""
    if isinstance(content, str):
        return content
    return [part.model_dump() for part in content]

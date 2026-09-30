"""Tests for the A2A client's handling of terminal task failure.

A specialist whose executor dies marks the task ``failed`` and the stream ends
with no artifacts. Both send paths must turn that state into an exception —
otherwise the orchestrator would relay an empty answer with no error.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from a2a import types as T

from erebots.agents.a2a.client import A2AClient, AgentTaskFailed


def _client_returning(responses: list) -> A2AClient:
    """An A2AClient whose pooled SDK client replays scripted responses."""

    async def _send_message(_request):
        for resp in responses:
            yield resp

    fake_sdk = SimpleNamespace(send_message=_send_message)
    client = A2AClient()

    async def _get_client(_agent_name, streaming):
        return fake_sdk

    client._get_client = _get_client
    return client


def _failed_status_update(detail_parts=()):
    message = (
        SimpleNamespace(parts=[SimpleNamespace(text=t) for t in detail_parts])
        if detail_parts
        else None
    )
    return SimpleNamespace(
        status_update=SimpleNamespace(
            task_id="t-1",
            status=SimpleNamespace(state=T.TaskState.TASK_STATE_FAILED, message=message),
        ),
        artifact_update=None,
        task=None,
    )


async def test_stream_raises_on_a_failed_task():
    client = _client_returning([_failed_status_update()])
    with pytest.raises(AgentTaskFailed, match="nutrition failed the task"):
        async for _chunk in client.send_message_stream("nutrition", "hi"):
            pass


async def test_stream_failure_carries_the_server_detail():
    client = _client_returning([_failed_status_update(["LLM backend unreachable"])])
    with pytest.raises(AgentTaskFailed, match="LLM backend unreachable"):
        async for _chunk in client.send_message_stream("nutrition", "hi"):
            pass


async def test_send_message_raises_on_a_failed_task():
    failed_task = SimpleNamespace(
        task=SimpleNamespace(
            status=SimpleNamespace(state=T.TaskState.TASK_STATE_FAILED, message=None),
            artifacts=[],
        ),
        status_update=None,
        artifact_update=None,
    )
    client = _client_returning([failed_task])
    with pytest.raises(AgentTaskFailed, match="nutrition failed the task"):
        await client.send_message("nutrition", "hi")


def _direct_reply():
    return T.Message(
        message_id="reply", role=T.Role.ROLE_AGENT,
        parts=[T.Part(text="First part"), T.Part(url="https://example.org/file"),
               T.Part(text="Second part")],
        metadata={"agentName": "remote", "trace": [
            {"agent": "remote", "action": "response", "detail": "Direct reply"},
        ]},
    )


async def test_direct_message_preserves_text_parts_and_trace():
    client = _client_returning([T.StreamResponse(message=_direct_reply())])
    response = await client.send_message("peer", "hi")
    assert response.content == "First part\nSecond part"
    assert response.agent_name == "remote"
    assert [(step.agent, step.action) for step in response.trace] == [("remote", "response")]


async def test_stream_accepts_a_direct_message():
    from erebots.agents.models import TraceStep

    client = _client_returning([T.StreamResponse(message=_direct_reply())])
    chunks = [chunk async for chunk in client.send_message_stream("peer", "hi")]
    assert isinstance(chunks[0], TraceStep)
    assert chunks[1:] == ["First part\nSecond part"]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("binding", ["JSONRPC", "HTTP+JSON"])
async def test_direct_message_round_trip_through_sdk(streaming, binding):
    """Exercise a peer executor that answers with Message, without creating a task."""
    import httpx
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from erebots.agents.a2a.server import build_routes, supported_interfaces

    class DirectExecutor(AgentExecutor):
        async def execute(self, context, event_queue):
            await event_queue.enqueue_event(T.Message(
                message_id="reply", role=T.Role.ROLE_AGENT,
                parts=[T.Part(text=f"Reply to {context.message.parts[0].text}")],
            ))

        async def cancel(self, context, event_queue):
            raise AssertionError("A direct reply has no task to cancel")

    base = "http://test/a2a/direct/"
    card = T.AgentCard(
        name="direct", description="Direct replies", version="1",
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        capabilities=T.AgentCapabilities(streaming=True),
        supported_interfaces=[i for i in supported_interfaces(base) if i.protocol_binding == binding],
    )
    handler = DefaultRequestHandler(
        agent_executor=DirectExecutor(), task_store=InMemoryTaskStore(), agent_card=card,
    )
    peer = Starlette(routes=build_routes(handler, card))
    app = Starlette(routes=[Mount("/a2a/direct", app=peer)])
    client = A2AClient(transport=httpx.ASGITransport(app=app))
    client.register_agent("direct", base)
    try:
        if streaming:
            chunks = [chunk async for chunk in client.send_message_stream("direct", "hello")]
            assert chunks == ["Reply to hello"]
        else:
            response = await client.send_message("direct", "hello")
            assert response.content == "Reply to hello"
            assert response.agent_name == "direct"
    finally:
        await client.aclose()

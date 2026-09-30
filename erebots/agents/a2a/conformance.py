"""A2A conformance endpoint driven by the ``a2a-tck`` suite.

The TCK verifies the *protocol* implementation — serialization, task states,
artifact shapes, error codes — and needs the system under test to produce
specific payloads on demand: a task parked in ``input-required``, an artifact
holding a file or a data part, a bare ``Message`` instead of a ``Task``. A real
agent cannot do that: what it emits is whatever the model decided to say.

The suite therefore selects a scenario through the ``messageId`` of the request
(``tck-<scenario>-<session>``, see ``tck.requirements.base.tck_id``); the message
text is a constant the SUT is meant to ignore. This module implements those
scenarios in a dedicated endpoint, mounted at ``/a2a/tck`` only when
``a2a_conformance_agent`` is enabled.

Keeping it separate from :mod:`erebots.agents.a2a.server` is deliberate: teaching
the production executor to read ``messageId`` would let any client pick the
protocol behaviour it wants — parking tasks, or getting fabricated artifacts
back — through a field that is only supposed to identify a message.
"""

from __future__ import annotations

import uuid

from a2a import types as T
from a2a.helpers.proto_helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore, TaskStore, TaskUpdater
from a2a.utils.errors import ExtensionSupportRequiredError
from google.protobuf import json_format, struct_pb2
from starlette.applications import Starlette
from starlette.middleware import Middleware

from erebots.agents.a2a.push import push_support
from erebots.agents.a2a.server import (
    AgentCardCaching,
    HistoryPreservingHandler,
    JsonContentTypeGuard,
    LenientJsonBodies,
    build_routes,
    supported_interfaces,
)

# The scenario names the TCK builds message ids from. Order matters: matching is
# longest-first, because ``artifact-file`` is a prefix of ``artifact-file-url``
# and the shorter name would otherwise swallow the longer one's requests.
_SCENARIOS = (
    "artifact-file-url",
    "artifact-file",
    "artifact-text",
    "artifact-data",
    "message-response",
    "input-required",
    "complete-task",
    "cap-ext-004",
)

# CORE-CAP-004: the extension the TCK expects the card to declare as required.
# Enforcement is scoped to the ``cap-ext-004`` scenario because the suite
# activates the extension in no other test — a card-wide gate would fail every
# requirement that sends a message without the header.
_REQUIRED_EXTENSION = "urn:a2a:tck:required-extension"

# Payloads the suite asserts on verbatim (tests/compatibility/core_operations/
# test_artifacts.py). They are fixtures, not behaviour: changing them fails the
# TCK rather than changing what any agent does.
_TEXT_CONTENT = "Generated text content"
_MESSAGE_CONTENT = "Direct message response"
_FILE_NAME = "output.txt"
_FILE_MEDIA_TYPE = "text/plain"
_FILE_URL = "https://example.com/output.txt"
_DATA_CONTENT = {"key": "value", "count": 42}

_QUESTION = "Conformance scenario: this task is parked awaiting input."


def scenario_of(message_id: str) -> str | None:
    """Return the TCK scenario encoded in ``message_id``, or ``None``.

    Ids look like ``tck-artifact-file-url-<session>``; the session suffix is
    fresh per run, so only the prefix identifies the scenario.
    """
    for name in _SCENARIOS:
        if message_id.startswith(f"tck-{name}-"):
            return name
    return None


def _data_part() -> T.Part:
    """A data part carrying :data:`_DATA_CONTENT`."""
    value = struct_pb2.Value()
    json_format.ParseDict(_DATA_CONTENT, value)
    return T.Part(data=value)


def _artifact_part(scenario: str) -> T.Part:
    """The single part the artifact scenarios are expected to return.

    One part, because the suite validates ``artifacts[0].parts[0]``.
    """
    if scenario == "artifact-file":
        return T.Part(
            raw=_TEXT_CONTENT.encode(),
            filename=_FILE_NAME,
            media_type=_FILE_MEDIA_TYPE,
        )
    if scenario == "artifact-file-url":
        return T.Part(
            url=_FILE_URL,
            filename=_FILE_NAME,
            media_type=_FILE_MEDIA_TYPE,
        )
    if scenario == "artifact-data":
        return _data_part()
    # artifact-text, complete-task and anything unrecognized.
    return T.Part(text=_TEXT_CONTENT)


class _ConformanceExecutor(AgentExecutor):
    """Run the scenario named by the request's ``messageId``."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Emit the payload the selected scenario prescribes.

        Each branch exists to produce a shape a real agent could not be asked
        for on demand: a bare ``Message`` with no task behind it, a task parked
        in ``input-required``, or an artifact holding a specific part type. An
        unrecognized (or absent) scenario falls through to the text artifact, so
        an unknown message id still completes a task normally.

        Args:
            context: The inbound request; its ``messageId`` selects the scenario.
            event_queue: Where the resulting events are published.

        Raises:
            ExtensionSupportRequiredError: The ``cap-ext-004`` scenario was
                requested without activating :data:`_REQUIRED_EXTENSION`.
        """
        message = context.message
        scenario = scenario_of(message.message_id) if message is not None else None

        if scenario == "cap-ext-004" and _REQUIRED_EXTENSION not in context.requested_extensions:
            # Raised before any task exists; the dispatchers map it to -32008
            # (JSON-RPC) and 400 FAILED_PRECONDITION / EXTENSION_SUPPORT_REQUIRED
            # (REST) on their own.
            raise ExtensionSupportRequiredError(data={"required": [_REQUIRED_EXTENSION]})

        if scenario == "message-response":
            # DM-MSG-001: the reply is a Message, so no task is created at all —
            # enqueueing one first is what makes a handler answer with a Task.
            await event_queue.enqueue_event(
                T.Message(
                    message_id=str(uuid.uuid4()),
                    context_id=context.context_id or "",
                    role=T.Role.ROLE_AGENT,
                    parts=[T.Part(text=_MESSAGE_CONTENT)],
                )
            )
            return

        task = context.current_task
        if task is None:
            task = new_task_from_user_message(message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        if scenario == "input-required":
            await updater.requires_input(
                message=updater.new_agent_message([T.Part(text=_QUESTION)])
            )
            return

        await updater.add_artifact(
            parts=[_artifact_part(scenario or "")],
            artifact_id=str(uuid.uuid4()),
            name="response",
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the task unconditionally.

        The scenarios hold no work to interrupt, so cancellation is just the
        state transition the suite checks for.

        Args:
            context: The request naming the task to cancel.
            event_queue: Where the cancellation is published.
        """
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


def _build_card(url: str, version: str, *, declare_capabilities: bool = True) -> T.AgentCard:
    """Card for the conformance endpoint.

    Capabilities mirror the real specialists' so the suite selects the same set
    of tests it would run against them: a card claiming less would silently skip
    the requirements we want covered.

    ``declare_capabilities=False`` turns the three optional ones off, which is
    the *only* way to reach `CORE-CAP-001/002/003`: those requirements pin the
    error a server owes a caller asking for a capability it does not have, so
    the suite skips them against a card that has it. A positive test and its
    negative can never both run against one card — hence a second endpoint
    rather than a different card here.
    """
    return T.AgentCard(
        name="tck",
        description=(
            "A2A conformance endpoint: replays the fixed payloads the a2a-tck "
            "suite selects through the request's messageId. Not an agent."
        ),
        version=version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=T.AgentCapabilities(
            streaming=declare_capabilities,
            push_notifications=declare_capabilities,
            extended_agent_card=declare_capabilities,
            # Declared so CORE-CAP-004 runs; only the cap-ext-004 scenario
            # enforces it (see _REQUIRED_EXTENSION above). Kept on both cards,
            # so they differ in exactly the three flags above.
            extensions=[
                T.AgentExtension(
                    uri=_REQUIRED_EXTENSION,
                    description=(
                        "TCK fixture extension: requests selecting the "
                        "cap-ext-004 scenario must activate it or receive "
                        "ExtensionSupportRequiredError."
                    ),
                    required=True,
                )
            ],
        ),
        supported_interfaces=supported_interfaces(url),
        # One skill, not none: CARD-STRUCT-001 requires the field to be present,
        # and protobuf drops an empty repeated field from the JSON entirely — so
        # declaring nothing reads on the wire as declaring no `skills` key.
        skills=[
            T.AgentSkill(
                id="conformance",
                name="conformance",
                description=(
                    "Replay the a2a-tck payload selected by the request's "
                    "messageId prefix: " + ", ".join(sorted(_SCENARIOS)) + "."
                ),
            )
        ],
    )


def _extended_card_of(card: T.AgentCard) -> T.AgentCard:
    """The card served to authenticated callers (CARD-EXT-001).

    The public card plus one skill it does not list — the difference is what
    proves the endpoint returns the *extended* card and not the public one.
    """
    extended = T.AgentCard()
    extended.CopyFrom(card)
    extended.skills.append(
        T.AgentSkill(
            id="conformance-extended",
            name="conformance-extended",
            description=(
                "Marker skill present only on the extended card, so a caller "
                "(and the TCK) can tell the two apart."
            ),
        )
    )
    return extended


def create_conformance_app(
    url: str = "http://localhost:7860/a2a/tck",
    version: str = "1.0.0",
    task_store: TaskStore | None = None,
    declare_capabilities: bool = True,
) -> Starlette:
    """Create the Starlette app serving the A2A conformance endpoint.

    Args:
        url: Public URL this endpoint is mounted at (advertised in the card).
        version: Version string for the agent card.
        task_store: Optional task store (defaults to in-memory).
        declare_capabilities: Whether the card claims streaming, push
            notifications and the extended card. Turning it off serves the
            same scenarios behind a card that declares none of the three,
            which is what makes `CORE-CAP-001/002/003` runnable — see
            :func:`_build_card`.

    Returns:
        A Starlette app serving the agent card and both bindings, exactly as
        :func:`erebots.agents.a2a.server.create_a2a_app` does.
    """
    url = url.rstrip("/") + "/"
    card = _build_card(url, version, declare_capabilities=declare_capabilities)
    push_store, push_sender = push_support()
    # HistoryPreservingHandler rather than the SDK default: the TCK's history
    # requirements (CORE-HIST-005/006) reuse message ids across sends, which
    # the SDK's id-only dedup would otherwise drop from history.
    handler = HistoryPreservingHandler(
        agent_executor=_ConformanceExecutor(),
        task_store=task_store or InMemoryTaskStore(),
        agent_card=card,
        push_config_store=push_store,
        push_sender=push_sender,
        # None when the card does not claim the capability: an extended card
        # behind a card that denies having one is the misconfiguration
        # `CARD-EXT-002` is about, not the absence `CORE-CAP-003` measures.
        extended_agent_card=_extended_card_of(card) if declare_capabilities else None,
    )
    # Same wire behaviour as the real endpoints — both bindings and the
    # content-type guard included: the suite checks error mapping here, so it
    # must not diverge from what a real agent serves.
    return Starlette(
        routes=build_routes(handler, card),
        middleware=[
            Middleware(JsonContentTypeGuard),
            Middleware(AgentCardCaching),
            Middleware(LenientJsonBodies),
        ],
    )

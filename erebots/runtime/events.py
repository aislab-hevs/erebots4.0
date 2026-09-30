"""Frontend-agnostic event protocol for orchestration.

A frontend (Gradio, CLI, REST/SSE, ...) consumes a stream of :class:`AgentEvent`
from the orchestrator's headless ``run()`` and renders it however it likes. The
core never imports a UI framework, and the same orchestration drives every
frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from erebots.agents.models import TraceStep

EventType = Literal[
    "routing",         # a routing decision was made (data["plan"] / data["target"])
    "token",           # a visible streamed text chunk (fallback path)
    "reasoning",       # a streamed <think> reasoning chunk
    "trace",           # a single TraceStep produced during the turn
    "input_required",  # a specialist parked its task waiting for the user
    "final",           # the final answer (text) and the agent that produced it
    "error",           # the turn failed (text holds the message)
]

# Where a streamed chunk belongs in the rendered turn. ``answer`` is the reply
# the user reads; ``side`` is a contributing specialist's own stream during a
# multi-agent turn, which a frontend shows next to (not inside) the answer.
Channel = Literal["answer", "side"]


@dataclass
class AgentEvent:
    """A single, UI-neutral event emitted while handling one user turn.

    Only the fields relevant to ``type`` are populated:

    * ``token`` / ``reasoning`` / ``final`` / ``error`` use ``text``.
    * ``trace`` uses ``trace_step``.
    * ``input_required`` uses ``text`` for the question and
      ``data["task_id"]`` for the parked task; echo ``{"agent", "task_id"}``
      back as ``context["pending"]`` on the next turn to resume it.
    * ``routing`` uses ``data["plan"]`` (the list of selected specialists) and
      ``data["target"]`` (the same, rendered as a display label).
    * ``agent`` names the agent the event pertains to, when meaningful.
    * ``channel`` splits concurrent specialist streams from the answer stream;
      it is ``answer`` everywhere except a fan-out turn's contributions, so a
      frontend that ignores it behaves exactly as before.
    """

    type: EventType
    text: str = ""
    agent: str = ""
    trace_step: TraceStep | None = None
    data: dict[str, Any] = field(default_factory=dict)
    channel: Channel = "answer"

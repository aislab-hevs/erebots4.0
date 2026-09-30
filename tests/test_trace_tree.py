"""Tests for trace-tree IDs, reparenting and markdown rendering."""

from __future__ import annotations

import pytest

from erebots.agents.models import TraceStep
from erebots.agents.orchestrator.agent import OrchestratorAgent
from examples.gradio_ui.chat import _format_trace_markdown


@pytest.fixture
def orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent(
        api_key="x",
        api_base_url="http://example.invalid",
        model="m",
    )


def test_trace_step_has_unique_ids():
    s1 = TraceStep(agent="o", action="x", detail="a")
    s2 = TraceStep(agent="o", action="x", detail="b")
    assert s1.step_id and s2.step_id
    assert s1.step_id != s2.step_id
    assert s1.parent_id is None


def test_reparent_specialist_trace_attaches_roots(orchestrator: OrchestratorAgent):
    parent_id = "a2a-id"
    specialist_trace = [
        TraceStep(agent="nutrition", action="processing", detail="root1"),
        TraceStep(agent="nutrition", action="response", detail="root2"),
    ]
    orchestrator._reparent_specialist_trace(specialist_trace, parent_id)
    assert all(s.parent_id == parent_id for s in specialist_trace)


def test_reparent_preserves_internal_hierarchy(orchestrator: OrchestratorAgent):
    parent = TraceStep(agent="nutrition", action="tool_call", detail="parent")
    child = TraceStep(
        agent="nutrition",
        action="tool_result",
        detail="child",
        parent_id=parent.step_id,
    )
    specialist_trace = [parent, child]

    orchestrator._reparent_specialist_trace(specialist_trace, "a2a-id")

    # root is re-parented under the orchestrator's a2a_call step
    assert parent.parent_id == "a2a-id"
    # internal edge is preserved
    assert child.parent_id == parent.step_id


def test_format_trace_markdown_indents_by_depth():
    root = TraceStep(agent="orchestrator", action="routing", detail="r")
    a2a = TraceStep(
        agent="orchestrator",
        action="a2a_call",
        detail="calling nutrition",
        parent_id=root.step_id,
    )
    spec = TraceStep(
        agent="nutrition",
        action="response",
        detail="answered",
        parent_id=a2a.step_id,
    )
    md = _format_trace_markdown("orchestrator", [root, a2a, spec], "nutrition")
    lines = md.splitlines()

    route_line = next(ln for ln in lines if "`routing`" in ln)
    a2a_line = next(ln for ln in lines if "`a2a_call`" in ln)
    spec_line = next(ln for ln in lines if "`response`" in ln)

    assert route_line.startswith("- ")          # depth 0
    assert a2a_line.startswith("    - ")        # depth 1
    assert spec_line.startswith("        - ")   # depth 2


def test_format_trace_includes_elapsed_and_tokens():
    step = TraceStep(
        agent="o",
        action="response",
        detail="x",
        elapsed_ms=123.4,
        tokens=42,
    )
    md = _format_trace_markdown("orchestrator", [step], "o")
    assert "123ms" in md
    assert "42 tok" in md


def test_a2a_serialize_roundtrip_preserves_step_id_and_parent():
    """A2A serialize → parse must keep ``step_id`` and ``parent_id``."""
    from erebots.agents.a2a.client import A2AClient
    from erebots.agents.a2a.server import _serialize_trace

    parent = TraceStep(agent="nutrition", action="tool_call", detail="p")
    child = TraceStep(
        agent="nutrition",
        action="tool_result",
        detail="c",
        parent_id=parent.step_id,
        elapsed_ms=50.0,
        tokens=10,
    )

    serialized = _serialize_trace([parent, child])
    parsed = A2AClient()._parse_trace(serialized)

    assert parsed[0].step_id == parent.step_id
    assert parsed[1].parent_id == parent.step_id
    assert parsed[1].elapsed_ms == 50.0
    assert parsed[1].tokens == 10

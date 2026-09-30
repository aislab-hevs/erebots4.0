"""Tests for multi-agent planning and fan-out in the orchestrator.

Two concerns, both mocked end-to-end (no network, no LLM):

* :meth:`OrchestratorAgent._parse_plan` — turning whatever a routing model
  replies into a list of registered agent names.
* the fan-out branch of ``run()`` — several specialists streamed in parallel on
  the ``side`` channel, then composed into one reply.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from erebots.agents.models import TraceStep
from erebots.agents.orchestrator import agent as agent_module
from erebots.agents.orchestrator.agent import OrchestratorAgent


def _orch(**overrides) -> OrchestratorAgent:
    """An orchestrator with two registered specialists and no live backend."""
    orch = OrchestratorAgent(api_key="x", api_base_url="http://invalid", model="m")
    orch.register_agent("nutrition", "http://x/a2a/nutrition", "Nutrition specialist.")
    orch.register_agent("personal", "http://x/a2a/personal", "Personal profile specialist.")
    return orch


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "orchestrator_max_specialists": 3,
        "multi_agent_composition": "synthesis",
        "routing_include_skills": True,
    }
    return SimpleNamespace(**{**base, **overrides})


async def _collect(agen) -> list:
    return [event async for event in agen]


# --- plan parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("nutrition", ["nutrition"]),
        ("none", []),
        ("personal, nutrition", ["personal", "nutrition"]),
        ("nutrition.", ["nutrition"]),                      # trailing punctuation
        ("Agent: nutrition", ["nutrition"]),                # preamble
        ("NUTRITION", ["nutrition"]),                       # casing
        ("nutrition, nutrition", ["nutrition"]),            # duplicates collapse
        ("nutrition, weather", ["nutrition"]),              # unknown name dropped
        ("weather", []),                                    # nothing registered
        ("", []),
    ],
)
def test_parse_plan_shapes(reply, expected):
    assert _orch()._parse_plan(reply) == expected


def test_parse_plan_ignores_reasoning_block():
    """A router that thinks out loud must not fan out to the agents it rejected."""
    orch = _orch()
    reply = "<think>personal could help, but this is purely dietary</think>\nnutrition"
    assert orch._parse_plan(reply) == ["nutrition"]


def test_parse_plan_reads_only_the_last_line():
    orch = _orch()
    reply = "Considering personal and nutrition...\nThe answer is:\nnutrition"
    assert orch._parse_plan(reply) == ["nutrition"]


def test_parse_plan_honours_the_cap(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(orchestrator_max_specialists=1))
    assert orch._parse_plan("personal, nutrition") == ["personal"]


def test_parse_plan_logs_when_nothing_matches(caplog):
    orch = _orch()
    with caplog.at_level("WARNING"):
        assert orch._parse_plan("I am not sure what to do here") == []
    assert "named no registered agent" in caplog.text


# --- routing context -------------------------------------------------------


async def test_plan_sees_recent_history():
    """A follow-up must reach the router with the turns that give it meaning."""
    orch = _orch()
    captured: dict = {}

    async def _chat(messages, system_prompt=None):
        captured["messages"] = messages
        return "nutrition"

    orch.generative.chat = _chat
    context = {
        "chat_history": [
            {"role": "user", "content": "how many calories do I need?"},
            {"role": "assistant", "content": "about 2450 kcal"},
        ]
    }

    assert await orch._plan("and for dinner?", context) == ["nutrition"]
    assert [m["content"] for m in captured["messages"]] == [
        "how many calories do I need?",
        "about 2450 kcal",
        "and for dinner?",
    ]


async def test_plan_flattens_multimodal_history():
    """Images in the history carry no routing signal and must not reach the router."""
    orch = _orch()
    captured: dict = {}

    async def _chat(messages, system_prompt=None):
        captured["messages"] = messages
        return "none"

    orch.generative.chat = _chat
    context = {
        "chat_history": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,zzz"}},
                ],
            }
        ]
    }

    await orch._plan("and now?", context)
    assert captured["messages"][0]["content"] == "what is this?"


# --- fan-out ---------------------------------------------------------------


def _fake_streams(chunks: dict[str, list], fail: dict[str, Exception] | None = None):
    """An ``a2a_client.send_message_stream`` yielding per-agent scripted chunks."""
    fail = fail or {}

    async def _stream(target, _message, _context=None, task_id=None):
        if target in fail:
            raise fail[target]
        for piece in chunks.get(target, []):
            await asyncio.sleep(0)  # let the sibling streams interleave
            yield piece

    return _stream


async def test_fanout_streams_each_specialist_on_the_side_channel(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="sections"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams(
        {"personal": ["74 kg, ", "34 years"], "nutrition": ["2450 ", "kcal"]}
    )

    events = await _collect(orch.run("calories and dinner?", context={}))

    side = [e for e in events if e.type == "token" and e.channel == "side"]
    assert "".join(e.text for e in side if e.agent == "personal") == "74 kg, 34 years"
    assert "".join(e.text for e in side if e.agent == "nutrition") == "2450 kcal"

    # The composed reply arrives on the default channel and carries both.
    final = next(e for e in events if e.type == "final")
    assert final.channel == "answer"
    assert "74 kg, 34 years" in final.text
    assert "2450 kcal" in final.text
    assert final.agent == orch.name


async def test_fanout_routing_event_exposes_the_plan(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="sections"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams({"personal": ["a"], "nutrition": ["b"]})

    events = await _collect(orch.run("x", context={}))

    assert events[0].type == "routing"
    assert events[0].data["plan"] == ["personal", "nutrition"]
    # ...and stays readable for consumers that only know the single-target field.
    assert events[0].data["target"] == "personal, nutrition"


async def test_fanout_synthesis_merges_the_contributions(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="synthesis"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams(
        {"personal": ["74 kg"], "nutrition": ["2450 kcal"]}
    )
    captured: dict = {}

    async def _chat_stream(messages, system_prompt=None):
        captured["prompt"] = messages[0]["content"]
        captured["system"] = system_prompt
        for piece in ["At 74 kg ", "you need 2450 kcal."]:
            yield piece

    orch.generative.chat_stream = _chat_stream

    events = await _collect(orch.run("calories?", context={}))

    # Both contributions are handed to the synthesis step...
    assert "74 kg" in captured["prompt"] and "2450 kcal" in captured["prompt"]
    assert "synthesis" in captured["system"].lower()
    # ...and its output is what the user reads, streamed on the answer channel.
    answer = [e for e in events if e.type == "token" and e.channel == "answer"]
    assert "".join(e.text for e in answer) == "At 74 kg you need 2450 kcal."
    assert next(e for e in events if e.type == "final").text == "At 74 kg you need 2450 kcal."


async def test_fanout_survives_one_failing_specialist(monkeypatch):
    """A dead specialist must not cost the turn the answers the others gave."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="sections"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams(
        {"nutrition": ["2450 kcal"]}, fail={"personal": RuntimeError("boom")}
    )

    events = await _collect(orch.run("x", context={}))

    assert not [e for e in events if e.type == "error"]
    final = next(e for e in events if e.type == "final")
    assert "2450 kcal" in final.text
    assert "personal" not in final.text  # no empty section for the failed agent
    # the failure is still visible in the trace
    errors = [e.trace_step for e in events if e.type == "trace" and e.trace_step.action == "error"]
    assert errors and "boom" in errors[0].detail


async def test_fanout_all_specialists_failing_emits_error(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams(
        {}, fail={"personal": RuntimeError("down"), "nutrition": RuntimeError("down")}
    )

    events = await _collect(orch.run("x", context={}))

    assert events[-1].type == "error"
    assert "down" in events[-1].text


async def test_fanout_reparents_specialist_trace_under_its_call(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="sections"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams(
        {
            "personal": [TraceStep(agent="personal", action="tool_call", detail="profile"), "ok"],
            "nutrition": ["fine"],
        }
    )

    events = await _collect(orch.run("x", context={}))
    steps = [e.trace_step for e in events if e.type == "trace"]
    by_id = {s.step_id: s for s in steps}

    plan_step = next(s for s in steps if s.action == "plan")
    calls = [s for s in steps if s.action == "a2a_call"]
    assert {s.parent_id for s in calls} == {plan_step.step_id}

    tool_step = next(s for s in steps if s.action == "tool_call")
    assert by_id[tool_step.parent_id].detail == "A2A request to personal"


async def test_synthesis_failure_falls_back_to_sections(monkeypatch):
    """A broken merge must not throw away work the specialists already did."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="synthesis"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams(
        {"personal": ["74 kg"], "nutrition": ["2450 kcal"]}
    )

    async def _boom(_messages, system_prompt=None):
        raise RuntimeError("synthesis backend down")
        yield  # pragma: no cover — makes this an async generator

    orch.generative.chat_stream = _boom

    events = await _collect(orch.run("x", context={}))

    final = next(e for e in events if e.type == "final")
    assert "74 kg" in final.text and "2450 kcal" in final.text


async def test_single_agent_plan_keeps_the_answer_channel(monkeypatch):
    """The one-specialist path must stream exactly as before the fan-out existed."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(return_value=["nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams({"nutrition": ["eat ", "oats"]})

    events = await _collect(orch.run("breakfast?", context={}))

    assert all(e.channel == "answer" for e in events)
    assert "".join(e.text for e in events if e.type == "token") == "eat oats"
    final = next(e for e in events if e.type == "final")
    assert final.agent == "nutrition"


# --- delegation -------------------------------------------------------------


async def test_fanout_gives_each_specialist_its_own_delegation_note(monkeypatch):
    """Each specialist must be told what the others are covering — and only that."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="sections"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    seen: dict[str, dict] = {}

    async def _stream(target, _message, context=None, task_id=None):
        seen[target] = context
        yield "ok"

    orch.a2a_client.send_message_stream = _stream
    await _collect(orch.run("x", context={"user_id": "u1"}))

    personal_note = seen["personal"]["delegation"]
    nutrition_note = seen["nutrition"]["delegation"]
    # each is addressed by name...
    assert "You are `personal`" in personal_note
    assert "You are `nutrition`" in nutrition_note
    # ...and the hands-off list holds the sibling and its scope, never itself
    assert "- `nutrition`: Nutrition specialist." in personal_note
    assert "- `personal`" not in personal_note
    assert "- `personal`: Personal profile specialist." in nutrition_note
    assert "- `nutrition`" not in nutrition_note
    assert "never claim to have performed them" in personal_note
    # the rest of the request context still reaches both
    assert seen["personal"]["user_id"] == seen["nutrition"]["user_id"] == "u1"


async def test_single_agent_gets_no_delegation_note(monkeypatch):
    """With one specialist there is nothing to stay out of."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(return_value=["nutrition"])
    seen: dict[str, dict] = {}

    async def _stream(target, _message, context=None, task_id=None):
        seen[target] = context
        yield "ok"

    orch.a2a_client.send_message_stream = _stream
    await _collect(orch.run("x", context={"user_id": "u1"}))

    assert "delegation" not in (seen["nutrition"] or {})


def test_delegation_note_is_empty_without_siblings():
    assert _orch()._delegation_note("nutrition", ["nutrition"]) == ""


def test_build_messages_injects_the_delegation_note():
    from erebots.agents.specialist import BaseSpecialist

    class _Specialist(BaseSpecialist):
        def setup_skills(self):
            pass

    specialist = _Specialist(
        agent_name="nutrition",
        api_key="x",
        api_base_url="http://invalid",
        model="m",
        system_prompt="domain prompt",
    )
    messages = specialist.build_messages(
        "question",
        {"user_context": "profile", "delegation": "stay out of X", "chat_history": []},
    )
    systems = [m["content"] for m in messages if m["role"] == "system"]
    assert systems == ["User context:\nprofile", "stay out of X"]


# --- timing -----------------------------------------------------------------


async def test_routing_cost_is_recorded_as_its_own_step(monkeypatch):
    """The routing LLM call runs on every turn; its cost must be visible."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(return_value=[])

    async def _stream(_messages, system_prompt=None):
        yield "hi"

    orch.generative.chat_stream = _stream
    events = await _collect(orch.run("ciao", context={}))

    step = next(
        e.trace_step for e in events if e.type == "trace" and e.trace_step.action == "routing"
    )
    assert step.elapsed_ms is not None and step.elapsed_ms >= 0
    assert "no specialist" in step.detail


async def test_each_a2a_call_step_is_closed_with_its_duration(monkeypatch):
    """Every specialist's step is re-emitted with elapsed_ms once its stream ends."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="sections"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams(
        {"personal": ["a"], "nutrition": ["b"]}
    )

    events = await _collect(orch.run("x", context={}))
    calls = [e.trace_step for e in events if e.type == "trace" and e.trace_step.action == "a2a_call"]

    # emitted twice per agent: once on dispatch, once closed with a duration
    assert len(calls) == 4
    closed = {s.step_id: s for s in calls if s.elapsed_ms is not None}
    assert len(closed) == 2  # one per specialist, keyed by the *same* step_id
    assert {s.detail for s in closed.values()} == {
        "A2A request to personal",
        "A2A request to nutrition",
    }


async def test_single_agent_call_step_is_closed_too(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(return_value=["nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams({"nutrition": ["ok"]})

    events = await _collect(orch.run("x", context={}))
    calls = [e.trace_step for e in events if e.type == "trace" and e.trace_step.action == "a2a_call"]
    assert len(calls) == 2 and calls[0].step_id == calls[1].step_id
    assert calls[1].elapsed_ms is not None


async def test_synthesis_step_carries_its_duration(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="synthesis"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])
    orch.a2a_client.send_message_stream = _fake_streams({"personal": ["a"], "nutrition": ["b"]})

    async def _chat_stream(_messages, system_prompt=None):
        yield "merged"

    orch.generative.chat_stream = _chat_stream
    events = await _collect(orch.run("x", context={}))

    step = next(
        e.trace_step for e in events if e.type == "trace" and e.trace_step.action == "synthesis"
    )
    assert step.elapsed_ms is not None


def test_gradio_adapter_replaces_a_step_instead_of_duplicating_it():
    from erebots.agents.models import TraceStep as TS
    from examples.gradio_ui.chat import _upsert_step

    trace: list = []
    step = TS(agent="orchestrator", action="a2a_call", detail="d")
    _upsert_step(trace, step)
    closed = TS(agent="orchestrator", action="a2a_call", detail="d", elapsed_ms=12.0)
    closed.step_id = step.step_id
    _upsert_step(trace, closed)

    assert len(trace) == 1
    assert trace[0].elapsed_ms == 12.0


# --- text-only deployments --------------------------------------------------


def _msgs():
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "cosa vedi?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,zz"}},
        ]},
        {"role": "assistant", "content": "plain text stays untouched"},
    ]


def test_multimodal_off_strips_images_and_keeps_text(monkeypatch):
    from erebots.agents import generative as gen_mod

    monkeypatch.setattr(gen_mod, "settings", SimpleNamespace(
        multimodal_enabled=False, enable_parser_retry=False))
    out = gen_mod._strip_image_parts(_msgs())

    assert isinstance(out[0]["content"], str)
    assert "cosa vedi?" in out[0]["content"]
    assert "text-only model" in out[0]["content"]
    assert "base64" not in out[0]["content"]
    assert out[1]["content"] == "plain text stays untouched"


def test_multimodal_off_image_only_turn_is_not_left_empty():
    from erebots.agents import generative as gen_mod

    only_image = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,zz"}}]}]
    out = gen_mod._strip_image_parts(only_image)
    # an empty user message makes a model answer nonsense; the placeholder does not
    assert out[0]["content"].strip()
    assert "cannot see it" in out[0]["content"]


def test_multimodal_on_leaves_the_request_untouched(monkeypatch):
    from erebots.agents import generative as gen_mod

    layer = gen_mod.GenerativeLayer.__new__(gen_mod.GenerativeLayer)
    layer.model, layer.thinking = "m", False
    monkeypatch.setattr(gen_mod, "settings", SimpleNamespace(
        multimodal_enabled=True, enable_parser_retry=False))
    kwargs = layer._build_kwargs(_msgs())
    assert isinstance(kwargs["messages"][0]["content"], list)


def test_multimodal_off_applies_to_every_call_path(monkeypatch):
    """The guard sits where messages meet the model, so no path can bypass it."""
    from erebots.agents import generative as gen_mod

    layer = gen_mod.GenerativeLayer.__new__(gen_mod.GenerativeLayer)
    layer.model, layer.thinking = "m", False
    monkeypatch.setattr(gen_mod, "settings", SimpleNamespace(
        multimodal_enabled=False, enable_parser_retry=False))
    for stream in (False, True):
        kwargs = layer._build_kwargs(_msgs(), stream=stream)
        assert isinstance(kwargs["messages"][0]["content"], str)


# --- image routing ----------------------------------------------------------


_IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,zz"}}


async def _captured_plan(orch, message, monkeypatch, multimodal: bool):
    monkeypatch.setattr(agent_module, "settings", _settings(multimodal_enabled=multimodal))
    seen: dict = {}

    async def _chat(messages, system_prompt=None):
        seen["content"] = messages[-1]["content"]
        return "nutrition"

    orch.generative.chat = _chat
    await orch._plan(message, None)
    return seen.get("content")


async def test_image_reaches_the_router_when_multimodal_is_on(monkeypatch):
    """The routing call already exists — the image rides along, no extra call."""
    content = await _captured_plan(_orch(), [_IMG], monkeypatch, multimodal=True)
    assert isinstance(content, list)
    assert content[-1] == _IMG
    assert content[0]["type"] == "text"


async def test_image_with_text_keeps_both_for_the_router(monkeypatch):
    message = [{"type": "text", "text": "quante calorie ha?"}, _IMG]
    content = await _captured_plan(_orch(), message, monkeypatch, multimodal=True)
    assert content[0]["text"] == "quante calorie ha?"
    assert content[-1] == _IMG


async def test_text_only_turn_is_unchanged_by_the_image_path(monkeypatch):
    content = await _captured_plan(_orch(), "solo testo", monkeypatch, multimodal=True)
    assert content == "solo testo"


async def test_text_only_deployment_routes_on_text_alone(monkeypatch):
    """multimodal off: the router must never be handed image parts."""
    content = await _captured_plan(_orch(), [_IMG], monkeypatch, multimodal=False)
    assert isinstance(content, str)
    assert "base64" not in content


async def test_image_only_turn_no_longer_dies_on_an_empty_plan(monkeypatch):
    """Before, an image with no text produced a placeholder no specialist matched."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multimodal_enabled=True))

    async def _chat(messages, system_prompt=None):
        return "nutrition"

    orch.generative.chat = _chat
    assert await orch._plan([_IMG], None) == ["nutrition"]


# --- a parked specialist surfaces as an event, not as the answer ------------


def _asking_stream(question="Quanti anni hai?", parked_id="t-1", text="Mi serve un dato. "):
    from erebots.agents.models import InputRequired

    async def _stream(_target, _message, _context=None, task_id=None):
        yield text
        yield InputRequired(question=question, task_id=parked_id)

    return _stream


async def test_single_agent_question_becomes_an_input_required_event(monkeypatch):
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(return_value=["nutrition"])
    orch.a2a_client.send_message_stream = _asking_stream()

    events = await _collect(orch.run("quante calorie?", context={}))

    asked = next(e for e in events if e.type == "input_required")
    assert asked.text == "Quanti anni hai?"
    assert asked.agent == "nutrition"
    assert asked.data["task_id"] == "t-1"


async def test_the_question_text_still_reaches_the_user(monkeypatch):
    """The marker is extra signal — the user must read the question regardless."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(return_value=["nutrition"])
    orch.a2a_client.send_message_stream = _asking_stream()

    events = await _collect(orch.run("x", context={}))
    assert "Mi serve un dato." in "".join(e.text for e in events if e.type == "token")


async def test_the_marker_never_leaks_into_the_answer_text(monkeypatch):
    """Regression: an unhandled marker would be stringified into the reply."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(return_value=["nutrition"])
    orch.a2a_client.send_message_stream = _asking_stream()

    events = await _collect(orch.run("x", context={}))
    final = next(e for e in events if e.type == "final")
    assert "InputRequired" not in final.text


async def test_a_waiting_contributor_does_not_stop_the_fanout(monkeypatch):
    """The others still answer, and their answers still get composed."""
    from erebots.agents.models import InputRequired

    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings(multi_agent_composition="sections"))
    orch.route_request = AsyncMock(return_value=["personal", "nutrition"])

    async def _stream(target, _message, _context=None, task_id=None):
        if target == "personal":
            yield InputRequired(question="Come ti chiami?", task_id="t-9")
        else:
            yield "2450 kcal"

    orch.a2a_client.send_message_stream = _stream
    events = await _collect(orch.run("x", context={}))

    asked = next(e for e in events if e.type == "input_required")
    assert (asked.agent, asked.data["task_id"]) == ("personal", "t-9")
    final = next(e for e in events if e.type == "final")
    assert "2450 kcal" in final.text
    assert "InputRequired" not in final.text


# --- resume: an answer goes back to the agent that asked, skipping routing --


def _pending_context(agent="personal", task_id="t-9") -> dict:
    return {"pending": {"agent": agent, "task_id": task_id}}


async def test_a_pending_answer_resumes_without_routing(monkeypatch):
    """Re-routing "38" could land it on a different specialist than the one
    that asked the age — resuming must bypass the router entirely."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(side_effect=AssertionError("routing must be skipped"))
    orch.a2a_client.can_resume = AsyncMock(return_value=True)
    seen = {}

    async def _stream(target, _message, _context=None, task_id=None):
        seen["target"], seen["task_id"] = target, task_id
        yield "Con 38 anni: 2000 kcal"

    orch.a2a_client.send_message_stream = _stream
    events = await _collect(orch.run("38", context=_pending_context()))

    assert (seen["target"], seen["task_id"]) == ("personal", "t-9")
    routing = next(e for e in events if e.type == "routing")
    assert routing.data["plan"] == ["personal"]
    assert routing.data.get("resumed") is True
    final = next(e for e in events if e.type == "final")
    assert "2000 kcal" in final.text


async def test_the_pre_flight_carries_the_turns_user_identity(monkeypatch):
    """``can_resume`` must know who is answering: the serving agent refuses a
    foreign resume, and the pre-flight mirrors that check to degrade a stolen
    or stale handle into normal routing instead of a failed turn."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.route_request = AsyncMock(side_effect=AssertionError("routing must be skipped"))
    orch.a2a_client.can_resume = AsyncMock(return_value=True)

    async def _stream(target, _message, _context=None, task_id=None):
        yield "ok"

    orch.a2a_client.send_message_stream = _stream
    context = {**_pending_context(), "user_id": "u-42"}
    await _collect(orch.run("38", context=context))

    orch.a2a_client.can_resume.assert_awaited_once_with(
        "personal", "t-9", user_id="u-42"
    )


async def test_a_stale_pending_handle_falls_back_to_routing(monkeypatch):
    """Task expired or server restarted: the turn must still work, routed."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.a2a_client.can_resume = AsyncMock(return_value=False)
    orch.route_request = AsyncMock(return_value=["nutrition"])
    seen = {}

    async def _stream(target, _message, _context=None, task_id=None):
        seen["target"], seen["task_id"] = target, task_id
        yield "2000 kcal"

    orch.a2a_client.send_message_stream = _stream
    events = await _collect(orch.run("38", context=_pending_context()))

    orch.route_request.assert_awaited_once()
    assert (seen["target"], seen["task_id"]) == ("nutrition", None)
    assert "2000 kcal" in next(e for e in events if e.type == "final").text


async def test_a_malformed_pending_handle_is_ignored(monkeypatch):
    """The handle is frontend-supplied input: garbage means "no pending"."""
    orch = _orch()
    monkeypatch.setattr(agent_module, "settings", _settings())
    orch.a2a_client.can_resume = AsyncMock(return_value=True)
    orch.route_request = AsyncMock(return_value=["nutrition"])

    async def _stream(_target, _message, _context=None, task_id=None):
        yield "ok"

    orch.a2a_client.send_message_stream = _stream
    await _collect(orch.run("x", context={"pending": {"agent": "", "task_id": 5}}))

    orch.route_request.assert_awaited_once()
    orch.a2a_client.can_resume.assert_not_awaited()

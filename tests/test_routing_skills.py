"""Tests that registered skills feed the routing/fallback prompts."""

from __future__ import annotations

import pytest

from erebots.agents.orchestrator.agent import OrchestratorAgent


@pytest.fixture
def orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent(
        api_key="x",
        api_base_url="http://example.invalid",
        model="m",
    )


def test_register_agent_stores_skills(orchestrator: OrchestratorAgent):
    orchestrator.register_agent(
        name="nutrition",
        a2a_url="http://x/a2a/nutrition",
        description="Nutrition specialist.",
        skills=[
            {"name": "search_recipes", "description": "find recipes by query"},
            {"name": "calculate_bmi", "description": "compute BMI"},
        ],
    )
    info = {a["name"]: a for a in orchestrator.get_agents_info()}
    assert info["nutrition"]["description"] == "Nutrition specialist."
    assert [s["name"] for s in info["nutrition"]["skills"]] == [
        "search_recipes",
        "calculate_bmi",
    ]


def test_routing_prompt_includes_skill_descriptions(orchestrator: OrchestratorAgent):
    orchestrator.register_agent(
        name="nutrition",
        a2a_url="http://x/a2a/nutrition",
        description="Nutrition, health and cooking specialist.",
        skills=[
            {"name": "search_recipes", "description": "find recipes by free-text query"},
        ],
    )
    prompt = orchestrator._routing_prompt
    assert "nutrition" in prompt
    # the granular skill signal must reach the router prompt
    assert "search_recipes" in prompt
    assert "find recipes by free-text query" in prompt


def test_lean_routing_prompt_omits_skills_but_keeps_agents(
    orchestrator: OrchestratorAgent, monkeypatch: pytest.MonkeyPatch
):
    """routing_include_skills=false → name+description only in the prompt;
    the registered skills stay available (card/UI are unaffected)."""
    from types import SimpleNamespace

    from erebots.agents.orchestrator import agent as agent_mod

    monkeypatch.setattr(
        agent_mod,
        "settings",
        SimpleNamespace(routing_include_skills=False, orchestrator_max_specialists=3),
    )
    orchestrator.register_agent(
        name="nutrition",
        a2a_url="http://x/a2a/nutrition",
        description="Nutrition specialist.",
        skills=[{"name": "search_recipes", "description": "find recipes"}],
    )
    prompt = orchestrator._routing_prompt
    assert "nutrition: Nutrition specialist." in prompt
    assert "search_recipes" not in prompt  # lean prompt
    # ...but the skills are still registered for the card and the UI.
    info = {a["name"]: a for a in orchestrator.get_agents_info()}
    assert [s["name"] for s in info["nutrition"]["skills"]] == ["search_recipes"]


def test_register_agent_without_skills_is_backward_compatible(
    orchestrator: OrchestratorAgent,
):
    orchestrator.register_agent(
        name="plain",
        a2a_url="http://x/a2a/plain",
        description="Plain agent.",
    )
    info = {a["name"]: a for a in orchestrator.get_agents_info()}
    assert info["plain"]["skills"] == []
    assert "plain" in orchestrator._routing_prompt

"""Tests for application configuration and A2A agent discovery."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import respx

from erebots.agents import loader


@pytest.fixture
def fake_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the loader to a temporary agents.yaml."""
    path = tmp_path / "agents.yaml"
    monkeypatch.setattr(loader, "settings", replace(loader.settings, agents_config_path=str(path)))
    return path


@respx.mock
async def test_load_agents_discovers_card(fake_yaml: Path):
    fake_yaml.write_text(
        "agents:\n  - a2a_url: http://specialist.local/a2a/foo\n"
    )
    respx.get("http://specialist.local/a2a/foo/.well-known/agent-card.json").mock(
        return_value=httpx.Response(
            200,
            json={"name": "foo", "description": "Foo specialist"},
        )
    )

    result = await loader.load_agents()

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.name == "foo"
    assert entry.description == "Foo specialist"
    assert entry.a2a_url == "http://specialist.local/a2a/foo"
    assert entry.skills == []


@respx.mock
async def test_load_agents_extracts_skills(fake_yaml: Path):
    fake_yaml.write_text(
        "agents:\n  - a2a_url: http://specialist.local/a2a/nut\n"
    )
    respx.get("http://specialist.local/a2a/nut/.well-known/agent-card.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "name": "nutrition",
                "description": "Nutrition specialist",
                "skills": [
                    {"name": "search_recipes", "description": "find recipes"},
                    {"id": "calculate_bmi", "description": "compute BMI"},
                    {"description": "nameless skill is skipped"},
                ],
            },
        )
    )

    result = await loader.load_agents()

    entry = result.entries[0]
    assert [s.name for s in entry.skills] == ["search_recipes", "calculate_bmi"]
    assert entry.skills[0].description == "find recipes"


@respx.mock
async def test_load_agents_skips_unreachable(fake_yaml: Path):
    fake_yaml.write_text(
        "agents:\n"
        "  - a2a_url: http://up.local/a2a/ok\n"
        "  - a2a_url: http://down.local/a2a/bad\n"
    )
    respx.get("http://up.local/a2a/ok/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json={"name": "ok", "description": "Up"})
    )
    respx.get("http://down.local/a2a/bad/.well-known/agent-card.json").mock(
        side_effect=httpx.ConnectError("boom")
    )

    result = await loader.load_agents()

    assert [e.name for e in result.entries] == ["ok"]


@respx.mock
async def test_load_agents_skips_card_without_name(fake_yaml: Path):
    fake_yaml.write_text("agents:\n  - a2a_url: http://x.local/a2a/x\n")
    respx.get("http://x.local/a2a/x/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json={"description": "no name"})
    )

    result = await loader.load_agents()

    assert result.entries == []


async def test_load_agents_missing_file_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(loader, "settings", replace(loader.settings, agents_config_path=str(tmp_path / "missing.yaml")))
    result = await loader.load_agents()
    assert result.entries == []


async def test_load_agents_missing_a2a_url_skipped(fake_yaml: Path):
    fake_yaml.write_text("agents:\n  - description: orphan entry\n")
    result = await loader.load_agents()
    assert result.entries == []


# --- following the declared endpoint ----------------------------------------


def _card(**over) -> dict:
    card = {
        "name": "nutrition",
        "description": "Nutrition specialist.",
        "skills": [{"name": "calculate_bmi", "description": "BMI"}],
        "supportedInterfaces": [
            {"url": "https://public.example/a2a/nutrition/", "protocolBinding": "JSONRPC"}
        ],
    }
    card.update(over)
    return card


def test_declared_endpoint_wins_over_the_configured_url():
    """Discovery must follow what the agent says about itself, not what we assumed."""
    from erebots.agents.loader import _declared_endpoint

    url = _declared_endpoint(_card(), "http://internal:7860/a2a/nutrition")
    assert url == "https://public.example/a2a/nutrition"  # trailing slash normalised


def test_declared_endpoint_falls_back_when_the_card_declares_nothing(caplog):
    """A peer that declares no endpoint stays reachable at the address we knew."""
    from erebots.agents.loader import _declared_endpoint

    fetched = "http://internal:7860/a2a/nutrition"
    with caplog.at_level("WARNING"):
        assert _declared_endpoint(_card(supportedInterfaces=[]), fetched) == fetched
    assert "no usable endpoint" in caplog.text


def test_declared_endpoint_skips_malformed_entries():
    from erebots.agents.loader import _declared_endpoint

    card = _card(supportedInterfaces=[
        "not-a-dict",
        {"protocolBinding": "JSONRPC"},           # no url
        {"url": "   "},                            # blank
        {"url": "https://good.example/a2a/x"},
    ])
    assert _declared_endpoint(card, "http://x") == "https://good.example/a2a/x"


def test_matching_endpoint_is_not_reported_as_a_divergence(caplog):
    from erebots.agents.loader import _declared_endpoint

    same = "https://public.example/a2a/nutrition"
    with caplog.at_level("INFO"):
        assert _declared_endpoint(_card(), same) == same
    assert "declares a different endpoint" not in caplog.text


# --- the address agents.yaml publishes --------------------------------------


def test_configured_urls_maps_names_to_published_addresses(tmp_path, monkeypatch):
    """The card's advertised URL comes from the same file the orchestrator dials."""
    from erebots.agents import loader

    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(
        "agents:\n"
        "  - https://app.example.org/a2a/nutrition\n"
        "  - http://localhost:7860/a2a/personal/\n"       # trailing slash
        "  - a2a_url: https://other.example/a2a/weather\n"  # dict form
    )
    monkeypatch.setattr(loader, "settings", replace(loader.settings, agents_config_path=str(yaml_file)))

    assert loader.configured_urls() == {
        "nutrition": "https://app.example.org/a2a/nutrition",
        "personal": "http://localhost:7860/a2a/personal",
        "weather": "https://other.example/a2a/weather",
    }


def test_configured_urls_is_empty_without_the_file(tmp_path, monkeypatch):
    from erebots.agents import loader

    monkeypatch.setattr(loader, "settings", replace(loader.settings, agents_config_path=str(tmp_path / "missing.yaml")))
    assert loader.configured_urls() == {}


@respx.mock
async def test_load_agents_reports_the_urls_it_could_not_discover(fake_yaml: Path):
    fake_yaml.write_text(
        "agents:\n"
        "  - http://specialist.local/a2a/up\n"
        "  - http://specialist.local/a2a/down\n"
        "  - http://specialist.local/a2a/nameless\n"
    )
    respx.get("http://specialist.local/a2a/up/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json={"name": "up"})
    )
    respx.get("http://specialist.local/a2a/down/.well-known/agent-card.json").mock(
        side_effect=httpx.ConnectError("refused")
    )
    respx.get("http://specialist.local/a2a/nameless/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json={})
    )

    result = await loader.load_agents()

    assert [e.name for e in result.entries] == ["up"]
    assert result.failed == [
        "http://specialist.local/a2a/down",
        "http://specialist.local/a2a/nameless",
    ]


@respx.mock
async def test_load_agents_with_urls_fetches_only_those(fake_yaml: Path):
    fake_yaml.write_text("agents:\n  - http://specialist.local/a2a/listed\n")
    listed = respx.get("http://specialist.local/a2a/listed/.well-known/agent-card.json")
    respx.get("http://specialist.local/a2a/retry/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json={"name": "retry"})
    )

    result = await loader.load_agents(urls=["http://specialist.local/a2a/retry"])

    assert [e.name for e in result.entries] == ["retry"]
    assert result.failed == []
    assert not listed.called


@respx.mock
async def test_explicit_config_is_shared_by_discovery_and_published_cards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("agents.yaml").write_text("agents:\n  - http://wrong/a2a/wrong\n")
    config = tmp_path / "deployment.yaml"
    config.write_text("agents:\n  - http://peer/a2a/personal/\n")
    respx.get("http://peer/a2a/personal/.well-known/agent-card.json").respond(
        200, json={"name": "personal"},
    )
    assert loader.configured_urls(config) == {"personal": "http://peer/a2a/personal"}
    loaded = await loader.load_agents(config_path=config)
    assert [entry.name for entry in loaded.entries] == ["personal"]


@respx.mock
async def test_default_config_is_relative_to_application_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loader, "settings", replace(loader.settings, agents_config_path="agents.yaml"))
    Path("agents.yaml").write_text("agents:\n  - http://peer/a2a/custom\n")
    respx.get("http://peer/a2a/custom/.well-known/agent-card.json").respond(
        200, json={"name": "custom"},
    )
    loaded = await loader.load_agents()
    assert [entry.name for entry in loaded.entries] == ["custom"]


async def test_explicit_empty_urls_do_not_read_yaml(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("[not valid yaml")
    loaded = await loader.load_agents(urls=[], config_path=config)
    assert loaded.entries == []
    assert loaded.failed == []

"""Test the example app composition (bootstrap.build_agents).

Runs offline (lazy engine, no LLM/network connections at build time) and proves
the graph wires up: the local specialists, their A2A apps, the tool servers and
the orchestrator.
"""

from __future__ import annotations

import asyncio

from bootstrap import build_agents
from erebots.agents.specialist import BaseSpecialist
from erebots.config import Settings
from specialists.registry import build_specialists

_EXPECTED_SPECIALISTS = {"conversion", "personal"}


def test_build_specialists_declares_the_local_agents():
    """The one place to add or remove a local specialist, decoupled from infra."""
    specialists = build_specialists(Settings(enable_auth=False))

    assert set(specialists) == _EXPECTED_SPECIALISTS
    assert all(isinstance(a, BaseSpecialist) for a in specialists.values())


def test_build_agents_wires_the_example_graph():
    agents = build_agents(Settings(enable_auth=False))

    assert set(agents.specialists) == _EXPECTED_SPECIALISTS
    assert set(agents.a2a_apps) == _EXPECTED_SPECIALISTS
    # No internal token provider when auth is disabled.
    assert agents.internal_token_provider is None


def test_build_agents_wires_persistence_independently_of_the_domain():
    """The store is built and injected whatever specialists exist.

    It used to live in an agent's package, so which specialists an application
    shipped decided whether it had persistence at all. It does not any more,
    and this is the assertion that says so: a composition with one specialist
    that touches no user data still gets an engine and a store.
    """
    agents = build_agents(Settings(enable_auth=False))

    assert agents.user_context_store is not None
    assert agents.engine is not None
    assert agents.orchestrator.conversation.persists_history


def test_build_agents_exposes_the_mcp_tool_servers_bound_to_app_state():
    agents = build_agents(Settings(enable_auth=False))

    # Only the servers a built specialist actually calls, so a composition
    # carries no server nobody talks to.
    assert set(agents.mcp_apps) == {"conversion", "personal"}
    # The names the tool loop calls have to be the names the servers publish.
    published = {
        name: {t.name for t in asyncio.run(server.list_tools())}
        for name, server in agents.mcp_apps.items()
    }
    assert published == {
        "conversion": {"convert_units"},
        "personal": {"update_profile", "manage_consent", "get_user_context"},
    }


def test_build_specialists_takes_domain_settings_explicitly():
    from specialists.config import DomainSettings

    specialists = build_specialists(
        Settings(enable_auth=False), DomainSettings(conversion_model="injected")
    )
    assert specialists["conversion"].generative.model == "injected"


def test_the_personal_agent_pins_the_user_id_server_side():
    """The agent that owns the profile must not let a model choose whose it reads.

    An attacker value is sent and has to be overwritten: filling in a blank
    ``user_id`` is not the property that matters. That the framework applies the
    hook at all is asserted in tests/test_specialist_stream.py, on both tool
    loops; this is the assertion that the shipped agent uses it.
    """
    agents = build_agents(Settings(enable_auth=False))

    pinned = agents.specialists["personal"].prepare_tool_args(
        {"user_id": "attacker"}, {"user_id": "u"}
    )
    assert pinned["user_id"] == "u"

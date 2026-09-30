"""Composition root for the example application.

Wires the infrastructure (orchestrator, persistence, auth, A2A apps) around the
specialists declared in ``specialists/registry.py``. This is *application* code,
so it imports the domain (``specialists``) directly — the framework (``erebots``)
never does. Adding/removing a local specialist is a change in
``specialists/registry.py``, not here; routing to a remote autonomous agent is a
URL in ``agents.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from a2a.server.tasks import (
    DatabasePushNotificationConfigStore,
    DatabaseTaskStore,
    TaskStore,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.applications import Starlette

from erebots.agents.a2a.server import create_a2a_app
from erebots.agents.loader import configured_urls
from erebots.agents.orchestrator import OrchestratorAgent
from erebots.agents.specialist import BaseSpecialist
from erebots.config import Settings
from erebots.persistence import UserContextStore, make_engine, make_session_factory
from erebots.security.internal_token import InternalTokenProvider
from specialists.config import DomainSettings
from specialists.conversion import build_conversion_mcp
from specialists.personal import ProfileStore, build_personal_mcp

# Secrets that must never guard a real deployment: the empty string and the
# placeholders shipped in env.example and the example Keycloak realm.
_INSECURE_INTERNAL_SECRETS = {
    "",
    "erebots-internal-secret",
    "change-me-internal-client-secret",
}


@dataclass
class AppAgents:
    """The agents, persistence and A2A apps for one app instance.

    ``specialists`` / ``a2a_apps`` are keyed by agent name; the app mounts and
    routes generically over them.
    """

    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    user_context_store: UserContextStore
    specialists: dict[str, BaseSpecialist]
    a2a_apps: dict[str, Starlette]
    orchestrator: OrchestratorAgent
    internal_token_provider: InternalTokenProvider | None
    a2a_task_store: TaskStore | None = None
    # MCP tool servers built with the SDK, keyed by mount name
    # (``/tools/<name>``). The framework mounts them and runs their lifespan.
    mcp_apps: dict[str, Any] = field(default_factory=dict)


def build_agents(
    settings: Settings,
    domain_settings: DomainSettings | None = None,
    *,
    specialists: dict[str, BaseSpecialist] | None = None,
) -> AppAgents:
    """Build an agent graph + persistence for one app instance.

    ``domain_settings`` (the specialists' own configuration) defaults to the
    environment as it is now. ``specialists`` defaults to the full example
    domain; ``demo.py`` passes a smaller set instead — everything below is
    generic given that dict.
    """
    if domain_settings is None:
        domain_settings = DomainSettings.from_env()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    store = UserContextStore(session_factory)

    # Fail fast on an insecure internal-auth setup: when auth is enabled the
    # service-account secret guards /a2a/* and /tools/* and must be a real,
    # rotated value — never the placeholder shipped in env.example / the realm.
    if settings.enable_auth and settings.internal_client_secret in _INSECURE_INTERNAL_SECRETS:
        raise RuntimeError(
            "enable_auth=true but internal_client_secret is empty or set to a "
            "well-known placeholder. Set a unique secret (see env.example) that "
            "matches the 'erebots-internal' client in your Keycloak realm."
        )

    internal_token_provider: InternalTokenProvider | None = None
    if settings.enable_auth and settings.internal_client_secret:
        token_url = (
            f"{settings.keycloak_internal_url.rstrip('/')}/realms/"
            f"{settings.keycloak_realm}/protocol/openid-connect/token"
        )
        internal_token_provider = InternalTokenProvider(
            token_url=token_url,
            client_id=settings.internal_client_id,
            client_secret=settings.internal_client_secret,
        )

    # The local specialists are declared in specialists/registry.py; adding one
    # never touches the infrastructure wiring below. The import stays deferred
    # so that composing a different domain does not pull the example set into
    # memory just to build the infrastructure around it.
    if specialists is None:
        from specialists.registry import build_specialists

        specialists = build_specialists(settings, domain_settings)

    # Each card advertises the URL agents.yaml already publishes the agent at:
    # that file is the single place declaring these addresses, and the card must
    # not carry a second copy that can drift from it. Only an agent missing from
    # the file falls back to a local default.
    published = configured_urls(settings.agents_config_path)
    fallback = f"http://localhost:{settings.app_port}"

    # Persist A2A tasks in the app DB when running multiple instances (opt-in);
    # otherwise each app uses its own in-memory store.
    # Push configs follow the tasks they belong to: with several instances, the
    # one that finishes a task must find the webhook the other one registered.
    task_store = DatabaseTaskStore(engine) if settings.a2a_persistent_tasks else None
    push_store = (
        DatabasePushNotificationConfigStore(engine) if settings.a2a_persistent_tasks else None
    )
    a2a_apps = {
        name: create_a2a_app(
            agent,
            url=published.get(name, f"{fallback}/a2a/{name}"),
            task_store=task_store,
            push_config_store=push_store,
        )
        for name, agent in specialists.items()
    }

    orchestrator = OrchestratorAgent(
        api_key=settings.orchestrator_api_key,
        api_base_url=settings.orchestrator_api_base_url,
        model=settings.orchestrator_model,
        user_context_store=store,
        thinking=settings.orchestrator_thinking,
    )

    if internal_token_provider is not None:
        token_getter = internal_token_provider.get_token
        for agent in specialists.values():
            agent.mcp_client.token_provider = token_getter
        orchestrator.a2a_client.token_provider = token_getter

    return AppAgents(
        engine=engine,
        session_factory=session_factory,
        user_context_store=store,
        specialists=specialists,
        a2a_apps=a2a_apps,
        orchestrator=orchestrator,
        internal_token_provider=internal_token_provider,
        a2a_task_store=task_store,
        mcp_apps=_tool_servers_for(specialists, store),
    )


def _tool_servers_for(specialists: dict, store) -> dict:
    """Mount only the tool servers some built specialist actually calls.

    Derived from the agents rather than listed unconditionally, because an
    unconditional list was wrong the moment a second composition existed: a
    composition building a subset was handed every tool server the full app
    has, including ones whose only caller it does not build. A mounted server
    nobody calls is dead weight, and a false signal in a demonstrator whose
    point is what it leaves out.

    A tool server needing the user-context store takes it as a closure argument
    here, which is how the personal agent's three tools reach it.

    The agent-to-tool mapping is spelled out rather than read back from the
    agents: ``BaseSpecialist`` keeps its ``mcp_server_urls`` in a private
    attribute, and application code has no business reaching into that.
    """
    tool_for_agent = {
        "conversion": "conversion",
        "personal": "personal",
    }
    builders = {
        "conversion": build_conversion_mcp,
        "personal": lambda: build_personal_mcp(ProfileStore(store)),
    }
    wanted = {tool_for_agent[name] for name in specialists if name in tool_for_agent}
    return {name: build() for name, build in builders.items() if name in wanted}

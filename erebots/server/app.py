"""Generic ASGI app factory for an erebots deployment.

Assembles the framework-level HTTP surface — CORS, auth, a health check, the
SSE chat endpoint and the A2A specialist mounts — around an orchestrator. It
**boots with zero specialists** (the orchestrator then answers directly), so a
minimal running app is just ``create_app(orchestrator)``.

It imports neither a specific frontend (``gradio``) nor the domain
(``specialists``): an example app builds those and layers them on top of the
returned FastAPI instance (root route, UI pages, MCP tool servers).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from erebots.agents.a2a.conformance import create_conformance_app
from erebots.agents.a2a.push import aclose_webhook_client
from erebots.agents.loader import load_agents
from erebots.api.auth_middleware import auth_dispatch, extract_user_token
from erebots.api.routes.router import router as api_router
from erebots.config import settings
from erebots.persistence import Base
from erebots.security.keycloak import KeycloakError, decode_user_token
from erebots.server.sse import create_chat_router

if TYPE_CHECKING:
    from a2a.server.tasks import TaskStore
    from mcp.server.fastmcp import FastMCP
    from sqlalchemy.ext.asyncio import AsyncEngine
    from starlette.applications import Starlette

    from erebots.agents.orchestrator import OrchestratorAgent


async def default_session_resolver(request: Request) -> dict:
    """Resolve a chat session from the auth cookie or bearer (anonymous if absent)."""
    token = extract_user_token(request)
    if not token:
        return {}
    try:
        payload = await decode_user_token(token)
    except KeycloakError:
        return {}
    user_id = payload.get("sub")
    return {"user_id": user_id, "jwt_claims": payload} if user_id else {}


def create_app(
    orchestrator: OrchestratorAgent,
    *,
    a2a_apps: dict[str, Starlette] | None = None,
    mcp_apps: dict[str, FastMCP] | None = None,
    engine: AsyncEngine | None = None,
    a2a_task_store: TaskStore | None = None,
    session_resolver: Callable[[Request], Awaitable[dict]] | None = None,
    token_provider: Callable[[], Awaitable[str]] | None = None,
    agents_config_path: str | Path | None = None,
    agent_urls: Iterable[str] | None = None,
) -> FastAPI:
    """Build the framework's FastAPI app around ``orchestrator``.

    Args:
        orchestrator: The routing agent driving the SSE chat endpoint.
        a2a_apps: Local specialists' A2A apps to mount at ``/a2a/<name>``.
        mcp_apps: Tool servers to mount at ``/tools/<name>``; their Streamable
            HTTP endpoint is then ``/tools/<name>/mcp``.
            Omit for a minimal app where the orchestrator answers directly.
        engine: If given, the DB schema is created in the lifespan and the
            engine disposed on shutdown.
        a2a_task_store: If given, initialized in the lifespan.
        session_resolver: Override how a chat session is derived from a request;
            defaults to :func:`default_session_resolver`.
        token_provider: Optional async callable returning an M2M bearer,
            forwarded to the A2A agent-card discovery.
        agents_config_path: Application-owned YAML configuration. Defaults to
            ``settings.agents_config_path``, relative to the working directory.
        agent_urls: Explicit discovery endpoints, overriding YAML. Pass an
            empty iterable to disable discovery from configuration.

    Returns:
        A configured FastAPI app. A frontend (e.g. gradio pages, the root route)
        is layered by the caller onto the return value.
    """
    a2a_apps = a2a_apps or {}
    mcp_apps = mcp_apps or {}
    initial_urls = tuple(agent_urls) if agent_urls is not None else None

    # Lazy agent discovery (agents.yaml → agent cards → orchestrator registry).
    # Runs on the first API hit, not at startup: the cards of *local* agents are
    # served by this very app, which is not accepting connections until the
    # lifespan yields. The agents found stay registered; the URLs that did not
    # answer are kept and retried on a later request, throttled by
    # ``discovery_retry_seconds``, so one that was down at the first request
    # is picked up when it comes back instead of being lost until a restart.
    discovered = False
    unreached: list[str] = []
    retry_after = 0.0
    agents_lock = asyncio.Lock()

    def discovery_due() -> bool:
        if not discovered:
            return True
        return bool(unreached) and time.monotonic() >= retry_after

    async def ensure_agents_registered() -> None:
        nonlocal discovered, unreached, retry_after
        if not discovery_due():
            return
        async with agents_lock:
            if not discovery_due():
                return
            if discovered:
                loaded = await load_agents(token_provider=token_provider, urls=unreached)
            else:
                loaded = await load_agents(
                    token_provider=token_provider, urls=initial_urls,
                    config_path=agents_config_path,
                )
            for entry in loaded.entries:
                orchestrator.register_agent(
                    name=entry.name,
                    a2a_url=entry.a2a_url,
                    description=entry.description,
                    skills=[
                        {"name": s.name, "description": s.description}
                        for s in entry.skills
                    ],
                )
            discovered = True
            unreached = list(loaded.failed)
            retry_after = time.monotonic() + settings.discovery_retry_seconds

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application startup and shutdown resources.

        Initialize the database schema and task store before serving requests.
        After startup succeeds, teardown closes the shared HTTP clients and
        database engine even when the running application fails.

        Args:
            app: The FastAPI application entering its lifespan.

        Yields:
            Control while the application is serving requests.
        """
        if engine is not None:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        if a2a_task_store is not None:
            await a2a_task_store.initialize()
        try:
            async with AsyncExitStack() as stack:
                # A mounted app gets no lifespan of its own, so each MCP
                # server's session manager has to be started here — without it
                # every tool call fails on a missing task group.
                for server in mcp_apps.values():
                    await stack.enter_async_context(server.session_manager.run())
                yield
        finally:
            # Close the orchestrator's pooled A2A connections, the shared
            # webhook client (a mounted A2A app gets no lifespan of its own,
            # so it cannot close it), then the DB.
            # Prefer the orchestrator's own shutdown: it knows whether it built
            # its A2A client or was handed one, and an injected client belongs
            # to whoever passed it in. The direct reach-in stays as the fallback
            # for orchestrators that predate `aclose` or are duck-typed in tests.
            aclose = getattr(orchestrator, "aclose", None) or getattr(
                getattr(orchestrator, "a2a_client", None), "aclose", None
            )
            if aclose is not None:
                await aclose()
            await aclose_webhook_client()
            if engine is not None:
                await engine.dispose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        """Health check endpoint for Docker and load balancers."""
        return {"status": "healthy", "service": "erebots-api"}

    app.middleware("http")(auth_dispatch)

    # Registered *after* the auth dispatcher, and the order is load-bearing.
    # Starlette inserts each middleware at index 0, so the last one added ends up
    # outermost — and CORS has to wrap authentication, for two reasons:
    #   * a browser preflight carries no cookie and no Authorization header, so
    #     auth would reject it before CORS ever answered it, and
    #   * a 401 on the *actual* request still has to carry
    #     Access-Control-Allow-Origin, or the browser reports an opaque CORS
    #     failure instead of the authentication error the user needs to see.
    # Passing OPTIONS through auth instead fixes only the first. See tests/test_cors.py.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # API router before any caller-layered UI mounts so /api takes priority.
    app.include_router(api_router)

    @app.get("/api/me")
    async def me(request: Request):
        """The current user's identity (from cookie or bearer), for frontends."""
        session = await (session_resolver or default_session_resolver)(request)
        claims = session.get("jwt_claims") or {}
        return {
            "authenticated": bool(session.get("user_id")),
            "name": claims.get("name") or claims.get("preferred_username") or "",
            "username": claims.get("preferred_username", ""),
            "email": claims.get("email", ""),
            "account_url": (
                f"{settings.keycloak_url}/realms/{settings.keycloak_realm}/account"
                if settings.enable_auth
                else None
            ),
        }

    @app.get("/api/agents")
    async def list_agents():
        """The orchestrator (router) plus every discovered specialist, with skills."""
        await ensure_agents_registered()
        return {
            "agents": [
                {**orchestrator.get_self_info(), "role": "router"},
                *({**info, "role": "specialist"} for info in orchestrator.get_agents_info()),
            ]
        }

    # Canonical SSE chat endpoint (POST /api/chat/stream): any frontend can
    # consume the orchestrator's AgentEvent stream over this standard path.
    app.include_router(
        create_chat_router(
            orchestrator,
            session_resolver=session_resolver or default_session_resolver,
            before_turn=ensure_agents_registered,
        )
    )

    # Mount each local specialist's A2A endpoint at /a2a/<name>.
    for name, a2a_app in a2a_apps.items():
        app.mount(f"/a2a/{name}", a2a_app)

    # Mount each MCP tool server at /tools/<name> (endpoint /tools/<name>/mcp).
    # Building the ASGI app here is also what creates the session manager the
    # lifespan above starts. The SDK's DNS-rebinding guard accepts only a
    # loopback Host with a port, which is what a specialist sends: these
    # servers are reachable from inside the deployment, not through a proxy.
    for name, server in mcp_apps.items():
        app.mount(f"/tools/{name}", server.streamable_http_app())

    # The conformance endpoint verifies this framework's A2A implementation, so
    # it is wired here rather than per application — but only on request: it
    # serves canned payloads and belongs to a test run, not to a deployment.
    if settings.a2a_conformance_agent:
        # Its own in-memory store on purpose: the suite's fixtures — including a
        # task parked forever — are test data and must not land in the
        # application database alongside real ones.
        app.mount(
            "/a2a/tck",
            create_conformance_app(url=f"http://localhost:{settings.app_port}/a2a/tck"),
        )
        # The same scenarios behind a card declaring none of the optional
        # capabilities. The suite skips a capability's *absence* tests against
        # a server that has it, so CORE-CAP-001/002/003 are only reachable on
        # a second endpoint; point a second TCK run at this one.
        app.mount(
            "/a2a/tck-minimal",
            create_conformance_app(
                url=f"http://localhost:{settings.app_port}/a2a/tck-minimal",
                declare_capabilities=False,
            ),
        )

    return app

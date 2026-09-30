"""The framework's ``create_app`` factory.

Proves the framework HTTP surface boots on its own — with **zero specialists**
(the orchestrator then answers directly) — and that passing specialists mounts
them at ``/a2a/<name>``.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

from erebots.agents.orchestrator import OrchestratorAgent
from erebots.server import create_app


def _orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent(api_key="EMPTY", api_base_url="http://x/v1", model="local")


def test_create_app_boots_with_zero_specialists():
    """A minimal deployment: no specialists, no DB engine, still serves."""
    app = create_app(_orchestrator())

    with TestClient(app) as client:  # runs the (no-op) lifespan
        r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert not any(getattr(route, "path", "").startswith("/a2a/") for route in app.routes)


def test_create_app_mounts_specialists_when_given():
    """Passing a2a_apps mounts each at /a2a/<name>."""
    app = create_app(_orchestrator(), a2a_apps={"foo": Starlette()})

    assert any(getattr(route, "path", "") == "/a2a/foo" for route in app.routes)


def _mounted_paths(app) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def test_conformance_endpoint_is_absent_by_default():
    """It replays canned payloads, so a deployment must not expose it unasked."""
    assert "/a2a/tck" not in _mounted_paths(create_app(_orchestrator()))


def test_conformance_endpoint_is_mounted_when_enabled(monkeypatch):
    from dataclasses import replace

    from erebots.server import app as server_app

    # Settings is frozen: swap the whole instance rather than a field.
    monkeypatch.setattr(
        server_app,
        "settings",
        replace(server_app.settings, a2a_conformance_agent=True),
    )

    mounted = _mounted_paths(create_app(_orchestrator()))
    assert "/a2a/tck" in mounted
    # Its twin, declaring none of the optional capabilities: the suite only
    # reaches CORE-CAP-001/002/003 against a server that lacks them.
    assert "/a2a/tck-minimal" in mounted


def test_conformance_endpoint_never_shares_the_application_task_store(monkeypatch):
    """Its fixtures — including a task parked forever — are not real tasks.

    Sharing the persistent store would accumulate them in the application
    database next to the genuine ones.
    """
    from dataclasses import replace

    from erebots.server import app as server_app

    seen: dict = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return Starlette()

    monkeypatch.setattr(
        server_app,
        "settings",
        replace(server_app.settings, a2a_conformance_agent=True),
    )
    monkeypatch.setattr(server_app, "create_conformance_app", _spy)

    create_app(_orchestrator(), a2a_task_store=object())

    assert seen.get("task_store") is None


def test_api_me_reports_anonymous_without_credentials(monkeypatch):
    """GET /api/me: anonymous (no cookie/bearer) → authenticated=false."""
    from erebots.api import auth_middleware as mw

    class _NoAuth:
        enable_auth = False
        jwt_cookie_name = "access_token"  # extract_user_token reads it

    monkeypatch.setattr(mw, "settings", _NoAuth())
    app = create_app(_orchestrator())

    with TestClient(app) as client:
        body = client.get("/api/me").json()

    assert body["authenticated"] is False
    assert body["name"] == ""


def test_api_agents_lists_router_and_registered_specialists(monkeypatch):
    """GET /api/agents: the orchestrator (router) plus registered specialists."""
    from erebots.agents.loader import LoadedAgents
    from erebots.api import auth_middleware as mw
    from erebots.server import app as server_app

    class _NoAuth:
        enable_auth = False

    monkeypatch.setattr(mw, "settings", _NoAuth())

    # Hermetic: the endpoint's lazy discovery must not depend on agents.yaml
    # or a live server on this machine (it would overwrite the registration).
    async def _no_discovery(token_provider=None, urls=None, *, config_path=None):
        return LoadedAgents()

    monkeypatch.setattr(server_app, "load_agents", _no_discovery)
    orch = _orchestrator()
    orch.register_agent(
        "nutrition", "http://x/a2a/nutrition", "Nutrition specialist",
        skills=[{"name": "search_recipes", "description": "find recipes"}],
    )
    app = create_app(orch)

    with TestClient(app) as client:
        body = client.get("/api/agents").json()

    roles = {a["name"]: a["role"] for a in body["agents"]}
    assert roles["orchestrator"] == "router"
    assert roles["nutrition"] == "specialist"
    nutrition = next(a for a in body["agents"] if a["name"] == "nutrition")
    assert nutrition["skills"] == [{"name": "search_recipes", "description": "find recipes"}]


def test_discovery_retries_the_agents_it_could_not_reach(monkeypatch):
    """One agent down at the first request is picked up later, without a restart,
    while the one found stays registered; retries are throttled."""
    from dataclasses import replace

    from erebots.agents.loader import AgentEntry, LoadedAgents
    from erebots.api import auth_middleware as mw
    from erebots.server import app as server_app

    class _NoAuth:
        enable_auth = False

    monkeypatch.setattr(mw, "settings", _NoAuth())
    calls: list = []

    async def _discovery(token_provider=None, urls=None, *, config_path=None):
        calls.append(urls)
        if urls is None:  # full discovery: nutrition answers, personal is down
            return LoadedAgents(
                entries=[AgentEntry("nutrition", "n", "http://x/a2a/nutrition")],
                failed=["http://x/a2a/personal"],
            )
        return LoadedAgents(entries=[AgentEntry("personal", "p", u) for u in urls])

    monkeypatch.setattr(server_app, "load_agents", _discovery)
    app = create_app(_orchestrator())

    def names(client):
        return {a["name"] for a in client.get("/api/agents").json()["agents"]}

    with TestClient(app) as client:
        assert names(client) == {"orchestrator", "nutrition"}
        # Within the throttle window nothing is retried.
        assert names(client) == {"orchestrator", "nutrition"}
        assert calls == [None]

        monkeypatch.setattr(
            server_app, "settings", replace(server_app.settings, discovery_retry_seconds=0)
        )
        # Throttle still armed from the first pass with the old value: force it.
        assert names(client) == {"orchestrator", "nutrition"}

    # A fresh app with no throttle retries exactly the unreached URL.
    calls.clear()
    app = create_app(_orchestrator())
    with TestClient(app) as client:
        assert names(client) == {"orchestrator", "nutrition"}
        assert names(client) == {"orchestrator", "nutrition", "personal"}
        assert names(client) == {"orchestrator", "nutrition", "personal"}
    assert calls == [None, ["http://x/a2a/personal"]]


def _mcp_payload(response) -> dict:
    """The JSON-RPC payload out of a Streamable HTTP (SSE-framed) response."""
    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError(f"no data frame in {response.text!r}")


def _echo_server():
    """A one-tool MCP server, stateless like the ones the app really mounts."""
    server = FastMCP("demo", stateless_http=True)

    @server.tool()
    async def echo(text: str) -> str:
        """Echo the given text."""
        return f"echoed {text}"

    return server


_MCP_HEADERS = {"Accept": "application/json, text/event-stream"}
# The MCP SDK's DNS-rebinding guard only accepts a loopback Host *with a port*,
# which is what the agents send; TestClient's default "testserver" is rejected.
_MCP_BASE_URL = "http://localhost:7860"
_LIST_TOOLS = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


def test_create_app_serves_a_mounted_mcp_tool_server(monkeypatch):
    """``mcp_apps`` are reachable over Streamable HTTP at /tools/<name>/mcp.

    Covers the wiring a mounted ASGI app does not get for free: the framework
    lifespan has to start each server's session manager, or every call fails.
    """
    import erebots.api.auth_middleware as mw

    class _NoAuth:
        enable_auth = False

    monkeypatch.setattr(mw, "settings", _NoAuth())
    app = create_app(_orchestrator(), mcp_apps={"demo": _echo_server()}, agent_urls=[])

    with TestClient(app, base_url=_MCP_BASE_URL) as client:
        listed = client.post("/tools/demo/mcp", headers=_MCP_HEADERS, json=_LIST_TOOLS)
        called = client.post(
            "/tools/demo/mcp",
            headers=_MCP_HEADERS,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"text": "hi"}},
            },
        )

    assert [t["name"] for t in _mcp_payload(listed)["result"]["tools"]] == ["echo"]
    assert "echoed hi" in json.dumps(_mcp_payload(called)["result"])


def test_mounted_mcp_tool_server_requires_the_internal_bearer():
    """A tool call is a POST under /tools/*, so the M2M regime applies to it."""
    app = create_app(_orchestrator(), mcp_apps={"demo": _echo_server()}, agent_urls=[])

    with TestClient(app, base_url=_MCP_BASE_URL) as client:
        r = client.post("/tools/demo/mcp", headers=_MCP_HEADERS, json=_LIST_TOOLS)

    assert r.status_code == 401


def test_mounted_mcp_tool_server_rejects_an_unauthenticated_loopback_get():
    """The default policy leaves nothing open under /tools/, GET included.

    Stated against the real factory with a real mount, because that is the
    surface the finding was about: the dispatcher's own unit tests build a
    hand-made app, and this one cannot drift from what ``create_app`` wires.
    """
    app = create_app(_orchestrator(), mcp_apps={"demo": _echo_server()}, agent_urls=[])

    with TestClient(app, base_url=_MCP_BASE_URL, client=("127.0.0.1", 0)) as client:
        r = client.get("/tools/demo/mcp")

    assert r.status_code == 401


def test_app_shutdown_does_not_close_an_injected_a2a_client():
    """The lifespan releases what the orchestrator owns, not what a caller lent it.

    This is the contract worth pinning: the shutdown hook is the code that would
    get it wrong, because it used to reach straight through to ``a2a_client``.
    """

    class _FakeA2AClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    fake = _FakeA2AClient()
    orchestrator = OrchestratorAgent(
        api_key="EMPTY", api_base_url="http://x/v1", model="local", a2a_client=fake
    )

    with TestClient(create_app(orchestrator, agent_urls=[])):
        pass

    assert fake.closed is False

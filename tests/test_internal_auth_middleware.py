"""Tests for the authentication dispatcher in ``erebots.api.auth_middleware``."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from erebots.api import auth_middleware as mw
from erebots.security.keycloak import KeycloakError


def _build_app(decoder) -> TestClient:
    """Build a tiny FastAPI app wired to ``auth_dispatch`` for testing."""
    app = FastAPI()
    app.middleware("http")(mw.auth_dispatch)

    @app.get("/a2a/nutrition/.well-known/agent-card.json")
    async def _agent_card():
        return {"name": "nutrition"}

    @app.post("/a2a/nutrition/")
    async def _a2a_post():
        return {"ok": True}

    @app.get("/tools/nutrition/config")
    async def _tools_config():
        return {"ok": True}

    @app.get("/health")
    async def _health():
        return {"status": "healthy"}

    @app.get("/ui")
    async def _ui():
        return {"page": "ui"}

    # Patch both decoders used inside the middleware module: the internal
    # path uses ``decode_token`` and the user-cookie path uses
    # ``decode_user_token``. Tests exercise one path at a time, so wiring
    # both to the same stub is harmless.
    mw.decode_token = decoder  # type: ignore[assignment]
    mw.decode_user_token = decoder  # type: ignore[assignment]
    return TestClient(app)


@pytest.fixture
def reset_decoder():
    """Restore the decoders after each test."""
    original = mw.decode_token
    original_user = mw.decode_user_token
    yield
    mw.decode_token = original  # type: ignore[assignment]
    mw.decode_user_token = original_user  # type: ignore[assignment]


def test_a2a_without_bearer_returns_401(reset_decoder):
    async def _never_called(_t):
        raise AssertionError("decode_token must not be called without a bearer")

    client = _build_app(_never_called)
    r = client.post("/a2a/nutrition/")
    assert r.status_code == 401
    assert r.json() == {"detail": "Missing bearer token"}


def test_a2a_with_invalid_bearer_returns_401(reset_decoder):
    async def _raise(_t):
        raise KeycloakError("nope")

    client = _build_app(_raise)
    r = client.post(
        "/a2a/nutrition/", headers={"Authorization": "Bearer bogus"}
    )
    assert r.status_code == 401
    assert r.json() == {"detail": "Invalid token"}


def test_a2a_with_wrong_azp_returns_403(reset_decoder):
    async def _decode(_t):
        return {"azp": "some-other-client"}

    client = _build_app(_decode)
    r = client.post(
        "/a2a/nutrition/", headers={"Authorization": "Bearer ok"}
    )
    assert r.status_code == 403


def test_a2a_with_correct_azp_passes(reset_decoder):
    async def _decode(_t):
        return {"azp": "erebots-internal"}

    client = _build_app(_decode)
    r = client.post(
        "/a2a/nutrition/", headers={"Authorization": "Bearer ok"}
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_a2a_agent_card_requires_bearer(reset_decoder):
    """The agent-card discovery endpoint is also under /a2a/ — must be protected."""

    async def _never_called(_t):
        raise AssertionError("must not decode")

    client = _build_app(_never_called)
    r = client.get("/a2a/nutrition/.well-known/agent-card.json")
    assert r.status_code == 401


def _loopback_client(decoder) -> TestClient:
    """A client whose requests carry a loopback peer address.

    ``TestClient`` puts ``("testclient", 50000)`` in the ASGI scope by default;
    the ``client`` kwarg is what makes the dispatcher see 127.0.0.1.
    """
    app = FastAPI()
    app.middleware("http")(mw.auth_dispatch)

    @app.get("/tools/nutrition/config")
    async def _tools_config():
        return {"ok": True}

    @app.post("/tools/nutrition/gradio_api/mcp")
    async def _mcp_call():
        return {"ok": True}

    @app.get("/a2a/nutrition/.well-known/agent-card.json")
    async def _agent_card():
        return {"name": "nutrition"}

    mw.decode_token = decoder  # type: ignore[assignment]
    mw.decode_user_token = decoder  # type: ignore[assignment]
    return TestClient(app, client=("127.0.0.1", 0))


def _allow_loopback_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt the deployment into the loopback carve-out.

    ``Settings`` is frozen, so the whole instance is swapped rather than a field.
    The ``_FakeSettings`` stub used further down cannot serve here: it disables
    authentication, and this regime needs it enabled.
    """
    monkeypatch.setattr(mw, "settings", replace(mw.settings, allow_loopback_tool_reads=True))


def test_a_non_loopback_get_on_tools_requires_the_bearer(reset_decoder):
    """The peer address is what the carve-out keys on, so a remote GET is refused."""

    async def _never_called(_t):
        raise AssertionError("no bearer present, decode must not run")

    # The default TestClient peer is "testclient", not loopback.
    client = _build_app(_never_called)
    r = client.get("/tools/nutrition/config")
    assert r.status_code == 401


def test_loopback_get_on_tools_requires_the_bearer_by_default(reset_decoder):
    """The framework's own tool servers speak POST, so nothing needs this open."""

    async def _never_called(_t):
        raise AssertionError("no bearer present, decode must not run")

    r = _loopback_client(_never_called).get("/tools/nutrition/config")
    assert r.status_code == 401


def test_loopback_get_on_tools_is_allowed_when_explicitly_enabled(
    reset_decoder, monkeypatch: pytest.MonkeyPatch
):
    """A deployment that mounts a re-entrant UI under /tools/ can opt in."""

    async def _never_called(_t):
        raise AssertionError("the carve-out must not decode a token")

    _allow_loopback_reads(monkeypatch)
    r = _loopback_client(_never_called).get("/tools/nutrition/config")
    assert r.status_code == 200


def test_loopback_post_requires_the_bearer_even_when_enabled(
    reset_decoder, monkeypatch: pytest.MonkeyPatch
):
    """MCP tool invocation is a POST and always authenticates, opt-in or not."""

    async def _never_called(_t):
        raise AssertionError("no bearer present, decode must not run")

    _allow_loopback_reads(monkeypatch)
    r = _loopback_client(_never_called).post("/tools/nutrition/gradio_api/mcp")
    assert r.status_code == 401


def test_a_non_loopback_get_is_rejected_even_when_enabled(
    reset_decoder, monkeypatch: pytest.MonkeyPatch
):
    """Opting in must not widen into "trust /tools/* from anywhere"."""

    async def _never_called(_t):
        raise AssertionError("no bearer present, decode must not run")

    _allow_loopback_reads(monkeypatch)
    r = _build_app(_never_called).get("/tools/nutrition/config")
    assert r.status_code == 401


def test_a2a_is_never_exempt_even_when_enabled(
    reset_decoder, monkeypatch: pytest.MonkeyPatch
):
    """The carve-out is scoped to /tools/; /a2a/ shares the prefix table, not this."""

    async def _never_called(_t):
        raise AssertionError("no bearer present, decode must not run")

    _allow_loopback_reads(monkeypatch)
    r = _loopback_client(_never_called).get("/a2a/nutrition/.well-known/agent-card.json")
    assert r.status_code == 401


def test_protected_html_with_valid_user_cookie_passes(reset_decoder):
    """A cookie accepted by decode_user_token (correct azp) reaches the page."""

    async def _decode_user(_t):
        return {"sub": "user-1", "azp": "erebots-webapp"}

    client = _build_app(_decode_user)
    client.cookies.set(mw.settings.jwt_cookie_name, "good-token")
    r = client.get("/ui", follow_redirects=False)
    assert r.status_code == 200
    assert r.json() == {"page": "ui"}


def test_protected_path_accepts_keycloak_bearer(reset_decoder):
    """A browser SPA may authenticate with a Keycloak bearer (not just a cookie)."""

    async def _decode_user(_t):
        return {"sub": "user-1", "azp": "erebots-webapp"}

    client = _build_app(_decode_user)
    r = client.get(
        "/ui",
        headers={"Authorization": "Bearer kc-token"},
        follow_redirects=False,
    )
    assert r.status_code == 200
    assert r.json() == {"page": "ui"}


def test_protected_html_with_rejected_user_cookie_redirects(reset_decoder):
    """A cookie rejected by decode_user_token (e.g. wrong azp) → login redirect.

    This is the C1 guard: an internal service-account token (azp mismatch)
    presented as a session cookie must not authenticate a user.
    """

    async def _reject(_t):
        raise KeycloakError("unexpected azp")

    client = _build_app(_reject)
    client.cookies.set(mw.settings.jwt_cookie_name, "internal-token")
    r = client.get("/ui", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/api/auth/login"


def test_health_is_public(reset_decoder):
    async def _never_called(_t):
        raise AssertionError("must not decode")

    client = _build_app(_never_called)
    r = client.get("/health")
    assert r.status_code == 200


def test_protected_html_without_cookie_redirects_to_login(reset_decoder):
    async def _never_called(_t):
        raise AssertionError("must not decode without cookie")

    client = _build_app(_never_called)
    r = client.get("/ui", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/api/auth/login"


def test_auth_disabled_passes_everything(
    reset_decoder, monkeypatch: pytest.MonkeyPatch
):
    # ``Settings`` is a frozen dataclass; swap the whole object on the
    # middleware module with a minimal stand-in.
    class _FakeSettings:
        enable_auth = False
        internal_client_id = "erebots-internal"
        jwt_cookie_name = "access_token"

    monkeypatch.setattr(mw, "settings", _FakeSettings())

    async def _never_called(_t):
        raise AssertionError("must not decode when auth is disabled")

    client = _build_app(_never_called)
    r = client.post("/a2a/nutrition/")
    assert r.status_code == 200

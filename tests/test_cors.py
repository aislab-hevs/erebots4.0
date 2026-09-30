"""CORS must wrap authentication, not the other way round.

A browser has to be able to tell "you are not logged in" apart from "this origin
is not allowed". That only holds when the CORS middleware is the outer one: a
preflight carries no credentials and has to be answered before auth sees it, and
an authentication failure on the *actual* request still has to carry the CORS
headers, or the browser reports an opaque network error instead of the 401.

Both halves are pinned below, because fixing only the first is a plausible and
wrong repair: passing ``OPTIONS`` through the auth dispatcher answers the
preflight but leaves the 401 header-less.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from erebots.agents.orchestrator import OrchestratorAgent
from erebots.server import app as server_app

_ALLOWED = "http://localhost:7860"
_DENIED = "http://evil.example"
_PROTECTED = "/api/chat/stream"


def _orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent(api_key="EMPTY", api_base_url="http://x/v1", model="local")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The real app, auth enabled, with one known allowed origin.

    ``Settings`` is frozen, so the whole instance is swapped rather than a field.
    Authentication stays *on*: the point of these tests is what an
    unauthenticated cross-origin caller sees. No request here reaches Keycloak —
    a missing token short-circuits before token validation.
    """
    monkeypatch.setattr(
        server_app, "settings", replace(server_app.settings, cors_origins=[_ALLOWED])
    )
    return TestClient(server_app.create_app(_orchestrator()))


def test_preflight_on_a_protected_route_is_answered_without_credentials(client: TestClient):
    """The original defect: auth rejected the preflight before CORS answered it."""
    response = client.options(
        _PROTECTED,
        headers={
            "Origin": _ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == _ALLOWED


def test_an_unauthenticated_request_still_carries_cors_headers(client: TestClient):
    """Without this, the browser cannot read the 401 and shows a CORS error instead."""
    response = client.post(_PROTECTED, json={"message": "hi"}, headers={"Origin": _ALLOWED})

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == _ALLOWED


def test_authentication_is_still_enforced_on_the_actual_request(client: TestClient):
    """Reordering the middleware must not let an unauthenticated request through."""
    response = client.post(_PROTECTED, json={"message": "hi"}, headers={"Origin": _ALLOWED})

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_a_disallowed_origin_gets_no_cors_headers(client: TestClient):
    """CORS being outermost must not turn into CORS being permissive."""
    response = client.options(
        _PROTECTED,
        headers={"Origin": _DENIED, "Access-Control-Request-Method": "POST"},
    )

    assert "access-control-allow-origin" not in response.headers

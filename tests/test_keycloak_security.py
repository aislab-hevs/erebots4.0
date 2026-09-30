"""Tests for the Keycloak token-validation hardening and the PKCE login flow.

Tokens are minted locally with a throwaway RSA key and the JWKS lookup is
stubbed, so ``decode_token``'s real validation path (signature, expiry, issuer,
azp) runs without a Keycloak server.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI

from erebots.api.routes import auth as auth_routes
from erebots.config import settings
from erebots.security import keycloak

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)

_ISSUER = f"{settings.keycloak_url.rstrip('/')}/realms/{settings.keycloak_realm}"


def _mint(**claims) -> str:
    payload = {
        "iss": _ISSUER,
        "azp": settings.keycloak_client_id,
        "sub": "user-1",
        "exp": int(time.time()) + 300,
        **claims,
    }
    return jwt.encode(payload, _KEY, algorithm="RS256")


@pytest.fixture(autouse=True)
def _stub_jwks(monkeypatch):
    fake = SimpleNamespace(
        get_signing_key_from_jwt=lambda _token: SimpleNamespace(key=_KEY.public_key())
    )
    monkeypatch.setattr(keycloak, "_get_jwk_client", lambda _uri: fake)


async def test_decode_accepts_a_token_from_the_configured_issuer():
    claims = await keycloak.decode_user_token(_mint())
    assert claims["sub"] == "user-1"


async def test_decode_rejects_a_foreign_issuer():
    """A correctly signed token from another realm/server must not pass: the
    signature alone proves the key, not that the realm is ours."""
    token = _mint(iss="https://evil.example/realms/erebots")
    with pytest.raises(keycloak.KeycloakError, match="issuer"):
        await keycloak.decode_user_token(token)


async def test_decode_rejects_a_missing_issuer():
    payload = {
        "azp": settings.keycloak_client_id,
        "sub": "user-1",
        "exp": int(time.time()) + 300,
    }
    token = jwt.encode(payload, _KEY, algorithm="RS256")
    with pytest.raises(keycloak.KeycloakError, match="issuer"):
        await keycloak.decode_user_token(token)


# --- PKCE ------------------------------------------------------------------


def test_login_redirect_carries_the_pkce_challenge():
    url = keycloak.get_login_redirect_url("some-state", code_challenge="abc123")
    query = parse_qs(urlparse(url).query)
    assert query["code_challenge"] == ["abc123"]
    assert query["code_challenge_method"] == ["S256"]


async def test_login_route_generates_state_and_verifier():
    app = FastAPI()
    app.include_router(auth_routes.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        r = await http.get("/api/auth/login", follow_redirects=False)

    assert r.status_code in (302, 307)
    query = parse_qs(urlparse(r.headers["location"]).query)
    state = query["state"][0]
    assert "code_challenge" in query
    # The verifier is parked server-side under the state, ready for the callback.
    _, verifier = auth_routes._auth_states.pop(state)
    assert len(verifier) >= 43  # RFC 7636 minimum


@respx.mock
async def test_code_exchange_sends_the_verifier():
    token_endpoint = (
        f"{keycloak.get_keycloak_server_url()}/realms/"
        f"{settings.keycloak_realm}/protocol/openid-connect/token"
    )
    route = respx.post(token_endpoint).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )

    await keycloak.exchange_code_for_token("the-code", code_verifier="the-verifier")

    sent = parse_qs(route.calls.last.request.content.decode())
    assert sent["code_verifier"] == ["the-verifier"]
    assert sent["code"] == ["the-code"]


async def test_jwks_lookup_does_not_block_the_event_loop(monkeypatch):
    """PyJWKClient fetches keys synchronously; a slow IdP must not stall other requests."""
    import asyncio
    import threading

    release = threading.Event()
    calls = {"n": 0}

    def slow_lookup(_token):
        calls["n"] += 1
        release.wait(timeout=5)
        return SimpleNamespace(key=_KEY.public_key())

    fake = SimpleNamespace(get_signing_key_from_jwt=slow_lookup)
    monkeypatch.setattr(keycloak, "_get_jwk_client", lambda _uri: fake)

    task = asyncio.create_task(keycloak.decode_user_token(_mint()))
    # The lookup is in flight in a worker thread while the loop keeps turning.
    while calls["n"] == 0:
        await asyncio.sleep(0.001)
    ticks = 0
    for _ in range(3):
        await asyncio.sleep(0.001)
        ticks += 1
    assert ticks == 3
    assert not task.done()

    release.set()
    claims = await asyncio.wait_for(task, timeout=5)
    assert claims["sub"] == "user-1"


async def test_concurrent_lookups_on_one_jwks_uri_are_serialized(monkeypatch):
    import asyncio
    import threading

    in_flight = {"now": 0, "max": 0}
    lock = threading.Lock()

    def lookup(_token):
        with lock:
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
        threading.Event().wait(0.01)
        with lock:
            in_flight["now"] -= 1
        return SimpleNamespace(key=_KEY.public_key())

    fake = SimpleNamespace(get_signing_key_from_jwt=lookup)
    monkeypatch.setattr(keycloak, "_get_jwk_client", lambda _uri: fake)

    await asyncio.gather(*(keycloak.decode_user_token(_mint()) for _ in range(5)))
    assert in_flight["max"] == 1


def test_jwk_client_has_an_explicit_fetch_timeout(monkeypatch):
    monkeypatch.undo()  # the autouse stub replaced the function under test
    client = keycloak._get_jwk_client("http://idp.test/certs")
    assert client.timeout == keycloak._JWKS_FETCH_TIMEOUT_SECONDS


_INTERNAL_ISSUER = f"{settings.keycloak_internal_url.rstrip('/')}/realms/{settings.keycloak_realm}"
_DISCOVERY = f"{settings.keycloak_internal_url.rstrip('/')}/realms/{settings.keycloak_realm}/.well-known/openid-configuration"


@pytest.fixture(autouse=True)
def _fresh_issuer_cache():
    keycloak._announced_issuers.clear()
    yield
    keycloak._announced_issuers.clear()


@respx.mock
async def test_decode_accepts_the_issuer_the_realm_announces_internally():
    """A bare KC_HOSTNAME makes backchannel tokens carry scheme+port of the request
    (``http://auth.example:8080``): the realm's discovery document names that
    issuer, so it is accepted without listing it in the configuration."""
    announced = "http://auth.erebots.local:8080/realms/erebots"
    route = respx.get(_DISCOVERY).mock(return_value=httpx.Response(200, json={"issuer": announced}))

    claims = await keycloak.decode_user_token(_mint(iss=announced))
    assert claims["sub"] == "user-1"
    # Fetched once, then cached.
    await keycloak.decode_user_token(_mint(iss=announced))
    assert route.call_count == 1


@respx.mock
async def test_decode_still_rejects_an_issuer_the_realm_does_not_announce():
    respx.get(_DISCOVERY).mock(return_value=httpx.Response(200, json={"issuer": "http://auth.erebots.local:8080/realms/erebots"}))
    with pytest.raises(keycloak.KeycloakError, match="issuer"):
        await keycloak.decode_user_token(_mint(iss="https://evil.example/realms/erebots"))


@respx.mock
async def test_configured_issuers_need_no_discovery_call():
    route = respx.get(_DISCOVERY).mock(return_value=httpx.Response(200, json={"issuer": "x"}))
    await keycloak.decode_user_token(_mint())
    assert route.call_count == 0


@respx.mock
async def test_unreachable_discovery_rejects_unknown_issuers_instead_of_failing_open():
    respx.get(_DISCOVERY).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(keycloak.KeycloakError, match="issuer"):
        await keycloak.decode_user_token(_mint(iss="http://auth.erebots.local:8080/realms/erebots"))

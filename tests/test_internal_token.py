"""Tests for ``source.security.internal_token.InternalTokenProvider``."""

from __future__ import annotations

import httpx
import pytest
import respx

from erebots.security.internal_token import InternalTokenProvider

TOKEN_URL = "https://kc.test/realms/erebots/protocol/openid-connect/token"


def _provider() -> InternalTokenProvider:
    return InternalTokenProvider(
        token_url=TOKEN_URL,
        client_id="erebots-internal",
        client_secret="s3cret",
        safety_margin_seconds=10,
    )


@respx.mock
async def test_fetches_token_on_first_call():
    route = respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "tok-1", "expires_in": 300}
        )
    )
    provider = _provider()

    token = await provider.get_token()

    assert token == "tok-1"
    assert route.called
    assert route.call_count == 1


@respx.mock
async def test_caches_token_within_expiry():
    route = respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "tok-cached", "expires_in": 300}
        )
    )
    provider = _provider()

    first = await provider.get_token()
    second = await provider.get_token()
    third = await provider.get_token()

    assert first == second == third == "tok-cached"
    assert route.call_count == 1


@respx.mock
async def test_refreshes_token_after_expiry(monkeypatch: pytest.MonkeyPatch):
    """Manually expire the cached token, then a second call must refetch."""
    responses = iter(
        [
            httpx.Response(200, json={"access_token": "tok-old", "expires_in": 60}),
            httpx.Response(200, json={"access_token": "tok-new", "expires_in": 60}),
        ]
    )
    route = respx.post(TOKEN_URL).mock(side_effect=lambda _req: next(responses))
    provider = _provider()

    first = await provider.get_token()
    assert first == "tok-old"
    assert route.call_count == 1

    # Force expiry: rewind the cache deadline into the past.
    provider._expires_at = 0.0

    second = await provider.get_token()
    assert second == "tok-new"
    assert route.call_count == 2


@respx.mock
async def test_propagates_http_errors():
    respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            401, json={"error": "unauthorized_client"}
        )
    )
    provider = _provider()

    with pytest.raises(httpx.HTTPStatusError):
        await provider.get_token()

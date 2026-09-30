"""Machine-to-machine token provider for internal A2A/MCP calls.

Obtains and caches an OAuth2 ``client_credentials`` access token from
Keycloak so that requests to ``/a2a/*`` and ``/tools/*`` can be
authenticated as the ``erebots-internal`` service account.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)


class InternalTokenProvider:
    """Fetch and cache an internal service-account access token.

    The token is refreshed on demand and reused until shortly before its
    expiry (``_safety_margin_seconds``). Concurrent callers share the
    same in-flight request through ``_lock``.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        safety_margin_seconds: int = 30,
    ):
        """Configure the provider; no token is fetched until first use.

        Args:
            token_url: Keycloak's ``client_credentials`` token endpoint.
            client_id: The service account to authenticate as.
            client_secret: Its secret.
            safety_margin_seconds: How long before the stated expiry a token
                is treated as stale, so one is never presented mid-flight to
                a peer that would reject it.
        """
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._safety_margin_seconds = safety_margin_seconds
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Return a valid access token, refreshing it if needed.

        Checked before and again inside the lock, so concurrent callers
        arriving on an expired token trigger one refresh, not several.

        Returns:
            A token valid for at least the safety margin.

        Raises:
            httpx.HTTPError: The IdP was unreachable or answered an error.
            KeyError: The response carried no ``access_token``.
        """
        if self._token and time.monotonic() < self._expires_at:
            return self._token
        async with self._lock:
            if self._token and time.monotonic() < self._expires_at:
                return self._token
            await self._fetch()
            return self._token  # type: ignore[return-value]

    async def _fetch(self) -> None:
        """Request a fresh token from Keycloak and cache it.

        The cached expiry is computed from ``time.monotonic``, so a wall-clock
        adjustment cannot make a live token look expired or the reverse.

        Raises:
            httpx.HTTPError: The IdP was unreachable or answered an error.
                Nothing is cached, so the next call retries.
            KeyError: The response carried no ``access_token``.
        """
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(self._token_url, data=data)
            response.raise_for_status()
            payload = response.json()
        self._token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 300))
        self._expires_at = time.monotonic() + max(
            expires_in - self._safety_margin_seconds, 1
        )
        logger.debug(
            "Refreshed internal token (expires_in=%ss, margin=%ss)",
            expires_in,
            self._safety_margin_seconds,
        )

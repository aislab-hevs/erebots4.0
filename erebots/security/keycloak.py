"""Keycloak OIDC integration for OAuth2 authentication.

Handles the authentication flow with Keycloak: login/logout URL generation,
JWT validation via remote JWKS keys (cached by PyJWKClient), and authorization
code exchange for access tokens.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt
from jwt import PyJWKClient

from erebots.config import settings

logger = logging.getLogger(__name__)

# JWKS cache duration: 1 hour (3600 seconds)
_JWKS_CACHE_TTL_SECONDS = 3600
# Network timeout for one JWKS fetch. PyJWKClient's own default is 30s, long
# enough for a slow IdP to hold every request behind it.
_JWKS_FETCH_TIMEOUT_SECONDS = 10

# Module-level PyJWKClient cache: built once per JWKS URI so its internal
# signing-key cache is reused across requests instead of being thrown away
# (creating a new client on every ``decode_token`` call defeats the cache).
_jwk_clients: dict[str, PyJWKClient] = {}
# One lock per JWKS URI: a burst of requests on a cold or expired key cache
# waits for one fetch instead of each starting its own.
_jwk_locks: dict[str, asyncio.Lock] = {}
# Issuer the realm announces on its discovery document, keyed by the internal
# base URL it was read from (see ``_announced_issuer``).
_announced_issuers: dict[str, str] = {}


def _get_jwk_client(jwks_uri: str) -> PyJWKClient:
    """Return a cached ``PyJWKClient`` for the given JWKS URI."""
    client = _jwk_clients.get(jwks_uri)
    if client is None:
        client = PyJWKClient(
            jwks_uri, lifespan=_JWKS_CACHE_TTL_SECONDS, timeout=_JWKS_FETCH_TIMEOUT_SECONDS
        )
        _jwk_clients[jwks_uri] = client
    return client


async def _signing_key(jwk_client, jwks_uri: str, token: str):
    """Resolve the token's signing key without blocking the event loop.

    ``PyJWKClient`` fetches the JWKS with ``urllib`` whenever its cache is cold
    or stale, so the lookup runs in a worker thread; the per-URI lock keeps
    concurrent requests from turning one cache miss into many fetches.
    """
    lock = _jwk_locks.get(jwks_uri)
    if lock is None:
        lock = _jwk_locks[jwks_uri] = asyncio.Lock()
    async with lock:
        return await asyncio.to_thread(jwk_client.get_signing_key_from_jwt, token)


async def _announced_issuer() -> str | None:
    """The issuer the realm announces on the internal URL (fetched once).

    Keycloak stamps tokens with its configured hostname, but when that is a
    bare host the scheme and port follow the request: a token the app itself
    obtains through the internal URL (``http://erebots-keycloak:8080``) comes
    back with ``iss = http://auth.erebots.local:8080/...``, which matches none
    of the configured URLs. The realm's own discovery document, read over the
    same trusted internal URL the JWKS come from, says what that issuer is.
    """
    base = get_keycloak_server_url()
    cached = _announced_issuers.get(base)
    if cached:
        return cached
    url = f"{base}/realms/{settings.keycloak_realm}/.well-known/openid-configuration"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(url)
            response.raise_for_status()
            issuer = str(response.json().get("issuer") or "")
    except Exception as exc:
        logger.warning("Could not read the realm discovery document at %s: %s", url, exc)
        return None
    if issuer:
        _announced_issuers[base] = issuer
    return issuer or None


class KeycloakError(Exception):
    """Base exception for Keycloak-related errors.

    Raised when an operation with the Keycloak server fails,
    such as JWKS key retrieval, JWT token validation, or
    authorization code exchange.
    """

    pass


def get_keycloak_server_url() -> str:
    """Return the internal Keycloak server URL.

    This URL is used for server-to-server communications within
    the Docker network (e.g. JWKS retrieval, token exchange).

    Returns:
        The internal Keycloak URL without trailing slash.
    """
    return settings.keycloak_internal_url.rstrip("/")


def get_keycloak_public_url() -> str:
    """Return the public Keycloak URL for browser redirects.

    This URL is used to build authorization and logout endpoints
    visible to the client (user's browser).

    Returns:
        The public Keycloak URL without trailing slash.
    """
    return settings.keycloak_public_url.rstrip("/")


def get_login_redirect_url(state: str, code_challenge: str | None = None) -> str:
    """Build the authorization URL for login via Keycloak.

    Generates the full URL with the OAuth2 parameters needed to start
    the Authorization Code flow. The ``state`` parameter is used for
    CSRF protection and to maintain request context.

    Args:
        state: Random value for CSRF protection, associated with
            the user's session.
        code_challenge: PKCE S256 challenge derived from the per-login
            code verifier. Binds the authorization code to this login
            attempt: an intercepted code is useless without the verifier.

    Returns:
        The full authorization URL with encoded query parameters.
    """
    auth_endpoint = f"{get_keycloak_public_url()}/realms/{settings.keycloak_realm}/protocol/openid-connect/auth"

    # OAuth2 parameters for the Authorization Code flow
    params = {
        "client_id": settings.keycloak_client_id,
        "response_type": "code",
        "scope": "openid profile email",
        "redirect_uri": settings.app_callback_url,
        "state": state,
    }
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"

    return f"{auth_endpoint}?{urlencode(params)}"


def get_logout_redirect_url(id_token_hint: str | None = None) -> str:
    """Build the logout URL to terminate the Keycloak session.

    After logout, the user is redirected to the application's main page.
    If an ``id_token_hint`` is provided, Keycloak can skip the logout
    confirmation.

    Args:
        id_token_hint: Optional ID token to identify the session to
            terminate without requiring user confirmation.

    Returns:
        The full logout URL with encoded query parameters.
    """
    logout_endpoint = (
        f"{get_keycloak_public_url()}/realms/"
        f"{settings.keycloak_realm}/protocol/openid-connect/logout"
    )

    # Derive the app base URL by removing the /api/auth/callback path
    base_url = settings.app_callback_url.rsplit("/api", 1)[0] if settings.app_callback_url else settings.keycloak_public_url

    params = {
        "post_logout_redirect_uri": base_url,
    }

    # Add the token hint if available to skip logout confirmation
    if id_token_hint:
        params["id_token_hint"] = id_token_hint

    return f"{logout_endpoint}?{urlencode(params)}"


def _get_kid_from_token(token: str) -> str | None:
    """Extract the ``kid`` (Key ID) from a JWT token header.

    The ``kid`` identifies which public key from the JWKS set should
    be used to verify the token's signature.

    Args:
        token: Encoded JWT token as a string.

    Returns:
        The ``kid`` field value from the JWT header, or ``None``
        if the token is invalid or the field is not present.
    """
    try:
        # Decode only the header without verifying the signature
        header = jwt.get_unverified_header(token)
        return header.get("kid")
    except jwt.DecodeError:
        return None


async def decode_token(token: str, *, expected_azp: str | None = None) -> dict[str, Any]:
    """Decode and verify a JWT token issued by Keycloak.

    Uses PyJWT's JWK client to automatically retrieve the correct
    signing key from the JWKS endpoint. The signature and expiry are
    always verified. The issuer (``iss``) must be the configured realm
    reached through any of the configured Keycloak URLs — Keycloak stamps
    the URL a token was minted through, so both the public and the
    internal form are legitimate here, and anything else is a token from
    a different realm or server.

    Audience (``aud``) is not enforced because it is not a reliable
    discriminator across the two mint URLs. The authorized party
    (``azp``) — the client that requested the token — is stable across
    URLs and is the correct way to tell apart the browser client from
    the internal service account. Pass ``expected_azp`` to enforce it.

    Args:
        token: Encoded JWT token string to validate.
        expected_azp: If set, the token's ``azp`` claim must equal this
            value, otherwise a :class:`KeycloakError` is raised. This is
            what prevents one realm client's token (e.g. the internal
            service account) from being accepted on another client's path
            (e.g. as a user session cookie).

    Returns:
        Dictionary with the decoded token payload (claims).

    Raises:
        KeycloakError: If token validation fails (invalid signature,
            expired token, malformed format, unexpected ``azp``, etc.).
    """
    try:
        # Build the JWKS URI for the configured realm
        jwks_uri = f"{get_keycloak_server_url()}/realms/{settings.keycloak_realm}/protocol/openid-connect/certs"
        jwk_client = _get_jwk_client(jwks_uri)

        # Retrieve the signing key matching the token's kid
        signing_key = await _signing_key(jwk_client, jwks_uri, token)

        # Decode and verify the token with RS256 algorithm
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except jwt.PyJWTError as e:
        logger.error(f"Token validation failed: {e}")
        raise KeycloakError(f"Invalid token: {e}") from e

    allowed_issuers = {
        f"{base.rstrip('/')}/realms/{settings.keycloak_realm}"
        for base in (
            settings.keycloak_url,
            settings.keycloak_public_url,
            settings.keycloak_internal_url,
        )
    }
    if payload.get("iss") not in allowed_issuers:
        # Not one of the configured forms: accept the form the realm itself
        # announces on the internal channel (the M2M tokens carry that one).
        announced = await _announced_issuer()
        if announced:
            allowed_issuers.add(announced)
    if payload.get("iss") not in allowed_issuers:
        logger.warning("Rejected token with issuer %r", payload.get("iss"))
        raise KeycloakError("Token was issued by an unexpected issuer (iss)")

    if expected_azp is not None and payload.get("azp") != expected_azp:
        logger.warning(
            "Rejected token with azp=%r (expected %r)",
            payload.get("azp"),
            expected_azp,
        )
        raise KeycloakError("Token has an unexpected authorized party (azp)")

    return payload


async def decode_user_token(token: str) -> dict[str, Any]:
    """Validate a browser/user access token from the session cookie.

    Beyond signature and expiry, this enforces that the token was issued
    to the browser client (``azp == keycloak_client_id``). Without this
    check, any RS256 token signed by the same realm — including the
    internal service-account token used on ``/a2a/*`` and ``/tools/*`` —
    would be accepted as a logged-in user (authentication bypass).

    Args:
        token: Encoded JWT access token taken from the session cookie.

    Returns:
        Dictionary with the decoded token payload (claims).

    Raises:
        KeycloakError: If validation fails or ``azp`` is not the browser
            client.
    """
    return await decode_token(token, expected_azp=settings.keycloak_client_id)


async def exchange_code_for_token(code: str, code_verifier: str | None = None) -> dict[str, Any]:
    """Exchange the OAuth2 authorization code for access tokens.

    Makes a POST request to Keycloak's token endpoint to complete
    the Authorization Code flow. Supports both public and confidential
    clients (with client secret).

    Args:
        code: Authorization code received from the callback redirect
            after user login.
        code_verifier: PKCE verifier matching the ``code_challenge`` sent
            on the authorization request; required by the server when the
            client enforces PKCE.

    Returns:
        Dictionary containing the tokens (``access_token``, ``id_token``,
        ``refresh_token``, etc.) and related metadata.

    Raises:
        KeycloakError: If the code exchange fails (expired code,
            invalid code, network error, etc.).
    """
    token_endpoint = f"{get_keycloak_server_url()}/realms/{settings.keycloak_realm}/protocol/openid-connect/token"

    # Required parameters for the Authorization Code flow
    data = {
        "grant_type": "authorization_code",
        "client_id": settings.keycloak_client_id,
        "code": code,
        "redirect_uri": settings.app_callback_url,
    }
    if code_verifier:
        data["code_verifier"] = code_verifier

    # Add the client secret for confidential clients
    if settings.keycloak_client_secret:
        data["client_secret"] = settings.keycloak_client_secret

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(token_endpoint, data=data)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to exchange code for token: {e}")
        raise KeycloakError(f"Token exchange failed: {e}") from e

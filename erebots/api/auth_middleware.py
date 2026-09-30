"""Authentication dispatcher for the FastAPI app.

Centralizes the access-control logic so it can be reused by the
application factory and exercised directly in unit tests.
"""

from __future__ import annotations

from fastapi import Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from erebots.config import settings
from erebots.security.keycloak import KeycloakError, decode_token, decode_user_token

_PUBLIC_PREFIXES = (
    "/api/auth/login",
    "/api/auth/callback",
    "/api/auth/logout",
    "/static",
    "/docs",
    "/openapi",
    "/favicon",
)
_PUBLIC_EXACT = ("/health",)
_INTERNAL_PREFIXES = ("/a2a/", "/tools/")


def extract_user_token(request: Request) -> str | None:
    """Return the user's Keycloak token from the cookie or the bearer header.

    Both are accepted so browser SPAs and mobile clients authenticate the same
    way; the cookie wins when both are present.

    Args:
        request: The inbound request.

    Returns:
        The raw token, or ``None`` when the request carries neither.
    """
    token = request.cookies.get(settings.jwt_cookie_name)
    if token:
        return token
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


async def auth_dispatch(request: Request, call_next):
    """Authenticate a request according to its route regime.

    Three regimes:
      - public routes (login, callback, static, health) → pass through.
      - internal M2M routes (``/a2a/*``, ``/tools/*``) → require a
        service-account bearer token whose ``azp`` matches
        ``settings.internal_client_id``. This holds for every method,
        including GET, unless the deployment opts out (below).
      - everything else → require a valid JWT cookie issued by Keycloak.

    ``settings.allow_loopback_tool_reads`` — off by default — additionally lets
    *loopback GET* requests on ``/tools/*`` through. It is a property of the
    deployment, not of the framework: the MCP tool servers mounted by
    :func:`erebots.server.create_app` speak Streamable HTTP over POST and send
    their own bearer, so they never need it. A deployment that also mounts a
    re-entrant UI under ``/tools/`` does — the example app's Gradio tool servers
    call themselves back over loopback with no ``Authorization`` header. POSTs
    stay authenticated either way, so MCP tool invocation and every mutating
    call always present the bearer.

    Enabling it trusts ``request.client.host``. See the field's comment in
    :mod:`erebots.config` for the residual risk that carries.

    Args:
        request: The inbound request, matched by path.
        call_next: The rest of the middleware chain, called only once the
            request is authorized.

    Returns:
        The downstream response, or a 401/403 (a redirect to login for a
        browser navigation) when authentication fails. Returns without checking
        anything when ``settings.enable_auth`` is off.
    """
    if not settings.enable_auth:
        return await call_next(request)

    path = request.url.path

    if path.startswith(_PUBLIC_PREFIXES) or path in _PUBLIC_EXACT:
        return await call_next(request)

    if path.startswith(_INTERNAL_PREFIXES):
        client_host = request.client.host if request.client else ""
        # Opt-in loopback carve-out, for a deployment that mounts a re-entrant UI
        # under /tools/ (see the docstring). Read-only, and never the
        # MCP/tool-invocation path, which is a POST and must always present the
        # internal bearer.
        if (
            settings.allow_loopback_tool_reads
            and path.startswith("/tools/")
            and request.method == "GET"
            and client_host in ("127.0.0.1", "::1")
        ):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse(
                {"detail": "Missing bearer token"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        bearer = auth_header.split(" ", 1)[1].strip()
        try:
            payload = await decode_token(bearer)
        except KeycloakError:
            return JSONResponse(
                {"detail": "Invalid token"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        if payload.get("azp") != settings.internal_client_id:
            return JSONResponse(
                {"detail": "Forbidden"},
                status_code=status.HTTP_403_FORBIDDEN,
            )
        return await call_next(request)

    accept = request.headers.get("accept", "")
    wants_json = path.startswith("/api") or "application/json" in accept

    token = extract_user_token(request)
    if not token:
        if wants_json:
            return JSONResponse(
                {"detail": "Not authenticated"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        return RedirectResponse(url="/api/auth/login")

    try:
        await decode_user_token(token)
    except KeycloakError:
        if wants_json:
            return JSONResponse(
                {"detail": "Invalid token"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        return RedirectResponse(url="/api/auth/login")

    return await call_next(request)

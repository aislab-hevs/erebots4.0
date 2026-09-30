"""OAuth2/OIDC authentication endpoints via Keycloak.

Handles the login, callback, logout flows and JWT token validation.
Uses HTTP-only cookies to securely store access and ID tokens.
"""

import base64
import hashlib
import secrets
import time

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse

from erebots.config import settings
from erebots.security.keycloak import (
    KeycloakError,
    exchange_code_for_token,
    get_login_redirect_url,
    get_logout_redirect_url,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# In-memory dictionary for CSRF protection during the OAuth2 flow.
# Each entry maps a state token to (creation timestamp, PKCE code verifier) —
# the verifier lives server-side only, never in the browser.
# In production, consider using Redis for multi-process environments.
_auth_states: dict[str, tuple[float, str]] = {}

# Maximum validity duration of a CSRF state token (5 minutes)
_STATE_EXPIRY_SECONDS = 300


def _cleanup_expired_states():
    """Remove expired CSRF states from the in-memory dictionary.

    Iterates over all entries in ``_auth_states`` and deletes those
    whose timestamp exceeds ``_STATE_EXPIRY_SECONDS``, preventing memory leaks.
    """
    now = time.time()
    expired = [
        k for k, (created, _) in _auth_states.items() if now - created > _STATE_EXPIRY_SECONDS
    ]
    for k in expired:
        del _auth_states[k]


@router.get("/login")
async def login():
    """Initiate the login flow by redirecting the user to Keycloak.

    Returns:
        A redirect to the Keycloak login page, carrying a generated state
        parameter for CSRF protection.
    """
    # Proactively clean up expired states
    _cleanup_expired_states()

    # Generate a cryptographically secure state token plus a PKCE verifier;
    # only the S256 challenge leaves the server (RFC 7636), so an intercepted
    # authorization code cannot be exchanged without this login's verifier.
    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(64)
    _auth_states[state] = (time.time(), code_verifier)

    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    redirect_url = get_login_redirect_url(state, code_challenge=code_challenge)
    return RedirectResponse(url=redirect_url)


@router.get("/callback")
async def callback(code: str, state: str, request: Request):
    """Handle the OAuth2 callback from Keycloak after login.

    Verifies the state parameter for CSRF protection, exchanges the
    authorization code for an access token, and sets secure cookies.

    Args:
        code: Authorization code received from Keycloak.
        state: State parameter for CSRF verification.
        request: FastAPI request object.

    Returns:
        A redirect to the app, with the auth cookies attached.

    Raises:
        HTTPException: If the state is invalid, the token is missing,
            or the code exchange fails.
    """
    # Verify the state parameter for CSRF protection
    if state not in _auth_states:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state parameter. Please try logging in again.",
        )
    _, code_verifier = _auth_states.pop(state)

    try:
        # Exchange the authorization code for tokens
        token_response = await exchange_code_for_token(code, code_verifier=code_verifier)
        access_token = token_response.get("access_token")
        id_token = token_response.get("id_token")

        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No access token in response",
            )

        expires_in = token_response.get("expires_in", 300)

        redirect = RedirectResponse(url=settings.ui_path, status_code=status.HTTP_302_FOUND)

        # Set the HTTP-only cookie with the access token
        redirect.set_cookie(
            key=settings.jwt_cookie_name,
            value=access_token,
            httponly=True,
            samesite="lax",
            secure=settings.jwt_cookie_secure,
            max_age=expires_in,
            path="/",
        )

        # Set the ID token cookie (required for logout)
        if id_token:
            redirect.set_cookie(
                key=settings.id_token_cookie_name,
                value=id_token,
                httponly=True,
                samesite="lax",
                secure=settings.jwt_cookie_secure,
                max_age=expires_in,
                path="/",
            )

        return redirect

    except KeycloakError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authentication failed: {str(e)}",
        ) from e


@router.post("/logout")
async def logout(request: Request, response: Response):
    """Log out the user by clearing cookies and returning the Keycloak logout URL.

    Args:
        request: Request object to retrieve the ID token from cookies.
        response: Response object to clear session cookies.

    Returns:
        dict: Dictionary with the Keycloak logout URL and a confirmation message.
    """
    # Clear authentication cookies
    response.delete_cookie(key=settings.jwt_cookie_name, path="/")
    response.delete_cookie(key=settings.id_token_cookie_name, path="/")

    # Retrieve the ID token for Keycloak-side logout
    id_token = request.cookies.get(settings.id_token_cookie_name)
    logout_url = get_logout_redirect_url(id_token)

    return {"logout_url": logout_url, "message": "Logged out"}

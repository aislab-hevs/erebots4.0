"""User profile menu component.

Provides the profile dropdown menu with the user's initials,
the link to the Keycloak profile, settings, and the logout button.
Also includes the button to return to the app selection page.
"""

import gradio as gr

from erebots.config import settings
from erebots.security.keycloak import (
    KeycloakError,
    decode_user_token,
    get_keycloak_public_url,
)


def _initials_from_name(name: str) -> str:
    """Extract initials from the user's full name.

    Args:
        name: The user's full name (e.g. "John Smith").

    Returns:
        A string with the uppercase initials. If the name has only one
        word, returns the first two characters. If the name is empty
        or missing, returns "U" as a default value.
    """
    if not name:
        return "U"
    parts = [part for part in name.strip().split() if part]
    if not parts:
        return "U"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return f"{parts[0][0]}{parts[-1][0]}".upper()


async def get_user_initials(request: gr.Request):
    """Retrieve the user's initials from the JWT token in the request.

    Reads the JWT cookie from the HTTP request, decodes the token via
    Keycloak, and extracts the name to compute the initials.

    Args:
        request: The Gradio request object containing HTTP cookies.

    Returns:
        A Gradio update with the user's initials, or "U" if the token
        is missing or invalid.
    """
    token = request.cookies.get(settings.jwt_cookie_name) if request else None
    if not token:
        return gr.update(value="U")
    try:
        # The user-scoped check, matching the rest of the session path; see
        # the note in page.py's resolver.
        payload = await decode_user_token(token)
    except KeycloakError:
        return gr.update(value="U")

    name = payload.get("name") or payload.get("preferred_username") or payload.get("email") or ""
    return gr.update(value=_initials_from_name(name))


def _toggle_dropdown(is_open: bool):
    """Toggle the visibility of the profile dropdown menu.

    Args:
        is_open: The current open state of the menu.

    Returns:
        A tuple with the new state (inverted) and a Gradio update
        for the dropdown visibility.
    """
    next_open = not is_open
    return next_open, gr.update(visible=next_open)


def profile_menu():
    """Build the Gradio user profile menu component.

    Creates a circular button with the user's initials that, on click,
    shows a dropdown menu with the options: user profile (Keycloak link),
    settings, and logout. Logout is handled via client-side JavaScript
    that calls the ``/api/auth/logout`` endpoint.

    Returns:
        The profile ``gr.Button`` component, useful for updating its value
        (initials) on page load, or ``None`` when authentication is off and
        there is no identity to show.
    """
    if not settings.enable_auth:
        # Nothing here works without Keycloak: there is no account to open and
        # no session to end, and the two links would fail in the browser rather
        # than explain themselves. A deployment with authentication off has no
        # identity, so it gets no identity menu.
        return None

    # The *public* URL, the one a browser can reach. Behind a proxy
    # ``keycloak_url`` is the address the server uses, which resolves to
    # nothing from the user's machine -- this link is followed by the browser.
    account_url = f"{get_keycloak_public_url()}/realms/{settings.keycloak_realm}/account"

    with gr.Column(elem_id="profile-menu"):
        is_open = gr.State(False)
        profile_button = gr.Button(
            value="U",
            elem_id="profile-button",
            elem_classes="profile-circle",
            scale=1,
        )
        with gr.Column(elem_id="profile-dropdown", visible=False) as dropdown:
            gr.Button(
                value="User profile",
                elem_classes="profile-menu-item",
                link=account_url,
            )
            gr.Button(
                value="Settings",
                elem_classes="profile-menu-item",
                interactive=False,
            )
            logout_button = gr.Button(
                value="Logout",
                elem_classes="profile-menu-item",
            )

    # Handle click on the profile button: toggle dropdown visibility
    profile_button.click(
        fn=_toggle_dropdown,
        inputs=is_open,
        outputs=[is_open, dropdown],
        show_progress=False,
    )

    # Handle logout via JavaScript call to the authentication endpoint
    logout_button.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="""
        async () => {
            const response = await fetch("/api/auth/logout", {
                method: "POST",
                credentials: "include"
            });

            if (response.ok) {
                const data = await response.json().catch(() => ({}));
                if (data.logout_url) {
                    window.location.href = data.logout_url;
                } else {
                    window.location.href = "/";
                }
            } else {
                alert("Logout failed");
            }
        }
        """,
    )

    return profile_button

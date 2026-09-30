"""Data model for user context.

Defines the ``UserContext`` dataclass that collects all information related to
a user: identification, roles, conversation history, profile, and privacy
consents. Data only -- turning it into the text a prompt carries is
:func:`erebots.runtime.render_user_context`, because that is a property of how
a turn is built rather than of how it is stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UserContext:
    """Context of a system user.

    Collects identification information, roles, conversation history,
    personalized profile, and privacy consents.

    Attributes:
        user_id: Unique user identifier, as the application resolves it from
            an authenticated session.
        username: Display name.
        email: Email address.
        roles: Roles the identity provider granted.
        chat_history: Conversation message history.
        profile: Dictionary with personalized profile information.
        consents: Dictionary with the state of privacy consents.
    """

    user_id: str
    username: str = ""
    email: str = ""
    roles: list[str] = field(default_factory=list)
    chat_history: list[dict] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    consents: dict = field(default_factory=dict)

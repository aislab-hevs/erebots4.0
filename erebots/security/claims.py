"""How an identity provider's token claims become a user's stored fields.

One home for the mapping, kept out of :mod:`erebots.persistence`, which stores
rows and has no business knowing what an OIDC provider calls things, and out of
:mod:`erebots.security.keycloak`, which validates tokens and deliberately reads
only ``iss``, ``azp`` and ``kid``.

The default below is Keycloak-shaped, because the bundled realm is. ``roles``
in particular comes from ``realm_access``, which is Keycloak's structure and
not part of OIDC. An application on another provider passes its own mapping to
:class:`~erebots.runtime.ConversationService` rather than editing this file.

Returns a plain dict rather than a ``UserContext`` on purpose: that keeps this
module free of any import from the persistence layer, so a mapping function is
a mapping function and nothing drags a database driver behind it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Injected wherever a first-sight user row has to be seeded from a token; see
# SessionResolver and TokenProvider for the same shape used elsewhere.
ClaimsMapper = Callable[[dict], dict[str, Any]]


def user_fields_from_claims(claims: dict) -> dict[str, Any]:
    """Map decoded Keycloak claims to the fields a new user row starts with.

    Only the fields an identity provider can tell us are mapped. ``profile``,
    ``consents`` and ``chat_history`` are the user's own and start empty; they
    are never seeded from a token.

    Args:
        claims: The decoded token payload.

    Returns:
        ``username``, ``email`` and ``roles``, each defaulted rather than
        absent, so a caller can hand the result straight to a row constructor.
        ``sub`` is not included: the caller already knows the user id, which is
        how it found the row in the first place.
    """
    return {
        "username": claims.get("preferred_username", ""),
        "email": claims.get("email", ""),
        "roles": claims.get("realm_access", {}).get("roles", []),
    }

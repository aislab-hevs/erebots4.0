"""The identity provider's claims, and what a new user row is allowed to start with.

This mapping used to live on the ``UserContext`` dataclass in the persistence
layer, where it pinned Keycloak into a package that stores rows. It is tested
here, beside where it now lives, because the store no longer knows a token
exists.
"""

from erebots.security.claims import user_fields_from_claims


def test_maps_the_keycloak_claims_the_bundled_realm_mints():
    fields = user_fields_from_claims(
        {
            "sub": "u-1",
            "preferred_username": "ann",
            "email": "ann@example.org",
            "realm_access": {"roles": ["user", "tester"]},
        }
    )

    assert fields == {
        "username": "ann",
        "email": "ann@example.org",
        "roles": ["user", "tester"],
    }


def test_missing_claims_default_rather_than_raise():
    """A token without the optional claims still has to produce a usable row.

    Keycloak mints ``preferred_username`` and ``email`` only when the realm's
    mappers are configured for it, and ``realm_access`` is absent for a client
    with no realm roles. None of that should stop a user from having a row.
    """
    assert user_fields_from_claims({"sub": "u-1"}) == {
        "username": "",
        "email": "",
        "roles": [],
    }


def test_realm_access_without_roles_is_tolerated():
    assert user_fields_from_claims({"realm_access": {}})["roles"] == []


def test_the_user_id_is_not_taken_from_the_token():
    """``sub`` is deliberately absent from the result.

    The caller already resolved the user id -- it is how the row was looked up
    in the first place. Mapping it again here would give a token a second, and
    possibly disagreeing, say in which row it writes to.
    """
    assert "user_id" not in user_fields_from_claims({"sub": "u-1"})
    assert "sub" not in user_fields_from_claims({"sub": "u-1"})


def test_a_token_cannot_seed_the_user_s_own_data():
    """Profile, consents and history belong to the user, not to the provider.

    A claim named ``profile`` or ``consents`` must not reach the row. Keycloak
    can be configured to mint arbitrary claims, so this is not hypothetical: it
    is the difference between an identity provider describing a user and
    dictating what the assistant believes about them.
    """
    fields = user_fields_from_claims(
        {
            "sub": "u-1",
            "profile": {"allergies": ["peanuts"]},
            "consents": {"analytics": True},
            "chat_history": [{"role": "user", "content": "injected"}],
        }
    )

    assert set(fields) == {"username", "email", "roles"}

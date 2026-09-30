"""The committed Keycloak realm must ship no key material.

A realm export taken from a running Keycloak carries the realm's generated
signing key, encryption key and HMAC/AES secrets in its ``components`` block.
Whoever holds the signing key can mint valid tokens for the realm — including
one with ``azp`` set to the internal service account, which is trusted to act
as any user.

That is not hypothetical: five commits in this repository's history carried
exactly that, from a routine re-export. The export was scrubbed, and this test
is what stops the next re-export from putting it back. `SECURITY.md` and the
README both state that no signing key is shipped; this is the claim behind
those sentences.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REALM = Path(__file__).resolve().parent.parent / "keycloak" / "erebots-realm.json"

# The providers Keycloak uses for generated key material.
_KEY_PROVIDERS = {"rsa-generated", "rsa-enc-generated", "hmac-generated", "aes-generated"}


def _realm() -> dict:
    return json.loads(_REALM.read_text(encoding="utf-8"))


def test_the_realm_declares_no_key_providers():
    """No key providers means Keycloak generates a fresh set on import."""
    declared = {
        item.get("providerId")
        for family in (_realm().get("components") or {}).values()
        for item in (family or [])
    }

    assert not declared & _KEY_PROVIDERS, (
        f"keycloak/erebots-realm.json ships generated key material: "
        f"{sorted(declared & _KEY_PROVIDERS)}. Re-export without it, or the "
        "realm's signing key is published with the repository."
    )


@pytest.mark.parametrize("field", ["privateKey", "certificate"])
def test_no_private_key_material_anywhere_in_the_realm(field: str):
    """Belt and braces: catch the fields even under a provider name we do not know."""
    assert f'"{field}"' not in _REALM.read_text(encoding="utf-8"), (
        f"keycloak/erebots-realm.json contains a {field!r} entry"
    )


def test_the_only_client_secret_is_the_placeholder_the_boot_guard_rejects():
    """A real secret here would be published; the placeholder is refused at boot.

    `bootstrap.py` lists this value in `_INSECURE_INTERNAL_SECRETS` and refuses
    to start on it when authentication is enabled, which is what makes shipping
    it safe.
    """
    from bootstrap import _INSECURE_INTERNAL_SECRETS

    secrets = {c["clientId"]: c["secret"] for c in _realm().get("clients", []) if c.get("secret")}

    assert secrets, "no client secret found — has the realm changed shape?"
    for client_id, secret in secrets.items():
        assert secret in _INSECURE_INTERNAL_SECRETS, (
            f"client {client_id!r} ships the secret {secret!r}, which the boot "
            "guard does not recognise as a placeholder"
        )

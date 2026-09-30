"""Domain configuration for the example app's specialists.

Settings that belong to the *example domain* live here, not in
:mod:`erebots.config`: the framework has no business knowing which specialists
an application ships. Adding a specialist adds its block here; nothing in
``erebots/`` changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from erebots.config import get_bool, get_str


@dataclass(frozen=True)
class DomainSettings:
    """Example-app configuration (immutable).

    Plain defaults here; :meth:`from_env` reads the environment *when called*,
    so the values reflect the environment at bootstrap, not at import. The
    composition root builds one instance and hands it to the components.
    """

    # Conversion agent. Its answers come from a deterministic tool rather than
    # from the model's arithmetic, so a small model is enough.
    conversion_api_key: str = "EMPTY"
    conversion_api_base_url: str = "http://localhost:8080/v1"
    conversion_model: str = "local"
    conversion_thinking: bool = False

    # Personal agent: the only agent that reads or writes a user's profile and
    # consents. Its tools are the way anything else asks for them.
    personal_api_key: str = "EMPTY"
    personal_api_base_url: str = "http://localhost:8080/v1"
    personal_model: str = "local"
    personal_thinking: bool = False

    @classmethod
    def from_env(cls) -> DomainSettings:
        """Build the settings from the environment as it is now."""
        defaults = cls()
        return cls(
            conversion_api_key=get_str("conversion_api_key", defaults.conversion_api_key),
            conversion_api_base_url=get_str(
                "conversion_api_base_url", defaults.conversion_api_base_url
            ),
            conversion_model=get_str("conversion_model", defaults.conversion_model),
            conversion_thinking=get_bool("conversion_thinking", defaults.conversion_thinking),
            personal_api_key=get_str("personal_api_key", defaults.personal_api_key),
            personal_api_base_url=get_str(
                "personal_api_base_url", defaults.personal_api_base_url
            ),
            personal_model=get_str("personal_model", defaults.personal_model),
            personal_thinking=get_bool("personal_thinking", defaults.personal_thinking),
        )


# Module-level instance for code configured at import time. Everything else
# takes a DomainSettings from the composition root.
domain_settings = DomainSettings.from_env()

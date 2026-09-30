"""DomainSettings reads the environment when built, not when imported."""

from __future__ import annotations

from specialists.config import DomainSettings


def test_from_env_reflects_the_environment_at_call_time(monkeypatch):
    """The module-level instance is built at import; from_env is not.

    A composition root that builds its settings after changing the
    environment has to see the change, which is the whole reason this is a
    classmethod rather than a module constant.
    """
    monkeypatch.delenv("conversion_model", raising=False)
    assert DomainSettings.from_env().conversion_model == "local"

    monkeypatch.setenv("conversion_model", "big-one")
    monkeypatch.setenv("conversion_thinking", "true")
    built = DomainSettings.from_env()
    assert built.conversion_model == "big-one"
    assert built.conversion_thinking is True

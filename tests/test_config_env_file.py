"""``.env`` belongs to the application, not to the installed framework.

The loader used to resolve the file relative to ``erebots/config.py``, which for
a wheel installation is ``site-packages/.env`` — a path that never exists. A
consumer's configuration was therefore read only when they happened to run from
a checkout of this repository, and ignored silently otherwise.

These tests pin where the file is looked for and the precedence around it.
"""

from __future__ import annotations

import os

import pytest

from erebots import config

_PROBE = "erebots_probe_setting"


@pytest.fixture
def environ(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """A throwaway ``os.environ`` the loader may write into.

    Two things make this necessary. ``_load_env_file`` applies values with
    ``os.environ.setdefault``, which monkeypatch cannot undo, so the real process
    environment would keep whatever a test loaded. And the root ``conftest.py``
    sets ``EREBOTS_IGNORE_DOTENV`` before anything imports this module, so the
    flag has to be cleared or the loader returns immediately.

    ``os.getenv`` looks up the module-global ``environ`` on every call, so
    replacing the attribute redirects the reads and the writes together.
    """
    env = dict(os.environ)
    env.pop("EREBOTS_IGNORE_DOTENV", None)
    monkeypatch.setattr(os, "environ", env)
    return env


def test_dotenv_is_read_from_the_working_directory(tmp_path, monkeypatch, environ):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"{_PROBE}=from-cwd\n", encoding="utf-8")

    config._load_env_file()

    assert environ[_PROBE] == "from-cwd"


def test_nothing_is_loaded_when_the_working_directory_has_no_env_file(
    tmp_path, monkeypatch, environ
):
    """The regression guard: the package's own directory is no longer consulted.

    A development checkout carries a ``.env`` at its root, and the old
    ``__file__``-relative lookup found it from any working directory. From an
    unrelated one the loader must now do nothing at all.
    """
    monkeypatch.chdir(tmp_path)
    before = dict(environ)

    config._load_env_file()

    assert environ == before


def test_an_explicit_env_file_wins_over_the_working_directory(tmp_path, monkeypatch, environ):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"{_PROBE}=from-cwd\n", encoding="utf-8")
    elsewhere = tmp_path / "deploy" / "production.env"
    elsewhere.parent.mkdir()
    elsewhere.write_text(f"{_PROBE}=from-override\n", encoding="utf-8")
    environ["EREBOTS_ENV_FILE"] = str(elsewhere)

    config._load_env_file()

    assert environ[_PROBE] == "from-override"


def test_an_existing_environment_variable_wins_over_the_file(tmp_path, monkeypatch, environ):
    """``setdefault`` semantics: the file never overwrites a real variable."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"{_PROBE}=from-file\n", encoding="utf-8")
    environ[_PROBE] = "already-set"

    config._load_env_file()

    assert environ[_PROBE] == "already-set"


def test_ignore_dotenv_skips_the_file_entirely(tmp_path, monkeypatch, environ):
    """The escape hatch the whole test suite depends on."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"{_PROBE}=from-file\n", encoding="utf-8")
    environ["EREBOTS_IGNORE_DOTENV"] = "1"

    config._load_env_file()

    assert _PROBE not in environ

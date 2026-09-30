"""The published API surface.

These import paths are the ones ``docs/api-stability.md`` promises to consumers,
so breaking one is a release-notes event rather than a refactor. The table below
mirrors that document; the two are meant to be changed together.
"""

from __future__ import annotations

import importlib
import importlib.resources
import tomllib
from pathlib import Path

import pytest

import erebots

_REPO_ROOT = Path(__file__).resolve().parent.parent

# (module path, exported name) — the Supported table in docs/api-stability.md.
_PUBLIC_API = [
    ("erebots.agents", "BaseAgent"),
    ("erebots.agents", "BaseSpecialist"),
    ("erebots.agents", "GenerativeLayer"),
    ("erebots.agents", "Skill"),
    ("erebots.agents", "SkillRegistry"),
    ("erebots.agents", "AgentResponse"),
    ("erebots.agents", "TraceStep"),
    ("erebots.agents", "InputRequired"),
    ("erebots.agents.specialist", "BaseSpecialist"),
    ("erebots.agents.orchestrator", "OrchestratorAgent"),
    ("erebots.agents.a2a", "A2AClient"),
    ("erebots.agents.a2a", "create_a2a_app"),
    ("erebots.agents.mcp", "MCPClient"),
    ("erebots.agents.mcp", "MCPToolError"),
    ("erebots.agents.loader", "load_agents"),
    ("erebots.agents.loader", "configured_urls"),
    ("erebots.server", "create_app"),
    ("erebots.server", "create_chat_router"),
    ("erebots.server", "default_session_resolver"),
    ("erebots.server", "format_sse_event"),
    ("erebots.server", "sse_response"),
    ("erebots.runtime", "AgentEvent"),
    ("erebots.runtime", "EventType"),
    ("erebots.runtime", "ConversationService"),
    ("erebots.runtime", "UserContextLike"),
    ("erebots.runtime", "UserContextRenderer"),
    ("erebots.runtime", "UserContextStoreProtocol"),
    ("erebots.runtime", "render_user_context"),
    ("erebots.persistence", "Base"),
    ("erebots.persistence", "UserContextStore"),
    ("erebots.persistence", "make_engine"),
    ("erebots.persistence", "make_session_factory"),
    ("erebots.config", "settings"),
    ("erebots.config", "Settings"),
    ("erebots.config", "get_bool"),
    ("erebots.config", "get_int"),
    ("erebots.config", "get_list"),
    ("erebots.config", "get_str"),
    ("erebots.security.keycloak", "decode_user_token"),
    ("erebots.security.internal_token", "InternalTokenProvider"),
    ("erebots.security.claims", "ClaimsMapper"),
    ("erebots.security.claims", "user_fields_from_claims"),
]


@pytest.mark.parametrize(("module_path", "name"), _PUBLIC_API)
def test_documented_public_imports_resolve(module_path: str, name: str):
    module = importlib.import_module(module_path)

    assert getattr(module, name, None) is not None, f"{module_path}.{name} is gone"


@pytest.mark.parametrize(
    "module_path",
    ["erebots.agents", "erebots.agents.a2a", "erebots.agents.mcp",
     "erebots.agents.orchestrator", "erebots.server", "erebots.runtime",
     "erebots.persistence"],
)
def test_exporting_packages_declare_what_they_export(module_path: str):
    """``__all__`` is what makes "supported" reviewable in a diff."""
    module = importlib.import_module(module_path)

    assert getattr(module, "__all__", None), f"{module_path} declares no __all__"
    for name in module.__all__:
        assert hasattr(module, name), f"{module_path}.__all__ names missing {name}"


def test_the_package_ships_the_typing_marker():
    """PEP 561: without this file a consumer's type checker ignores our hints.

    Asserted through the imported package rather than a repository path, because
    the claim is about what an installed distribution carries.
    """
    marker = importlib.resources.files("erebots").joinpath("py.typed")

    assert marker.is_file()


def test_version_matches_the_project_metadata():
    """``importlib.metadata`` would only work when installed; the file is truth."""
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert erebots.__version__ == pyproject["project"]["version"]

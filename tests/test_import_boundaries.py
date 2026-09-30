"""Architecture guardrail: the framework core must not depend on the domain or a frontend.

A "serious framework" requires the generic core to be decoupled from (a) the example
domain (``specialists/``) and (b) any specific frontend (``gradio``). This test enforces
that boundary by **static analysis** (AST) on a curated list of core modules that are
already clean today. It deliberately does not *import* the modules, so it stays fast and
free of heavy optional dependencies (torch, openai, ...).

Notes:

* Application composition (which specialists to host, persistence wiring) lives
  in ``bootstrap.py`` at the repo root — *app* code that imports ``specialists``
  directly. The framework (``erebots/``) never imports the domain.
* Frontends live **outside** the package: the Gradio chat UI is an example
  adapter (``examples/gradio_ui/``), like the CLI and the Vue SPA. The whole
  ``erebots/`` tree is swept — no module may import a forbidden root.

When a milestone lands, move the newly-clean modules into ``CORE_CLEAN`` so regressions
are caught.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level packages the core must never import.
_FORBIDDEN_ROOTS = ("specialists", "gradio")

# Core modules verified decoupled today. Extend this list as the refactor proceeds.
CORE_CLEAN = [
    "erebots/agents/base.py",
    "erebots/agents/models.py",
    "erebots/agents/skill.py",
    "erebots/agents/generative.py",
    "erebots/agents/loader.py",
    "erebots/agents/specialist.py",
    "erebots/agents/orchestrator/agent.py",
    "erebots/agents/a2a/client.py",
    "erebots/agents/a2a/server.py",
    "erebots/agents/mcp/client.py",
    "erebots/runtime/events.py",
    "erebots/runtime/conversation.py",
    "erebots/security/keycloak.py",
    "erebots/security/internal_token.py",
    "erebots/persistence/db.py",
    "erebots/persistence/models.py",
    "erebots/server/sse.py",
    "erebots/server/app.py",
]


def _imported_roots(path: Path) -> set[str]:
    """Return the set of top-level package names imported by a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("rel_path", CORE_CLEAN)
def test_core_module_is_decoupled_from_domain_and_frontend(rel_path: str) -> None:
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"listed core module no longer exists: {rel_path}"
    offending = sorted(_imported_roots(path) & set(_FORBIDDEN_ROOTS))
    assert not offending, (
        f"{rel_path} must stay decoupled from {_FORBIDDEN_ROOTS}, "
        f"but it imports: {offending}"
    )


def test_no_module_in_the_package_imports_a_forbidden_root() -> None:
    """Package-wide sweep: the installable ``erebots`` wheel is frontend- and
    domain-free — no module under ``erebots/`` imports gradio or specialists."""
    offenders: dict[str, list[str]] = {}
    for path in sorted((_REPO_ROOT / "erebots").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        offending = sorted(_imported_roots(path) & set(_FORBIDDEN_ROOTS))
        if offending:
            offenders[str(path.relative_to(_REPO_ROOT))] = offending
    assert not offenders, f"framework modules import forbidden roots: {offenders}"

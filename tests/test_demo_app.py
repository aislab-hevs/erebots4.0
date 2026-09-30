"""The demonstrator app (``demo.py``).

Its reason to exist is what it leaves out: the whole protocol path — routing,
A2A, MCP tools, persistence — running without Gradio, the nutrition example or
the RAG stack. That is an *import* property, so the check runs in a subprocess:
by the time this suite gets here, other tests have long since imported gradio,
and ``demo.py`` builds its app at import time (as ``app.py`` does, so uvicorn
can find it).

The child therefore gets a deliberately narrow environment: none of the
developer's application configuration, but enough of the operating system to
start Python. See :func:`_subprocess_env` for why the second half matters.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Variables the child needs from the *operating system*, as opposed to the
# application configuration this environment deliberately withholds. Windows
# locates its Winsock provider through SYSTEMROOT, so dropping it makes the
# child fail at ``import asyncio`` with ``OSError: [WinError 10106]`` long
# before it reaches the checks below.
_OS_PASSTHROUGH = ("PATH", "SYSTEMROOT")

_CHECK = """
import sys
import demo

leaked = sorted(
    m for m in sys.modules
    if m == "gradio"
    or m.startswith(("gradio.", "torch", "transformers", "sentence_transformers"))
)
assert not leaked, f"demo.py pulled in {leaked}"

agents = demo.app.state.agents
assert sorted(agents.specialists) == ["conversion", "personal"], sorted(agents.specialists)
assert sorted(agents.mcp_apps) == ["conversion", "personal"], sorted(agents.mcp_apps)

mounted = {getattr(r, "path", "") for r in demo.app.routes}
missing = {"/a2a/conversion", "/a2a/personal", "/tools/conversion", "/tools/personal"} - mounted
assert not missing, f"not mounted: {missing}"

# Each specialist has to call the endpoint the framework actually mounts.
urls = {name: a._mcp_server_urls for name, a in agents.specialists.items()}
assert urls == {
    "conversion": ["http://localhost:7860/tools/conversion/mcp"],
    "personal": ["http://localhost:7860/tools/personal/mcp"],
}, urls
print("ok")
"""


def _subprocess_env(tmp_path: Path) -> dict[str, str]:
    """The child's environment: our application settings, plus the platform's own.

    The sanitisation targets *application* configuration — the child must not
    inherit the developer's exported ``erebots`` settings, and
    ``EREBOTS_IGNORE_DOTENV`` keeps it off their ``.env`` as well. It does not
    target the operating system: passing :data:`_OS_PASSTHROUGH` through is what
    keeps this test runnable on Windows as well as POSIX.
    """
    env = {
        "EREBOTS_IGNORE_DOTENV": "1",
        "enable_auth": "false",
        "agents_config_path": "agents.demo.yaml",
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'demo.db'}",
    }
    for name in _OS_PASSTHROUGH:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def test_demo_app_boots_without_gradio_nutrition_or_rag(tmp_path):
    env = _subprocess_env(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout

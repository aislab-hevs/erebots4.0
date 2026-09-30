"""Erebots4.0 demonstrator: the protocol path with no Gradio and no RAG.

The smallest deployment that still exercises the whole path — routing, A2A
between orchestrator and specialist, MCP tool calls, persistence:

* **PersonalAgent** — the user's profile and consents. It is the only agent
  allowed to touch them; anything else asks it.
* **ConversionAgent** — unit conversion through a deterministic MCP tool,
  answering from arithmetic rather than from a model's arithmetic.

It reuses the same composition root as the full example app
(:func:`bootstrap.build_agents`) with its own specialist set, so nothing here
duplicates the infrastructure wiring, and it imports no Gradio. The frontend is
the static SPA in ``examples/web`` talking to ``/api/chat/stream``, or any
client of that endpoint (see ``examples/cli_chat.py``).

Run::

    agents_config_path=agents.demo.yaml enable_auth=false \\
        database_url=sqlite+aiosqlite:///./demo.db \\
        uv run --frozen python demo.py

Then open http://localhost:7860/ (or point a client at ``/api/chat/stream``).
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from bootstrap import build_agents
from erebots.agents.specialist import BaseSpecialist
from erebots.config import Settings, settings
from erebots.server import create_app as create_framework_app
from specialists.config import DomainSettings
from specialists.conversion import ConversionAgent
from specialists.personal import PersonalAgent


def build_demo_specialists(
    settings: Settings, domain_settings: DomainSettings
) -> dict[str, BaseSpecialist]:
    """The demonstrator's specialists, keyed by name.

    The key is the A2A mount path (``/a2a/<name>``) and must match the URL
    listed for it in ``agents.demo.yaml``.
    """

    def mcp(tool: str) -> str:
        return f"http://localhost:{settings.app_port}/tools/{tool}/mcp"

    return {
        "personal": PersonalAgent(
            api_key=domain_settings.personal_api_key,
            api_base_url=domain_settings.personal_api_base_url,
            model=domain_settings.personal_model,
            mcp_server_urls=[mcp("personal")],
            thinking=domain_settings.personal_thinking,
        ),
        "conversion": ConversionAgent(
            api_key=domain_settings.conversion_api_key,
            api_base_url=domain_settings.conversion_api_base_url,
            model=domain_settings.conversion_model,
            mcp_server_urls=[mcp("conversion")],
            thinking=domain_settings.conversion_thinking,
        ),
    }


def create_demo_app() -> FastAPI:
    """Build the demonstrator app."""
    domain_settings = DomainSettings.from_env()
    agents = build_agents(
        settings,
        domain_settings,
        specialists=build_demo_specialists(settings, domain_settings),
    )

    app = create_framework_app(
        agents.orchestrator,
        agents_config_path=settings.agents_config_path,
        a2a_apps=agents.a2a_apps,
        mcp_apps=agents.mcp_apps,
        engine=agents.engine,
        a2a_task_store=agents.a2a_task_store,
        token_provider=(
            agents.internal_token_provider.get_token
            if agents.internal_token_provider is not None
            else None
        ),
    )
    app.state.agents = agents

    # Mounted last: it is a catch-all, so /api, /a2a/* and /tools/* have to be
    # registered before it.
    app.mount(
        settings.ui_path,
        StaticFiles(directory="examples/web", html=True),
        name="web-ui",
    )
    return app


app = create_demo_app()


if __name__ == "__main__":
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)

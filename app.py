"""Erebots4.0 example app: the framework factory plus a concrete domain.

Builds the agent graph (``bootstrap``), assembles the framework HTTP surface via
:func:`erebots.server.create_app`, then layers this application's own pieces on
top: the root redirect and the Gradio chat UI. The generic assembly (auth,
health, SSE, A2A mounts) lives in the framework and boots with zero
specialists; everything below is application code.
"""

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from bootstrap import build_agents
from erebots.config import settings
from erebots.security.keycloak import KeycloakError, decode_user_token
from erebots.server import create_app as create_framework_app
from examples.gradio_ui.page import build_page_blocks

# HTML tag for loading the custom font in Gradio pages
_HEAD = """<link href="https://fonts.googleapis.com/css2?family=Zalando+Sans+Expanded:ital,wght@0,200..900;1,200..900&display=swap" rel="stylesheet">"""


def create_app() -> FastAPI:
    """Build the example app: the framework factory plus this domain's UI."""
    agents = build_agents(settings)

    # The framework assembles CORS, auth, health, the SSE chat endpoint and the
    # A2A specialist mounts. It boots with zero specialists; here we pass ours.
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

    # With ui_path="/" the chat UI is mounted directly at the root (the auth
    # middleware already redirects anonymous browsers to login). On a sub-path,
    # keep a root redirect pointing at it.
    if settings.ui_path != "/":

        @app.get("/")
        async def root(request: Request):
            """Root endpoint: redirect to the chat UI (or login)."""
            if not settings.enable_auth:
                return RedirectResponse(url=settings.ui_path)

            token = request.cookies.get(settings.jwt_cookie_name)
            if token:
                try:
                    await decode_user_token(token)
                    return RedirectResponse(url=settings.ui_path)
                except KeycloakError:
                    pass

            return RedirectResponse(url="/api/auth/login")

    # Every tool server this application ships is built with the MCP SDK and
    # mounted by the composition root, so nothing is mounted here. Gradio-hosted
    # tool servers used to be, which is the only reason
    # ``allow_loopback_tool_reads`` exists; with none left, that setting has no
    # subject in this tree.

    # Mount the chat UI last: with ui_path="/" it is a catch-all mount, so every
    # more specific route (/api, /a2a/*, /tools/*) must be registered before it.
    # Which UI is served is config (frontend=gradio|web); both consume the same core.
    if settings.frontend == "web":
        # Static Vue SPA (examples/web): same-origin — cookie auth, /api/* calls.
        app.mount(
            settings.ui_path,
            StaticFiles(directory="examples/web", html=True),
            name="web-ui",
        )
    else:
        app = gr.mount_gradio_app(
            app,
            build_page_blocks(
                agents.orchestrator,
                agents_config_path=settings.agents_config_path,
                token_provider=(
                    agents.internal_token_provider.get_token
                    if agents.internal_token_provider is not None
                    else None
                ),
            ),
            path=settings.ui_path,
            head=_HEAD,
            css_paths=["examples/gradio_ui/css/app.css", "examples/gradio_ui/css/page.css"],
            footer_links=["gradio", "settings"],
        )

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
    )

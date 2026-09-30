"""The assistant page.

A Gradio interface over the orchestrator: the brand mark, the chat, and the
profile menu. What it is called on screen comes from ``settings.brand_name``
rather than from this file, so a deployment brands it without editing code.

Built as a factory (``build_page_blocks``) bound to one orchestrator instance.
"""

import asyncio
from pathlib import Path

import gradio as gr

from erebots.agents.loader import load_agents
from erebots.agents.orchestrator import OrchestratorAgent
from erebots.config import settings
from erebots.security.keycloak import KeycloakError, decode_user_token
from examples.gradio_ui.chat import GradioChat
from examples.gradio_ui.profile_menu import get_user_initials, profile_menu


def build_page_blocks(
    orchestrator: OrchestratorAgent, token_provider=None,
    *, agents_config_path: str | Path | None = None,
) -> gr.Blocks:
    """Build the page bound to a specific orchestrator instance.

    Args:
        orchestrator: The orchestrator agent driving the chat.
        token_provider: Optional async callable returning a bearer token,
            forwarded to the A2A agent-card discovery (``load_agents``).
        agents_config_path: Application routing configuration, shared with
            bootstrap and the HTTP frontend.
    """
    chat = GradioChat(orchestrator)

    async def get_session_context(request: gr.Request):
        """Extract the user's session context from the JWT token."""
        token = request.cookies.get(settings.jwt_cookie_name) if request else None
        if not token:
            return {}
        try:
            # decode_user_token, not decode_token: it pins ``azp`` to the
            # browser client. The looser call accepts any token the realm
            # signed, including the service-account token the internal
            # ``/a2a/*`` and ``/tools/*`` channels carry, which would let a
            # machine credential arrive here as a signed-in person.
            payload = await decode_user_token(token)
        except KeycloakError:
            return {}

        user_id = payload.get("sub")
        if not user_id:
            return {}
        return {
            "user_id": user_id,
            "jwt_claims": payload,
        }

    agents_loaded = False
    agents_loaded_lock = asyncio.Lock()

    async def _ensure_agents_loaded():
        nonlocal agents_loaded
        if agents_loaded:
            return
        async with agents_loaded_lock:
            if agents_loaded:
                return
            loaded = await load_agents(
                token_provider=token_provider, config_path=agents_config_path,
            )
            for entry in loaded.entries:
                orchestrator.register_agent(
                    name=entry.name,
                    a2a_url=entry.a2a_url,
                    description=entry.description,
                    skills=[
                        {"name": s.name, "description": s.description}
                        for s in entry.skills
                    ],
                )
            # Only latch once discovery actually found agents: a transient IdP or
            # network hiccup at startup returns no entries (the loader swallows
            # per-agent errors), and latching then would disable routing until a
            # manual restart. Leave it False so the next page load retries.
            agents_loaded = bool(loaded.entries)

    def _skills_markdown(agent: dict) -> str:
        """Bullet list of an agent's skills (``name — description``)."""
        skills = agent.get("skills") or []
        if not skills:
            return "_No skills advertised._"
        lines = []
        for skill in skills:
            s_name = skill.get("name", "")
            s_desc = skill.get("description", "")
            lines.append(f"- `{s_name}` — {s_desc}" if s_desc else f"- `{s_name}`")
        return "\n".join(lines)

    async def _load_agents_info() -> list[dict]:
        """Discover specialists, then return the orchestrator + specialists info.

        The orchestrator (router / entry point) comes first, then each
        discovered A2A specialist. Each entry is tagged with a display label.
        """
        await _ensure_agents_loaded()
        info = [{**orchestrator.get_self_info(), "label": "router"}]
        info += [{**agent, "label": "A2A"} for agent in orchestrator.get_agents_info()]
        return info

    with gr.Blocks(
        title=settings.brand_name,
    ) as page:
        # None when authentication is off: there is no user to show.
        profile_button = profile_menu()
        session_state = gr.State({})

        if profile_button is not None:
            page.load(
                fn=get_user_initials,
                inputs=None,
                outputs=profile_button,
                show_progress=False,
            )
        session_ready = page.load(
            fn=get_session_context,
            inputs=None,
            outputs=session_state,
            show_progress=False,
        )

        gr.HTML(value=f'<div id="brand-logo">{settings.brand_name}</div>')

        chatbot = gr.Chatbot(
            buttons=["copy"],
            height=0,
            elem_id="chatbot",
            editable="user",
            show_label=False,
            autoscroll=True,
            sanitize_html=False,
            allow_file_downloads=False,
        )

        # Chained after the session load, so the identity is known: the
        # chatbot opens on the user's stored conversation instead of an empty
        # box it would then persist over the top of it.
        session_ready.then(
            fn=chat.restore_history,
            inputs=session_state,
            outputs=chatbot,
            show_progress=False,
        )

        chat_input = gr.MultimodalTextbox(
            elem_id="chat-input",
            stop_btn=False,
            autofocus=True,
            submit_btn=True,
            show_label=False,
            sources=["upload"],
            file_types=[".png", ".jpg", ".jpeg"],
            placeholder=f"Ask {settings.brand_name}...",
        )

        # Agents & Skills: one collapsible section per agent; expand an agent
        # to reveal its skills (the list can be long, so each stays folded).
        agents_state = gr.State([])
        with gr.Accordion("Agents & Skills", open=False, elem_id="debug-accordion"):

            @gr.render(inputs=agents_state)
            def _render_agents(agents):
                if not agents:
                    gr.Markdown("_No agents discovered yet._")
                    return
                for agent in agents:
                    header = f"{agent['name']} · {agent['label']}"
                    with gr.Accordion(header, open=False):
                        description = agent.get("description") or ""
                        gr.Markdown(f"_{description}_\n\n{_skills_markdown(agent)}")

        # Discover A2A specialists and populate the accordion in one shot.
        page.load(
            fn=_load_agents_info,
            inputs=None,
            outputs=agents_state,
            show_progress=False,
        )

        # Every way a turn starts (submit, edit, retry) runs ``chat.chat`` as
        # its own queued event. The stop button cancels *those events for this
        # session* — Gradio scopes ``cancels`` by session — so one user's stop
        # never touches another user's turn, and the cancellation unwinds the
        # whole turn (LLM streams, A2A streams) instead of only the
        # orchestrator's own generation.
        turn_events = []
        for trigger, prepare, inputs, outputs in (
            (
                chat_input.submit,
                chat.submit,
                [chat_input, chatbot, session_state],
                [chat_input, chatbot, chatbot],
            ),
            (chatbot.edit, chat.handle_edit, [chatbot, session_state], [chat_input, chatbot]),
            (chatbot.retry, chat.handle_retry, [chatbot, session_state], [chat_input, chatbot]),
        ):
            turn = trigger(
                fn=prepare,
                inputs=inputs,
                outputs=outputs,
                api_name=False,
                show_progress="hidden",
            ).then(
                fn=chat.chat,
                inputs=[chatbot, session_state],
                outputs=[chatbot],
                api_name=False,
                show_progress="hidden",
            )
            turn.then(
                fn=chat.enable_input,
                inputs=None,
                outputs=[chat_input],
                api_name=False,
                show_progress="hidden",
            )
            turn_events.append(turn)

        chat_input.stop(
            fn=chat.stopped,
            inputs=[chatbot, session_state],
            outputs=[chatbot],
            cancels=turn_events,
            api_name=False,
            show_progress="hidden",
        ).then(
            fn=chat.enable_input,
            inputs=None,
            outputs=[chat_input],
            api_name=False,
            show_progress="hidden",
        )

        chatbot.clear(
            fn=chat.handle_clear,
            inputs=[session_state],
            outputs=[chatbot],
            show_progress="hidden",
            api_name=False,
        )

    return page

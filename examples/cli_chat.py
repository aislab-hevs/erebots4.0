"""Minimal CLI frontend for Erebots4.0 — proof that the core is frontend-agnostic.

Drives the headless :meth:`OrchestratorAgent.run` event stream straight to the
terminal: no Gradio, no web server, no UI types. The exact same orchestrator
powers the Gradio app; this client just renders ``AgentEvent`` differently.

It goes through the same :class:`~erebots.runtime.conversation.ConversationService`
as the HTTP adapter, so identity, history and persistence mean the same thing
here as they do there. It keeps no transcript of its own: each turn the service
loads the user's stored history, and the turn is persisted at the end — which is
also what lets a later session pick the conversation up.

With ``--user`` the turns belong to that identity (its profile, consents and
history); without it they run anonymously, as before, and nothing is persisted.
``--user`` *selects* an identity, it does not authenticate one: there is no
credential and no token check here, so it is a development convenience, not an
access-control boundary. The authenticated path is the HTTP one, where the
session resolver derives the identity from a Keycloak-issued token.
The orchestrator runs here, in the terminal; the specialists it routes to are
the ones a running deployment serves at ``/a2a/*``, reached over A2A with the
same service-account token the server uses.

Run (with an OpenAI-compatible LLM configured via the usual env vars)::

    uv run python examples/cli_chat.py
    uv run python examples/cli_chat.py --user alice
    echo "how many pounds is 70 kg?" | uv run python examples/cli_chat.py
"""

from __future__ import annotations

import argparse
import asyncio

from erebots.agents.loader import load_agents
from erebots.agents.orchestrator import OrchestratorAgent
from erebots.config import settings
from erebots.persistence import Base


async def chat_once(
    orchestrator: OrchestratorAgent,
    message: str,
    pending: dict | None = None,
    session: dict | None = None,
) -> tuple[dict | None, str]:
    """Stream one turn's events to stdout.

    Returns the handle of a specialist's unanswered question — so the next turn
    resumes that task instead of routing — together with the answer as rendered,
    which the caller persists.

    ``session`` is the identity of the turn (``{"user_id": ...}``); the
    conversation service turns it into the turn's context, loading this user's
    stored profile and history.
    """
    print("bot> ", end="", flush=True)
    context = dict(await orchestrator.conversation.build_request_context(session) or {})
    if pending:
        context["pending"] = pending
    pending = None
    answer = ""
    answer_streamed = False
    async for event in orchestrator.run(message, context):
        if event.type == "routing":
            target = event.data.get("target")
            if target and target != "none":
                print(f"[routing → {target}] ", end="", flush=True)
        elif event.type == "token":
            # Contributions of a fan-out turn would interleave on one line; the
            # composed answer that follows already carries their content.
            if event.channel != "side":
                print(event.text, end="", flush=True)
                answer += event.text
                answer_streamed = answer_streamed or bool(event.text)
        elif event.type == "input_required":
            if event.data.get("task_id"):
                pending = {"agent": event.agent, "task_id": event.data["task_id"]}
        elif event.type == "final" and event.channel != "side":
            # Final repeats the accumulated answer. Legacy/final-only peers
            # still need rendering, but streamed answers are already visible.
            if not answer_streamed:
                print(event.text, end="", flush=True)
                answer += event.text
        elif event.type == "error":
            print(f"\n[error] {event.text}", end="", flush=True)
    print()
    return pending, answer


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user",
        default="",
        help="run the turns as this user id (profile, consents and history) — "
        "selects an identity, does not authenticate it; "
        "omit to run anonymously and persist nothing",
    )
    args = parser.parse_args()

    # The example app's composition root with no local specialists: this
    # process hosts the orchestrator and the store, and the specialists live
    # in the deployment it discovers below. It also hands the orchestrator the
    # service-account token, without which /a2a/* answers 401.
    from bootstrap import build_agents

    agents = build_agents(settings, specialists={})
    orchestrator = agents.orchestrator

    # This process owns its schema: the server does this in its lifespan, and
    # without it the very first identified turn finds no tables.
    async with agents.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    loaded = await load_agents(
        token_provider=(
            agents.internal_token_provider.get_token
            if agents.internal_token_provider is not None
            else None
        ),
        config_path=settings.agents_config_path,
    )
    for entry in loaded.entries:
        orchestrator.register_agent(
            name=entry.name,
            description=entry.description,
            a2a_url=entry.a2a_url,
            skills=[{"name": s.name, "description": s.description} for s in entry.skills],
        )
    if loaded.failed:
        print(f"[warning] no agent card at: {', '.join(loaded.failed)} — routing without them")

    session = {"user_id": args.user} if args.user else None
    who = args.user or "anonymous"
    print(f"Erebots4.0 CLI — headless orchestrator, {who} (Ctrl-D to quit)")

    pending: dict | None = None
    try:
        while True:
            try:
                message = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            pending, answer = await chat_once(orchestrator, message, pending, session)
            if session is not None:
                # Appended, not written back: this client never holds the
                # transcript (the turn reads it from the store, already capped
                # for the prompt), so persisting "its" history would delete
                # every turn older than that cap.
                await orchestrator.conversation.append_turn(
                    session,
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": answer},
                    ],
                )
    finally:
        await orchestrator.a2a_client.aclose()
        await agents.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

"""Main orchestrator agent.

Manages the dispatching of user requests to specialized agents
via the A2A (Agent-to-Agent) protocol. A request may need more than one
specialist: they then run in parallel and their answers are composed into a
single reply. When no specialized agent matches, the orchestrator generates a
response directly with its own LLM.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator

from erebots.agents.a2a.client import A2AClient
from erebots.agents.base import BaseAgent
from erebots.agents.models import (
    AgentResponse,
    InputRequired,
    TraceStep,
    elapsed_ms_since,
)
from erebots.agents.orchestrator.prompts import (
    build_orchestrator_prompt,
    build_routing_prompt,
    build_synthesis_prompt,
)
from erebots.agents.orchestrator.think_parser import ThinkSplitter
from erebots.agents.skill import Skill
from erebots.config import settings
from erebots.runtime.conversation import ConversationService
from erebots.runtime.events import AgentEvent

logger = logging.getLogger(__name__)

# Identifier-shaped tokens, used to pick registered agent names out of whatever
# prose the routing model wraps its answer in.
_AGENT_TOKEN_RE = re.compile(r"[a-z0-9_-]+")

# Recent turns shown to the routing model so follow-ups ("and for dinner?")
# keep their topic. Deliberately shorter than the specialists' history budget:
# routing needs the thread, not the transcript.
_ROUTING_HISTORY_MESSAGES = 4

# Sentinel marking "this specialist's stream ended" on the fan-out queue.
_STREAM_DONE = object()


def _pending_of(context: dict | None) -> tuple[str, str] | None:
    """The ``(agent, task_id)`` of a pending question, if this turn answers one.

    The handle comes from the frontend (echoed back from an ``input_required``
    event), so it is untrusted input: anything but two non-empty strings is
    treated as "no pending question" rather than an error.
    """
    pending = (context or {}).get("pending")
    if not isinstance(pending, dict):
        return None
    agent, task_id = pending.get("agent"), pending.get("task_id")
    if isinstance(agent, str) and agent and isinstance(task_id, str) and task_id:
        return agent, task_id
    return None


class OrchestratorAgent(BaseAgent):
    """Main agent that dispatches requests to specialized agents via A2A.

    The orchestrator analyzes the user's message, determines which
    specialized agent is best suited to handle it, and forwards the
    request via the A2A protocol. If no agent matches, it responds
    directly with its own LLM.
    """

    def __init__(
        self,
        api_key: str,
        api_base_url: str,
        model: str,
        user_context_store=None,
        thinking: bool = False,
        a2a_client: A2AClient | None = None,
    ):
        """Initialize the orchestrator agent.

        Args:
            api_key: API key for the LLM service.
            api_base_url: Base URL of the LLM API.
            model: Name of the LLM model to use.
            user_context_store: Optional :class:`~erebots.persistence.UserContextStore` used to
                read the user's profile and persist chat history.
            thinking: If True, enables extended reasoning mode.
            a2a_client: Talk to specialists through this client instead of a
                fresh one. The seam for an in-process round trip: ``A2AClient``
                takes an ``httpx`` transport, so a caller can point the
                orchestrator at an ASGI app without a listening socket.

                Ownership follows construction. A client passed in belongs to
                the caller and :meth:`aclose` leaves it alone; one built here is
                closed there. Nothing else changes — the pool, the token
                provider and the registered agents behave identically.
        """
        super().__init__(
            agent_name="orchestrator",
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            system_prompt="",
            thinking=thinking,
        )
        # ``is None``, not truthiness: a client with an empty pool is falsy on
        # no sensible reading, but must not be silently replaced either way.
        self.a2a_client = A2AClient() if a2a_client is None else a2a_client
        self._owns_a2a_client = a2a_client is None
        self.user_context_store = user_context_store
        self.conversation = ConversationService(user_context_store)
        self._agent_descriptions: dict[str, str] = {}
        # name -> list of {"name", "description"} skill dicts
        self._agent_skills: dict[str, list[dict]] = {}
        self._routing_prompt = ""
        self._fallback_prompt = ""
        self._synthesis_prompt = build_synthesis_prompt()
        self.setup_skills()
        self._refresh_prompts()

    async def aclose(self) -> None:
        """Release what this orchestrator owns (call on application shutdown).

        Only the A2A client it built itself. One handed to the constructor stays
        open: the caller may still be using it, and closing somebody else's
        connection pool from a shutdown hook is the kind of thing that shows up
        as an unexplained ``RuntimeError`` two layers away.
        """
        if self._owns_a2a_client:
            await self.a2a_client.aclose()

    def setup_skills(self):
        """Register the orchestrator's skills.

        Configures the ``route_request`` skill that allows routing
        user requests to the appropriate specialized agents.
        """
        self.skills.register(
            Skill(
                name="route_request",
                description=(
                    "Route a user request to the specialist agents that can handle it, "
                    "and compose their answers into one reply"
                ),
            )
        )

    def register_agent(
        self,
        name: str,
        a2a_url: str,
        description: str,
        skills: list[dict] | None = None,
    ):
        """Register a specialized agent reachable via A2A.

        Args:
            name: Identifying name of the agent.
            a2a_url: URL of the agent's A2A endpoint.
            description: Short, stable identity of the agent.
            skills: Optional list of ``{"name", "description"}`` dicts
                from the agent card; these provide the granular routing
                signal so the agent ``description`` can stay a one-liner.
        """
        self.a2a_client.register_agent(name, a2a_url)
        self._agent_descriptions[name] = description
        self._agent_skills[name] = skills or []
        self._refresh_prompts()

    def get_agents_info(self) -> list[dict]:
        """Return information about registered agents.

        Returns:
            List of dictionaries with ``name``, ``description`` and
            ``skills`` (list of ``{"name", "description"}``) for each
            registered agent. Used by the debug UI.
        """
        return [
            {
                "name": name,
                "description": desc,
                "skills": self._agent_skills.get(name, []),
            }
            for name, desc in self._agent_descriptions.items()
        ]

    def get_self_info(self) -> dict:
        """Return the orchestrator's own identity and skills for the debug UI.

        Mirrors the shape of :meth:`get_agents_info` entries so the UI can
        render the router alongside the discovered specialists.
        """
        return {
            "name": self.name,
            "description": (
                "Router: analyzes each message and dispatches it to the best "
                "specialist, or answers directly when none fits."
            ),
            "skills": [
                {"name": s.name, "description": s.description} for s in self.skills.list_skills()
            ],
        }

    def _agent_block(self, name: str, desc: str) -> str:
        """Render one agent for the routing/fallback prompts.

        With ``routing_include_skills`` off, only ``name: description`` is
        rendered — the lean form for small routing models. The agent card
        still advertises every skill either way.
        """
        lines = [f"- {name}: {desc}"]
        if settings.routing_include_skills:
            for skill in self._agent_skills.get(name, []):
                s_name = skill.get("name", "")
                s_desc = skill.get("description", "")
                if not s_name:
                    continue
                lines.append(f"    - skill `{s_name}`: {s_desc}")
        return "\n".join(lines)

    def _refresh_prompts(self):
        """Rebuild system prompts after changes to registered agents.

        Both the routing prompt and the fallback prompt list each agent
        with its skills, so capability-level matching does not depend on
        hand-tuning the agent description.
        """
        descriptions = "\n".join(
            self._agent_block(name, desc) for name, desc in self._agent_descriptions.items()
        )
        self._routing_prompt = build_routing_prompt(
            descriptions, max_specialists=max(1, settings.orchestrator_max_specialists)
        )
        self._fallback_prompt = build_orchestrator_prompt(agent_descriptions=descriptions)
        self.generative.system_prompt = self._fallback_prompt

    def _extract_routing_text(self, content: str | list[dict]) -> str:
        """Extract routing-relevant text from message content.

        Handles both simple text content and multimodal content
        (list of text/image parts). If only an image is present
        without text, returns a generic fallback text.

        Args:
            content: Message content, as a string or list of
                multimodal parts.

        Returns:
            Extracted text for the routing decision.
        """
        if isinstance(content, str):
            return content.strip()

        text_parts: list[str] = []
        has_image = False
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and part.get("text"):
                text_parts.append(part["text"].strip())
            elif part.get("type") == "image_url" and part.get("image_url", {}).get("url"):
                has_image = True

        joined = "\n".join(part for part in text_parts if part).strip()
        if joined:
            return joined
        if has_image:
            return "Analyze the provided image."
        return ""

    def _reparent_specialist_trace(
        self,
        specialist_trace: list[TraceStep],
        parent_id: str,
    ) -> None:
        """Attach a specialist's root steps under ``parent_id``.

        Steps already pointing to another node inside the specialist's
        own trace keep their internal edge; only those with
        ``parent_id is None`` are re-parented to the orchestrator's
        ``a2a_call`` step.
        """
        internal_ids = {step.step_id for step in specialist_trace}
        for step in specialist_trace:
            if step.parent_id is None or step.parent_id not in internal_ids:
                step.parent_id = parent_id

    def _routing_history(self, context: dict | None) -> list[dict]:
        """Return the last few turns as plain text, for the routing decision.

        Multimodal turns are flattened to their text parts: an image tells the
        router nothing it can act on and would only bloat the prompt.
        """
        history = (context or {}).get("chat_history") or []
        messages: list[dict] = []
        for turn in history[-_ROUTING_HISTORY_MESSAGES:]:
            text = self._extract_routing_text(turn.get("content", ""))
            if text:
                messages.append({"role": turn.get("role", "user"), "content": text})
        return messages

    def _routing_content(self, message: str | list[dict]) -> str | list[dict]:
        """Build what the router actually sees for this turn.

        Text by default. When the message carries images and the deployment
        declares its models multimodal, the image parts are passed to the
        routing call itself, so the router decides on what the picture *shows*
        rather than on a placeholder string. This costs no extra LLM call — the
        routing call already exists — only a larger routing prompt.

        With ``multimodal_enabled`` off the parts are stripped downstream in
        :meth:`GenerativeLayer._build_kwargs` anyway, so returning them here is
        harmless and the router falls back to the text path on its own.
        """
        text = self._extract_routing_text(message)
        if not settings.multimodal_enabled or not isinstance(message, list):
            return text

        images = [
            part
            for part in message
            if isinstance(part, dict)
            and part.get("type") == "image_url"
            and part.get("image_url", {}).get("url")
        ]
        if not images:
            return text
        return [{"type": "text", "text": text or "Route this image."}, *images]

    def _parse_plan(self, response: str) -> list[str]:
        """Extract the registered agent names the router asked for.

        Robust to the shapes small routing models actually produce: reasoning
        in ``<think>`` blocks, a trailing period, a "Agent: x" preamble, or a
        comma-separated list. Reasoning is stripped first (it often *mentions*
        the agents it then rejects) and only the last non-empty line is read,
        so a model that thinks out loud before answering still routes correctly.
        Unknown names are dropped; a reply naming nothing registered is logged
        instead of silently degrading to the fallback.
        """
        splitter = ThinkSplitter()
        visible = "".join(
            piece
            for kind, piece in [*splitter.feed(response), *splitter.flush()]
            if kind != "think"
        )
        lines = [line for line in visible.splitlines() if line.strip()]
        if not lines:
            return []

        candidates = _AGENT_TOKEN_RE.findall(lines[-1].lower())
        plan: list[str] = []
        for name in candidates:
            if name in self._agent_descriptions and name not in plan:
                plan.append(name)
        if not plan and "none" not in candidates:
            logger.warning(
                "Router reply named no registered agent (falling back): %r", visible[-200:]
            )
        return plan[: max(1, settings.orchestrator_max_specialists)]

    async def _plan(self, message: str | list[dict], context: dict | None = None) -> list[str]:
        """Decide which specialists should handle this turn.

        Args:
            message: User message (string or multimodal content).
            context: Request context; its ``chat_history`` gives the router the
                thread needed to resolve follow-up messages.

        Returns:
            Registered agent names, in the order the router named them. Empty
            when no specialist fits and the orchestrator should answer itself.
        """
        content = self._routing_content(message)
        if not content:
            return []

        messages = [*self._routing_history(context), {"role": "user", "content": content}]
        response = await self.generative.chat(
            messages,
            system_prompt=self._routing_prompt,
        )
        return self._parse_plan(response)

    async def route_request(
        self, message: str | list[dict], context: dict | None = None
    ) -> list[str]:
        """Plan the specialists for a request (exposed as a skill)."""
        return await self._plan(message, context)

    def _build_fallback_messages(
        self, message: str | list[dict], context: dict | None
    ) -> list[dict]:
        """Assemble the LLM message list for a direct orchestrator answer.

        Shape: ``[user-context?] + chat_history + [user turn]`` — shared by the
        headless ``run()`` and the A2A ``process()`` paths.
        """
        messages: list[dict] = []
        if context and context.get("user_context"):
            messages.append(
                {"role": "system", "content": f"User context:\n{context['user_context']}"}
            )
        if context and context.get("chat_history"):
            messages.extend(context["chat_history"])
        messages.append({"role": "user", "content": message})
        return messages

    async def run(
        self, message: str | list[dict], context: dict | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Headless orchestration of one user turn, as a stream of events.

        Fully async and UI-neutral — the single basis every frontend (Gradio,
        CLI, REST/SSE) consumes. Dispatches to one specialist over A2A, fans out
        to several when the request spans domains, or streams the orchestrator's
        own answer when nothing matches. ``context`` is the already-built request
        context (``user_id``/``user_context``/``chat_history``); persistence is
        the caller's concern.

        ``context["pending"]`` — ``{"agent", "task_id"}``, echoed back by the
        frontend from an earlier ``input_required`` event — marks this turn as
        the answer to a specialist's question: routing is skipped and the same
        agent resumes the same task. A stale handle (task expired, server
        restarted) — or one belonging to a different user than this turn's —
        silently falls back to normal routing.

        Yields:
            The turn's events: ``routing``/``trace`` for observability,
            ``token``/``reasoning`` as the answer is produced, ``final`` once,
            and ``input_required`` when a specialist stops to ask the user
            something. Fan-out contributions carry ``channel="side"``.
        """
        pending = _pending_of(context)
        if pending is not None:
            target, task_id = pending
            user_id = (context or {}).get("user_id")
            if await self.a2a_client.can_resume(target, task_id, user_id=user_id):
                # The user is answering, not asking: re-routing "38" through
                # the LLM router could land it on a different specialist than
                # the one that asked. Resuming bypasses routing by design.
                yield AgentEvent(
                    "routing",
                    agent=self.name,
                    data={"plan": [target], "target": target, "resumed": True},
                )
                yield AgentEvent(
                    "trace",
                    trace_step=TraceStep(
                        agent=self.name,
                        action="resume",
                        detail=f"Answer routed back to {target}'s pending question",
                    ),
                )
                async for event in self._run_single(target, message, context, task_id=task_id):
                    yield event
                return

        routing_started = time.monotonic()
        try:
            plan = await self.route_request(message, context)
        except Exception as exc:
            # The first LLM call of every turn: if it dies the stream must still
            # end with an explicit error event, not a dropped connection.
            logger.exception("Routing failed")
            yield AgentEvent(
                "error",
                text=f"Routing failed: {exc}",
                agent=self.name,
                trace_step=TraceStep(agent=self.name, action="error", detail=str(exc)),
            )
            return
        routing_ms = elapsed_ms_since(routing_started)
        yield AgentEvent(
            "routing",
            agent=self.name,
            # ``plan`` is the machine-readable decision; ``target`` renders the
            # same thing as a label, so single-target consumers are unaffected.
            data={"plan": plan, "target": ", ".join(plan) if plan else "none"},
        )
        # The routing call is a full LLM round-trip on every single turn; without
        # its own step its cost is invisible in the trace.
        yield AgentEvent(
            "trace",
            trace_step=TraceStep(
                agent=self.name,
                action="routing",
                detail=f"Selected: {', '.join(plan) if plan else 'no specialist'}",
                elapsed_ms=routing_ms,
                prompt_tokens=self.generative.last_prompt_tokens or None,
            ),
        )

        if len(plan) == 1:
            async for event in self._run_single(plan[0], message, context):
                yield event
        elif plan:
            async for event in self._run_fanout(plan, message, context):
                yield event
        else:
            async for event in self._run_fallback(message, context):
                yield event

    async def _run_single(
        self,
        target: str,
        message: str | list[dict],
        context: dict | None,
        task_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream one specialist's answer straight through to the user.

        ``task_id`` resumes that parked task instead of opening a new one.

        Args:
            target: The specialist to dispatch to.
            message: The user's turn.
            context: The request context forwarded to the specialist.
            task_id: Parked task to resume, if this turn answers a question.

        Yields:
            The specialist's events on the ``answer`` channel, ending in
            ``final`` or ``input_required``.
        """
        a2a_step = TraceStep(agent=self.name, action="a2a_call", detail=f"A2A request to {target}")
        yield AgentEvent("trace", trace_step=a2a_step)
        started = time.monotonic()
        splitter = ThinkSplitter()
        text = ""
        pending: InputRequired | None = None
        try:
            async for chunk in self.a2a_client.send_message_stream(
                target, message, context, task_id=task_id
            ):
                if isinstance(chunk, InputRequired):
                    # Not an answer: the specialist is waiting. The question has
                    # already streamed as text, so this only carries the handle
                    # a later turn needs to resume the same task.
                    pending = chunk
                    continue
                if isinstance(chunk, TraceStep):
                    # Specialist trace marker: re-parent under our a2a_call.
                    chunk.parent_id = a2a_step.step_id
                    yield AgentEvent("trace", trace_step=chunk, agent=target)
                    continue
                for kind, piece in splitter.feed(chunk):
                    if kind == "think":
                        yield AgentEvent("reasoning", text=piece, agent=target)
                    else:
                        text += piece
                        yield AgentEvent("token", text=piece, agent=target)
            for kind, piece in splitter.flush():
                if kind == "think":
                    yield AgentEvent("reasoning", text=piece, agent=target)
                else:
                    text += piece
                    yield AgentEvent("token", text=piece, agent=target)
        except Exception as exc:
            logger.exception("A2A communication failed for %s", target)
            yield AgentEvent(
                "error",
                text=f"Error communicating with `{target}`: {exc}",
                agent=target,
                trace_step=TraceStep(
                    agent=target,
                    action="error",
                    detail=str(exc),
                    parent_id=a2a_step.step_id,
                ),
            )
            return
        # Re-emitted with its duration now that the call is done; consumers
        # upsert trace steps by ``step_id``.
        a2a_step.elapsed_ms = elapsed_ms_since(started)
        yield AgentEvent("trace", trace_step=a2a_step, agent=target)
        yield AgentEvent(
            "trace",
            trace_step=TraceStep(
                agent=self.name,
                action="relay",
                detail=f"Streamed response generated by {target}",
            ),
        )
        if pending is not None:
            yield AgentEvent(
                "input_required",
                text=pending.question or text,
                agent=target,
                data={"task_id": pending.task_id},
            )
        yield AgentEvent("final", text=text, agent=target)

    def _delegation_note(self, me: str, plan: list[str]) -> str:
        """Tell one specialist which parts of the request are somebody else's.

        Every specialist receives the user's message in full, so without this it
        answers the parts it merely *saw* — in the worst case claiming to have
        performed an action it has no tool for. Naming the sibling and its scope
        gives the model a reason to stay out, which a bare prohibition does not:
        the point is not that it may not help, but that the help is already
        covered and duplicating it produces a second, conflicting opinion.

        Built from the registered agent cards, so it costs no extra LLM call and
        cannot invent an assignment that was never planned.
        """
        others = [name for name in plan if name != me]
        if not others:
            return ""
        blocks = "\n".join(
            f"- `{name}`: {self._agent_descriptions.get(name, '')}" for name in others
        )
        return (
            "This request is being handled by several specialists at once, each answering "
            f"from its own domain. You are `{me}`.\n\n"
            "The following parts are already being handled, in parallel, by others. Do not "
            "address them, do not perform them, and never claim to have performed them:\n"
            f"{blocks}\n\n"
            "Answer only what belongs to your own domain, and say nothing at all about "
            "the rest — not a handoff, not an apology, not a note about your own limits. "
            "The answers are merged before the user sees them, so any such sentence "
            "reaches them as confusing noise about machinery they cannot see. Never "
            'write anything like "I\'ll leave that to the other specialist", "that '
            'part is handled elsewhere", or "I cannot help with that, it is outside '
            'my functions". Just answer your part and stop: saying nothing about the '
            "rest is correct, not an omission.\n\n"
            "And never invent the system around you. You do not know whether this "
            "product has account settings, a support desk, a dashboard, or any other "
            "way for the user to obtain what you are not answering — do not send them "
            "anywhere and do not describe how the system works. If something is not "
            "yours to answer, the correct behaviour is silence on that point."
        )

    async def _merge_specialist_streams(
        self, plan: list[str], message: str | list[dict], contexts: dict[str, dict]
    ) -> AsyncIterator[tuple[str, object]]:
        """Yield ``(agent_name, item)`` from every specialist's stream at once.

        ``item`` is a text chunk, a :class:`TraceStep`, or the exception that
        ended that specialist's stream. The specialists run concurrently and
        their chunks arrive interleaved but tagged, so a frontend can render
        each stream live in its own place. One failure never stops the others.

        ``contexts`` is per-agent: each carries the same request context plus the
        delegation note written for that specialist.

        Yields:
            ``(agent_name, item)`` pairs, interleaved across specialists.
        """
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

        async def pump(name: str) -> None:
            """Drain one specialist's stream into the shared queue."""
            try:
                async for chunk in self.a2a_client.send_message_stream(
                    name, message, contexts[name]
                ):
                    queue.put_nowait((name, chunk))
            except Exception as exc:  # reported to the consumer, not raised
                queue.put_nowait((name, exc))
            finally:
                queue.put_nowait((name, _STREAM_DONE))

        tasks = [asyncio.create_task(pump(name)) for name in plan]
        pending = len(tasks)
        try:
            while pending:
                name, item = await queue.get()
                if item is _STREAM_DONE:
                    pending -= 1
                # Forwarded too: the caller closes that specialist's trace step.
                yield name, item
        finally:
            for task in tasks:
                task.cancel()

    async def _run_fanout(
        self, plan: list[str], message: str | list[dict], context: dict | None
    ) -> AsyncIterator[AgentEvent]:
        """Run several specialists in parallel and compose one reply.

        Each specialist answers the whole request from its own angle; their
        streams are emitted on the ``side`` channel so a frontend can show them
        filling live, and the composed reply follows on the default channel.

        Yields:
            Each specialist's events on ``side``, then the composed answer on
            ``answer``. A specialist that fails is dropped from the
            composition rather than failing the turn.
        """
        plan_step = TraceStep(
            agent=self.name, action="plan", detail=f"Fan-out to {', '.join(plan)}"
        )
        yield AgentEvent("trace", trace_step=plan_step)

        call_steps: dict[str, TraceStep] = {}
        for name in plan:
            step = TraceStep(
                agent=self.name,
                action="a2a_call",
                detail=f"A2A request to {name}",
                parent_id=plan_step.step_id,
            )
            call_steps[name] = step
            yield AgentEvent("trace", trace_step=step)

        splitters = {name: ThinkSplitter() for name in plan}
        answers = dict.fromkeys(plan, "")
        failures: dict[str, str] = {}
        started = dict.fromkeys(plan, time.monotonic())

        # One context per specialist: same request data, its own delegation note.
        contexts = {
            name: {**(context or {}), "delegation": self._delegation_note(name, plan)}
            for name in plan
        }

        async for name, item in self._merge_specialist_streams(plan, message, contexts):
            if item is _STREAM_DONE:
                call_steps[name].elapsed_ms = elapsed_ms_since(started[name])
                yield AgentEvent("trace", trace_step=call_steps[name], agent=name)
                continue
            if isinstance(item, BaseException):
                failures[name] = str(item)
                logger.warning("A2A communication failed for %s: %s", name, item)
                yield AgentEvent(
                    "trace",
                    agent=name,
                    trace_step=TraceStep(
                        agent=name,
                        action="error",
                        detail=str(item),
                        parent_id=call_steps[name].step_id,
                    ),
                )
                continue
            if isinstance(item, InputRequired):
                # One contributor is waiting; the others keep going and their
                # answers still get composed. Without this the marker would be
                # stringified straight into the reply.
                yield AgentEvent(
                    "input_required",
                    text=item.question,
                    agent=name,
                    data={"task_id": item.task_id},
                    channel="side",
                )
                continue
            if isinstance(item, TraceStep):
                item.parent_id = call_steps[name].step_id
                yield AgentEvent("trace", trace_step=item, agent=name)
                continue
            for kind, piece in splitters[name].feed(str(item)):
                if kind == "think":
                    yield AgentEvent("reasoning", text=piece, agent=name, channel="side")
                else:
                    answers[name] += piece
                    yield AgentEvent("token", text=piece, agent=name, channel="side")

        for name, splitter in splitters.items():
            for kind, piece in splitter.flush():
                if kind == "think":
                    yield AgentEvent("reasoning", text=piece, agent=name, channel="side")
                else:
                    answers[name] += piece
                    yield AgentEvent("token", text=piece, agent=name, channel="side")

        contributions = {name: text.strip() for name, text in answers.items() if text.strip()}
        if not contributions:
            detail = (
                "; ".join(f"{name}: {err}" for name, err in failures.items())
                or "no specialist returned any content"
            )
            yield AgentEvent(
                "error",
                text=f"No specialist produced an answer ({detail}).",
                agent=self.name,
                trace_step=TraceStep(agent=self.name, action="error", detail=detail),
            )
            return

        if settings.multi_agent_composition == "sections":
            async for event in self._compose_sections(contributions):
                yield event
        else:
            async for event in self._compose_synthesis(message, contributions):
                yield event

    async def _compose_sections(self, contributions: dict[str, str]) -> AsyncIterator[AgentEvent]:
        """Concatenate the specialist answers under a per-agent heading.

        No extra LLM call, so nothing can distort what a specialist said — the
        baseline to fall back on when the synthesis is under suspicion.

        Yields:
            One ``token`` event with the whole text, a ``trace`` step, and
            ``final``.
        """
        started = time.monotonic()
        text = "\n\n".join(f"## {name}\n\n{body}" for name, body in contributions.items())
        yield AgentEvent("token", text=text, agent=self.name)
        yield AgentEvent(
            "trace",
            trace_step=TraceStep(
                agent=self.name,
                action="compose",
                detail=f"Concatenated {len(contributions)} specialist answers",
                elapsed_ms=elapsed_ms_since(started),
            ),
        )
        yield AgentEvent("final", text=text, agent=self.name)

    async def _compose_synthesis(
        self, message: str | list[dict], contributions: dict[str, str]
    ) -> AsyncIterator[AgentEvent]:
        """Merge the specialist answers into one reply, streamed to the user.

        Yields:
            ``token``/``reasoning`` events as the synthesis is produced, a
            ``trace`` step, then ``final``.
        """
        block = "\n\n".join(f"### {name}\n{body}" for name, body in contributions.items())
        prompt = (
            f"User request:\n{self._extract_routing_text(message)}\n\n"
            f"Specialist answers:\n\n{block}"
        )
        started = time.monotonic()
        splitter = ThinkSplitter()
        text = ""
        try:
            async for chunk in self.generative.chat_stream(
                [{"role": "user", "content": prompt}],
                system_prompt=self._synthesis_prompt,
            ):
                for kind, piece in splitter.feed(chunk):
                    if kind == "think":
                        yield AgentEvent("reasoning", text=piece, agent=self.name)
                    else:
                        text += piece
                        yield AgentEvent("token", text=piece, agent=self.name)
            for kind, piece in splitter.flush():
                if kind == "think":
                    yield AgentEvent("reasoning", text=piece, agent=self.name)
                else:
                    text += piece
                    yield AgentEvent("token", text=piece, agent=self.name)
        except Exception:
            # The specialists already did the work: fall back to concatenation
            # rather than losing their answers to a failed merge.
            logger.exception("Synthesis failed; falling back to concatenated sections")
            async for event in self._compose_sections(contributions):
                yield event
            return

        yield AgentEvent(
            "trace",
            trace_step=TraceStep(
                agent=self.name,
                action="synthesis",
                detail=f"Merged answers from {', '.join(contributions)}",
                elapsed_ms=elapsed_ms_since(started),
                tokens=self.generative.last_completion_tokens or None,
                prompt_tokens=self.generative.last_prompt_tokens or None,
            ),
        )
        yield AgentEvent("final", text=text, agent=self.name)

    async def _run_fallback(
        self, message: str | list[dict], context: dict | None
    ) -> AsyncIterator[AgentEvent]:
        """Stream the orchestrator's own answer when no specialist matched.

        Yields:
            ``token``/``reasoning`` events, a ``trace`` step, then ``final``.
        """
        started = time.monotonic()
        messages = self._build_fallback_messages(message, context)
        splitter = ThinkSplitter()
        response_text = ""
        try:
            async for chunk in self.generative.chat_stream(messages):
                for kind, piece in splitter.feed(chunk):
                    if kind == "think":
                        yield AgentEvent("reasoning", text=piece, agent=self.name)
                    else:
                        response_text += piece
                        yield AgentEvent("token", text=piece, agent=self.name)
            for kind, piece in splitter.flush():
                if kind == "think":
                    yield AgentEvent("reasoning", text=piece, agent=self.name)
                else:
                    response_text += piece
                    yield AgentEvent("token", text=piece, agent=self.name)
        except Exception as exc:
            logger.exception("Orchestrator fallback response failed")
            yield AgentEvent(
                "error",
                text=f"Error generating orchestrator response: {exc}",
                agent=self.name,
                trace_step=TraceStep(agent=self.name, action="error", detail=str(exc)),
            )
            return

        yield AgentEvent(
            "trace",
            trace_step=TraceStep(
                agent=self.name,
                action="response",
                detail="Generated orchestrator response",
                elapsed_ms=elapsed_ms_since(started),
                tokens=self.generative.last_completion_tokens or None,
                prompt_tokens=self.generative.last_prompt_tokens or None,
            ),
        )
        yield AgentEvent("final", text=response_text, agent=self.name)

    async def process(
        self, message: str | list[dict], context: dict | None = None
    ) -> AgentResponse:
        """Non-streaming variant of :meth:`run` (the BaseAgent contract).

        Collects the event stream into a single :class:`AgentResponse`, so there
        is one orchestration path (``run``) and this just consumes it.
        """
        tokens: list[str] = []
        trace: list[TraceStep] = []
        final_text = ""
        agent_name = self.name
        async for event in self.run(message, context):
            # Side-channel chunks are a fan-out turn's raw specialist streams;
            # the composed answer arrives on the default channel, so collecting
            # both would duplicate every contribution in the content.
            if event.type == "token" and event.channel != "side":
                tokens.append(event.text)
            elif event.type == "trace" and event.trace_step:
                trace.append(event.trace_step)
            elif event.type == "final":
                final_text, agent_name = event.text, event.agent
            elif event.type == "error":
                final_text = event.text
        return AgentResponse(
            content="".join(tokens) or final_text,
            agent_name=agent_name,
            trace=trace,
        )

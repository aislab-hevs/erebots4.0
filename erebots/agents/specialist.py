"""Base class for MCP-backed specialist agents.

A *specialist* is a domain agent that answers via an LLM augmented with MCP
tools. This base class owns the two pieces every specialist needs and that
were previously copy-pasted into each one:

* lazy MCP tool **discovery** across one or more servers (double-checked
  locking, resilient to a single server failing), and
* the native **tool-calling loop** (LLM → tool → observe, bounded by
  ``settings.tool_loop_max_iterations``) with trace emission.

Subclasses provide their system prompt, description and skills, and may
override :meth:`prepare_tool_args` to enforce server-side arguments: an agent
whose tools act on a user record pins ``user_id`` to the trusted request
identity there, so a model cannot redirect the lookup by supplying its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from erebots.agents.base import BaseAgent
from erebots.agents.mcp.client import MCPClient, MCPToolError
from erebots.agents.models import (
    AgentResponse,
    InputRequired,
    TraceStep,
    elapsed_ms_since,
)
from erebots.config import settings

logger = logging.getLogger(__name__)

# Offered to every specialist alongside its MCP tools. Calling it is how an
# agent says "I cannot answer without this" in a way the protocol can carry:
# the turn ends, the task parks in input-required, and the answer arrives as a
# later message on the same task instead of as an unrelated new turn.
ASK_USER_TOOL = "ask_user"
_ASK_USER_SPEC = {
    "type": "function",
    "function": {
        "name": ASK_USER_TOOL,
        "description": (
            "Ask the user for information that you need and cannot obtain from "
            "your tools or from the context you were given. Use it only when the "
            "request genuinely cannot be answered without that information — not "
            "to confirm, not to make conversation. Your turn ends here; the "
            "answer reaches you as a later message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to ask, phrased for the user, in their language.",
                }
            },
            "required": ["question"],
        },
    },
}


def _tool_args(raw: str | None) -> dict:
    """Parse a tool call's ``arguments`` into a dict, whatever the model sent.

    Models occasionally emit JSON that parses fine but is not an object —
    ``"null"``, a bare string, a list. Catching only :class:`json.JSONDecodeError`
    lets those through and the caller then explodes on ``.get``, so the type is
    checked too: a malformed call must degrade to "no arguments", never to a
    failed task.
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class BaseSpecialist(BaseAgent):
    """LLM + MCP-tools specialist with shared discovery and tool-loop."""

    def __init__(
        self,
        *,
        agent_name: str,
        api_key: str,
        api_base_url: str,
        model: str,
        system_prompt: str,
        description: str = "",
        mcp_server_urls: list[str] | None = None,
        thinking: bool = False,
    ):
        """Initialize the specialist and the MCP client its tools come from.

        Args:
            agent_name: Unique identifier name of the agent.
            api_key: API key for the LLM service.
            api_base_url: Base URL of the OpenAI-compatible API endpoint.
            model: LLM model identifier to use.
            system_prompt: System prompt to prepend to conversations.
            description: Short description the orchestrator's router reads to
                discriminate this specialist from the others.
            mcp_server_urls: Servers to discover tools from on first use.
                Duplicates and blanks are dropped, order preserved.
            thinking: If True, enables the model's explicit reasoning mode.
        """
        super().__init__(
            agent_name=agent_name,
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            system_prompt=system_prompt,
            description=description,
            thinking=thinking,
        )
        # Deduplicate the MCP URLs while preserving order; drop blanks.
        seen: set[str] = set()
        self._mcp_server_urls: list[str] = [
            u for u in (mcp_server_urls or []) if u and not (u in seen or seen.add(u))
        ]
        self.mcp_client = MCPClient()
        # Servers whose tools are not registered yet; discovery drains it.
        self._mcp_pending: list[str] = list(self._mcp_server_urls)
        self._mcp_retry_after = 0.0
        self._discovery_lock = asyncio.Lock()

    def _mcp_discovery_due(self) -> bool:
        return bool(self._mcp_pending) and time.monotonic() >= self._mcp_retry_after

    async def _ensure_tools_discovered(self) -> None:
        """Discover MCP tools on first use.

        A server that fails discovery is skipped for now and retried, throttled
        by ``discovery_retry_seconds``, on a later turn: the tools of the
        servers that answered are usable meanwhile, and one that was down is
        picked up when it comes back instead of being lost until a restart.
        """
        if not self._mcp_discovery_due():
            return
        async with self._discovery_lock:
            if not self._mcp_discovery_due():
                return
            any_succeeded = False
            still_pending: list[str] = []
            for server_url in self._mcp_pending:
                try:
                    discovered = await self.mcp_client.discover_tools(server_url)
                except Exception as exc:
                    logger.warning(
                        "%s MCP discovery failed for %s: %s", self.name, server_url, exc
                    )
                    still_pending.append(server_url)
                    continue
                for tool in discovered:
                    self.mcp_client.register_tool(
                        name=tool["name"],
                        url=server_url,
                        description=tool.get("description", ""),
                        input_schema=tool.get("input_schema", {}),
                    )
                any_succeeded = True
            self._mcp_pending = still_pending
            self._mcp_retry_after = time.monotonic() + settings.discovery_retry_seconds
            if any_succeeded:
                self.mcp_client._build_openai_tools()

    def prepare_tool_args(self, tool_args: dict, context: dict | None) -> dict:
        """Hook to mutate tool arguments before each MCP call.

        Default is identity. Override to enforce server-side values that the
        LLM must not control (e.g. ``user_id``).
        """
        return tool_args

    def build_messages(self, message: str | list[dict], context: dict | None) -> list[dict]:
        """Assemble the message list: [user-context?] + [delegation?] + history + user turn.

        ``delegation`` is set by the orchestrator when this turn is shared with
        other specialists. It names the parts somebody else is answering, so a
        specialist that receives the full user message does not answer past its
        own domain.
        """
        messages: list[dict] = []
        if context and context.get("user_context"):
            messages.append(
                {"role": "system", "content": f"User context:\n{context['user_context']}"}
            )
        if context and context.get("delegation"):
            messages.append({"role": "system", "content": context["delegation"]})
        if context and context.get("chat_history"):
            messages.extend(context["chat_history"])
        messages.append({"role": "user", "content": message})
        return messages

    async def process(
        self, message: str | list[dict], context: dict | None = None
    ) -> AgentResponse:
        """Run the bounded tool-calling loop and return the final response."""
        trace = [
            TraceStep(
                agent=self.name,
                action="processing",
                detail=f"Processing {self.name} query",
            )
        ]

        started = time.monotonic()
        await self._ensure_tools_discovered()
        messages = self.build_messages(message, context)
        openai_tools = self.mcp_client.get_openai_tools() or None

        for _ in range(settings.tool_loop_max_iterations):
            try:
                response = await self.generative.chat(messages, tools=openai_tools)
            except Exception:
                logger.exception("%s generation failed", self.name)
                raise

            # Without tools the generative layer returns a plain string.
            if not openai_tools:
                trace.append(
                    TraceStep(
                        agent=self.name,
                        action="response",
                        detail=f"Generated {self.name} response",
                        elapsed_ms=elapsed_ms_since(started),
                        tokens=self.generative.last_completion_tokens or None,
                        prompt_tokens=self.generative.last_prompt_tokens or None,
                    )
                )
                return AgentResponse(content=response.strip(), agent_name=self.name, trace=trace)

            choice = response.choices[0]
            tool_calls = choice.message.tool_calls
            if not tool_calls:
                content = choice.message.content or ""
                trace.append(
                    TraceStep(
                        agent=self.name,
                        action="response",
                        detail=f"Generated {self.name} response",
                        elapsed_ms=elapsed_ms_since(started),
                        tokens=self.generative.last_completion_tokens or None,
                        prompt_tokens=self.generative.last_prompt_tokens or None,
                    )
                )
                return AgentResponse(content=content.strip(), agent_name=self.name, trace=trace)

            messages.append(choice.message.model_dump())
            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = self.prepare_tool_args(_tool_args(tc.function.arguments), context)

                trace.append(
                    TraceStep(agent=self.name, action="tool_call", detail=f"Calling MCP tool: {tool_name}")
                )
                call_started = time.monotonic()
                action = "tool_result"
                try:
                    result = await self.mcp_client.call_tool(tool_name, tool_args)
                except MCPToolError as exc:
                    # The tool answered, and its answer is "no". Its own message is
                    # already written for a reader, so it goes to the model as-is;
                    # wrapping it would only say "error" twice.
                    logger.warning("%s MCP tool %s reported failure: %s", self.name, tool_name, exc)
                    result = str(exc)
                    action = "tool_error"
                except Exception as exc:
                    logger.warning("%s MCP tool call failed for %s: %s", self.name, tool_name, exc)
                    result = f"Error calling tool {tool_name}: {exc}"
                    action = "tool_error"

                trace.append(
                    TraceStep(
                        agent=self.name,
                        action=action,
                        detail=f"Tool {tool_name}: {str(result)[:100]}",
                        elapsed_ms=elapsed_ms_since(call_started),
                    )
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
                )

        return AgentResponse(
            content="Unable to complete the request after multiple tool calls.",
            agent_name=self.name,
            trace=trace,
        )

    async def process_stream(
        self, message: str | list[dict], context: dict | None = None
    ) -> AsyncIterator[str | TraceStep | InputRequired]:
        """Stream the answer token-by-token, resolving MCP tool calls as it goes.

        Reasoning is delimited inline by ``<think>`` tags; tool calls are
        resolved between answer chunks.

        Args:
            message: Text message or list of dictionaries (multi-modal format).
            context: Optional additional context for processing.

        Yields:
            Text chunks interleaved with :class:`TraceStep` markers (the same
            steps the non-streaming :meth:`process` records), so callers can
            surface the full trace while streaming, and at most one
            :class:`InputRequired` when the agent stops to ask the user
            something.
        """
        started = time.monotonic()
        yield TraceStep(
            agent=self.name, action="processing", detail=f"Processing {self.name} query"
        )
        await self._ensure_tools_discovered()
        messages = self.build_messages(message, context)
        # ask_user rides along with the agent's own tools; it does not create a
        # tools array where there was none. An agent that exposes no tool keeps
        # calling the model without the field, exactly as before — some backends
        # reject `tools` outright, and that path used to work.
        discovered = self.mcp_client.get_openai_tools() or []
        tools = [*discovered, _ASK_USER_SPEC] if discovered else None

        for _ in range(settings.tool_loop_max_iterations):
            tool_calls = None
            async for kind, val in self.generative.stream_with_tools(messages, tools):
                if kind == "text":
                    yield val
                elif kind == "tools":
                    tool_calls = val

            # Asking wins over any other call in the same round: without the
            # answer the rest of the round is guesswork anyway.
            asked = next((c for c in tool_calls or [] if c["name"] == ASK_USER_TOOL), None)
            if asked is not None:
                question = str(_tool_args(asked["args"]).get("question", "")).strip()
                yield TraceStep(
                    agent=self.name,
                    action="ask_user",
                    detail=question or "(no question given)",
                    elapsed_ms=elapsed_ms_since(started),
                    prompt_tokens=self.generative.last_prompt_tokens or None,
                )
                yield InputRequired(question=question or "I need more information to continue.")
                return

            if not tool_calls:
                yield TraceStep(
                    agent=self.name,
                    action="response",
                    detail=f"Generated {self.name} response",
                    elapsed_ms=elapsed_ms_since(started),
                    tokens=self.generative.last_completion_tokens or None,
                    prompt_tokens=self.generative.last_prompt_tokens or None,
                )
                return

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["args"]},
                    }
                    for c in tool_calls
                ],
            })
            for c in tool_calls:
                args = self.prepare_tool_args(_tool_args(c["args"]), context)
                yield TraceStep(
                    agent=self.name, action="tool_call", detail=f"Calling MCP tool: {c['name']}"
                )
                call_started = time.monotonic()
                action = "tool_result"
                try:
                    result = await self.mcp_client.call_tool(c["name"], args)
                except MCPToolError as exc:
                    # Same split as the non-streaming loop: the tool's own message
                    # reaches the model unchanged, and only the log and the trace
                    # learn that it was a failure.
                    logger.warning("%s MCP tool %s reported failure: %s", self.name, c["name"], exc)
                    result = str(exc)
                    action = "tool_error"
                except Exception as exc:
                    logger.warning("%s MCP tool call failed for %s: %s", self.name, c["name"], exc)
                    result = f"Error calling tool {c['name']}: {exc}"
                    action = "tool_error"
                yield TraceStep(
                    agent=self.name,
                    action=action,
                    detail=f"Tool {c['name']}: {str(result)[:100]}",
                    elapsed_ms=elapsed_ms_since(call_started),
                )
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": str(result)})

        # Tool-loop budget exhausted (the model kept requesting tools every
        # round): surface it instead of ending the stream empty — mirrors the
        # non-streaming fallback in :meth:`process`.
        yield "Unable to complete the request after multiple tool calls."
        yield TraceStep(
            agent=self.name,
            action="response",
            detail=f"Tool-loop budget exhausted after {settings.tool_loop_max_iterations} rounds",
            elapsed_ms=elapsed_ms_since(started),
        )

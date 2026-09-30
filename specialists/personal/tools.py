"""The personal agent's MCP tool server.

The three tools are plain async functions bound to a
:class:`~specialists.personal.store.ProfileStore`; the store is injected
at application startup, so nothing here is a module-level singleton.

Every tool takes ``user_id``, but the agent overrides it server-side with the
trusted request identity (see :meth:`PersonalAgent.prepare_tool_args`) — the
model never decides which user record is touched.
"""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP

from erebots.runtime import render_user_context
from specialists.personal.store import ProfileStore


def make_personal_tools(store: ProfileStore) -> dict[str, Callable]:
    """The personal agent's tool functions, bound to ``store``.

    Each is an async callable whose signature and docstring become the schema
    and description the MCP server publishes.
    """

    async def update_profile(user_id: str, updates: dict) -> str:
        """Update fields in the user's profile.

        Use this when the user shares facts about themselves (name, age,
        weight, height, dietary preferences, allergies, lifestyle, hobbies).
        Provide only the fields that are new or changing.

        Args:
            user_id: Server-injected user identifier; do not rely on it.
            updates: Dictionary of profile fields to merge in.

        Returns:
            Confirmation string.
        """
        ctx = await store.update_profile(user_id, updates)
        return f"Profile updated for {ctx.username or user_id}: {updates}"

    async def manage_consent(user_id: str, consents: dict) -> str:
        """Update privacy/consent flags for the user.

        Use this when the user grants or revokes a consent
        (analytics, marketing, ...). Boolean values.

        Args:
            user_id: Server-injected user identifier; do not rely on it.
            consents: Dictionary of consent flag updates.

        Returns:
            Confirmation string.
        """
        ctx = await store.update_consents(user_id, consents)
        return f"Consents updated for {ctx.username or user_id}: {consents}"

    async def get_user_context(user_id: str) -> str:
        """Retrieve the latest profile, preferences, and consents.

        Args:
            user_id: Server-injected user identifier; do not rely on it.

        Returns:
            Human-readable summary of the user's stored context.
        """
        ctx = await store.read(user_id)
        return render_user_context(ctx)

    return {
        "update_profile": update_profile,
        "manage_consent": manage_consent,
        "get_user_context": get_user_context,
    }


def build_personal_mcp(store: ProfileStore) -> FastMCP:
    """MCP server exposing the personal tools bound to ``store``.

    ``stateless_http`` matches how :class:`~erebots.agents.mcp.client.MCPClient`
    calls it — one connection per call, no session to keep alive — so the server
    never accumulates sessions nobody will resume.
    """
    server = FastMCP("personal", stateless_http=True)
    for fn in make_personal_tools(store).values():
        server.tool()(fn)
    return server

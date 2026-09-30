"""Profile and consent access, owned by the personal agent.

The user's profile and consents live in the same row as the conversation, but
they are not the framework's to write. This agent is the only one allowed to
touch them; every other agent that needs them asks it through its MCP tools
rather than reaching into the database.

That separation is the reason this file exists rather than two more methods on
:class:`~erebots.persistence.UserContextStore`. The framework owns the row and
the conversation and offers a concurrency-safe way to merge one JSON column;
which columns are *this* agent's is stated here, in the agent that owns them.
"""

from __future__ import annotations

from erebots.persistence import UserContext, UserContextStore

# The two columns of the shared row this agent owns. Naming them here rather
# than in the framework is the point: a second agent adding its own data would
# add its own store beside this one, not a method to the framework's.
_PROFILE = "profile"
_CONSENTS = "consents"


class ProfileStore:
    """Read and write the parts of a user's record this agent owns.

    Wraps the framework's store rather than the database: row creation, the row
    lock and the column-wise merge are infrastructure and stay there. What this
    class adds is the statement of which columns belong to the personal agent.
    """

    def __init__(self, user_context_store: UserContextStore):
        self._store = user_context_store

    async def read(self, user_id: str) -> UserContext:
        """The user's stored context, created on first sight.

        Returns the whole context rather than the profile alone because that is
        what the framework's store returns; the caller uses the two fields this
        agent owns.
        """
        return await self._store.get_or_create(user_id)

    async def update_profile(self, user_id: str, updates: dict) -> UserContext:
        """Merge ``updates`` into the user's profile, other columns untouched."""
        return await self._store.merge_json_column(user_id, _PROFILE, updates)

    async def update_consents(self, user_id: str, updates: dict) -> UserContext:
        """Merge ``updates`` into the user's consents, other columns untouched."""
        return await self._store.merge_json_column(user_id, _CONSENTS, updates)

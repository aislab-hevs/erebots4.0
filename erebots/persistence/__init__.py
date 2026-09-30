"""Persistence layer: the async engine, the ORM models, and the user-context store.

The user context is owned here rather than by any one agent, because it has
never belonged to one. The orchestrator reads it to build a turn's context,
and any specialist may read or write it -- a profile written by one agent is
read by another. That is also why the writes are column-wise under a row
lock: two agents touching one user at the same time is the normal case, not
the exceptional one.
"""

from erebots.persistence.db import Base, make_engine, make_session_factory
from erebots.persistence.store import UserContextStore
from erebots.persistence.user_context import UserContext

__all__ = [
    "Base",
    "UserContext",
    "UserContextStore",
    "make_engine",
    "make_session_factory",
]

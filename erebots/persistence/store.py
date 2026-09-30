"""Persistent store for ``UserContext``.

Wraps SQLAlchemy with a tight, agent-friendly API. The store is the
single place that knows how to read and write the ``user_context``
table; everything else manipulates ``UserContext`` dataclasses.
"""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from erebots.persistence.models import UserContextRow
from erebots.persistence.user_context import UserContext


def _row_to_context(row: UserContextRow) -> UserContext:
    """Convert an ORM row to a ``UserContext`` dataclass."""
    return UserContext(
        user_id=row.user_id,
        username=row.username or "",
        email=row.email or "",
        roles=list(row.roles or []),
        chat_history=list(row.chat_history or []),
        profile=dict(row.profile or {}),
        consents=dict(row.consents or {}),
    )


class UserContextStore:
    """Async persistence for ``UserContext``.

    The framework's own concern here is the conversation: creating a user's row
    and reading and writing ``chat_history``. Profile and consent data lives in
    the same row but belongs to the agent that owns it, which reaches it through
    :meth:`merge_json_column` rather than through a method named after it.

    Writes are per column, so callers working on the same user at the same time
    (a specialist's tool while the frontend persists the conversation, two tabs
    of one user) cannot overwrite each other's fields.
    Merges into one JSON column read and rewrite the row under a row lock
    (``SELECT ... FOR UPDATE`` where the database has it). Appending to the
    chat history goes further and holds on SQLite too, which has no row locks;
    see :meth:`append_chat_history`.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def get_or_create(
        self,
        user_id: str,
        seed: dict | None = None,
    ) -> UserContext:
        """Return the stored context for ``user_id``, creating it on first sight.

        Two first requests for the same user may both try to create it; the one
        that loses the primary-key race reads back what the winner wrote.

        Args:
            user_id: The row's primary key.
            seed: Field values for a row that does not exist yet, ignored when
                it does. Where they come from is the caller's business: this
                layer stores rows and does not know what an identity provider
                calls things. :func:`erebots.security.claims.user_fields_from_claims`
                is what the framework passes.

        Returns:
            The stored context, existing or newly created.
        """
        async with self._session_factory() as session:
            row = await session.get(UserContextRow, user_id)
            if row is not None:
                return _row_to_context(row)

            fields = seed or {}
            row = UserContextRow(
                user_id=user_id,
                username=fields.get("username", ""),
                email=fields.get("email", ""),
                roles=list(fields.get("roles", [])),
                profile=dict(fields.get("profile", {})),
                consents=dict(fields.get("consents", {})),
                chat_history=list(fields.get("chat_history", [])),
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                row = await session.get(UserContextRow, user_id)
                if row is None:  # pragma: no cover - the conflict was not ours
                    raise
            return _row_to_context(row)

    async def append_chat_history(self, user_id: str, messages: list[dict]) -> None:
        """Append ``messages`` to the user's chat history, atomically.

        Read-modify-write in one transaction, and the *first* statement is a
        write, so the row is locked before it is read. That order is the whole
        point:

        * PostgreSQL takes a row lock (``UPDATE`` blocks a concurrent
          ``UPDATE`` of the same row until this transaction commits, then
          re-reads the committed row) — so the second appender sees the first
          one's turn and adds to it.
        * SQLite has no row locks and ignores ``FOR UPDATE`` entirely. What it
          does have is a database write lock, taken when a transaction issues
          its first write; a concurrent transaction waits for it (``busy
          timeout``) and proceeds after the commit. Reading first and trying to
          upgrade later is the case SQLite refuses to wait for: it returns
          "database is locked" instead of serializing.

        The locking statement assigns the column to itself: a real write as far
        as the lock is concerned, and a no-op as far as the data is concerned,
        so nothing is lost if the transaction rolls back before the append.
        """
        if not messages:
            return
        await self.get_or_create(user_id)
        async with self._session_factory() as session:
            await session.execute(
                update(UserContextRow)
                .where(UserContextRow.user_id == user_id)
                .values(chat_history=UserContextRow.chat_history)
            )
            stored = (
                await session.execute(
                    select(UserContextRow.chat_history).where(
                        UserContextRow.user_id == user_id
                    )
                )
            ).scalar_one()
            await session.execute(
                update(UserContextRow)
                .where(UserContextRow.user_id == user_id)
                .values(chat_history=list(stored or []) + list(messages))
            )
            await session.commit()

    async def set_chat_history(self, user_id: str, history: list[dict]) -> None:
        """Replace the user's chat history; other columns untouched.

        A replace, for a caller that owns the whole conversation. A caller
        adding to it wants :meth:`append_chat_history`, which does not depend
        on the caller holding what is already there.
        """
        await self.get_or_create(user_id)
        async with self._session_factory() as session:
            row = await session.get(UserContextRow, user_id, with_for_update=True)
            row.chat_history = list(history)
            await session.commit()

    async def merge_json_column(self, user_id: str, column: str, updates: dict) -> UserContext:
        """Merge ``updates`` into one JSON column, leaving every other untouched.

        The write primitive an application builds its own accessor on. The
        framework does not name the columns it holds beyond the conversation:
        ``profile`` and ``consents`` belong to whichever agent owns that data,
        and in the bundled example that is the personal agent, which is the
        only one allowed to touch them.

        Safe against a concurrent write to a different column: the row is read
        back under a lock and only the named column is reassigned, so two
        agents working on one user at the same time cannot overwrite each
        other's fields.

        Args:
            user_id: The row to merge into, created first if absent.
            column: The JSON column's name.
            updates: Keys to merge in; existing keys are replaced, others kept.

        Returns:
            The stored context after the merge.
        """
        await self.get_or_create(user_id)
        async with self._session_factory() as session:
            row = await session.get(UserContextRow, user_id, with_for_update=True)
            # A new dict, not an in-place update: the ORM tracks the column as
            # changed by reassignment, and only changed columns reach the UPDATE.
            setattr(row, column, {**(getattr(row, column) or {}), **updates})
            await session.commit()
            return _row_to_context(row)

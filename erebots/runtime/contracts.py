"""The shapes the runtime expects, written down where they can be checked.

:class:`~erebots.runtime.ConversationService` has always taken a duck-typed
store, and the framework ships one in :mod:`erebots.persistence`. But the store
is replaceable by design -- an application with its own tables writes its own --
and until now the only statement of what a replacement had to provide was a
docstring. That docstring named a ``save(ctx)`` method which has never existed
anywhere in this project, and nothing could contradict it, because prose cannot
fail a test.

So the contract lives here instead, and ``tests/test_store_contract.py`` holds
the shipped implementation to it.

Two notes on what these protocols deliberately do not do.

They name no concrete type from :mod:`erebots.persistence`. A store returns
something with the right attributes; it need not be the dataclass this project
happens to store. Keeping the import out also keeps a database driver from
being dragged in behind a type annotation.

``append_chat_history`` is absent. It is an optional fast path the service
probes with ``getattr`` and falls back from, the same way
:func:`erebots.server.create_app` probes ``aclose``. Putting it in the protocol
would oblige every implementer to write a method the framework only uses when
it happens to be there.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class UserContextLike(Protocol):
    """What a store hands back, as far as the runtime is concerned.

    Attributes:
        username: Display name, used when a turn is rendered for a prompt.
        profile: What the user has told the assistant about themselves.
        consents: Privacy flags the user has set.
        chat_history: The stored conversation, oldest first.
    """

    username: str
    profile: dict
    consents: dict
    chat_history: list[dict]


@runtime_checkable
class UserContextStoreProtocol(Protocol):
    """The two calls :class:`~erebots.runtime.ConversationService` makes."""

    async def get_or_create(
        self, user_id: str, seed: dict | None = None
    ) -> UserContextLike:
        """Return the stored context for ``user_id``, creating it if absent.

        Args:
            user_id: The identity the application resolved for this turn.
            seed: Field values for a row that does not exist yet, ignored when
                it does. The framework passes the result of
                :func:`erebots.security.claims.user_fields_from_claims`.
        """
        ...

    async def set_chat_history(self, user_id: str, history: list[dict]) -> None:
        """Replace the stored conversation for ``user_id``.

        A replace rather than an append: only a caller holding the whole
        transcript calls this.
        """
        ...

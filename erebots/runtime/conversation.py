"""Frontend-agnostic conversation/session service.

Owns the two session concerns the orchestrator used to embed: building the
per-turn request context (identity + persisted user context) and persisting the
chat history. Wraps a user-context *store* duck-typed as
``get_or_create(user_id, seed=None)`` (returning an object with ``username``,
``profile``, ``consents`` and ``chat_history``) /
``set_chat_history(user_id, history)`` — no domain import — so every frontend
(Gradio, CLI, REST) reuses the same logic. Async-native.

Rendering that stored context into prompt text lives here too, in
:func:`render_user_context`, because what a model is told about a user is a
property of how a turn is built rather than of how it is stored.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any

from erebots.config import settings
from erebots.runtime.contracts import UserContextStoreProtocol
from erebots.security.claims import ClaimsMapper, user_fields_from_claims

logger = logging.getLogger(__name__)

# Swap the rendering without subclassing anything, the way SessionResolver and
# TokenProvider are swapped elsewhere in the framework.
UserContextRenderer = Callable[[Any], str]


def render_user_context(ctx: Any) -> str:
    """Render a stored user context into the text a prompt carries.

    Read with ``getattr`` rather than attribute access on purpose. The only
    caller runs inside the context build, where an exception is caught and
    logged, so a store returning something slightly different would silently
    cost the turn its user context instead of failing loudly. A conforming
    store provides all three; this survives one that does not.

    Args:
        ctx: The stored context, with ``username``, ``profile`` and
            ``consents``.

    Returns:
        One line per populated field, empty fields omitted.
    """
    parts = [f"User: {getattr(ctx, 'username', '') or ''}"]
    profile = getattr(ctx, "profile", None)
    if profile:
        parts.append(f"Profile: {profile}")
    consents = getattr(ctx, "consents", None)
    if consents:
        parts.append(f"Consents: {consents}")
    return "\n".join(parts)


class ConversationService:
    """Builds request context and persists chat history via a user-context store."""

    def __init__(
        self,
        user_context_store: UserContextStoreProtocol | None = None,
        *,
        render_context: UserContextRenderer = render_user_context,
        seed_from_claims: ClaimsMapper = user_fields_from_claims,
    ):
        """Bind the service to the store it reads and writes user context through.

        Args:
            user_context_store: The store to read and write user context
                through, normally :class:`~erebots.persistence.UserContextStore`.
                Structurally typed rather than inherited: anything matching
                :class:`~erebots.runtime.UserContextStoreProtocol` works, and
                that protocol is the contract a replacement is held to. Left as
                ``None`` for a stateless deployment: the service then builds
                context from the session alone and persists nothing.
            render_context: Turns the stored context into the prompt text an
                agent receives. Defaults to :func:`render_user_context`;
                replace it to change what the model is told about a user
                without replacing the store.
            seed_from_claims: Turns a token's claims into the fields a
                first-sight user row starts with. Defaults to the Keycloak
                mapping in :mod:`erebots.security.claims`; replace it for
                another identity provider. The store never sees the claims.
        """
        self._store = user_context_store
        self._render = render_context
        self._seed_from_claims = seed_from_claims

    @property
    def persists_history(self) -> bool:
        """Is there anywhere to keep a conversation at all?

        An identity is not enough: with no store configured, every append is a
        no-op. A frontend that stops sending its transcript on the strength of
        "there is a user" would hand its conversation to nobody.
        """
        return self._store is not None

    @staticmethod
    def sanitize_history(messages: list[dict]) -> list[dict]:
        """Keep only plain user/assistant turns (drop UI metadata bubbles).

        Args:
            messages: The raw history as the frontend holds it.

        Returns:
            A deep copy of the conversational turns, so later mutation of the
            caller's list cannot reach what gets persisted.
        """
        return [
            copy.deepcopy(message)
            for message in messages
            if message.get("role") in {"user", "assistant"} and "metadata" not in message
        ]

    async def build_request_context(
        self,
        session_context: dict | None = None,
        chat_history: list[dict] | None = None,
    ) -> dict | None:
        """Assemble the context forwarded to specialists for one turn.

        Includes ``user_id`` and, when a store is configured, the persisted
        ``user_context`` rendered for prompting, plus ``chat_history``. The
        token's claims are deliberately not included: this dict travels to
        every specialist, and they are needed only to seed a first-sight row.

        The history is the caller's when it supplies one, and the user's stored
        history otherwise — never both. So a frontend may own the transcript or
        own nothing, and both see the same conversation.

        "Supplies one" means ``chat_history is not None``: an explicit ``[]`` is
        a caller stating this conversation is empty (a cleared chat, a fresh
        tab that owns its transcript), and is honoured as such. Only omitting
        the argument asks the service to load the stored history.

        A store that fails is logged and skipped: the turn proceeds without the
        persisted context rather than failing on it.

        Args:
            session_context: Identity for this turn (``user_id``, and
                ``jwt_claims`` when the request was authenticated), as resolved
                from the request. The claims are read here and not forwarded.
            chat_history: The conversation so far. Capped at
                ``settings.chat_history_max_messages`` most recent messages,
                which ``0`` disables.

        Returns:
            The context to forward to specialists, or ``None`` when there is
            nothing to say about this turn — an anonymous turn with no history.
        """
        context: dict = {}
        session = session_context or {}
        stored_history: list[dict] | None = None

        user_id = session.get("user_id")
        if user_id:
            context["user_id"] = user_id
            # The claims stay out of the context on purpose. This dict is
            # serialized whole into A2A message metadata and reaches every
            # specialist the turn touches, including agents on hosts this
            # deployment does not own -- a specialist is just a URL. Nothing
            # downstream reads the claims, so sending a user's decoded token to
            # a third party bought nothing. They are used here and go no further.
            jwt_claims = session.get("jwt_claims")
            if self._store is not None:
                try:
                    # The claims are mapped here, not in the store: what an
                    # identity provider calls a username is not a storage
                    # concern, and a deployment on another provider swaps the
                    # mapper rather than the persistence layer.
                    seed = (
                        self._seed_from_claims(jwt_claims)
                        if isinstance(jwt_claims, dict)
                        else None
                    )
                    user_ctx = await self._store.get_or_create(user_id, seed)
                    context["user_context"] = self._render(user_ctx)
                    stored_history = getattr(user_ctx, "chat_history", None)
                except Exception:
                    logger.exception("Failed to fetch user context for %s", user_id)

        # A frontend that owns the transcript sends it (the SSE client, the
        # Gradio chatbot) and that wins; one that keeps no state sends nothing
        # and gets the persisted history instead, so a thin client still holds
        # a conversation. Exactly one of the two — merging them is how a turn
        # ends up in the prompt twice.
        history = chat_history if chat_history is not None else stored_history
        if history:
            # Only user/assistant turns may reach a prompt: the HTTP adapter
            # rejects anything else, and this keeps every other frontend from
            # smuggling a ``system`` turn in by mistake. Then the server-side
            # cap: whatever a client sends, only the most recent messages
            # reach the LLM, so prompts stay bounded in long chats.
            history = self.sanitize_history(history)
            limit = settings.chat_history_max_messages
            if limit > 0:
                history = history[-limit:]
            if history:
                context["chat_history"] = history

        return context or None

    async def load_history(self, session_context: dict | None) -> list[dict]:
        """The user's full stored history, sanitized and *uncapped*.

        The cap in :meth:`build_request_context` bounds what reaches a prompt;
        it must never bound what is kept. A frontend that seeds its transcript
        from the store reads it here, so replacing that transcript later cannot
        silently discard the turns the prompt was not going to see anyway.
        """
        if self._store is None or not session_context:
            return []
        user_id = session_context.get("user_id")
        if not user_id:
            return []
        try:
            user_ctx = await self._store.get_or_create(user_id)
        except Exception:
            logger.exception("Failed to fetch stored history for %s", user_id)
            return []
        return self.sanitize_history(list(getattr(user_ctx, "chat_history", None) or []))

    async def append_turn(self, session_context: dict | None, messages: list[dict]) -> None:
        """Append ``messages`` to the user's stored history.

        The path for a frontend that keeps no transcript of its own (the CLI,
        a thin SSE client): it never holds the full conversation, so it must
        not be the one to write it back. Appending to what the store already
        has is what keeps a long conversation whole — persisting the caller's
        view instead turns the prompt cap into a retention cap.

        Two turns appended at the same instant — one user with two tabs open,
        not just two instances — must both survive, so the append happens in
        the store when it offers ``append_chat_history``: one transaction that
        locks the row before reading it. Doing it here instead means reading
        and writing in two steps with nothing in between, which loses a turn.
        That is the fallback for a store that offers only ``set_chat_history``
        (the service duck-types the store), and it is the one path where a
        concurrent append can still be lost.
        """
        if self._store is None or not session_context or not messages:
            return
        user_id = session_context.get("user_id")
        if not user_id:
            return
        clean = self.sanitize_history(list(messages))
        if not clean:
            return
        append = getattr(self._store, "append_chat_history", None)
        if append is not None:
            try:
                await append(user_id, clean)
            except Exception:
                logger.exception("Failed to append to chat history for %s", user_id)
            return
        stored = await self.load_history(session_context)
        await self.persist_history(session_context, stored + clean)

    async def persist_history(
        self, session_context: dict | None, history: list[dict]
    ) -> None:
        """Replace the stored chat history, preserving profile/consents.

        Sanitized, and a *replace*: only a frontend that owns the whole
        transcript may call this (an edit, a retry and a clear are all
        replacements). One that owns nothing calls :meth:`append_turn`.

        Read-modify-write on the stored context, so the fields this service does
        not own are carried over untouched. A missing store, an anonymous turn,
        or a failing write are all no-ops: history is a convenience, and losing
        it must not fail the turn that produced it.

        Args:
            session_context: Identity for this turn; nothing is written without
                a ``user_id``.
            history: The raw history, sanitized before being stored.
        """
        if self._store is None or not session_context:
            return
        user_id = session_context.get("user_id")
        if not user_id:
            return
        clean = self.sanitize_history(history)
        try:
            # One column: a profile or consent write landing at the same time
            # (a specialist's tool, another tab) is not overwritten.
            await self._store.set_chat_history(user_id, clean)
        except Exception:
            logger.exception("Failed to persist chat history for %s", user_id)

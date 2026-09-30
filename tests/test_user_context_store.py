"""Tests for ``UserContextStore``: per-column writes and race-safe creation.

The store used to rewrite the whole row on every save, so two callers that
had read the same state overwrote each other's fields (a consent save wiped a
profile update). Writes are now per column; these tests pin that down on a
real database (SQLite on disk, the same engine path the app uses).
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from erebots.persistence.db import Base, make_engine, make_session_factory
from erebots.persistence.store import UserContextStore


@pytest.fixture
async def store(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'ctx.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield UserContextStore(make_session_factory(engine))
    finally:
        await engine.dispose()


async def test_get_or_create_applies_the_seed_only_on_creation(store):
    first = await store.get_or_create("u1", {"username": "ann", "email": "a@x"})
    again = await store.get_or_create("u1", {"username": "changed"})
    assert first.username == "ann"
    assert again.username == "ann"


async def test_get_or_create_without_a_seed_creates_an_empty_row(store):
    """A caller with no identity provider still gets a row to write into."""
    ctx = await store.get_or_create("u1")
    assert ctx.user_id == "u1"
    assert (ctx.username, ctx.email, ctx.roles) == ("", "", [])


async def test_profile_consents_and_history_writes_do_not_overwrite_each_other(store):
    """The review's reproduction: a profile update survives a later consent save."""
    await store.get_or_create("u1")

    await store.merge_json_column("u1", "profile", {"name": "Ann"})
    await store.merge_json_column("u1", "consents", {"analytics": True})
    await store.set_chat_history("u1", [{"role": "user", "content": "hi"}])
    await store.merge_json_column("u1", "profile", {"age": 30})

    ctx = await store.get_or_create("u1")
    assert ctx.profile == {"name": "Ann", "age": 30}
    assert ctx.consents == {"analytics": True}
    assert ctx.chat_history == [{"role": "user", "content": "hi"}]


async def test_concurrent_writers_to_different_columns_all_land(store):
    await store.get_or_create("u1")
    await asyncio.gather(
        store.merge_json_column("u1", "profile", {"name": "Ann"}),
        store.merge_json_column("u1", "consents", {"analytics": True}),
        store.set_chat_history("u1", [{"role": "user", "content": "hi"}]),
    )
    ctx = await store.get_or_create("u1")
    assert ctx.profile == {"name": "Ann"}
    assert ctx.consents == {"analytics": True}
    assert ctx.chat_history == [{"role": "user", "content": "hi"}]


async def test_column_writes_create_the_row_for_an_unseen_user(store):
    ctx = await store.merge_json_column("new", "profile", {"name": "Bo"})
    assert ctx.user_id == "new"
    assert ctx.profile == {"name": "Bo"}
    await store.set_chat_history("newer", [])
    assert (await store.get_or_create("newer")).chat_history == []


async def test_losing_the_creation_race_returns_the_winner_row(store, monkeypatch):
    """Two first requests: the loser's INSERT conflicts and it reads the winner's row."""
    await store.get_or_create("u1", {"username": "winner"})

    real_get = AsyncSession.get
    misses = {"left": 1}

    async def stale_get(self, *args, **kwargs):
        # The first lookup ran before the winner committed: it saw no row.
        if misses["left"]:
            misses["left"] -= 1
            return None
        return await real_get(self, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "get", stale_get)
    ctx = await store.get_or_create("u1", {"username": "loser"})

    assert ctx.username == "winner"


async def test_reopening_a_long_conversation_does_not_truncate_what_is_stored(
    tmp_path, monkeypatch
):
    """Against the real store on disk, the way a thin frontend actually uses it.

    The prompt cap bounds what the LLM sees; it must not bound what is kept.
    A frontend that keeps no transcript (the CLI, an SSE client) reads its
    history back already capped, so it can only be the service that writes the
    conversation — otherwise the first turn after reopening a long chat deletes
    every message older than the cap.
    """
    from types import SimpleNamespace as NS

    from erebots.runtime import conversation as conv_mod
    from erebots.runtime.conversation import ConversationService

    monkeypatch.setattr(conv_mod, "settings", NS(chat_history_max_messages=4))

    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'chat.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        service = ConversationService(UserContextStore(make_session_factory(engine)))
        session = {"user_id": "u1"}

        # A long conversation, one turn at a time, as a frontend would run it.
        for i in range(10):
            context = await service.build_request_context(session)
            assert len(context.get("chat_history", [])) <= 4, "the prompt stays bounded"
            await service.append_turn(
                session,
                [
                    {"role": "user", "content": f"ask {i}"},
                    {"role": "assistant", "content": f"answer {i}"},
                ],
            )

        # Reopened later (a new process, a new browser tab): the whole
        # conversation is still there, in order.
        kept = await service.load_history(session)
        assert [m["content"] for m in kept] == [
            part for i in range(10) for part in (f"ask {i}", f"answer {i}")
        ]
    finally:
        await engine.dispose()


async def test_two_simultaneous_appends_both_survive(tmp_path):
    """One user, two tabs, two turns landing at the same instant.

    Read-modify-write in the service lost one of them — reproduced before the
    fix. The store now appends inside one transaction that writes before it
    reads, so the second appender waits for the first and adds to its result
    instead of overwriting it. Both turns have to be there, in one order or
    the other, with the earlier conversation intact.
    """
    import asyncio

    from erebots.runtime.conversation import ConversationService

    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'race.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        service = ConversationService(UserContextStore(make_session_factory(engine)))
        session = {"user_id": "u1"}
        earlier = [{"role": "user", "content": "earlier"}]
        await service.persist_history(session, earlier)

        await asyncio.gather(
            service.append_turn(session, [{"role": "user", "content": "A"}]),
            service.append_turn(session, [{"role": "user", "content": "B"}]),
        )

        kept = [m["content"] for m in await service.load_history(session)]
        assert kept[0] == "earlier", "an append must never drop an earlier turn"
        assert kept in (["earlier", "A", "B"], ["earlier", "B", "A"]), kept
    finally:
        await engine.dispose()


async def test_many_simultaneous_appends_all_survive(tmp_path):
    """The same guarantee under real contention, not just two racers."""
    import asyncio

    from erebots.runtime.conversation import ConversationService

    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'race-many.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        service = ConversationService(UserContextStore(make_session_factory(engine)))
        session = {"user_id": "u1"}
        await asyncio.gather(
            *(
                service.append_turn(session, [{"role": "user", "content": str(i)}])
                for i in range(12)
            )
        )
        kept = [m["content"] for m in await service.load_history(session)]
        assert sorted(kept, key=int) == [str(i) for i in range(12)], kept
    finally:
        await engine.dispose()


async def test_append_is_atomic_against_a_concurrent_replace(tmp_path):
    """An append must not resurrect a conversation a replace just cleared.

    The two writers of ``chat_history`` (an owning frontend replacing it, a
    thin one appending) touch the same column, so the append has to see the
    replace's result rather than a copy it read beforehand.
    """
    import asyncio

    from erebots.runtime.conversation import ConversationService

    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'replace.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        service = ConversationService(UserContextStore(make_session_factory(engine)))
        session = {"user_id": "u1"}
        await service.persist_history(session, [{"role": "user", "content": "old"}])

        await asyncio.gather(
            service.persist_history(session, []),  # the user cleared the chat
            service.append_turn(session, [{"role": "user", "content": "new"}]),
        )

        kept = [m["content"] for m in await service.load_history(session)]
        # Either order is correct; what must not happen is "old" coming back.
        assert kept in ([], ["new"]), kept
    finally:
        await engine.dispose()

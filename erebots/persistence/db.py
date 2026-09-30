"""SQLAlchemy async engine and session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def make_engine(url: str) -> AsyncEngine:
    """Create a SQLAlchemy async engine from a database URL.

    Args:
        url: SQLAlchemy URL (e.g. ``postgresql+asyncpg://...`` or
            ``sqlite+aiosqlite:///:memory:`` for tests).

    Returns:
        The engine. Network connections are opened lazily, so connection
        failures surface on first use rather than here.
    """
    return create_async_engine(url, future=True)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to ``engine``.

    Sessions do not expire attributes on commit, so an object stays readable
    after the transaction that wrote it.

    Args:
        engine: The engine the sessions connect through.

    Returns:
        The session factory.
    """
    return async_sessionmaker(engine, expire_on_commit=False)

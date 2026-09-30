"""ORM models for application data."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from erebots.persistence.db import Base

# JSONB on Postgres (operators + GIN indexes), plain JSON elsewhere (e.g. SQLite for tests).
JsonCol = JSON().with_variant(JSONB(), "postgresql")


class UserContextRow(Base):
    """Persistent ``UserContext``: one row per user, keyed by the user id.

    Attributes:
        user_id: The user identifier, and the primary key.
        username: Display name.
        email: Email address.
        roles: Realm roles carried by the token at the last write.
        profile: Application-defined user data, shape owned by the app.
        consents: Per-topic consent flags the user granted or withdrew.
        chat_history: The conversation turns kept for this user.
        updated_at: Set by the database on insert and on every update.
    """

    __tablename__ = "user_context"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    username: Mapped[str] = mapped_column(String(255), default="")
    email: Mapped[str] = mapped_column(String(255), default="")
    roles: Mapped[list] = mapped_column(JsonCol, default=list)
    profile: Mapped[dict] = mapped_column(JsonCol, default=dict)
    consents: Mapped[dict] = mapped_column(JsonCol, default=dict)
    chat_history: Mapped[list] = mapped_column(JsonCol, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
    )

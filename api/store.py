"""Postgres persistence for API users and their conversations (schema `app`)."""

from __future__ import annotations

import uuid
from datetime import datetime
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from agent.turn import TurnState
from api.sse import to_json as _dumps


_USER_COLS = "id, email, name, image, preferences"
TITLE_MAX_CHARS = 60


class ConversationBusy(Exception):
    """Another turn is already running for this conversation."""


def conversation_title(messages: list[dict]) -> str | None:
    """The first user message's text, whitespace collapsed, cut to 60 characters, right-trimmed."""
    for m in messages:
        if m.get("role") == "user":
            return " ".join(str(m.get("content", "")).split())[:TITLE_MAX_CHARS].rstrip() or None
    return None


class ConversationStore:
    def __init__(self, engine: Engine):
        self.engine = engine
        # Turn locks live on their own unpooled connections so a long turn never
        # holds a connection (or an open transaction) from the shared pool.
        self._lock_engine = create_engine(
            engine.url, poolclass=NullPool, isolation_level="AUTOCOMMIT"
        )

    # ── Users ─────────────────────────────────────────────────────────────

    def upsert_user(self, email: str, name: str | None, image: str | None) -> dict:
        """Create the user for this email (trimmed, lowercased) or update name, image and last_seen_at."""
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "INSERT INTO app.users (id, email, name, image) VALUES (:id, :email, :name, :image) "
                    "ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name, image = EXCLUDED.image, "
                    "last_seen_at = now() RETURNING " + _USER_COLS
                ),
                {"id": uuid.uuid4(), "email": email.strip().lower(), "name": name, "image": image},
            ).one()
        return dict(row._mapping)

    def get_user(self, user_id: uuid.UUID) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT {_USER_COLS} FROM app.users WHERE id = :id"), {"id": user_id}
            ).first()
        return dict(row._mapping) if row else None

    def update_preferences(self, user_id: uuid.UUID, patch: dict) -> dict:
        """Shallow-merge patch into the user's preferences; returns the merged preferences."""
        with self.engine.begin() as conn:
            return conn.execute(
                text(
                    "UPDATE app.users SET preferences = preferences || CAST(:patch AS jsonb) "
                    "WHERE id = :id RETURNING preferences"
                ),
                {"id": user_id, "patch": _dumps(patch)},
            ).scalar_one()

    # ── Conversations (every read and write checks the owner) ─────────────

    def create(self, user_id: uuid.UUID) -> uuid.UUID:
        cid = uuid.uuid4()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO app.conversations (id, user_id, prompt_cache_key) VALUES (:id, :uid, :key)"
                ),
                {"id": cid, "uid": user_id, "key": TurnState.new().prompt_cache_key},
            )
        return cid

    def load(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> TurnState | None:
        """The conversation's state, or None when it does not exist or belongs to someone else."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT prompt_cache_key, session_memory, previous_execution_results "
                    "FROM app.conversations WHERE id = :id AND user_id = :uid"
                ),
                {"id": conversation_id, "uid": user_id},
            ).first()
            if row is None:
                return None
            messages = conn.execute(
                text("SELECT body FROM app.messages WHERE conversation_id = :id ORDER BY position"),
                {"id": conversation_id},
            ).scalars().all()
        return TurnState(
            messages=list(messages),
            session_memory=row.session_memory,
            previous_execution_results=row.previous_execution_results,
            prompt_cache_key=row.prompt_cache_key,
        )

    def summary(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> dict | None:
        """{id, title, updated_at}, or None when not found or not owned."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT id, title, updated_at FROM app.conversations WHERE id = :id AND user_id = :uid"
                ),
                {"id": conversation_id, "uid": user_id},
            ).first()
        return dict(row._mapping) if row else None

    def list_conversations(self, user_id: uuid.UUID) -> list[dict]:
        """The user's 100 most recently updated conversations, newest first."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, title, updated_at FROM app.conversations WHERE user_id = :uid "
                    "ORDER BY updated_at DESC LIMIT 100"
                ),
                {"uid": user_id},
            ).all()
        return [dict(r._mapping) for r in rows]

    def rename(self, conversation_id: uuid.UUID, user_id: uuid.UUID, title: str) -> dict | None:
        """Set the title (`updated_at` stays: renaming is not activity); returns {id, title, updated_at}, or None."""
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "UPDATE app.conversations SET title = :title "
                    "WHERE id = :id AND user_id = :uid RETURNING id, title, updated_at"
                ),
                {"id": conversation_id, "uid": user_id, "title": title},
            ).first()
        return dict(row._mapping) if row else None

    def delete(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM app.conversations WHERE id = :id AND user_id = :uid"),
                {"id": conversation_id, "uid": user_id},
            )
        return result.rowcount > 0

    def save(self, conversation_id: uuid.UUID, state: TurnState) -> None:
        with self.engine.begin() as conn:
            stored = conn.execute(
                text("SELECT count(*) FROM app.messages WHERE conversation_id = :id"),
                {"id": conversation_id},
            ).scalar_one()
            for position, body in enumerate(state.messages[stored:], start=stored):
                conn.execute(
                    text(
                        "INSERT INTO app.messages (conversation_id, position, body) "
                        "VALUES (:id, :pos, CAST(:body AS jsonb))"
                    ),
                    {"id": conversation_id, "pos": position, "body": _dumps(body)},
                )
            conn.execute(
                text(
                    "UPDATE app.conversations SET updated_at = now(), "
                    "title = COALESCE(title, :title), "
                    "session_memory = CAST(:mem AS jsonb), "
                    "previous_execution_results = CAST(:res AS jsonb) WHERE id = :id"
                ),
                {
                    "id": conversation_id,
                    "title": conversation_title(state.messages),
                    "mem": _dumps(state.session_memory),
                    "res": _dumps(state.previous_execution_results),
                },
            )

    def try_start_turn(self, cap: int, at: datetime | None = None) -> bool:
        """Count one turn start in its UTC minute; False once that minute's count exceeds cap.

        `at` defaults to the database clock; rows older than an hour are deleted in passing.
        """
        with self.engine.begin() as conn:
            minute, n = conn.execute(
                text(
                    "INSERT INTO app.turn_starts (minute, n) VALUES ("
                    "date_trunc('minute', COALESCE(CAST(:at AS timestamptz), now()) AT TIME ZONE 'UTC') AT TIME ZONE 'UTC', 1) "
                    "ON CONFLICT (minute) DO UPDATE SET n = app.turn_starts.n + 1 RETURNING minute, n"
                ),
                {"at": at},
            ).one()
            conn.execute(text("DELETE FROM app.turn_starts WHERE minute < :m - interval '1 hour'"), {"m": minute})
        return n <= cap

    @contextmanager
    def lock(self, conversation_id: uuid.UUID) -> Iterator[None]:
        """Hold an exclusive advisory lock for one conversation, or raise ConversationBusy."""
        key = {"id": str(conversation_id)}
        with self._lock_engine.connect() as conn:
            acquired = conn.execute(
                text("SELECT pg_try_advisory_lock(hashtext(CAST(:id AS text)))"), key
            ).scalar_one()
            if not acquired:
                raise ConversationBusy(str(conversation_id))
            try:
                yield
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(hashtext(CAST(:id AS text)))"), key)

"""Postgres persistence for API conversations (schema `app`)."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import text
from sqlalchemy.engine import Engine

from agent.turn import TurnState
from api.sse import to_json as _dumps


class ConversationBusy(Exception):
    """Another turn is already running for this conversation."""


class ConversationStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def create(self) -> uuid.UUID:
        cid = uuid.uuid4()
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO app.conversations (id, prompt_cache_key) VALUES (:id, :key)"),
                {"id": cid, "key": TurnState.new().prompt_cache_key},
            )
        return cid

    def load(self, conversation_id: uuid.UUID) -> TurnState | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT prompt_cache_key, session_memory, previous_execution_results "
                    "FROM app.conversations WHERE id = :id"
                ),
                {"id": conversation_id},
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
                    "session_memory = CAST(:mem AS jsonb), "
                    "previous_execution_results = CAST(:res AS jsonb) WHERE id = :id"
                ),
                {
                    "id": conversation_id,
                    "mem": _dumps(state.session_memory),
                    "res": _dumps(state.previous_execution_results),
                },
            )

    @contextmanager
    def lock(self, conversation_id: uuid.UUID) -> Iterator[None]:
        """Hold an exclusive advisory lock for one conversation, or raise ConversationBusy."""
        with self.engine.connect() as conn:
            with conn.begin():
                acquired = conn.execute(
                    text("SELECT pg_try_advisory_xact_lock(hashtext(CAST(:id AS text)))"),
                    {"id": str(conversation_id)},
                ).scalar_one()
                if not acquired:
                    raise ConversationBusy(str(conversation_id))
                yield

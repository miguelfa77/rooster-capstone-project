import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from agent.turn import TurnState

TEST_DB = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")

DDL = Path(__file__).resolve().parents[2] / "sql" / "migrate_app_conversations.sql"


@pytest.fixture
def engine():
    eng = create_engine(TEST_DB)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS app CASCADE"))
        conn.execute(text(DDL.read_text()))
    yield eng
    eng.dispose()


@pytest.fixture
def store(engine):
    from api.store import ConversationStore

    return ConversationStore(engine)


def _state(n_messages: int) -> TurnState:
    state = TurnState(prompt_cache_key="rooster_abc")
    state.messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n_messages)]
    state.session_memory = {"a": 1}
    state.previous_execution_results = [{"tool": "select_metrics", "rows": [{"x": 1}]}]
    return state


def test_create_then_load_returns_empty_state(store):
    cid = store.create()
    state = store.load(cid)
    assert state.messages == [] and state.session_memory == {} and state.previous_execution_results == []
    assert state.prompt_cache_key.startswith("rooster_")


def test_save_appends_messages_and_round_trips(store):
    cid = store.create()
    store.save(cid, _state(2))
    loaded, expected = store.load(cid), _state(2)
    assert (loaded.messages, loaded.session_memory, loaded.previous_execution_results) == (
        expected.messages, expected.session_memory, expected.previous_execution_results
    )


def test_save_is_append_only(store, engine):
    cid = store.create()
    store.save(cid, _state(2))
    store.save(cid, _state(4))
    with engine.connect() as conn:
        positions = conn.execute(
            text("SELECT position FROM app.messages WHERE conversation_id = :c ORDER BY position"), {"c": cid}
        ).scalars().all()
    assert positions == [0, 1, 2, 3]
    assert store.load(cid).messages == _state(4).messages


def test_load_unknown_returns_none(store):
    assert store.load(uuid.uuid4()) is None


def test_lock_is_exclusive(store):
    from api.store import ConversationBusy

    cid = store.create()
    with store.lock(cid):
        with pytest.raises(ConversationBusy):
            with store.lock(cid):
                pass
    with store.lock(cid):
        pass  # released after the first block


def test_non_json_values_are_stored(store):
    from decimal import Decimal

    cid = store.create()
    state = _state(0)
    state.previous_execution_results = [{"v": Decimal("6.2")}]
    store.save(cid, state)
    assert store.load(cid).previous_execution_results[0]["v"] in (6.2, "6.2")

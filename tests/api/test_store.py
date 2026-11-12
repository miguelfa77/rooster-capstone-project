import os
import uuid
from datetime import datetime, timedelta, timezone
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


def test_lock_does_not_use_the_main_pool():
    from api.store import ConversationStore

    eng = create_engine(TEST_DB, pool_size=1, max_overflow=0, pool_timeout=1)
    try:
        store = ConversationStore(eng)
        cid = uuid.uuid4()
        with store.lock(cid):
            with eng.connect() as conn:  # would time out if the lock held the only pooled connection
                assert conn.execute(text("SELECT 1")).scalar_one() == 1
    finally:
        eng.dispose()


def test_lock_leaves_no_transaction_open(store, engine):
    cid = uuid.uuid4()
    with store.lock(cid):
        with engine.connect() as conn:
            states = conn.execute(
                text("SELECT state FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()")
            ).scalars().all()
    assert "idle in transaction" not in states


def test_lock_released_when_holder_connection_dies(store, engine):
    from api.store import ConversationBusy

    cid = uuid.uuid4()
    with engine.connect() as holder:
        holder.execute(text("SELECT pg_advisory_lock(hashtext(CAST(:id AS text)))"), {"id": str(cid)})
        holder.commit()
        with pytest.raises(ConversationBusy):
            with store.lock(cid):
                pass
        holder.invalidate()  # closes the DBAPI connection abruptly; Postgres drops its session locks
    with store.lock(cid):
        pass


def test_non_json_values_are_stored(store):
    from decimal import Decimal

    cid = store.create()
    state = _state(0)
    state.previous_execution_results = [{"v": Decimal("6.2"), "missing": float("nan")}]
    store.save(cid, state)
    assert store.load(cid).previous_execution_results[0] == {"v": 6.2, "missing": None}


T0 = datetime(2026, 11, 12, 10, 0, 5, tzinfo=timezone.utc)


def test_turn_starts_count_up_to_the_cap(store, engine):
    assert [store.try_start_turn(2, at=T0 + timedelta(seconds=i)) for i in range(3)] == [True, True, False]
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT minute, n FROM app.turn_starts")).all()
    assert rows == [(datetime(2026, 11, 12, 10, 0, tzinfo=timezone.utc), 3)]


def test_turn_starts_each_minute_is_independent(store):
    assert [store.try_start_turn(1, at=T0) for _ in range(2)] == [True, False]
    assert store.try_start_turn(1, at=T0 + timedelta(minutes=1)) is True


def test_turn_starts_default_to_the_database_clock(store, engine):
    assert store.try_start_turn(5) is True
    with engine.connect() as conn:
        n, age = conn.execute(text("SELECT n, now() - minute FROM app.turn_starts")).one()
    assert n == 1 and timedelta(0) <= age < timedelta(minutes=2)


def test_turn_starts_older_than_an_hour_are_deleted(store, engine):
    store.try_start_turn(5, at=T0)
    store.try_start_turn(5, at=T0 + timedelta(minutes=30))
    store.try_start_turn(5, at=T0 + timedelta(minutes=61))
    with engine.connect() as conn:
        minutes = conn.execute(text("SELECT minute FROM app.turn_starts ORDER BY minute")).scalars().all()
    assert [m.minute for m in minutes] == [30, 1]


def test_migration_is_idempotent(engine):
    with engine.begin() as conn:
        conn.execute(text(DDL.read_text()))

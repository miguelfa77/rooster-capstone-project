import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from agent.turn import TurnState

TEST_DB = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
DDL = SQL_DIR / "migrate_app_conversations.sql"
USERS_DDL = SQL_DIR / "migrate_app_users.sql"


@pytest.fixture
def engine():
    eng = create_engine(TEST_DB)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS app CASCADE"))
        conn.execute(text(DDL.read_text()))
        conn.execute(text(USERS_DDL.read_text()))
    yield eng
    eng.dispose()


@pytest.fixture
def store(engine):
    from api.store import ConversationStore

    return ConversationStore(engine)


@pytest.fixture
def uid(store):
    return store.upsert_user("ana@example.com", "Ana", None)["id"]


def _state(n_messages: int) -> TurnState:
    state = TurnState(prompt_cache_key="rooster_abc")
    state.messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n_messages)]
    state.session_memory = {"a": 1}
    state.previous_execution_results = [{"tool": "select_metrics", "rows": [{"x": 1}]}]
    return state


def test_create_then_load_returns_empty_state(store, uid):
    cid = store.create(uid)
    state = store.load(cid, uid)
    assert state.messages == [] and state.session_memory == {} and state.previous_execution_results == []
    assert state.prompt_cache_key.startswith("rooster_")


def test_save_appends_messages_and_round_trips(store, uid):
    cid = store.create(uid)
    store.save(cid, _state(2))
    loaded, expected = store.load(cid, uid), _state(2)
    assert (loaded.messages, loaded.session_memory, loaded.previous_execution_results) == (
        expected.messages, expected.session_memory, expected.previous_execution_results
    )


def test_save_is_append_only(store, engine, uid):
    cid = store.create(uid)
    store.save(cid, _state(2))
    store.save(cid, _state(4))
    with engine.connect() as conn:
        positions = conn.execute(
            text("SELECT position FROM app.messages WHERE conversation_id = :c ORDER BY position"), {"c": cid}
        ).scalars().all()
    assert positions == [0, 1, 2, 3]
    assert store.load(cid, uid).messages == _state(4).messages


def test_load_unknown_returns_none(store, uid):
    assert store.load(uuid.uuid4(), uid) is None


def test_lock_is_exclusive(store, uid):
    from api.store import ConversationBusy

    cid = store.create(uid)
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


def test_non_json_values_are_stored(store, uid):
    from decimal import Decimal

    cid = store.create(uid)
    state = _state(0)
    state.previous_execution_results = [{"v": Decimal("6.2"), "missing": float("nan")}]
    store.save(cid, state)
    assert store.load(cid, uid).previous_execution_results[0] == {"v": 6.2, "missing": None}


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
        conn.execute(text(USERS_DDL.read_text()))


def test_migration_deletes_ownerless_conversations_and_is_idempotent(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA app CASCADE"))
        conn.execute(text(DDL.read_text()))
        old = uuid.uuid4()
        conn.execute(text("INSERT INTO app.conversations (id, prompt_cache_key) VALUES (:id, 'k')"), {"id": old})
        conn.execute(text("INSERT INTO app.messages (conversation_id, position, body) VALUES (:id, 0, '{}')"), {"id": old})
    for _ in range(2):
        with engine.begin() as conn:
            conn.execute(text(USERS_DDL.read_text()))
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM app.conversations")).scalar_one() == 0
        assert conn.execute(text("SELECT count(*) FROM app.messages")).scalar_one() == 0
        nullable = conn.execute(text(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'app' AND table_name = 'conversations' AND column_name = 'user_id'"
        )).scalar_one()
        indexes = conn.execute(text(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = 'app' AND tablename = 'conversations'"
        )).scalars().all()
    assert nullable == "NO"
    assert any("(user_id, updated_at DESC)" in d for d in indexes)


# ── Users ─────────────────────────────────────────────────────────────────


def test_upsert_user_creates_then_updates_and_lowercases(store, engine):
    first = store.upsert_user("  Ana@Example.COM ", "Ana", None)
    assert set(first) == {"id", "email", "name", "image", "preferences"}
    assert first["email"] == "ana@example.com" and first["preferences"] == {}
    with engine.connect() as conn:
        seen_before = conn.execute(text("SELECT last_seen_at FROM app.users")).scalar_one()
    again = store.upsert_user("ana@example.com", "Ana M", "https://x/p.png")
    assert again["id"] == first["id"]
    assert (again["name"], again["image"]) == ("Ana M", "https://x/p.png")
    with engine.connect() as conn:
        n, seen_after = conn.execute(text("SELECT count(*), max(last_seen_at) FROM app.users")).one()
    assert n == 1 and seen_after > seen_before


def test_get_user(store, uid):
    assert store.get_user(uid)["email"] == "ana@example.com"
    assert store.get_user(uuid.uuid4()) is None


def test_update_preferences_is_a_shallow_merge(store, uid):
    assert store.update_preferences(uid, {"theme": "dark"}) == {"theme": "dark"}
    assert store.update_preferences(uid, {"chat_open": True}) == {"theme": "dark", "chat_open": True}
    assert store.update_preferences(uid, {"theme": "light"}) == {"theme": "light", "chat_open": True}
    assert store.get_user(uid)["preferences"] == {"theme": "light", "chat_open": True}


# ── Ownership ─────────────────────────────────────────────────────────────


def test_load_of_another_users_conversation_is_none(store, uid):
    other = store.upsert_user("luis@example.com", None, None)["id"]
    cid = store.create(uid)
    assert store.load(cid, other) is None
    assert store.summary(cid, other) is None
    assert store.load(cid, uid) is not None


def test_list_conversations_newest_first_only_mine_max_100(store, uid, engine):
    other = store.upsert_user("luis@example.com", None, None)["id"]
    ids = [store.create(uid) for _ in range(102)]
    store.create(other)
    with engine.begin() as conn:  # deterministic order: later index = newer
        for i, cid in enumerate(ids):
            conn.execute(text("UPDATE app.conversations SET updated_at = :t WHERE id = :id"),
                         {"t": T0 + timedelta(minutes=i), "id": cid})
    listed = store.list_conversations(uid)
    assert len(listed) == 100
    assert [c["id"] for c in listed] == list(reversed(ids))[:100]
    assert set(listed[0]) == {"id", "title", "updated_at"} and listed[0]["title"] is None


def test_rename_sets_title_bumps_updated_at_and_checks_owner(store, uid):
    other = store.upsert_user("luis@example.com", None, None)["id"]
    cid = store.create(uid)
    before = store.summary(cid, uid)["updated_at"]
    assert store.rename(cid, other, "no") is None
    row = store.rename(cid, uid, "Ruzafa")
    assert row == store.summary(cid, uid)
    assert row["id"] == cid and row["title"] == "Ruzafa" and row["updated_at"] > before
    assert store.rename(uuid.uuid4(), uid, "x") is None


def test_delete_checks_owner_and_removes_messages(store, uid, engine):
    other = store.upsert_user("luis@example.com", None, None)["id"]
    cid = store.create(uid)
    store.save(cid, _state(2))
    assert store.delete(cid, other) is False
    assert store.delete(cid, uid) is True
    assert store.load(cid, uid) is None
    assert store.delete(cid, uid) is False
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM app.messages")).scalar_one() == 0


def test_deleting_a_user_deletes_their_conversations(store, uid, engine):
    store.create(uid)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM app.users WHERE id = :id"), {"id": uid})
        assert conn.execute(text("SELECT count(*) FROM app.conversations")).scalar_one() == 0


def test_save_sets_title_once_from_first_user_message(store, uid):
    cid = store.create(uid)
    before = store.summary(cid, uid)["updated_at"]
    state = TurnState(prompt_cache_key="k")
    question = "  ¿Cuál es   el barrio\ncon mejor rentabilidad bruta de Valencia, para invertir hoy?"
    state.messages = [{"role": "user", "content": question}, {"role": "assistant", "summary": "Benimaclet"}]
    store.save(cid, state)
    first = store.summary(cid, uid)
    assert first["title"] == "¿Cuál es el barrio con mejor rentabilidad bruta de Valencia,"
    assert len(first["title"]) == 60 and first["updated_at"] > before
    state.messages = [{"role": "user", "content": "otra"}] + state.messages
    store.save(cid, state)
    assert store.summary(cid, uid)["title"] == first["title"]


def test_save_keeps_a_renamed_title(store, uid):
    cid = store.create(uid)
    store.rename(cid, uid, "Mi título")
    store.save(cid, _state(2))
    assert store.summary(cid, uid)["title"] == "Mi título"

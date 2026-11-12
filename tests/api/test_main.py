import itertools
import json
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from agent import ui_es as UI
from agent.turn import (
    DoneEvent,
    ErrorEvent,
    FollowUpsEvent,
    PrimitiveEvent,
    TextDeltaEvent,
    ThoughtEvent,
    TurnState,
)

KEY = "test-key"
USER_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
AUTH = {"X-Rooster-Key": KEY, "X-Rooster-User": str(USER_ID)}
T0 = datetime(2026, 11, 12, tzinfo=timezone.utc)


class FakeStore:
    """In-memory ConversationStore; titles come from the same conversation_title as the real one."""

    def __init__(self):
        self.users: dict[uuid.UUID, dict] = {}
        self.states: dict[uuid.UUID, TurnState] = {}
        self.meta: dict[uuid.UUID, dict] = {}  # cid -> {user_id, title, updated_at}
        self.busy: set[uuid.UUID] = set()
        self.locks_taken: list[uuid.UUID] = []
        self.turn_starts = 0  # one fixed window; minute rollover is covered in test_store.py
        self._clock = itertools.count(1)

    def _now(self):
        return T0 + timedelta(seconds=next(self._clock))

    def try_start_turn(self, cap):
        self.turn_starts += 1
        return self.turn_starts <= cap

    def upsert_user(self, email, name, image):
        email = email.strip().lower()
        user = next((u for u in self.users.values() if u["email"] == email), None)
        if user is None:
            user = {"id": uuid.uuid4(), "email": email, "preferences": {}}
            self.users[user["id"]] = user
        user.update(name=name, image=image)
        return dict(user)

    def get_user(self, user_id):
        user = self.users.get(user_id)
        return dict(user) if user else None

    def update_preferences(self, user_id, patch):
        self.users[user_id]["preferences"] = {**self.users[user_id]["preferences"], **patch}
        return dict(self.users[user_id]["preferences"])

    def create(self, user_id):
        cid = uuid.uuid4()
        self.states[cid] = TurnState.new()
        self.meta[cid] = {"user_id": user_id, "title": None, "updated_at": self._now()}
        return cid

    def _owned(self, cid, user_id):
        return cid in self.meta and self.meta[cid]["user_id"] == user_id

    def load(self, cid, user_id):
        return self.states[cid] if self._owned(cid, user_id) else None

    def summary(self, cid, user_id):
        if not self._owned(cid, user_id):
            return None
        return {"id": cid, "title": self.meta[cid]["title"], "updated_at": self.meta[cid]["updated_at"]}

    def list_conversations(self, user_id):
        mine = [self.summary(c, user_id) for c in self.meta if self._owned(c, user_id)]
        return sorted(mine, key=lambda s: s["updated_at"], reverse=True)[:100]

    def save(self, cid, state):
        from api.store import conversation_title

        self.states[cid] = state
        meta = self.meta[cid]
        meta["title"] = meta["title"] or conversation_title(state.messages)
        meta["updated_at"] = self._now()

    def rename(self, cid, user_id, title):
        if not self._owned(cid, user_id):
            return None
        self.meta[cid].update(title=title, updated_at=self._now())
        return self.summary(cid, user_id)

    def delete(self, cid, user_id):
        if not self._owned(cid, user_id):
            return False
        del self.meta[cid], self.states[cid]
        return True

    @contextmanager
    def lock(self, cid):
        from api.store import ConversationBusy

        if cid in self.busy:
            raise ConversationBusy(str(cid))
        self.locks_taken.append(cid)
        self.busy.add(cid)
        try:
            yield
        finally:
            self.busy.discard(cid)


def scripted_turn(user_input, state, ctx, emit):
    emit(ThoughtEvent(text="Pensando"))
    emit(TextDeltaEvent(text="Hola"))
    emit(PrimitiveEvent(primitive={"kind": "bar", "data": []}))
    emit(FollowUpsEvent(items=["¿Y en alquiler?"]))
    emit(DoneEvent())
    return replace(state, messages=state.messages + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "summary": "Hola"},
    ])


def failing_turn(user_input, state, ctx, emit):
    emit(ErrorEvent(reason="boom", suggestion=UI.CHAT_EXCEPTION_SUGGESTION))
    emit(DoneEvent())
    return replace(state, messages=state.messages + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "graceful_fallback": {"reason": "boom"}},
    ])


@pytest.fixture
def store():
    s = FakeStore()
    s.users[USER_ID] = {"id": USER_ID, "email": "ana@example.com", "name": "Ana", "image": None, "preferences": {}}
    return s


@pytest.fixture
def client(monkeypatch, store):
    import api.main as main

    monkeypatch.setenv("ROOSTER_API_KEY", KEY)
    monkeypatch.setattr(main, "run_turn", scripted_turn)
    main.app.dependency_overrides[main.get_store] = lambda: store
    main.app.dependency_overrides[main.get_turn_context] = lambda: None
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def _events(response):
    frames = [f for f in response.text.split("\n\n") if f.strip()]
    return [(f.split("\n")[0].removeprefix("event: "), json.loads(f.split("\n")[1].removeprefix("data: "))) for f in frames]


def _create(client):
    return client.post("/conversations", headers=AUTH).json()["id"]


def test_health_no_auth(client):
    assert client.get("/health").json() == {"ok": True}


def test_missing_key_401(client):
    assert client.post("/conversations").status_code == 401


def test_wrong_key_401(client):
    assert client.post("/conversations", headers={"X-Rooster-Key": "nope"}).status_code == 401


def test_unset_server_key_rejects_everything(client, monkeypatch):
    monkeypatch.delenv("ROOSTER_API_KEY")
    assert client.post("/conversations", headers={"X-Rooster-Key": ""}).status_code == 401


def test_create_conversation_201(client, store):
    r = client.post("/conversations", headers=AUTH)
    assert r.status_code == 201
    assert uuid.UUID(r.json()["id"]) in store.states


def test_message_streams_events_in_order(client):
    cid = _create(client)
    r = client.post(f"/conversations/{cid}/messages", json={"text": "hola"}, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r)
    assert [t for t, _ in events] == ["thought", "text_delta", "primitive", "follow_ups", "done"]
    assert events[2][1]["primitive"]["kind"] == "bar"
    history = client.get(f"/conversations/{cid}", headers=AUTH).json()
    assert history["id"] == cid
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]


def test_done_is_sent_after_the_turn_is_saved(client, store, monkeypatch):
    import api.main as main

    def turn(user_input, state, ctx, emit):
        new = scripted_turn(user_input, state, ctx, lambda e: None)
        emit(DoneEvent())  # run_turn's own done is held back until after the save
        return new

    monkeypatch.setattr(main, "run_turn", turn)
    cid = _create(client)
    events = _events(client.post(f"/conversations/{cid}/messages", json={"text": "hola"}, headers=AUTH))
    assert [t for t, _ in events] == ["done"]
    assert len(store.states[uuid.UUID(cid)].messages) == 2


def test_unknown_conversation_404(client):
    missing = uuid.uuid4()
    assert client.post(f"/conversations/{missing}/messages", json={"text": "hola"}, headers=AUTH).status_code == 404
    assert client.get(f"/conversations/{missing}", headers=AUTH).status_code == 404


def test_malformed_conversation_id_404(client):
    assert client.get("/conversations/not-a-uuid", headers=AUTH).status_code == 404


def test_busy_conversation_409(client, store):
    cid = _create(client)
    store.busy.add(uuid.UUID(cid))
    r = client.post(f"/conversations/{cid}/messages", json={"text": "hola"}, headers=AUTH)
    assert r.status_code == 409 and r.json() == {"detail": "busy"}


def _post(client, cid, text="hola"):
    return client.post(f"/conversations/{cid}/messages", json={"text": text}, headers=AUTH)


def test_turns_under_the_cap_are_accepted(client, monkeypatch):
    monkeypatch.setenv("ROOSTER_MAX_TURNS_PER_MINUTE", "3")
    cid = _create(client)
    assert [_post(client, cid).status_code for _ in range(3)] == [200, 200, 200]


def test_turn_over_the_cap_429(client, monkeypatch):
    monkeypatch.setenv("ROOSTER_MAX_TURNS_PER_MINUTE", "2")
    cid = _create(client)
    assert [_post(client, cid).status_code for _ in range(2)] == [200, 200]
    r = _post(client, cid)
    assert r.status_code == 429 and r.json() == {"detail": "rate_limited"}


def test_turn_limit_zero_disables(client, store, monkeypatch):
    monkeypatch.setenv("ROOSTER_MAX_TURNS_PER_MINUTE", "0")
    cid = _create(client)
    assert all(_post(client, cid).status_code == 200 for _ in range(40))
    assert store.turn_starts == 0  # the counter is not touched when the cap is off


@pytest.mark.parametrize("raw", ["abc", "30.5", " "])
def test_invalid_turn_limit_falls_back_to_30(client, monkeypatch, raw):
    monkeypatch.setenv("ROOSTER_MAX_TURNS_PER_MINUTE", raw)
    cid = _create(client)
    assert all(_post(client, cid).status_code == 200 for _ in range(30))
    assert _post(client, cid).status_code == 429


def test_negative_turn_limit_disables(client, monkeypatch):
    monkeypatch.setenv("ROOSTER_MAX_TURNS_PER_MINUTE", "-1")
    cid = _create(client)
    assert all(_post(client, cid).status_code == 200 for _ in range(40))


def test_lock_released_after_turn(client, store):
    cid = _create(client)
    client.post(f"/conversations/{cid}/messages", json={"text": "uno"}, headers=AUTH)
    r = client.post(f"/conversations/{cid}/messages", json={"text": "dos"}, headers=AUTH)
    assert r.status_code == 200 and not store.busy


@pytest.mark.parametrize("text", ["", "   ", "x" * 2001])
def test_bad_text_422(client, text):
    cid = _create(client)
    assert client.post(f"/conversations/{cid}/messages", json={"text": text}, headers=AUTH).status_code == 422


def test_turn_exception_streams_error_then_done(client, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "run_turn", failing_turn)
    cid = _create(client)
    events = _events(client.post(f"/conversations/{cid}/messages", json={"text": "hola"}, headers=AUTH))
    assert [t for t, _ in events] == ["error", "done"]
    monkeypatch.setattr(main, "run_turn", scripted_turn)
    assert client.post(f"/conversations/{cid}/messages", json={"text": "otra"}, headers=AUTH).status_code == 200


def test_lock_released_when_load_fails_after_locking(client, store, monkeypatch):
    cid = _create(client)
    real_load, calls = store.load, []

    def load_then_fail(c, user_id):
        calls.append(c)
        if len(calls) > 1:  # the second load runs while the lock is held
            raise RuntimeError("db down")
        return real_load(c, user_id)

    monkeypatch.setattr(store, "load", load_then_fail)
    with pytest.raises(RuntimeError):
        _post(client, cid)
    assert not store.busy


def test_save_failure_streams_error_then_done(client, store, monkeypatch):
    def broken_save(cid, state):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "save", broken_save)
    cid = _create(client)
    events = _events(client.post(f"/conversations/{cid}/messages", json={"text": "hola"}, headers=AUTH))
    assert [t for t, _ in events][-2:] == ["error", "done"]
    assert not store.busy


def test_openapi_export_contains_event_schemas(tmp_path):
    from api.export_openapi import build_openapi

    schemas = build_openapi()["components"]["schemas"]
    for name in ("ThoughtEvent", "TextDeltaEvent", "PrimitiveEvent", "FollowUpsEvent", "ErrorEvent",
                 "DoneEvent", "TurnEvent", "ChoroplethDescriptor", "CompositePrimitive", "PrimitiveBlock",
                 "UserOut", "Preferences", "ConversationSummary"):
        assert name in schemas, name


def test_eval_runner_via_api_chains_follow_ups(client, store):
    from tests.eval.run_eval import run_all_via_api

    questions = [
        {"id": "a", "question": "uno"},
        {"id": "b", "question": "dos", "follow_up_of": "a"},
        {"id": "c", "question": "tres"},
    ]
    entries = run_all_via_api(questions, client, KEY)
    assert [e["id"] for e in entries] == ["a", "b", "c"]
    assert entries[0]["primitive_kinds"] == ["text"] and entries[0]["text"] == "Hola"
    sizes = sorted(len(s.messages) for s in store.states.values())
    assert sizes == [2, 4]  # "b" continued "a"'s conversation; "c" started its own
    owners = {m["user_id"] for m in store.meta.values()}
    assert [store.users[o]["email"] for o in owners] == ["eval@rooster.local"]

import json
import uuid
from contextlib import contextmanager
from dataclasses import replace

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
AUTH = {"X-Rooster-Key": KEY}


class FakeStore:
    def __init__(self):
        self.states: dict[uuid.UUID, TurnState] = {}
        self.busy: set[uuid.UUID] = set()

    def create(self):
        cid = uuid.uuid4()
        self.states[cid] = TurnState.new()
        return cid

    def load(self, cid):
        return self.states.get(cid)

    def save(self, cid, state):
        self.states[cid] = state

    @contextmanager
    def lock(self, cid):
        from api.store import ConversationBusy

        if cid in self.busy:
            raise ConversationBusy(str(cid))
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
    return FakeStore()


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
                 "DoneEvent", "TurnEvent", "ChoroplethDescriptor", "CompositePrimitive", "PrimitiveBlock"):
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

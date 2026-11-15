"""Users, preferences and user-owned conversations over HTTP (fake store from test_main)."""

import uuid

import pytest

from tests.api.test_main import AUTH, KEY, USER_ID, client, store  # noqa: F401  (fixtures)

KEY_ONLY = {"X-Rooster-Key": KEY}


def _as(user_id) -> dict:
    return {"X-Rooster-Key": KEY, "X-Rooster-User": str(user_id)}


def _other_user(client) -> dict:
    uid = client.put("/users", json={"email": "luis@example.com"}, headers=KEY_ONLY).json()["id"]
    return _as(uid)


def _create(client, headers=AUTH) -> str:
    return client.post("/conversations", headers=headers).json()["id"]


def _say(client, cid, text, headers=AUTH):
    return client.post(f"/conversations/{cid}/messages", json={"text": text}, headers=headers)


def test_put_users_upserts_and_lowercases(client, store):
    r = client.put("/users", json={"email": "  Luis@Example.COM ", "name": "Luis"}, headers=KEY_ONLY)
    assert r.status_code == 200
    first = r.json()
    assert first["email"] == "luis@example.com" and first["name"] == "Luis"
    assert first["image"] is None and first["preferences"] == {}
    again = client.put(
        "/users", json={"email": "luis@example.com", "name": "Luis M", "image": "https://x/p.png"}, headers=KEY_ONLY
    ).json()
    assert again["id"] == first["id"]
    assert (again["name"], again["image"]) == ("Luis M", "https://x/p.png")


@pytest.mark.parametrize("email", ["", "   ", "no-at-sign"])
def test_put_users_rejects_bad_email_422(client, email):
    assert client.put("/users", json={"email": email}, headers=KEY_ONLY).status_code == 422


def test_put_users_requires_key(client):
    assert client.put("/users", json={"email": "a@b.c"}).status_code == 401


def test_me_requires_user_header_401(client):
    r = client.get("/me", headers=KEY_ONLY)
    assert r.status_code == 401 and r.json() == {"detail": "user_required"}


@pytest.mark.parametrize("path,method", [
    ("/me", "get"), ("/conversations", "get"), ("/conversations", "post"),
])
def test_user_routes_require_user_header_401(client, path, method):
    r = getattr(client, method)(path, headers=KEY_ONLY)
    assert r.status_code == 401 and r.json() == {"detail": "user_required"}


@pytest.mark.parametrize("user", [str(uuid.uuid4()), "not-a-uuid"])
def test_unknown_user_401(client, user):
    r = client.get("/me", headers={"X-Rooster-Key": KEY, "X-Rooster-User": user})
    assert r.status_code == 401 and r.json() == {"detail": "user_required"}


def test_missing_key_wins_over_missing_user(client):
    assert client.get("/me").json() == {"detail": "unauthorized"}


def test_me_returns_user(client):
    r = client.get("/me", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"id": str(USER_ID), "email": "ana@example.com", "name": "Ana", "image": None, "preferences": {}}


def test_preferences_merge_and_reject_unknown_key_422(client):
    assert client.patch("/me/preferences", json={"theme": "dark"}, headers=AUTH).json() == {"theme": "dark"}
    merged = client.patch("/me/preferences", json={"chat_open": True, "ticker_visible": False}, headers=AUTH)
    assert merged.status_code == 200
    assert merged.json() == {"theme": "dark", "chat_open": True, "ticker_visible": False}
    assert client.patch("/me/preferences", json={"font": "big"}, headers=AUTH).status_code == 422
    assert client.patch("/me/preferences", json={"theme": "blue"}, headers=AUTH).status_code == 422
    assert client.get("/me", headers=AUTH).json()["preferences"] == {
        "theme": "dark", "chat_open": True, "ticker_visible": False
    }


def test_conversations_listed_newest_first_only_mine(client):
    other = _other_user(client)
    first, second = _create(client), _create(client)
    _create(client, other)
    _say(client, first, "hola")  # bumps `first` above `second`
    listed = client.get("/conversations", headers=AUTH).json()
    assert [c["id"] for c in listed] == [first, second]
    assert listed[0]["title"] == "hola" and listed[1]["title"] is None
    assert {"id", "title", "updated_at"} == set(listed[0])


def test_other_users_conversation_is_404(client, store):
    cid = _create(client)
    other = _other_user(client)
    assert client.get(f"/conversations/{cid}", headers=other).status_code == 404
    assert client.patch(f"/conversations/{cid}", json={"title": "mío"}, headers=other).status_code == 404
    assert client.delete(f"/conversations/{cid}", headers=other).status_code == 404
    assert _say(client, cid, "hola", other).status_code == 404
    assert store.turn_starts == 0 and store.locks_taken == []
    assert client.get(f"/conversations/{cid}", headers=AUTH).status_code == 200


def test_rename_trims_and_validates_length(client):
    cid = _create(client)
    r = client.patch(f"/conversations/{cid}", json={"title": "  Ruzafa vs Benimaclet  "}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["id"] == cid and r.json()["title"] == "Ruzafa vs Benimaclet"
    assert client.get(f"/conversations/{cid}", headers=AUTH).json()["title"] == "Ruzafa vs Benimaclet"
    for bad in ["", "   ", "x" * 121]:
        assert client.patch(f"/conversations/{cid}", json={"title": bad}, headers=AUTH).status_code == 422
    assert client.patch(f"/conversations/{cid}", json={"title": "x" * 120}, headers=AUTH).status_code == 200


def test_rename_does_not_move_the_conversation_up_the_recent_list(client):
    first, second = _create(client), _create(client)
    before = client.get("/conversations", headers=AUTH).json()
    assert [c["id"] for c in before] == [second, first]
    renamed = client.patch(f"/conversations/{first}", json={"title": "Renombrada"}, headers=AUTH).json()
    after = client.get("/conversations", headers=AUTH).json()
    assert [c["id"] for c in after] == [second, first]
    assert renamed["updated_at"] == next(c for c in before if c["id"] == first)["updated_at"]


def test_delete_then_404(client):
    cid = _create(client)
    assert client.delete(f"/conversations/{cid}", headers=AUTH).status_code == 204
    assert client.get(f"/conversations/{cid}", headers=AUTH).status_code == 404
    assert client.delete(f"/conversations/{cid}", headers=AUTH).status_code == 404
    assert client.get("/conversations", headers=AUTH).json() == []


def test_title_set_from_first_message_60_chars(client):
    cid = _create(client)
    assert client.get(f"/conversations/{cid}", headers=AUTH).json()["title"] is None
    long = "¿Cuál   es el\nbarrio con mejor rentabilidad bruta de Valencia para invertir hoy?"
    _say(client, cid, long)
    _say(client, cid, "otra pregunta")
    title = client.get(f"/conversations/{cid}", headers=AUTH).json()["title"]
    assert title == "¿Cuál es el barrio con mejor rentabilidad bruta de Valencia"  # 60-char cut, right-trimmed


def test_stored_unknown_preference_key_is_ignored_on_output(client, store):
    store.users[USER_ID]["preferences"] = {"theme": "dark", "legacy_flag": 1}
    assert client.get("/me", headers=AUTH).json()["preferences"] == {"theme": "dark"}
    r = client.put("/users", json={"email": "ana@example.com", "name": "Ana"}, headers=KEY_ONLY)
    assert r.status_code == 200 and r.json()["preferences"] == {"theme": "dark"}
    r = client.patch("/me/preferences", json={"chat_open": True}, headers=AUTH)
    assert r.status_code == 200 and r.json() == {"theme": "dark", "chat_open": True}


def test_conversation_title_has_no_trailing_space():
    from api.store import conversation_title

    text = "a" * 59 + " bcd"  # the 60-char cut ends on the space
    assert conversation_title([{"role": "user", "content": text}]) == "a" * 59


def test_conversation_title(client):
    from api.store import conversation_title

    assert conversation_title([]) is None
    assert conversation_title([{"role": "assistant", "summary": "x"}]) is None
    assert conversation_title([{"role": "assistant"}, {"role": "user", "content": "  a \n b  "}]) == "a b"


def test_intel_needs_no_user(client, monkeypatch):
    import agent.intel_data as intel
    import api.main as main

    monkeypatch.setattr(intel, "market_framing", lambda: {"city_median_yield": 5.1})
    main.clear_intel_cache()
    try:
        assert client.get("/intel/framing", headers=KEY_ONLY).status_code == 200
    finally:
        main.clear_intel_cache()
    assert client.get("/health").status_code == 200

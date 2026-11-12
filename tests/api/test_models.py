import json
from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from agent.turn import TurnState
from api.models import PrimitivePayload
from tests.api.test_main import AUTH, USER_ID, client, store  # noqa: F401  (fixtures)

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "primitive_payloads.json").read_text(encoding="utf-8"))
PAYLOAD = TypeAdapter(PrimitivePayload)


def test_fixture_payloads_validate():
    validated = [PAYLOAD.validate_python(p) for p in FIXTURE["payloads"] + [FIXTURE["folium_code"]]]
    assert {p.kind for p in validated} == {
        "text", "kpi", "table", "code", "choropleth", "bar", "scatter", "line", "point_map", "composite",
    }
    assert validated[-1].output_type == "folium_html"


def test_payload_ignores_extra_keys():
    bar = next(p for p in FIXTURE["payloads"] if p["kind"] == "bar")
    assert PAYLOAD.validate_python({**bar, "extra": 1}).kind == "bar"


def test_get_conversation_filters_internal_fields(client, store):  # noqa: F811
    cid = store.create(USER_ID)
    bar = next(p for p in FIXTURE["payloads"] if p["kind"] == "bar")
    assistant = {
        "role": "assistant",
        "summary": "Benimaclet lidera.",
        "blocks": [{"kind": "text", "content": "Benimaclet lidera."}, bar],
        "primitive_payloads": [bar],
        "follow_ups": ["¿Y en alquiler?"],
        "execution_results": [{"tool": "select_metrics", "success": True, "rows": bar["data"]}],
        "sql": "SELECT 1",
        "rows": bar["data"],
    }
    store.states[cid] = replace(TurnState.new(), messages=[{"role": "user", "content": "hola"}, assistant])
    r = client.get(f"/conversations/{cid}", headers=AUTH)
    assert r.status_code == 200
    msg = r.json()["messages"][1]
    assert not {"execution_results", "sql", "rows"} & msg.keys()
    assert [b["kind"] for b in msg["blocks"]] == ["text", "bar"]
    assert msg["blocks"][1]["data"] == bar["data"]


def test_openapi_has_contract_models():
    from api.export_openapi import build_openapi

    schemas = build_openapi()["components"]["schemas"]
    for name in ("PrimitivePayload", "CodePayload", "BarPayload", "ConversationOut", "AssistantMessage", "NeighborhoodsOut"):
        assert name in schemas, name
    assert schemas["PrimitiveEvent"]["properties"]["primitive"] == {"$ref": "#/components/schemas/PrimitivePayload"}


def test_openapi_json_is_fresh():
    from api.export_openapi import OUT_PATH, build_openapi

    assert json.loads(OUT_PATH.read_text(encoding="utf-8")) == build_openapi()


def test_text_kpi_table_payloads_ignore_extra_keys():
    for kind in ("text", "kpi", "table"):
        p = next(p for p in FIXTURE["payloads"] if p["kind"] == kind)
        assert PAYLOAD.validate_python({**p, "extra": 1}).kind == kind


def test_get_conversation_tolerates_extra_key_on_text_block(client, store):  # noqa: F811
    cid = store.create(USER_ID)
    assistant = {"role": "assistant", "blocks": [{"kind": "text", "content": "Hola", "extra": 1}]}
    store.states[cid] = replace(TurnState.new(), messages=[{"role": "user", "content": "hola"}, assistant])
    r = client.get(f"/conversations/{cid}", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["messages"][1]["blocks"][0]["content"] == "Hola"

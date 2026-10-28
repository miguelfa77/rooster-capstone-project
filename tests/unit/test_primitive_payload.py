import json
from datetime import datetime
from decimal import Decimal

import pytest

import agent.primitive_payload as pp
from agent.primitive_payload import build_primitive_payload

GEO_ROWS = [{"neighborhood_name": "A", "geom": "{}", "yield": 6.0}]
LABEL_ROWS = [{"neighborhood_name": "A", "yield": 6.0}]
POINT_ROWS = [{"lat": 39.4, "lng": -0.37, "price": 200000}]


def _ok(rows, tool="t"):
    return {"tool": tool, "success": True, "rows": rows}


def test_choropleth_gets_rows_with_geom_and_metric():
    results = [_ok(LABEL_ROWS), _ok(GEO_ROWS)]
    out = build_primitive_payload({"kind": "choropleth", "metric": "yield"}, results)
    assert out["data"] == GEO_ROWS


def test_bar_picks_result_with_columns():
    results = [_ok([{"other": 1}]), _ok(LABEL_ROWS)]
    out = build_primitive_payload({"kind": "bar", "value_field": "yield", "label_field": "neighborhood_name"}, results)
    assert out["data"] == LABEL_ROWS


def test_scatter_and_line_use_x_and_y():
    rows = [{"x": 1, "y": 2}]
    for kind in ("scatter", "line"):
        assert build_primitive_payload({"kind": kind, "x_field": "x", "y_field": "y"}, [_ok(rows)])["data"] == rows


def test_point_map_uses_lat_lng():
    out = build_primitive_payload({"kind": "point_map"}, [_ok(LABEL_ROWS), _ok(POINT_ROWS)])
    assert out["data"] == POINT_ROWS


def test_failed_results_are_skipped():
    results = [{"tool": "t", "success": False, "rows": LABEL_ROWS}, _ok(LABEL_ROWS, tool="u")]
    out = build_primitive_payload({"kind": "bar", "value_field": "yield", "label_field": "neighborhood_name"}, results)
    assert out["data"] == LABEL_ROWS


def test_missing_columns_gives_empty_data():
    out = build_primitive_payload({"kind": "point_map"}, [_ok(LABEL_ROWS)])
    assert out["data"] == []


def test_text_kpi_table_unchanged():
    for prim in ({"kind": "text", "content": "x"}, {"kind": "kpi", "label": "l", "value": "1"},
                 {"kind": "table", "rows_json": "[]", "columns": []}):
        assert build_primitive_payload(prim, [_ok(LABEL_ROWS)]) == prim


def test_input_primitive_is_not_mutated():
    prim = {"kind": "point_map"}
    build_primitive_payload(prim, [_ok(POINT_ROWS)])
    assert prim == {"kind": "point_map"}


def test_code_success_attaches_output(monkeypatch):
    sent = {}

    def fake_sandbox(code, data):
        sent["data"] = data
        return {"success": True, "output_type": "plotly_json", "output": '{"data": []}'}

    monkeypatch.setattr(pp, "call_sandbox", fake_sandbox)
    out = build_primitive_payload({"kind": "code", "visualization_type": "plotly", "code": "c"}, [_ok(LABEL_ROWS)])
    assert out["output_type"] == "plotly_json" and out["output"] == '{"data": []}'
    assert sent["data"] == LABEL_ROWS


def test_code_failure_falls_back_to_table(monkeypatch):
    monkeypatch.setattr(pp, "call_sandbox", lambda code, data: {"success": False, "error": "boom"})
    out = build_primitive_payload({"kind": "code", "visualization_type": "folium", "code": "c"}, [_ok(LABEL_ROWS)])
    assert out == {"kind": "table", "rows_json": json.dumps(LABEL_ROWS), "columns": []}


def test_code_without_rows_falls_back_to_empty_table(monkeypatch):
    monkeypatch.setattr(pp, "call_sandbox", lambda code, data: pytest.fail("sandbox must not be called"))
    out = build_primitive_payload({"kind": "code", "visualization_type": "plotly", "code": "c"}, [])
    assert out == {"kind": "table", "rows_json": "[]", "columns": []}


def test_composite_recurses():
    prim = {"kind": "composite", "heading": "h", "blocks": [{"kind": "text", "content": "x"}, {"kind": "point_map"}]}
    out = build_primitive_payload(prim, [_ok(POINT_ROWS)])
    assert out["blocks"][0] == {"kind": "text", "content": "x"}
    assert out["blocks"][1]["data"] == POINT_ROWS


def test_payload_is_json_serializable():
    from shapely.geometry import Point

    from api.sse import to_json

    rows = [{"lat": 1.0, "lng": 2.0, "v": Decimal("6.2"), "t": datetime(2026, 1, 1), "g": Point(0, 1)}]
    out = build_primitive_payload({"kind": "point_map"}, [_ok(rows)])
    assert json.loads(to_json(out))["data"][0]["v"] == 6.2

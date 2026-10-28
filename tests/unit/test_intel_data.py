import subprocess
import sys

import pandas as pd
import pytest

import agent.intel_data as intel


def test_geojson_rejects_unknown_metric(monkeypatch):
    monkeypatch.setattr(intel, "get_pg_conn", lambda: pytest.fail("must validate before connecting"))
    with pytest.raises(ValueError):
        intel.neighborhood_geojson("price; DROP TABLE x")


@pytest.mark.parametrize("operation,expected_params", [("bogus", None), ("All", None), ("venta", ["venta"])])
def test_listings_operation_filter(monkeypatch, operation, expected_params):
    seen = {}

    def fake_read(q, engine, params=None):
        seen["q"], seen["params"] = q, params
        return pd.DataFrame()

    monkeypatch.setattr(intel, "analytics_views_available", lambda: {"neighborhood_metrics": True})
    monkeypatch.setattr(intel, "get_pg_engine", lambda: None)
    monkeypatch.setattr(intel.pd, "read_sql_query", fake_read)
    intel.listings(operation)
    assert seen["params"] == expected_params
    assert ("l.operation = %s" in seen["q"]) == (expected_params is not None)


def test_framing_none_without_analytics_views(monkeypatch):
    monkeypatch.setattr(intel, "analytics_views_available", lambda: {"neighborhood_metrics": False})
    assert intel.market_framing() is None


def test_ranking_empty_without_analytics_views(monkeypatch):
    monkeypatch.setattr(intel, "analytics_views_available", lambda: {"neighborhood_metrics": False})
    assert intel.neighborhood_ranking().empty and intel.yield_liquidity().empty


def test_intel_module_has_no_streamlit():
    code = "import sys, agent.intel_data; print('streamlit' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"

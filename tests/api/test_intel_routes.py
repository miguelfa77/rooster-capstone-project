import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent import ui_es as UI

KEY = "test-key"
AUTH = {"X-Rooster-Key": KEY}
FC = {"type": "FeatureCollection", "features": []}
DF = pd.DataFrame([{"neighborhood_name": "A", "yield_pct": float("nan"), "score": 4.2}])


@pytest.fixture
def client(monkeypatch):
    import agent.intel_data as intel
    import api.main as main

    monkeypatch.setenv("ROOSTER_API_KEY", KEY)
    monkeypatch.setattr(intel, "market_framing", lambda: {"city_median_yield": 5.1})
    monkeypatch.setattr(intel, "neighborhood_geojson", lambda metric: (FC, 1.0, 9.0))
    for name in ("transit_overlay", "tourism_overlay", "neighborhood_ranking", "yield_liquidity"):
        monkeypatch.setattr(intel, name, lambda: DF)
    monkeypatch.setattr(intel, "listings", lambda operation: DF.assign(op=operation))
    monkeypatch.setattr(intel, "parcels_geojson", lambda: FC)
    main.clear_intel_cache()
    yield TestClient(main.app)
    main.clear_intel_cache()


ROWS = [{"neighborhood_name": "A", "yield_pct": None, "score": 4.2}]


@pytest.mark.parametrize("path,expected", [
    ("/intel/framing", {"city_median_yield": 5.1}),
    ("/intel/overlays/transit", ROWS),
    ("/intel/overlays/tourism", ROWS),
    ("/intel/parcels", FC),
    ("/intel/ranking", ROWS),
    ("/intel/yield-liquidity", ROWS),
])
def test_routes_return_payload(client, path, expected):
    r = client.get(path, headers=AUTH)
    assert r.status_code == 200 and r.json() == expected


def test_neighborhoods_returns_geojson_and_range(client):
    metric = next(iter(UI.CHOROPLETH_METRICS.values()))
    r = client.get("/intel/neighborhoods", params={"metric": metric}, headers=AUTH)
    assert r.json() == {"geojson": FC, "vmin": 1.0, "vmax": 9.0}


def test_neighborhoods_bad_metric_422(client):
    assert client.get("/intel/neighborhoods", params={"metric": "nope"}, headers=AUTH).status_code == 422


def test_listings_passes_operation(client):
    assert client.get("/intel/listings", params={"operation": "venta"}, headers=AUTH).json()[0]["op"] == "venta"
    assert client.get("/intel/listings", params={"operation": "bogus"}, headers=AUTH).status_code == 422


def test_intel_requires_key_401(client):
    assert client.get("/intel/framing").status_code == 401


def test_framing_none_returns_null(client, monkeypatch):
    import agent.intel_data as intel

    monkeypatch.setattr(intel, "market_framing", lambda: None)
    import api.main as main

    main.clear_intel_cache()
    r = client.get("/intel/framing", headers=AUTH)
    assert r.status_code == 200 and r.json() is None


def test_responses_are_cached(client, monkeypatch):
    import agent.intel_data as intel

    calls = []
    monkeypatch.setattr(intel, "neighborhood_ranking", lambda: calls.append(1) or DF)
    client.get("/intel/ranking", headers=AUTH)
    client.get("/intel/ranking", headers=AUTH)
    assert len(calls) == 1

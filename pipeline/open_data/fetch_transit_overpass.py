"""
Fetch public transport nodes in Valencia via Overpass API, load raw + core.transit_stops,
assign neighborhood_id, refresh core.listings.nearest_stop_m.

Run from repo root:
  python -m pipeline.open_data.fetch_transit_overpass
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.raw.db_utils import get_pg_conn

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Valencia bounding box (south, west, north, east) — ~ city + suburbs
QUERY = """
[out:json][timeout:120];
(
  node["public_transport"="stop_position"](39.40,-0.45,39.55,-0.30);
  node["public_transport"="platform"](39.40,-0.45,39.55,-0.30);
  node["highway"="bus_stop"](39.40,-0.45,39.55,-0.30);
  node["railway"="station"](39.40,-0.45,39.55,-0.30);
  node["railway"="halt"](39.40,-0.45,39.55,-0.30);
  node["railway"="tram_stop"](39.40,-0.45,39.55,-0.30);
  node["station"="subway"](39.40,-0.45,39.55,-0.30);
);
out body;
"""


def infer_stop_type(tags: dict) -> str:
    t = {k.lower(): v for k, v in tags.items()}
    if t.get("railway") in ("station", "halt"):
        return "rail"
    if t.get("station") == "subway" or t.get("subway") == "yes":
        return "metro"
    if t.get("railway") in ("tram_stop", "tram") or t.get("tram") == "yes":
        return "tram"
    if t.get("highway") == "bus_stop" or t.get("bus") == "yes":
        return "bus"
    if t.get("public_transport") in ("stop_position", "platform"):
        return "transit"
    return t.get("public_transport") or "other"


def fetch_overpass() -> dict:
    req = urllib.request.Request(
        OVERPASS_URL,
        data=QUERY.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def refresh_nearest_stop_m(cur) -> None:
    cur.execute(
        """
        UPDATE core.listings
        SET nearest_stop_m = NULL
        WHERE NOT EXISTS (SELECT 1 FROM core.transit_stops LIMIT 1);
        """
    )
    cur.execute(
        """
        UPDATE core.listings l
        SET nearest_stop_m = (
            SELECT ROUND(
                ST_Distance(
                    ST_SetSRID(ST_MakePoint(l.lng, l.lat), 4326)::geography,
                    t.geom::geography
                )
            )::integer
            FROM core.transit_stops t
            ORDER BY ST_SetSRID(ST_MakePoint(l.lng, l.lat), 4326) <-> t.geom
            LIMIT 1
        )
        WHERE l.lat IS NOT NULL
          AND l.lng IS NOT NULL
          AND EXISTS (SELECT 1 FROM core.transit_stops LIMIT 1);
        """
    )
    cur.execute(
        """
        UPDATE core.listings
        SET nearest_stop_m = NULL
        WHERE lat IS NULL OR lng IS NULL;
        """
    )


def main() -> None:
    print("Fetching Overpass…")
    try:
        data = fetch_overpass()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"Overpass fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    elements = data.get("elements") or []
    rows: list[tuple] = []
    for el in elements:
        if el.get("type") != "node":
            continue
        lat, lon = el.get("lat"), el.get("lon")
        if lat is None or lon is None:
            continue
        osm_id = int(el["id"])
        tags = el.get("tags") or {}
        name = (tags.get("name") or tags.get("ref") or "")[:500]
        stop_type = infer_stop_type(tags)
        rows.append((osm_id, name, stop_type, float(lat), float(lon)))

    print(f"Parsed {len(rows)} stops")

    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        cur.execute("TRUNCATE raw.transit_stops;")
        cur.execute("TRUNCATE core.transit_stops;")

        if rows:
            cur.executemany(
                """
                INSERT INTO raw.transit_stops (osm_id, name, stop_type, lat, lng)
                VALUES (%s, %s, %s, %s, %s)
                """,
                rows,
            )

            cur.executemany(
                """
                INSERT INTO core.transit_stops (osm_id, name, stop_type, lat, lng, geom, neighborhood_id)
                VALUES (
                    %s, %s, %s, %s, %s,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                    NULL
                )
                """,
                [(r[0], r[1], r[2], r[3], r[4], r[4], r[3]) for r in rows],
            )

            cur.execute(
                """
                UPDATE core.transit_stops t
                SET neighborhood_id = (
                    SELECT n.id
                    FROM core.neighborhoods n
                    WHERE ST_Within(t.geom, n.geom)
                    LIMIT 1
                );
                """
            )

        print("Refreshing nearest_stop_m on listings…")
        refresh_nearest_stop_m(cur)

        conn.commit()
        cur.close()
        print("Done.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Load Valencia neighborhood polygons from barris-barrios.geojson into raw.neighborhoods_raw.

Source:
  pipeline/barrios/data/barris-barrios.geojson

Assumptions:
  - GeoJSON has a property with the neighborhood name (e.g. 'NOM' or 'name').
  - Geometry is in EPSG:4326 or can be interpreted as such.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd

from .db_utils import get_pg_conn

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = PIPELINE_ROOT / "barrios" / "data" / "barris-barrios.geojson"


def detect_name_column(df: gpd.GeoDataFrame) -> str:
    candidates = ["name", "nombre", "nom", "NOM", "NOMBRE"]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        if col != df.geometry.name:
            return col
    raise ValueError("Could not detect a suitable name column in barrios GeoJSON.")


def main() -> None:
    if not GEOJSON_PATH.exists():
        raise SystemExit(f"GeoJSON not found: {GEOJSON_PATH}")

    gdf: gpd.GeoDataFrame = gpd.read_file(GEOJSON_PATH)
    if gdf.empty:
        raise SystemExit("Neighborhood GeoJSON is empty.")

    name_col = detect_name_column(gdf)

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    conn = get_pg_conn()
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS raw.neighborhoods_raw (
            id   TEXT,
            name TEXT,
            geom geometry(MULTIPOLYGON, 4326)
        )
        """
    )
    cur.execute("TRUNCATE raw.neighborhoods_raw")

    count = 0
    for idx, row in gdf.iterrows():
        name_val: Any = row.get(name_col)
        geom_wkb = row.geometry.wkb
        cur.execute(
            """
            INSERT INTO raw.neighborhoods_raw (id, name, geom)
            VALUES (%s, %s, ST_SetSRID(%s::geometry, 4326))
            """,
            (str(idx), str(name_val) if name_val is not None else None, geom_wkb),
        )
        count += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {count} neighborhoods into raw.neighborhoods_raw (CRS=4326)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Load Catastro parcels (Valencia city) into raw.parcels_raw in PostgreSQL.

Source:
  pipeline/catastro/data/46_UA_23012026_SHP.zip
  → extract Valencia PARCELA.shp

Fields used:
  REFCAT, MUNICIPIO, MASA, PARCELA, VIA, NUMERO, AREA, COORX, COORY, geometry
"""

import zipfile
from pathlib import Path

import shapefile
from shapely.geometry import shape

from .db_utils import get_pg_conn

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
CATASTRO_DATA = PIPELINE_ROOT / "catastro" / "data"
SHP_ZIP = CATASTRO_DATA / "46_UA_23012026_SHP.zip"
VALENCIA_SUBFOLDER = "46900uA 46005 VALENCIA"
PARCEL_ZIP = "46900uA_46005_23012026_PARCELA.ZIP"


def ensure_valencia_parcels_extracted() -> Path:
    extract_root = CATASTRO_DATA / "46_UA_23012026_SHP"
    valencia_dir = extract_root / VALENCIA_SUBFOLDER
    shp_dir = valencia_dir

    if not SHP_ZIP.exists():
        raise FileNotFoundError(f"Catastro zip not found: {SHP_ZIP}")

    if not extract_root.exists():
        with zipfile.ZipFile(SHP_ZIP, "r") as z:
            z.extractall(CATASTRO_DATA)

    parcel_zip = shp_dir / PARCEL_ZIP
    shp_file = shp_dir / "PARCELA.shp"
    if not shp_file.exists() and parcel_zip.exists():
        with zipfile.ZipFile(parcel_zip, "r") as z:
            z.extractall(shp_dir)
    if not shp_file.exists():
        raise FileNotFoundError(f"PARCELA.shp not found in {shp_dir}")
    return shp_dir


def main() -> None:
    shp_dir = ensure_valencia_parcels_extracted()
    shp_path = shp_dir / "PARCELA.shp"

    sf = shapefile.Reader(str(shp_path))
    field_names = [f[0] for f in sf.fields[1:]]

    conn = get_pg_conn()
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS raw.parcels_raw (
            refcat   TEXT,
            municipio TEXT,
            masa     TEXT,
            parcela  TEXT,
            via      TEXT,
            numero   TEXT,
            area     DOUBLE PRECISION,
            coorx    DOUBLE PRECISION,
            coory    DOUBLE PRECISION,
            geom     geometry(POLYGON, 25830)
        )
        """
    )
    cur.execute("TRUNCATE raw.parcels_raw")

    count = 0
    for shape_rec, dbf_rec in zip(sf.shapes(), sf.records()):
        rec = dict(zip(field_names, dbf_rec))
        refcat = (rec.get("REFCAT") or "").strip()
        if not refcat:
            continue

        try:
            geom_dict = {
                "type": "Polygon",
                "coordinates": [[[float(x), float(y)] for x, y in shape_rec.points]],
            }
            geom = shape(geom_dict)
            wkb = geom.wkb
        except Exception:
            wkb = None

        area_val = rec.get("AREA")
        try:
            area = float(area_val) if area_val is not None else None
        except (TypeError, ValueError):
            area = None
        coorx = rec.get("COORX")
        coory = rec.get("COORY")
        try:
            coorx = float(coorx) if coorx is not None else None
        except (TypeError, ValueError):
            coorx = None
        try:
            coory = float(coory) if coory is not None else None
        except (TypeError, ValueError):
            coory = None

        cur.execute(
            """
            INSERT INTO raw.parcels_raw
            (refcat, municipio, masa, parcela, via, numero, area, coorx, coory, geom)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s IS NULL THEN NULL ELSE ST_SetSRID(%s::geometry, 25830) END)
            """,
            (
                refcat,
                str(rec.get("MUNICIPIO", "")).strip() or None,
                str(rec.get("MASA", "")).strip() or None,
                str(rec.get("PARCELA", "")).strip() or None,
                str(rec.get("VIA", "")).strip() or None,
                str(rec.get("NUMERO", "")).strip() or None,
                area,
                coorx,
                coory,
                wkb,
                wkb,
            ),
        )
        count += 1
        if count % 5000 == 0:
            print(f"  Loaded {count} parcels...")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {count} parcels into raw.parcels_raw")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Load Catastro vías CSV into raw.vias_raw in PostgreSQL.

Source (tried in order):
  pipeline/catastro/data/catastro_vias.csv
  pipeline/catastro/vias.csv

Columns: provincia, nombreMunicipio, CodigoVia, tipoVia, nombreVia.
Optional: via_code, street_name (derived from tipoVia + nombreVia if missing).
"""

import csv
from pathlib import Path

from .db_utils import get_pg_conn

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
VIAS_CANDIDATES = [
    PIPELINE_ROOT / "catastro" / "data" / "catastro_vias.csv",
    PIPELINE_ROOT / "catastro" / "vias.csv",
]


def _resolve_vias_path() -> Path:
    for p in VIAS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(f"Vías CSV not found. Tried: {VIAS_CANDIDATES}")


def _row_to_values(row: dict) -> tuple:
    provincia = (row.get("provincia") or "").strip()
    nombre_municipio = (row.get("nombreMunicipio") or "").strip()
    codigo_via = str(row.get("CodigoVia") or row.get("codigo_via") or "").strip()
    tipo_via = (row.get("tipoVia") or row.get("tipo_via") or "").strip()
    nombre_via = (row.get("nombreVia") or row.get("nombre_via") or "").strip()
    via_code = (row.get("via_code") or codigo_via).strip()
    street_name = (row.get("street_name") or "").strip()
    if not street_name:
        street_name = f"{tipo_via} {nombre_via}".strip() or via_code
    return provincia, nombre_municipio, codigo_via, tipo_via, nombre_via, via_code, street_name


def main() -> None:
    vias_path = _resolve_vias_path()

    conn = get_pg_conn()
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS raw.vias_raw (
            provincia        TEXT,
            nombre_municipio TEXT,
            codigo_via       TEXT,
            tipo_via         TEXT,
            nombre_via       TEXT,
            via_code         TEXT,
            street_name      TEXT
        )
        """
    )

    cur.execute("TRUNCATE raw.vias_raw")

    count = 0
    with vias_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = _row_to_values(row)
            cur.execute(
                """
                INSERT INTO raw.vias_raw
                (provincia, nombre_municipio, codigo_via, tipo_via, nombre_via, via_code, street_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                vals,
            )
            count += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {count} vías into raw.vias_raw from {vias_path.name}")


if __name__ == "__main__":
    main()

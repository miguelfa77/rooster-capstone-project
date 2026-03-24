#!/usr/bin/env python3
"""
Load raw.vias_raw → core.streets (Valencia city streets).
"""

from pipeline.raw.db_utils import get_pg_conn


def main() -> None:
    conn = get_pg_conn()
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("TRUNCATE core.streets")
    cur.execute(
        """
        INSERT INTO core.streets (nombre_municipio, codigo_via, street_name, tipo_via, nombre_via)
        SELECT nombre_municipio, COALESCE(codigo_via, via_code), street_name, tipo_via, nombre_via
        FROM raw.vias_raw
        WHERE UPPER(TRIM(nombre_municipio)) = 'VALENCIA'
          AND (codigo_via IS NOT NULL OR via_code IS NOT NULL)
        ON CONFLICT (nombre_municipio, codigo_via) DO UPDATE SET
            street_name = EXCLUDED.street_name,
            tipo_via = EXCLUDED.tipo_via,
            nombre_via = EXCLUDED.nombre_via
        """
    )
    count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {count} streets into core.streets")

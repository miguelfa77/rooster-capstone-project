#!/usr/bin/env python3
"""
Load raw.parcels_raw → core.parcels.
"""

from pipeline.raw.db_utils import get_pg_conn


def main() -> None:
    conn = get_pg_conn()
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("TRUNCATE core.parcels")
    cur.execute(
        """
        INSERT INTO core.parcels (refcat, municipio, via, numero, area, geom)
        SELECT DISTINCT ON (refcat) refcat, municipio, via, numero, area, geom
        FROM raw.parcels_raw
        WHERE refcat IS NOT NULL AND refcat != ''
        ORDER BY refcat
        """
    )
    count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {count} parcels into core.parcels")

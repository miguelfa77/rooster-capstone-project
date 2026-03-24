#!/usr/bin/env python3
"""
Load raw.neighborhoods_raw → core.neighborhoods.
"""

from pipeline.raw.db_utils import get_pg_conn


def main() -> None:
    conn = get_pg_conn()
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("TRUNCATE core.neighborhoods")
    cur.execute(
        """
        INSERT INTO core.neighborhoods (id, name, geom)
        SELECT id, COALESCE(name, id), geom
        FROM raw.neighborhoods_raw
        """
    )
    count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {count} neighborhoods into core.neighborhoods")

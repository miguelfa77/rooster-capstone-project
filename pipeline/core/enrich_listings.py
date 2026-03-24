#!/usr/bin/env python3
"""
Recompute derived listing fields after load_listings (and after streets/parcels exist for geocode).

- Amenities from description (regex)
- floor_int, minutes_to_center
- Optional: lat/lng via pg_trgm + cadastre parcels (GEOCODE_LISTINGS=1)

Prerequisites: sql/migrate_listing_enrichment.sql (or fresh core_tables.sql with new columns).

Run from repo root:
  python -m pipeline.core.enrich_listings
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pipeline.raw.db_utils import get_pg_conn

ROOT = Path(__file__).resolve().parents[2]


def _exec_sql_script(cur, text: str) -> None:
    """Execute semicolon-separated statements; strip line comments."""
    lines_out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines_out.append(line)
    full = "\n".join(lines_out) + "\n"
    parts = re.split(r";\s*\n", full)
    for part in parts:
        stmt = part.strip()
        if stmt:
            cur.execute(stmt)


def main() -> None:
    refresh_path = ROOT / "sql" / "enrich_listings_refresh.sql"
    geo_path = ROOT / "sql" / "enrich_geocode.sql"

    if not refresh_path.exists():
        raise FileNotFoundError(refresh_path)

    conn = get_pg_conn()
    conn.autocommit = False
    cur = conn.cursor()
    try:
        _exec_sql_script(cur, refresh_path.read_text(encoding="utf-8"))
        conn.commit()
        print("enrich_listings: refreshed amenities, floor_int, minutes_to_center")

        if os.getenv("GEOCODE_LISTINGS", "").strip().lower() in ("1", "true", "yes"):
            if not geo_path.exists():
                raise FileNotFoundError(geo_path)
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
            _exec_sql_script(cur, geo_path.read_text(encoding="utf-8"))
            conn.commit()
            print("enrich_listings: ran parcel-based geocode (GEOCODE_LISTINGS=1)")
    except Exception as e:
        conn.rollback()
        print(
            "enrich_listings failed — if columns are missing, run:\n"
            "  psql -d rooster -f sql/migrate_listing_enrichment.sql\n"
            "  psql -d rooster -f sql/analytics_views.sql"
        )
        raise e
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()

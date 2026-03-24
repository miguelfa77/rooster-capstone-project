#!/usr/bin/env python3
"""
Load raw.listings_raw → core.listings (upsert by url) + append core.listing_snapshots.

- first_seen_at / last_seen_at: set on first insert; last_seen_at updated each load.
- price_int_previous: previous price_int before each update (price change tracking).
- listing_snapshots: one row per (url, scraped_at) per raw row for time series.
  Raw may contain many rows per url (different scraped_at); process in url + time order
  so core.listings ends on the latest observation.

Set LISTINGS_TRUNCATE_BEFORE_LOAD=1 to wipe core.listings + core.listing_snapshots and rebuild
from raw (destructive).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Optional

from pipeline.raw.db_utils import get_pg_conn


def _parse_price(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = re.sub(r"[^\d]", "", str(s).strip())
    return int(s) if s else None


def _parse_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        return int(float(str(s).strip().replace(",", ".")))
    except (ValueError, TypeError):
        return None


def _parse_area_sqm(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = str(s).strip().replace(",", ".").replace(" ", "")
    try:
        return float(re.sub(r"[^\d.]", "", s))
    except (ValueError, TypeError):
        return None


def _parse_scraped_ts(scraped_at: Optional[str]) -> datetime:
    """Parse Idealista ISO timestamp; fall back to UTC now."""
    if scraped_at and isinstance(scraped_at, str) and scraped_at.strip():
        text = scraped_at.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _extract_location(heading: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Extract street_name_raw and neighborhood_raw from Idealista heading.
    Pattern: "Type en [Street], [Neighborhood], València" or "Type en [Neighborhood], València"
    """
    if not heading or not isinstance(heading, str):
        return None, None
    heading = heading.strip()
    loc = re.sub(r",\s*València\s*$", "", heading, flags=re.I).strip()
    if not loc:
        return None, None
    m = re.search(r"\ben\s+(.+)$", loc, re.I)
    if m:
        loc = m.group(1).strip()
    parts = [p.strip() for p in loc.split(",") if p.strip()]
    if not parts:
        return None, None
    if len(parts) == 1:
        return None, parts[0]
    neighborhood = parts[-1]
    street = parts[-2] if len(parts) >= 2 else None
    return street, neighborhood


UPSERT_SQL = """
INSERT INTO core.listings (
    url, operation, heading, price, price_int, price_int_previous, currency, period,
    rooms, rooms_int, area, area_sqm, floor, time_to_center, description,
    street_name_raw, neighborhood_raw, scraped_at, first_seen_at, last_seen_at
) VALUES (
    %s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (url) DO UPDATE SET
    price_int_previous = core.listings.price_int,
    price_int = EXCLUDED.price_int,
    price = EXCLUDED.price,
    heading = EXCLUDED.heading,
    operation = EXCLUDED.operation,
    currency = EXCLUDED.currency,
    period = EXCLUDED.period,
    rooms = EXCLUDED.rooms,
    rooms_int = EXCLUDED.rooms_int,
    area = EXCLUDED.area,
    area_sqm = EXCLUDED.area_sqm,
    floor = EXCLUDED.floor,
    time_to_center = EXCLUDED.time_to_center,
    description = EXCLUDED.description,
    street_name_raw = EXCLUDED.street_name_raw,
    neighborhood_raw = EXCLUDED.neighborhood_raw,
    scraped_at = EXCLUDED.scraped_at,
    first_seen_at = core.listings.first_seen_at,
    last_seen_at = EXCLUDED.last_seen_at
"""

SNAPSHOT_SQL = """
INSERT INTO core.listing_snapshots (url, price_int, scraped_at)
VALUES (%s, %s, %s)
ON CONFLICT (url, scraped_at) DO NOTHING
"""


def main() -> None:
    conn = get_pg_conn()
    conn.autocommit = False
    cur = conn.cursor()

    if os.getenv("LISTINGS_TRUNCATE_BEFORE_LOAD", "").strip() in ("1", "true", "yes"):
        cur.execute("TRUNCATE core.listing_snapshots")
        cur.execute("TRUNCATE core.listings")

    cur.execute(
        """
        SELECT url, operation, heading, price, currency, period, rooms, area,
               floor, time_to_center, description, scraped_at
        FROM raw.listings_raw
        ORDER BY url, scraped_at NULLS LAST
        """
    )
    rows = cur.fetchall()

    n_snapshots = 0
    for row in rows:
        url, operation, heading, price, currency, period, rooms, area, floor, time_to_center, description, scraped_at = row
        if not url or not operation:
            continue
        price_int = _parse_price(price)
        rooms_int = _parse_int(rooms)
        area_sqm = _parse_area_sqm(area)
        street_raw, neighborhood_raw = _extract_location(heading)
        ts = _parse_scraped_ts(scraped_at)

        cur.execute(
            UPSERT_SQL,
            (
                url,
                operation,
                heading,
                price,
                price_int,
                currency,
                period,
                rooms,
                rooms_int,
                area,
                area_sqm,
                floor,
                time_to_center,
                description,
                street_raw,
                neighborhood_raw,
                scraped_at,
                ts,
                ts,
            ),
        )
        cur.execute(SNAPSHOT_SQL, (url, price_int, ts))
        n_snapshots += cur.rowcount

    conn.commit()
    cur.close()
    conn.close()
    print(f"Processed {len(rows)} raw rows → core.listings (upsert); {n_snapshots} new snapshot row(s)")


if __name__ == "__main__":
    main()

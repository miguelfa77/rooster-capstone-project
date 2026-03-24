#!/usr/bin/env python3
"""
Load Idealista CSVs into raw.listings_raw in PostgreSQL.

Source files (pipe-separated):
  pipeline/idealista/data/idealista_alquiler.csv
  pipeline/idealista/data/idealista_venta.csv

Primary key is (url, scraped_at) so each scrape observation is kept for time series
(core.listing_snapshots). Re-scrapes of the same URL with a new scraped_at insert
a new raw row.

Columns (from scraper):
  operation, heading, price, currency, period,
  rooms, area, floor, time_to_center, description,
  url, page, scraped_at
"""

import csv
from pathlib import Path

from .db_utils import get_pg_conn

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PIPELINE_ROOT / "idealista" / "data"

CSV_SEPARATOR = "|"
FIELDNAMES = [
    "operation",
    "heading",
    "price",
    "currency",
    "period",
    "rooms",
    "area",
    "floor",
    "time_to_center",
    "description",
    "url",
    "page",
    "scraped_at",
]


def load_csv(cur, csv_path: Path) -> int:
    if not csv_path.exists():
        print(f"Skip (not found): {csv_path}")
        return 0
    count = 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=CSV_SEPARATOR, fieldnames=FIELDNAMES)
        next(reader, None)  # skip header
        for row in reader:
            if not row.get("url") or not row.get("operation"):
                continue
            cur.execute(
                """
                INSERT INTO raw.listings_raw
                (operation, heading, price, currency, period, rooms, area, floor,
                 time_to_center, description, url, page, scraped_at)
                VALUES (%(operation)s, %(heading)s, %(price)s, %(currency)s, %(period)s,
                        %(rooms)s, %(area)s, %(floor)s, %(time_to_center)s, %(description)s,
                        %(url)s, %(page)s, %(scraped_at)s)
                ON CONFLICT (url, scraped_at) DO NOTHING
                """,
                row,
            )
            count += cur.rowcount
    return count


def main() -> None:
    conn = get_pg_conn()
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS raw.listings_raw (
            operation      TEXT,
            heading        TEXT,
            price          TEXT,
            currency       TEXT,
            period         TEXT,
            rooms          TEXT,
            area           TEXT,
            floor          TEXT,
            time_to_center TEXT,
            description    TEXT,
            url            TEXT NOT NULL,
            page           TEXT,
            scraped_at     TEXT NOT NULL,
            PRIMARY KEY (url, scraped_at)
        )
        """
    )

    total = 0
    for name in ("idealista_alquiler", "idealista_venta"):
        csv_path = DATA_DIR / f"{name}.csv"
        n = load_csv(cur, csv_path)
        total += n
        print(f"Loaded {n} rows from {csv_path.name}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Total rows inserted into raw.listings_raw: {total}")


if __name__ == "__main__":
    main()

"""
Apply sql/migrate_tourist_apartments_geocode.sql: parse addresses, fuzzy-match streets,
join parcels, fallback street centroid, neighborhoods, then refresh core.tourist_apartments.

Requires: core.streets, core.parcels (Valencia city), core.neighborhoods, raw.tourist_apartments loaded.

Run:
  python -m pipeline.open_data.geocode_tourist_apartments
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.raw.db_utils import get_pg_conn, load_pipeline_env


def _run_sql_psql(sql_path: Path) -> None:
    psql = shutil.which("psql")
    assert psql is not None
    dsn = os.getenv("DATABASE_URL")
    cmd = [psql, "-v", "ON_ERROR_STOP=1", "-f", str(sql_path)]
    if dsn:
        cmd.insert(1, dsn)
    subprocess.run(cmd, check=True)


def _run_sql_psycopg2(sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    conn = get_pg_conn()
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    sql_path = ROOT / "sql" / "migrate_tourist_apartments_geocode.sql"
    if not sql_path.is_file():
        print(f"Missing {sql_path}", file=sys.stderr)
        sys.exit(1)

    load_pipeline_env()
    if shutil.which("psql"):
        try:
            _run_sql_psql(sql_path)
        except subprocess.CalledProcessError:
            print("psql failed; retrying with psycopg2…", file=sys.stderr)
            _run_sql_psycopg2(sql_path)
    else:
        _run_sql_psycopg2(sql_path)

    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(geocode_quality, '(null)') AS geocode_quality,
                   COUNT(*)::bigint AS n,
                   ROUND(COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER (), 0), 1) AS pct
            FROM raw.tourist_apartments
            GROUP BY geocode_quality
            ORDER BY n DESC
            """
        )
        rows = cur.fetchall()
        print("Coverage (raw.tourist_apartments by geocode_quality):")
        for gq, n, pct in rows:
            pct_s = f"{pct}%" if pct is not None else "n/a"
            print(f"  {gq}: {n} ({pct_s})")
        cur.close()
    finally:
        conn.close()


if __name__ == "__main__":
    main()

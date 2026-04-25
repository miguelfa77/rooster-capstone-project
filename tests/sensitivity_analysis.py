#!/usr/bin/env python3
"""
Sensitivity analysis for the Rooster investment score yield weight.

Loads analytics.neighborhood_profile from a local PostgreSQL database, recomputes
score for yield_weight in {0.3, 0.4, 0.5, 0.6, 0.7} (baseline 0.5), compares ranks,
and exports ``tests/sensitivity_results.csv`` next to this script.

Uses only the standard library + psycopg2 (no numpy/pandas/scipy) so it runs on
Apple Silicon even when a system Python has x86_64-only wheels.

Usage (from repo root; use ``python3`` on macOS if ``python`` is not found):

  python3 tests/sensitivity_analysis.py

Connection: ``tests/.env`` (loaded automatically), or ``ROOSTER_PG_*`` / standard
``PGHOST``, ``PGPORT``, ``PGUSER``, ``PGPASSWORD``, ``PGDATABASE`` in the environment.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

_PG_ENV_KEYS = frozenset(
    {"DATABASE_URL", "PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"}
)


def _should_merge_dotenv_key(key: str) -> bool:
    if key not in os.environ:
        return True
    if key in _PG_ENV_KEYS and os.environ.get(key) == "":
        return True
    return False


def _load_tests_dotenv() -> None:
    """
    Merge ``tests/.env`` into the process environment (same line format as
    ``pipeline/.env``). Does not override non-empty vars except empty PG* keys.
    """
    path = Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and _should_merge_dotenv_key(key):
            os.environ[key] = value


_load_tests_dotenv()


def _env(*names: str, default: str = "") -> str:
    """First non-empty value among ``names``."""
    for n in names:
        v = os.getenv(n)
        if v is not None and v != "":
            return v
    return default


# ROOSTER_PG_* (project convention) or libpq-style PG* (same as pipeline/.env)
DB_HOST = _env("ROOSTER_PG_HOST", "PGHOST", default="localhost")
DB_PORT = int(_env("ROOSTER_PG_PORT", "PGPORT", default="5432"))
DB_NAME = _env("ROOSTER_PG_DATABASE", "PGDATABASE", default="rooster")
DB_USER = _env("ROOSTER_PG_USER", "PGUSER", default="postgres")
DB_PASSWORD = _env("ROOSTER_PG_PASSWORD", "PGPASSWORD")

YIELD_WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7)
BASELINE_WEIGHT = 0.5

SQL = """
SELECT
    neighborhood_name,
    gross_rental_yield_pct,
    avg_dist_to_stop_m,
    tourist_density_pct,
    investment_score,
    total_count
FROM analytics.neighborhood_profile
WHERE gross_rental_yield_pct IS NOT NULL
  AND COALESCE(total_count, 0) > 0
"""


def _is_nullish(x: object) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def _transport_bonus(avg_dist_to_stop_m: object) -> float:
    """Match production: COALESCE(avg_dist_to_stop_m, 99999) for thresholds."""
    if _is_nullish(avg_dist_to_stop_m):
        d = 99999.0
    else:
        d = float(avg_dist_to_stop_m)
    if d < 400:
        return 2.0
    if d < 700:
        return 1.0
    return 0.0


def _tourism_bonus(tourist_density_pct: object) -> float:
    """Match production: COALESCE(tourist_density_pct, 0) < 10 → 1.0 else 0.0."""
    if _is_nullish(tourist_density_pct):
        td = 0.0
    else:
        td = float(tourist_density_pct)
    return 1.0 if td < 10 else 0.0


def investment_score_row(row: dict[str, object], yield_weight: float) -> float:
    """
    score = COALESCE(gross_rental_yield_pct, 0) * yield_weight
          + transport_bonus
          + tourism_bonus
    """
    y = row["gross_rental_yield_pct"]
    if _is_nullish(y):
        yv = 0.0
    else:
        yv = float(y)
    raw = (
        yv * yield_weight
        + _transport_bonus(row["avg_dist_to_stop_m"])
        + _tourism_bonus(row["tourist_density_pct"])
    )
    return round(raw, 2)


def rank_descending_average(scores: list[float]) -> list[float]:
    """
    Average ranks: rank 1 = best (highest score). Ties share the mean of their rank positions.
    """
    n = len(scores)
    if n == 0:
        return []
    idx = list(range(n))
    idx.sort(key=lambda i: scores[i], reverse=True)
    ranks = [0.0] * n
    pos = 0
    while pos < n:
        val = scores[idx[pos]]
        end = pos
        while end + 1 < n and scores[idx[end + 1]] == val:
            end += 1
        mean_rank = (pos + 1 + end + 1) / 2.0
        for k in range(pos, end + 1):
            ranks[idx[k]] = mean_rank
        pos = end + 1
    return ranks


def pearson_rho(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient (Spearman uses Pearson of ranks)."""
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = sum((xi - mx) ** 2 for xi in x)
    deny = sum((yi - my) ** 2 for yi in y)
    if denx <= 0 or deny <= 0:
        return float("nan")
    return num / math.sqrt(denx * deny)


def fisher_pvalue_correlation(rho: float, n: int) -> float:
    """
    Two-tailed p-value for H0: rho=0 using Fisher z transform (standard for r).
    Adequate for n > ~10; exact small-sample behavior differs from scipy.stats.
    """
    if n <= 3 or math.isnan(rho) or abs(rho) >= 1.0:
        return float("nan")
    z = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
    se = 1.0 / math.sqrt(n - 3)
    z_stat = z / se
    # two-tailed vs standard normal
    p = 2.0 * (1.0 - _normal_cdf(abs(z_stat)))
    return min(p, 1.0)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_rows() -> list[dict[str, object]]:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(SQL)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def main() -> None:
    if not DB_PASSWORD:
        print(
            "No password: add PGPASSWORD or ROOSTER_PG_PASSWORD to tests/.env "
            "or export it in your shell."
        )

    rows = load_rows()
    if not rows:
        print("No rows after filters. Check analytics.neighborhood_profile and filters.")
        return

    n = len(rows)
    scores_by_w: dict[float, list[float]] = {}
    for w in YIELD_WEIGHTS:
        scores_by_w[w] = [investment_score_row(r, w) for r in rows]

    ranks_by_w: dict[float, list[float]] = {}
    for w in YIELD_WEIGHTS:
        ranks_by_w[w] = rank_descending_average(scores_by_w[w])

    baseline_ranks = ranks_by_w[BASELINE_WEIGHT]
    # Top 10 by baseline rank (smaller rank number = better)
    order = sorted(range(n), key=lambda i: baseline_ranks[i])
    top10_idx = order[:10]

    display_cols = ["neighborhood_name"]
    for w in YIELD_WEIGHTS:
        display_cols.append(f"score_w{w}")
        display_cols.append(f"rank_w{w}")

    print("\n=== Top 10 neighborhoods by baseline rank (yield_weight = 0.5) ===\n")
    header = "".join(f"{c:>14}" for c in display_cols)
    print(header)
    print("-" * len(header))
    for i in top10_idx:
        r = rows[i]
        line = [str(r.get("neighborhood_name") or "")]
        for w in YIELD_WEIGHTS:
            line.append(f"{scores_by_w[w][i]:.2f}")
            line.append(f"{ranks_by_w[w][i]:.2f}")
        print("".join(f"{x:>14}" for x in line))

    print("\n=== Spearman rank correlation vs baseline (w = 0.5) ===\n")
    print("(ρ = Pearson correlation of rank vectors; p-value via Fisher z approximation.)\n")
    for w in YIELD_WEIGHTS:
        if w == BASELINE_WEIGHT:
            continue
        rho = pearson_rho(baseline_ranks, ranks_by_w[w])
        pval = fisher_pvalue_correlation(rho, n)
        print(f"  w={w}  ρ = {rho:.4f}  p-value = {pval:.4g}")

    out_path = Path(__file__).resolve().parent / "sensitivity_results.csv"
    fieldnames = ["neighborhood_name"]
    for w in YIELD_WEIGHTS:
        fieldnames.append(f"score_w{w}")
        fieldnames.append(f"rank_w{w}")
    extra_keys = [
        k
        for k in rows[0].keys()
        if k not in fieldnames and k != "neighborhood_name"
    ]
    fieldnames_full = fieldnames + extra_keys

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_full, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows):
            out = {k: row.get(k) for k in extra_keys}
            out["neighborhood_name"] = row.get("neighborhood_name")
            for wt in YIELD_WEIGHTS:
                out[f"score_w{wt}"] = scores_by_w[wt][i]
                out[f"rank_w{wt}"] = ranks_by_w[wt][i]
            w.writerow(out)

    print(f"\nWrote full results to {out_path.resolve()}")


if __name__ == "__main__":
    main()

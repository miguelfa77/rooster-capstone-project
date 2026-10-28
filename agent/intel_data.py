"""Intel-tab data queries, independent of any UI.

Moved from app.py so the Streamlit app and the HTTP API share them.
"""

from __future__ import annotations

import json

import pandas as pd
from psycopg2 import errors as pg_errors
from psycopg2 import sql

from agent import ui_es as UI
from agent.llm_sql import get_pg_conn, get_pg_engine

CHOROPLETH_METRICS = UI.CHOROPLETH_METRICS
CHOROPLETH_METRICS_PROFILE_ONLY_COLUMNS = UI.CHOROPLETH_METRICS_PROFILE_ONLY_COLUMNS


def analytics_views_available() -> dict[str, bool]:
    """Whether key analytics views exist (schema may be empty until SQL is applied)."""
    names = {"listing_summary": False, "neighborhood_metrics": False}
    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT table_name FROM information_schema.views
            WHERE table_schema = 'analytics'
              AND table_name IN ('listing_summary', 'neighborhood_metrics')
            """
        )
        for (tname,) in cur.fetchall():
            if tname in names:
                names[tname] = True
        cur.close()
    except Exception:
        pass
    finally:
        conn.close()
    return names


def neighborhood_profile_available() -> bool:
    """True if analytics.neighborhood_profile exists (transport/tourism/investment choropleth)."""
    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1 FROM information_schema.views
            WHERE table_schema = 'analytics' AND table_name = 'neighborhood_profile'
            """
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def neighborhood_geojson(metric_column: str) -> tuple[dict, float, float]:
    """Return GeoJSON FeatureCollection and (vmin, vmax). Uses analytics.neighborhood_profile when present, else neighborhood_metrics, else core.neighborhoods (no metric shading)."""
    allowed = set(CHOROPLETH_METRICS.values())
    if metric_column not in allowed:
        raise ValueError("Invalid metric column")

    conn = get_pg_conn()
    rows: list[tuple] = []
    try:
        cur = conn.cursor()
        profile_ok = neighborhood_profile_available()

        def _fetch_metric_rows(src_tbl: str, col_name: str) -> list[tuple]:
            q = sql.SQL(
                "SELECT neighborhood_name, ST_AsGeoJSON(geom)::text, {col} AS v, "
                "       total_count, gross_rental_yield_pct "
                "FROM {src}"
            ).format(
                col=sql.Identifier(col_name),
                src=sql.Identifier("analytics", src_tbl),
            )
            cur.execute(q)
            return cur.fetchall()

        try:
            if profile_ok:
                rows = _fetch_metric_rows("neighborhood_profile", metric_column)
            else:
                col_nm = (
                    metric_column
                    if metric_column not in CHOROPLETH_METRICS_PROFILE_ONLY_COLUMNS
                    else "gross_rental_yield_pct"
                )
                rows = _fetch_metric_rows("neighborhood_metrics", col_nm)
        except (pg_errors.UndefinedTable, pg_errors.UndefinedColumn):
            conn.rollback()
            rows = []
            try:
                rows = _fetch_metric_rows("neighborhood_metrics", "gross_rental_yield_pct")
            except Exception:
                conn.rollback()
                rows = []

        if not rows:
            cur.execute(
                """
                SELECT name,
                       ST_AsGeoJSON(
                           ST_SimplifyPreserveTopology(geom, 0.00004),
                           4
                       )::text,
                       NULL::double precision,
                       NULL::bigint,
                       NULL::double precision
                FROM core.neighborhoods
                WHERE geom IS NOT NULL AND name IS NOT NULL
                """
            )
            rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    features: list[dict] = []
    values: list[float] = []
    for row in rows:
        if len(row) < 3:
            continue
        name, gj, v = row[0], row[1], row[2]
        tc = row[3] if len(row) > 3 else None
        yp = row[4] if len(row) > 4 else None
        if gj is None:
            continue
        geom = json.loads(gj)
        fv: float | None = float(v) if v is not None else None
        if fv is not None:
            if metric_column in (
                "total_count",
                "transit_stop_count",
                "tourist_density_pct",
                "investment_score",
            ):
                if fv >= 0:
                    values.append(fv)
            elif fv > 0:
                values.append(fv)
        tc_i = int(tc) if tc is not None else None
        yp_f = float(yp) if yp is not None else None
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "name": name or "",
                    "value": round(fv, 4) if fv is not None else None,
                    "listings": tc_i,
                    "yield_pct": round(yp_f, 2) if yp_f is not None else None,
                },
            }
        )

    fc = {"type": "FeatureCollection", "features": features}
    if not values:
        return fc, 0.0, 1.0
    return fc, min(values), max(values)


def parcels_geojson() -> dict:
    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ST_AsGeoJSON(
                ST_SimplifyPreserveTopology(ST_Transform(geom, 4326), 0.00004),
                4
            )::text
            FROM core.parcels
            WHERE geom IS NOT NULL
            """
        )
        feats = []
        for (gj,) in cur.fetchall():
            if not gj:
                continue
            feats.append(
                {
                    "type": "Feature",
                    "geometry": json.loads(gj),
                    "properties": {},
                }
            )
        cur.close()
        return {"type": "FeatureCollection", "features": feats}
    finally:
        conn.close()


def transit_overlay() -> pd.DataFrame:
    try:
        engine = get_pg_engine()
        return pd.read_sql(
            """
            SELECT lat, lng, COALESCE(name, '') AS name, COALESCE(stop_type, '') AS stop_type
            FROM core.transit_stops
            WHERE lat IS NOT NULL AND lng IS NOT NULL
            LIMIT 8000
            """,
            engine,
        )
    except Exception:
        return pd.DataFrame()


def tourism_overlay() -> pd.DataFrame:
    try:
        engine = get_pg_engine()
        return pd.read_sql(
            """
            SELECT lat, lng, COALESCE(address, '') AS address
            FROM core.tourist_apartments
            WHERE lat IS NOT NULL AND lng IS NOT NULL
            LIMIT 8000
            """,
            engine,
        )
    except Exception:
        return pd.DataFrame()


def listings(operation_filter: str) -> pd.DataFrame:
    """Listings + canonical barrio name and metrics via neighborhood_id (spatial FK)."""
    if operation_filter not in ("All", "venta", "alquiler"):
        operation_filter = "All"
    has_nm = analytics_views_available().get("neighborhood_metrics", False)
    engine = get_pg_engine()
    if has_nm:
        q = """
            SELECT
                l.url,
                l.operation,
                l.price_int,
                l.area_sqm,
                l.rooms_int,
                ROUND(l.floor_int)::integer AS floor_int,
                l.lat,
                l.lng,
                l.geocode_quality,
                l.neighborhood_id,
                n.name AS neighborhood_name,
                l.has_parking,
                l.has_terrace,
                l.has_elevator,
                l.is_exterior,
                l.is_renovated,
                l.has_ac,
                l.nearest_stop_m,
                ROUND((l.price_int::numeric / NULLIF(l.area_sqm, 0)::numeric), 0) AS eur_per_sqm,
                nm.median_venta_eur_per_sqm AS neighborhood_median_sqm,
                nm.median_venta_price AS neighborhood_median_sale,
                nm.median_alquiler_price AS neighborhood_median_rent,
                nm.gross_rental_yield_pct AS yield_pct,
                CASE
                    WHEN nm.median_venta_price IS NULL AND nm.median_alquiler_price IS NULL THEN NULL
                    WHEN l.operation = 'venta'
                         AND nm.median_venta_price IS NOT NULL
                         AND l.price_int < nm.median_venta_price
                    THEN TRUE
                    WHEN l.operation = 'alquiler'
                         AND nm.median_alquiler_price IS NOT NULL
                         AND l.price_int < nm.median_alquiler_price
                    THEN TRUE
                    ELSE FALSE
                END AS below_median
            FROM core.listings l
            JOIN core.neighborhoods n ON n.id = l.neighborhood_id
            LEFT JOIN analytics.neighborhood_metrics nm ON nm.neighborhood_id = l.neighborhood_id
            WHERE l.price_int > 0 AND l.area_sqm > 0
        """
    else:
        q = """
            SELECT
                l.url,
                l.operation,
                l.price_int,
                l.area_sqm,
                l.rooms_int,
                ROUND(l.floor_int)::integer AS floor_int,
                l.lat,
                l.lng,
                l.geocode_quality,
                l.neighborhood_id,
                n.name AS neighborhood_name,
                l.has_parking,
                l.has_terrace,
                l.has_elevator,
                l.is_exterior,
                l.is_renovated,
                l.has_ac,
                l.nearest_stop_m,
                ROUND((l.price_int::numeric / NULLIF(l.area_sqm, 0)::numeric), 0) AS eur_per_sqm,
                NULL::double precision AS neighborhood_median_sqm,
                NULL::double precision AS neighborhood_median_sale,
                NULL::double precision AS neighborhood_median_rent,
                NULL::double precision AS yield_pct,
                NULL::boolean AS below_median
            FROM core.listings l
            JOIN core.neighborhoods n ON n.id = l.neighborhood_id
            WHERE l.price_int > 0 AND l.area_sqm > 0
        """
    if operation_filter != "All":
        q = q.rstrip() + " AND l.operation = %s"
        return pd.read_sql_query(q, engine, params=[operation_filter])
    return pd.read_sql_query(q, engine)


def market_framing() -> dict | None:
    """City-wide median yield, yield spread (pp), and count of barrios >6% yield."""
    if not analytics_views_available().get("neighborhood_metrics"):
        return None
    eng = get_pg_engine()
    try:
        row = pd.read_sql_query(
            """
            SELECT
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gross_rental_yield_pct) AS city_median_yield,
                (MAX(gross_rental_yield_pct) - MIN(gross_rental_yield_pct)) AS spread_pp,
                COUNT(*) FILTER (WHERE gross_rental_yield_pct > 6) AS n_strong,
                COUNT(*) AS n_with_yield
            FROM analytics.neighborhood_metrics
            WHERE gross_rental_yield_pct IS NOT NULL
            """,
            eng,
        ).iloc[0]
    except Exception:
        return None
    return row.to_dict()


def neighborhood_ranking() -> pd.DataFrame:
    if not analytics_views_available().get("neighborhood_metrics"):
        return pd.DataFrame()
    eng = get_pg_engine()
    if neighborhood_profile_available():
        try:
            return pd.read_sql_query(
                """
                SELECT
                    neighborhood_name,
                    ROUND(gross_rental_yield_pct::numeric, 1) AS yield_pct,
                    ROUND(median_venta_eur_per_sqm::numeric, 0) AS eur_per_sqm,
                    transit_stop_count,
                    transport_rating,
                    ROUND(tourist_density_pct::numeric, 1) AS tourist_pct,
                    tourism_pressure,
                    ROUND(investment_score::numeric, 1) AS score,
                    venta_count,
                    alquiler_count
                FROM analytics.neighborhood_profile
                WHERE gross_rental_yield_pct IS NOT NULL
                  AND venta_count >= 3
                  AND alquiler_count >= 3
                ORDER BY investment_score DESC NULLS LAST
                """,
                eng,
            )
        except Exception:
            pass
    return pd.read_sql_query(
        """
        SELECT
            neighborhood_name,
            ROUND(gross_rental_yield_pct::numeric, 1) AS yield_pct,
            ROUND(median_venta_eur_per_sqm::numeric, 0) AS eur_per_sqm,
            NULL::bigint AS transit_stop_count,
            NULL::text AS transport_rating,
            NULL::numeric AS tourist_pct,
            NULL::text AS tourism_pressure,
            ROUND(gross_rental_yield_pct::numeric, 1) AS score,
            venta_count,
            alquiler_count
        FROM analytics.neighborhood_metrics
        WHERE gross_rental_yield_pct IS NOT NULL
          AND venta_count >= 3
          AND alquiler_count >= 3
        ORDER BY gross_rental_yield_pct DESC NULLS LAST
        """,
        eng,
    )


def yield_liquidity() -> pd.DataFrame:
    if not analytics_views_available().get("neighborhood_metrics"):
        return pd.DataFrame()
    eng = get_pg_engine()
    if neighborhood_profile_available():
        try:
            return pd.read_sql_query(
                """
                SELECT
                    neighborhood_name,
                    gross_rental_yield_pct AS yield_pct,
                    total_count AS listing_count,
                    investment_score,
                    transport_rating,
                    tourism_pressure,
                    median_venta_eur_per_sqm AS eur_per_sqm
                FROM analytics.neighborhood_profile
                WHERE gross_rental_yield_pct IS NOT NULL
                  AND total_count >= 3
                """,
                eng,
            )
        except Exception:
            pass
    return pd.read_sql_query(
        """
        SELECT
            neighborhood_name,
            gross_rental_yield_pct AS yield_pct,
            total_count AS listing_count,
            NULL::double precision AS investment_score,
            NULL::text AS transport_rating,
            NULL::text AS tourism_pressure,
            median_venta_eur_per_sqm AS eur_per_sqm
        FROM analytics.neighborhood_metrics
        WHERE gross_rental_yield_pct IS NOT NULL
          AND total_count >= 3
        """,
        eng,
    )

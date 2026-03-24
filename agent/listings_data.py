"""Load listing frames for chart renderers (no Streamlit dependency)."""

from __future__ import annotations

import pandas as pd

from agent.llm_sql import get_pg_conn, get_pg_engine


def _has_neighborhood_metrics_view() -> bool:
    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1 FROM information_schema.views
            WHERE table_schema = 'analytics' AND table_name = 'neighborhood_metrics'
            """
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def load_listings_frame(operation_filter: str = "All") -> pd.DataFrame:
    """Same columns as app `_load_listings_for_charts` (price/area listings)."""
    if operation_filter not in ("All", "venta", "alquiler"):
        operation_filter = "All"
    has_nm = _has_neighborhood_metrics_view()
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

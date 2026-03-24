import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

# Streamlit executes this file with a script run context; plain `python app.py` does not — that hangs or does nothing useful.
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = lambda: None  # type: ignore
    if get_script_run_ctx() is None:
        print(
            "\nRooster must be started with Streamlit (not `python app.py`).\n\n"
            f"  cd {ROOT}\n"
            "  ./bin/python -m streamlit run app.py\n\n"
            "Or:\n  ./run_app.sh\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

# Load agent/.env (file wins over shell so API keys match what you edit)
agent_env = ROOT / "agent" / ".env"
if agent_env.exists():
    for line in agent_env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("\"'")
        if k:
            os.environ[k] = v
    if os.getenv("OPENAI_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium
import pandas as pd
import plotly.express as px
from branca.colormap import LinearColormap
from psycopg2 import errors as pg_errors
from psycopg2 import sql
from streamlit_folium import st_folium

from agent.agent_pipeline import (
    CONVERSATIONAL_SYNTH_MAX_TOKENS,
    build_render_stack,
    extract_neighborhood_names_from_schema,
    format_confirmed_visuals,
    format_last_assistant_for_planner,
    get_live_schema_context,
    is_conversational_message,
    pick_fast_path_conversational_reply,
    run_openai_function_calling_pipeline,
    run_synthesiser,
    stream_canned_text_word_by_word,
    stream_openai_final_response_messages,
    update_conversation_state,
)
from agent.llm_sql import (
    DEFAULT_SYNTHESISER_MODEL_OPENAI,
    SUMMARIZE_TIMEOUT_SEC,
    get_pg_conn,
    get_pg_engine,
    get_schema_context,
    summarize_conversation_memo,
)
from agent import ui_es as UI
from agent.renderers import dispatch

st.set_page_config(page_title=UI.PAGE_TITLE, page_icon="🏠", layout="wide")


def _setup_rooster_logging() -> None:
    """
    Send Rooster agent logs to stderr (Streamlit terminal) for debugging.
    Set ROOSTER_LOG_LEVEL=DEBUG|INFO|WARNING (default INFO).
    """
    level_name = (os.getenv("ROOSTER_LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO
    log = logging.getLogger("rooster")
    log.setLevel(level)
    if log.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    log.addHandler(handler)
    log.propagate = False


_setup_rooster_logging()
_LOG = logging.getLogger("rooster.chat")


@st.cache_resource
def _cached_schema_context() -> str:
    return get_schema_context()


@st.cache_data(ttl=300)
def _cached_live_schema_context() -> str:
    """Live DB snapshot for the planner; refreshed at most every 5 minutes."""
    return get_live_schema_context()


QUERY_TIMEOUT_SEC = 60

CHOROPLETH_METRICS = UI.CHOROPLETH_METRICS
CHOROPLETH_METRICS_PROFILE_ONLY_COLUMNS = UI.CHOROPLETH_METRICS_PROFILE_ONLY_COLUMNS


def _is_summary_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) > 220:
        return False
    triggers = (
        "summarise",
        "summarize",
        "summary",
        "wrap up",
        "wrap-up",
        "what have we found",
        "what we've found",
        "recap",
        "investment memo",
        "brief memo",
        "give me a summary",
        "resumen",
        "resúmen",
        "resume",
        "sintetiza",
        "sintetizar",
        "memo de inversión",
        "memo inversion",
        "conclusión",
        "conclusiones",
        "qué hemos visto",
        "que hemos visto",
        "cerrar",
        "cierra el resumen",
    )
    return any(x in t for x in triggers)


def _format_conversation_for_memo(messages: list) -> str:
    lines: list[str] = []
    for m in messages:
        if m.get("role") == "user":
            lines.append(f"User: {m.get('content', '')}")
        elif m.get("role") == "assistant":
            parts: list[str] = []
            if m.get("interpretation"):
                parts.append(str(m["interpretation"]))
            if m.get("summary"):
                parts.append(str(m["summary"]))
            if m.get("empty_narrative"):
                parts.append(str(m["empty_narrative"]))
            if m.get("error"):
                parts.append(f"(error: {m['error']})")
            body = " ".join(parts) if parts else "(no text reply)"
            lines.append(f"Rooster: {body}")
    return "\n".join(lines)


def build_conversation_context(chat_history: list, max_turns: int = 3) -> str:
    """Last N user/assistant turns as plain text for the combined LLM (no tables)."""
    recent = chat_history[-(max_turns * 2) :]
    lines: list[str] = []
    for msg in recent:
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                lines.append(f"User: {c.strip()}")
        else:
            t = _assistant_context_line(msg)
            if t:
                lines.append(f"Rooster: {t}")
    return "\n".join(lines)


def _assistant_context_line(msg: dict) -> str | None:
    if msg.get("error"):
        return f"[{UI.ASSISTANT_ERROR_PREFIX.format(err=str(msg['error'])[:200])}]"
    if msg.get("empty"):
        return (msg.get("empty_narrative") or UI.ASSISTANT_EMPTY_SEARCH).strip()
    parts: list[str] = []
    if msg.get("interpretation"):
        parts.append(msg["interpretation"])
    if msg.get("summary"):
        parts.append(msg["summary"])
    elif msg.get("rows"):
        parts.append(UI.ROWS_RETURNED.format(n=len(msg["rows"])))
    if msg.get("agent_turn") and msg.get("render_stack"):
        rs = msg["render_stack"]
        parts.append(
            "["
            + ", ".join(str(b.get("intent") or "?") for b in rs)
            + "]"
        )
    if parts:
        return " ".join(parts)
    return None


_ANALYTICS_SETUP_MD = UI.ANALYTICS_SETUP_MD


@st.cache_data(ttl=300, show_spinner=False)
def _analytics_views_available() -> dict[str, bool]:
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


@st.cache_data(ttl=300, show_spinner=False)
def _neighborhood_profile_available() -> bool:
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


def _warn_analytics_missing_once() -> None:
    av = _analytics_views_available()
    if av["listing_summary"] and av["neighborhood_metrics"]:
        return
    if st.session_state.get("_rooster_analytics_warned"):
        return
    st.session_state["_rooster_analytics_warned"] = True
    st.warning(_ANALYTICS_SETUP_MD)


TIMEOUT_PHASE_HINTS = UI.TIMEOUT_PHASE_HINTS


def _dataframe_with_links(df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return
    col_cfg = {}
    if "url" in df.columns:
        col_cfg["url"] = st.column_config.LinkColumn("Anuncio", display_text="Abrir ↗")
    st.dataframe(
        df,
        column_config=col_cfg or None,
        use_container_width=True,
        hide_index=True,
    )


def _build_neighborhood_geojson(metric_column: str) -> tuple[dict, float, float]:
    """Return GeoJSON FeatureCollection and (vmin, vmax). Uses analytics.neighborhood_profile when present, else neighborhood_metrics, else core.neighborhoods (no metric shading)."""
    allowed = set(CHOROPLETH_METRICS.values())
    if metric_column not in allowed:
        raise ValueError("Invalid metric column")

    conn = get_pg_conn()
    rows: list[tuple] = []
    try:
        cur = conn.cursor()
        profile_ok = _neighborhood_profile_available()

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


@st.cache_data(ttl=600, show_spinner="Cargando parcelas catastrales…")
def _load_parcels_geojson() -> dict:
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


@st.cache_data(ttl=600, show_spinner=False)
def _load_intel_transit_overlay() -> pd.DataFrame:
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


@st.cache_data(ttl=600, show_spinner=False)
def _load_intel_tourism_overlay() -> pd.DataFrame:
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


_INTEL_TRANSIT_COLORS: dict[str, str] = {
    "station": "#534AB7",
    "halt": "#534AB7",
    "stop_position": "#1D9E75",
    "tram_stop": "#534AB7",
    "": "#888780",
}


def _intel_toggle(label: str, *, value: bool = False, key: str) -> bool:
    """Pill-style on/off; prefers ``st.toggle`` when the Streamlit version supports it."""
    fn = getattr(st, "toggle", None)
    if callable(fn):
        return bool(fn(label, value=value, key=key))
    return st.checkbox(label, value=value, key=key)


def _render_intel_explorer_map(
    *,
    show_barrios: bool,
    metric_column: str,
    metric_caption: str,
    show_transport: bool,
    show_vut: bool,
    show_listings: bool,
    show_parcels: bool,
    height: int = 560,
    folium_key: str = "rooster_intel_explorer",
) -> None:
    """Blank base map; layers added only for toggles that are on. No Folium LayerControl."""
    m = folium.Map(location=[39.47, -0.37], zoom_start=13, tiles="CartoDB dark_matter")

    if show_barrios:
        try:
            fc, vmin, vmax = _build_neighborhood_geojson(metric_column)
        except Exception as e:
            st.error(UI.MAP_LOAD_ERR.format(err=e))
            st.info(_ANALYTICS_SETUP_MD)
            return

        if not fc.get("features"):
            st.warning(UI.MAP_NO_POLYGONS)

        if vmin >= vmax:
            vmax = vmin + 1.0
        cmap = LinearColormap(
            colors=["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30"],
            vmin=vmin,
            vmax=vmax,
            caption=metric_caption,
        )

        def style_fn(feature: dict) -> dict:
            v = feature["properties"].get("value")
            if v is None:
                return {
                    "fillColor": "#d9d9d9",
                    "fillOpacity": 0.35,
                    "color": "#999",
                    "weight": 0.5,
                }
            return {
                "fillColor": cmap(float(v)),
                "fillOpacity": 0.72,
                "color": "#333333",
                "weight": 0.5,
            }

        folium.GeoJson(
            fc,
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "value", "listings", "yield_pct"],
                aliases=[UI.TOOLTIP_NEIGHBORHOOD, metric_caption, UI.TOOLTIP_LISTINGS, UI.TOOLTIP_YIELD],
                localize=True,
            ),
            popup=folium.GeoJsonPopup(
                fields=["name", "value", "listings", "yield_pct"],
                aliases=[UI.TOOLTIP_NEIGHBORHOOD, metric_caption, UI.TOOLTIP_LISTINGS, UI.TOOLTIP_YIELD],
                localize=True,
            ),
        ).add_to(m)
        cmap.add_to(m)

    if show_transport:
        to = _load_intel_transit_overlay()
        for _, row in to.dropna(subset=["lat", "lng"]).iterrows():
            st_raw = row.get("stop_type")
            sk = (
                str(st_raw).lower()
                if st_raw is not None and not (isinstance(st_raw, float) and pd.isna(st_raw))
                else ""
            ) or ""
            color = _INTEL_TRANSIT_COLORS.get(sk, "#888780")
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lng"])],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(
                    f"<b>{row.get('name') or '—'}</b><br>{st_raw or ''}",
                    max_width=140,
                ),
            ).add_to(m)

    if show_vut:
        tour = _load_intel_tourism_overlay()
        for _, row in tour.dropna(subset=["lat", "lng"]).iterrows():
            addr = str(row.get("address") or "")[:60]
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lng"])],
                radius=4,
                color="#EF9F27",
                fill=True,
                fill_opacity=0.65,
                weight=1,
                popup=folium.Popup(addr, max_width=160),
            ).add_to(m)

    if show_listings:
        lo = _load_listings_for_charts("All")
        if not lo.empty and "lat" in lo.columns and "lng" in lo.columns:
            lm = lo.dropna(subset=["lat", "lng"])
            if not lm.empty and "eur_per_sqm" in lm.columns:
                vals = pd.to_numeric(lm["eur_per_sqm"], errors="coerce").dropna()
                if not vals.empty:
                    vmin_l = float(vals.quantile(0.05))
                    vmax_l = float(vals.quantile(0.95))
                    if vmin_l >= vmax_l:
                        vmax_l = vmin_l + 1.0
                    lcmap = LinearColormap(
                        colors=["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30"],
                        vmin=vmin_l,
                        vmax=vmax_l,
                        caption=UI.LEGEND_EUR_M2,
                    )
                    for _, row in lm.iterrows():
                        raw = row.get("eur_per_sqm")
                        try:
                            fv = float(raw) if raw is not None and not (isinstance(raw, float) and pd.isna(raw)) else vmin_l
                        except (TypeError, ValueError):
                            fv = vmin_l
                        c = lcmap(max(vmin_l, min(vmax_l, fv)))
                        price = int(row["price_int"]) if pd.notna(row.get("price_int")) else 0
                        url = row.get("url") or "#"
                        folium.CircleMarker(
                            location=[float(row["lat"]), float(row["lng"])],
                            radius=4,
                            color=c,
                            fill=True,
                            fill_opacity=0.8,
                            weight=1,
                            popup=folium.Popup(
                                f"<b>€{price:,}</b> · {row.get('area_sqm', '—')} m²<br>"
                                f"<a href=\"{url}\" target=\"_blank\">{UI.POPUP_IDEALISTA}</a>",
                                max_width=200,
                            ),
                        ).add_to(m)
                    lcmap.add_to(m)

    if show_parcels:
        try:
            parcel_fc = _load_parcels_geojson()
        except Exception as e:
            st.error(UI.PARCEL_LAYER_ERR.format(err=e))
            parcel_fc = None
        if parcel_fc and parcel_fc.get("features"):
            folium.GeoJson(
                parcel_fc,
                style_function=lambda _f: {
                    "fillColor": "#00000000",
                    "fillOpacity": 0,
                    "color": "#1a5276",
                    "weight": 0.35,
                    "opacity": 0.55,
                },
            ).add_to(m)
            st.caption(UI.PARCEL_CAPTION.format(n=len(parcel_fc["features"])))

    st_folium(
        m,
        use_container_width=True,
        height=height,
        returned_objects=[],
        key=folium_key,
    )

    if show_barrios and metric_column == "investment_score" and _neighborhood_profile_available():
        st.caption(UI.MAP_LEGEND_INVESTMENT)


@st.cache_data(ttl=300, show_spinner="Cargando anuncios…")
def _load_listings_for_charts(operation_filter: str) -> pd.DataFrame:
    """Listings + canonical barrio name and metrics via neighborhood_id (spatial FK)."""
    if operation_filter not in ("All", "venta", "alquiler"):
        operation_filter = "All"
    has_nm = _analytics_views_available().get("neighborhood_metrics", False)
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


@st.cache_data(ttl=300, show_spinner=False)
def _load_market_framing_metrics() -> dict | None:
    """City-wide median yield, yield spread (pp), and count of barrios >6% yield."""
    if not _analytics_views_available().get("neighborhood_metrics"):
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


@st.cache_data(ttl=300, show_spinner=False)
def _load_neighborhood_ranking_df() -> pd.DataFrame:
    if not _analytics_views_available().get("neighborhood_metrics"):
        return pd.DataFrame()
    eng = get_pg_engine()
    if _neighborhood_profile_available():
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


@st.cache_data(ttl=300, show_spinner=False)
def _load_yield_liquidity_df() -> pd.DataFrame:
    if not _analytics_views_available().get("neighborhood_metrics"):
        return pd.DataFrame()
    eng = get_pg_engine()
    if _neighborhood_profile_available():
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


def _render_intelligence_framing_metrics(m: dict | None) -> None:
    if not m:
        st.info(UI.INTEL_NO_FRAMING)
        return
    med = m.get("city_median_yield")
    sp = m.get("spread_pp")
    ns = m.get("n_strong")
    nw = m.get("n_with_yield")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        UI.INTEL_METRIC_MEDIAN_YIELD,
        f"{float(med):.1f}%" if med is not None and pd.notna(med) else "—",
        help=UI.INTEL_METRIC_MEDIAN_YIELD_HELP,
    )
    c2.metric(
        UI.INTEL_METRIC_SPREAD,
        f"{float(sp):.1f} pp" if sp is not None and pd.notna(sp) else "—",
        help=UI.INTEL_METRIC_SPREAD_HELP,
    )
    c3.metric(
        UI.INTEL_METRIC_STRONG_YIELD,
        f"{int(ns)} de {int(nw)}" if ns is not None and nw is not None else "—",
        help=UI.INTEL_METRIC_STRONG_YIELD_HELP,
    )


def _render_neighborhood_ranking_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info(UI.INTEL_NO_RANKING)
        return
    show = df.copy()
    if "tourist_pct" in show.columns:
        show["tourist_pct"] = pd.to_numeric(show["tourist_pct"], errors="coerce").fillna(0)
    if show["transport_rating"].notna().any():
        show["motivo"] = show.apply(
            lambda r: f"{r['yield_pct']:.1f}% rent. · {r.get('transport_rating') or '—'} · {r.get('tourism_pressure') or '—'}",
            axis=1,
        )
    else:
        show["motivo"] = show.apply(
            lambda r: f"{r['yield_pct']:.1f}% rent. · {int(r['venta_count'])} venta / {int(r['alquiler_count'])} alquiler",
            axis=1,
        )
    cfg: dict = {
        "neighborhood_name": st.column_config.TextColumn("Barrio", width="medium"),
        "yield_pct": st.column_config.ProgressColumn(
            "Rent. %",
            min_value=0,
            max_value=10,
            format="%.1f%%",
        ),
        "eur_per_sqm": st.column_config.NumberColumn("€/m² (venta)", format="€%d"),
        "transit_stop_count": st.column_config.NumberColumn(
            "Paradas transporte",
            help="Paradas en el polígono del barrio",
        ),
        "transport_rating": st.column_config.TextColumn("Transporte"),
        "tourist_pct": st.column_config.ProgressColumn(
            "Densidad turística %",
            min_value=0,
            max_value=50,
            format="%.1f%%",
            help="Mayor % = más competencia tipo Airbnb / riesgo regulatorio",
        ),
        "tourism_pressure": st.column_config.TextColumn("Presión turística"),
        "score": st.column_config.ProgressColumn(
            "Puntuación inversión",
            min_value=0,
            max_value=10,
            format="%.1f",
        ),
        "venta_count": st.column_config.NumberColumn(
            "Anuncios venta",
            help="Indicador de liquidez",
        ),
        "alquiler_count": st.column_config.NumberColumn("Anuncios alquiler"),
        "motivo": st.column_config.TextColumn("Resumen", width="large"),
    }
    for k in list(cfg.keys()):
        if k not in show.columns:
            del cfg[k]
    st.dataframe(
        show,
        column_config=cfg,
        hide_index=True,
        use_container_width=True,
        height=420,
    )
    st.caption(UI.INTEL_RANK_CAPTION)


def _render_yield_liquidity_scatter(df: pd.DataFrame) -> None:
    if df.empty:
        st.info(UI.INTEL_NO_SCATTER)
        return
    if df["yield_pct"].notna().sum() < 2:
        st.info(UI.INTEL_NO_SCATTER)
        return
    has_score = df["investment_score"].notna().any()
    if has_score:
        fig = px.scatter(
            df,
            x="listing_count",
            y="yield_pct",
            size="eur_per_sqm",
            color="investment_score",
            color_continuous_scale=["#3B8BD4", "#1D9E75", "#EF9F27"],
            hover_name="neighborhood_name",
            hover_data={
                "transport_rating": True,
                "tourism_pressure": True,
                "listing_count": True,
                "yield_pct": ":.1f",
                "investment_score": ":.1f",
                "eur_per_sqm": False,
            },
            template="plotly_dark",
            labels={
                "listing_count": "Anuncios totales (liquidez →)",
                "yield_pct": "Rentabilidad bruta % (rentabilidad ↑)",
                "investment_score": "Puntuación",
            },
        )
        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="Puntuación"),
        )
    else:
        fig = px.scatter(
            df,
            x="listing_count",
            y="yield_pct",
            size="eur_per_sqm",
            hover_name="neighborhood_name",
            template="plotly_dark",
            labels={
                "listing_count": "Anuncios totales (liquidez →)",
                "yield_pct": "Rentabilidad bruta % (rentabilidad ↑)",
            },
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0))
    y_med = float(df["yield_pct"].median())
    x_med = float(df["listing_count"].median())
    # Plotly shape lines do not accept 8-digit #RRGGBBAA; use rgba for translucent guides.
    guide_line = "rgba(255,255,255,0.12)"
    fig.add_hline(
        y=y_med,
        line_dash="dot",
        line_color=guide_line,
        annotation_text="Mediana rentabilidad ciudad",
        annotation_position="right",
    )
    fig.add_vline(
        x=x_med,
        line_dash="dot",
        line_color=guide_line,
        annotation_text="Mediana oferta",
        annotation_position="top",
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(UI.INTEL_SCATTER_CAPTION)


def _load_chat_briefing_top_pick() -> dict | None:
    if not _analytics_views_available().get("neighborhood_metrics"):
        return None
    eng = get_pg_engine()
    if _neighborhood_profile_available():
        try:
            df = pd.read_sql_query(
                """
                SELECT neighborhood_name, investment_score, gross_rental_yield_pct,
                       transport_rating, tourism_pressure, venta_count
                FROM analytics.neighborhood_profile
                WHERE venta_count >= 5
                  AND gross_rental_yield_pct IS NOT NULL
                ORDER BY investment_score DESC NULLS LAST
                LIMIT 1
                """,
                eng,
            )
            if not df.empty:
                return df.iloc[0].to_dict()
        except Exception:
            pass
    try:
        df = pd.read_sql_query(
            """
            SELECT neighborhood_name,
                   gross_rental_yield_pct AS investment_score,
                   gross_rental_yield_pct,
                   NULL::text AS transport_rating,
                   NULL::text AS tourism_pressure,
                   venta_count
            FROM analytics.neighborhood_metrics
            WHERE venta_count >= 5
              AND gross_rental_yield_pct IS NOT NULL
            ORDER BY gross_rental_yield_pct DESC NULLS LAST
            LIMIT 1
            """,
            eng,
        )
        if not df.empty:
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return None


def _load_chat_briefing_best_value() -> tuple[dict | None, float | None]:
    """Lowest €/m² among barrios with yield above city median."""
    if not _analytics_views_available().get("neighborhood_metrics"):
        return None, None
    eng = get_pg_engine()
    try:
        med_city = pd.read_sql_query(
            """
            SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gross_rental_yield_pct) AS med
            FROM analytics.neighborhood_metrics
            WHERE gross_rental_yield_pct IS NOT NULL
            """,
            eng,
        ).iloc[0]["med"]
        row = pd.read_sql_query(
            """
            WITH city AS (
                SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gross_rental_yield_pct) AS med
                FROM analytics.neighborhood_metrics
                WHERE gross_rental_yield_pct IS NOT NULL
            )
            SELECT nm.neighborhood_name, nm.median_venta_eur_per_sqm, nm.gross_rental_yield_pct
            FROM analytics.neighborhood_metrics nm, city
            WHERE nm.gross_rental_yield_pct > city.med
              AND nm.median_venta_eur_per_sqm IS NOT NULL
              AND nm.median_venta_eur_per_sqm > 0
              AND nm.venta_count >= 3
            ORDER BY nm.median_venta_eur_per_sqm ASC NULLS LAST
            LIMIT 1
            """,
            eng,
        )
    except Exception:
        return None, None
    med_f = float(med_city) if med_city is not None and pd.notna(med_city) else None
    if row.empty:
        return None, med_f
    return row.iloc[0].to_dict(), med_f


def _load_chat_briefing_price_signal() -> dict | None:
    eng = get_pg_engine()
    try:
        df = pd.read_sql_query(
            """
            WITH ranked_dates AS (
                SELECT scraped_at,
                       ROW_NUMBER() OVER (ORDER BY scraped_at DESC) AS rn
                FROM (SELECT DISTINCT scraped_at FROM core.listing_snapshots) t
            ),
            t1 AS (SELECT scraped_at FROM ranked_dates WHERE rn = 1),
            t2 AS (SELECT scraped_at FROM ranked_dates WHERE rn = 2)
            SELECT
                n.name AS neighborhood_name,
                AVG(s1.price_int)::numeric AS cur_avg,
                AVG(s2.price_int)::numeric AS prev_avg,
                COUNT(*)::int AS n_listings
            FROM core.listing_snapshots s1
            JOIN core.listing_snapshots s2 ON s2.url = s1.url
            JOIN core.listings l ON l.url = s1.url
            JOIN core.neighborhoods n ON n.id = l.neighborhood_id
            WHERE l.operation = 'venta'
              AND s1.scraped_at = (SELECT scraped_at FROM t1)
              AND s2.scraped_at = (SELECT scraped_at FROM t2)
              AND EXISTS (SELECT 1 FROM ranked_dates WHERE rn = 2)
              AND s1.price_int > 0 AND s2.price_int > 0
            GROUP BY n.name
            HAVING COUNT(*) >= 3
               AND AVG(s2.price_int) > 0
            ORDER BY ABS(
                (AVG(s1.price_int) - AVG(s2.price_int)) / AVG(s2.price_int)
            ) DESC
            LIMIT 1
            """,
            eng,
        )
    except Exception:
        return None
    if df.empty:
        return None
    r = df.iloc[0]
    cur = float(r["cur_avg"])
    prev = float(r["prev_avg"])
    if prev == 0:
        return None
    chg = (cur - prev) / prev * 100.0
    return {
        "neighborhood_name": r["neighborhood_name"],
        "change_pct": chg,
        "n": int(r["n_listings"]),
    }


def _render_chat_briefing_cards() -> None:
    """Empty-state briefing: top pick, best €/m², price movement signal."""
    c1, c2, c3 = st.columns(3)
    top = _load_chat_briefing_top_pick()
    with c1:
        st.markdown(f"**{UI.BRIEF_TOP_PICK}**")
        if top:
            st.markdown(
                UI.BRIEF_TOP_PICK_BODY.format(
                    name=str(top["neighborhood_name"]),
                    score=float(top["investment_score"]),
                    yld=float(top["gross_rental_yield_pct"]),
                    tr=str(top.get("transport_rating") or "—"),
                    tp=str(top.get("tourism_pressure") or "—"),
                    vc=int(top["venta_count"]),
                )
            )
        else:
            st.caption(UI.BRIEF_NO_TOP_PICK)
    val, city_med = _load_chat_briefing_best_value()
    with c2:
        st.markdown(f"**{UI.BRIEF_BEST_VALUE}**")
        if val and city_med is not None:
            st.markdown(
                UI.BRIEF_VALUE_BODY.format(
                    name=str(val["neighborhood_name"]),
                    m2=int(val["median_venta_eur_per_sqm"]),
                    yld=float(val["gross_rental_yield_pct"]),
                    med=city_med,
                )
            )
        else:
            st.caption(UI.BRIEF_NO_VALUE)
    sig = _load_chat_briefing_price_signal()
    with c3:
        st.markdown(f"**{UI.BRIEF_MARKET_SIGNAL}**")
        if sig:
            st.markdown(
                UI.BRIEF_SIGNAL_BODY.format(
                    name=str(sig["neighborhood_name"]),
                    chg=float(sig["change_pct"]),
                    n=int(sig["n"]),
                )
            )
        else:
            st.caption(UI.BRIEF_NO_SIGNAL)


def _floor_label_for_chart(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        v = int(x)
    except (TypeError, ValueError):
        return ""
    if v == 0:
        return UI.FLOOR_GROUND
    if v == 99:
        return UI.FLOOR_PENTHOUSE
    return UI.FLOOR_N.format(n=v)


def _floor_sort_key(lbl: str) -> tuple:
    if lbl == UI.FLOOR_GROUND:
        return (0,)
    if lbl == UI.FLOOR_PENTHOUSE:
        return (999,)
    if lbl.startswith("Planta "):
        try:
            return (1, int(lbl.split()[1]))
        except (ValueError, IndexError):
            return (2, lbl)
    return (2, lbl)


def render_dashboard() -> None:
    st.subheader(UI.PAGE_SUBTITLE_INTEL)
    st.caption(UI.INTEL_CAPTION)
    _warn_analytics_missing_once()

    st.markdown(UI.INTEL_S1_TITLE)
    _render_intelligence_framing_metrics(_load_market_framing_metrics())

    st.markdown(UI.INTEL_S2_TITLE)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        show_barrios = _intel_toggle(UI.INTEL_TOGGLE_BARRIOS, value=False, key="intel_t_barrios")
    with col2:
        show_transport = _intel_toggle(UI.INTEL_TOGGLE_TRANSPORT, value=False, key="intel_t_transport")
    with col3:
        show_vut = _intel_toggle(UI.INTEL_TOGGLE_VUT, value=False, key="intel_t_vut")
    with col4:
        show_listings = _intel_toggle(UI.INTEL_TOGGLE_LISTINGS, value=False, key="intel_t_listings")
    with col5:
        show_parcels = _intel_toggle(UI.INTEL_TOGGLE_PARCELS, value=False, key="intel_t_parcels")

    metric_col = "total_count"
    metric_caption = ""
    if show_barrios:
        choropleth_labels = (
            list(CHOROPLETH_METRICS.keys())
            if _neighborhood_profile_available()
            else [
                k
                for k, v in CHOROPLETH_METRICS.items()
                if v not in CHOROPLETH_METRICS_PROFILE_ONLY_COLUMNS
            ]
        )
        metric_label = st.selectbox(
            UI.INTEL_MAP_COLOR_BY,
            choropleth_labels,
            label_visibility="collapsed",
            key="intel_choropleth_metric_pill",
        )
        metric_col = CHOROPLETH_METRICS[metric_label]
        metric_caption = metric_label.replace("Median ", "").strip()

    _render_intel_explorer_map(
        show_barrios=show_barrios,
        metric_column=metric_col,
        metric_caption=metric_caption,
        show_transport=show_transport,
        show_vut=show_vut,
        show_listings=show_listings,
        show_parcels=show_parcels,
        height=560,
        folium_key="rooster_intel_explorer",
    )

    st.markdown(UI.INTEL_S3_TITLE)
    if not _analytics_views_available().get("neighborhood_metrics"):
        st.info(_ANALYTICS_SETUP_MD)
    else:
        _render_neighborhood_ranking_table(_load_neighborhood_ranking_df())

    st.markdown(UI.INTEL_S4_TITLE)
    if not _analytics_views_available().get("neighborhood_metrics"):
        st.info(_ANALYTICS_SETUP_MD)
    else:
        _render_yield_liquidity_scatter(_load_yield_liquidity_df())


def _replay_message(msg: dict, idx: int = 0) -> None:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
        return
    with st.chat_message("assistant", avatar="🐓"):
        if msg.get("error"):
            st.error(msg["error"])
            return
        if msg.get("empty"):
            if msg.get("empty_narrative"):
                for block in msg["empty_narrative"].strip().split("\n\n"):
                    if block.strip():
                        st.markdown(block.strip())
            else:
                st.info(UI.CHAT_EMPTY)
            return
        if "render_stack" in msg:
            stack = msg.get("render_stack") or []
            summary = (msg.get("summary") or "").strip()
            if summary:
                st.markdown(summary.replace("**", "").replace("__", ""))
            val_errs = msg.get("validation_errors") or []
            for block in stack:
                intent = block.get("intent") or "search"
                rows = block.get("rows") or []
                meta = dict(block.get("meta") or {})
                meta["geo_key"] = idx
                if intent == "ranking":
                    meta.setdefault("metric_label", UI.RANKING_METRIC_DEFAULT)
                dispatch(intent, rows, meta, "")
            if val_errs:
                st.caption("Note: " + ". ".join(str(e) for e in val_errs))
            return
        intent = msg.get("intent", "search")
        if msg.get("is_memo") or intent == "memo":
            dispatch(
                "memo",
                [],
                {"memo_text": (msg.get("summary") or "").strip(), "geo_key": idx},
                "",
            )
            return
        if intent == "combined_map":
            dispatch(
                "combined_map",
                [],
                {
                    "geo_key": idx,
                    "rows_listings": msg.get("rows_listings") or [],
                    "rows_transit": msg.get("rows_transit") or [],
                    "rows_tourism": msg.get("rows_tourism") or [],
                    "caveat": (msg.get("caveat") or "").strip(),
                },
                msg.get("summary") or "",
            )
            return
        rows = msg.get("rows")
        meta: dict = {
            "geo_key": idx,
            "caveat": (msg.get("caveat") or "").strip(),
        }
        if intent == "chart":
            meta["chart_type"] = msg.get("chart_type") or "scatter"
        if intent == "ranking":
            meta["metric_label"] = (msg.get("reasoning_focus") or "").strip() or UI.RANKING_METRIC_DEFAULT
        dispatch(intent, rows or [], meta, msg.get("summary") or "")


def _dispatch_render_stack_blocks(render_stack: list, geo_key: int) -> None:
    """Render agent render_stack (maps, tables, charts) without duplicate prose."""
    for block in render_stack:
        intent = block.get("intent") or "search"
        rows = block.get("rows") or []
        meta = dict(block.get("meta") or {})
        meta["geo_key"] = geo_key
        if intent == "ranking":
            meta.setdefault("metric_label", UI.RANKING_METRIC_DEFAULT)
        dispatch(intent, rows, meta, "")


def render_chat() -> None:
    st.markdown(f"### {UI.PAGE_SUBTITLE_CHAT}")
    _warn_analytics_missing_once()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = {
            "neighborhoods_discussed": [],
            "user_priorities": [],
            "operation_focus": "venta",
            "price_ceiling": None,
            "stage": "orienting",
            "last_intent": None,
            "turns": 0,
        }

    timeout_sec = float(st.session_state.get("query_timeout", QUERY_TIMEOUT_SEC))
    model_choice = st.session_state.get("selected_model") or DEFAULT_SYNTHESISER_MODEL_OPENAI

    if not st.session_state.messages:
        _render_chat_briefing_cards()

    for i, msg in enumerate(st.session_state.messages):
        _replay_message(msg, idx=i)

    user_input = st.chat_input(UI.CHAT_INPUT_PLACEHOLDER)
    if st.session_state.get("ask_auto_submit"):
        user_input = st.session_state.pop("ask_auto_submit")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if _is_summary_request(user_input):
            progress_memo: dict = {}
            memo_text = ""
            with st.status(UI.STATUS_THINKING, expanded=False) as status:
                try:
                    status.update(label=UI.STATUS_MEMO)
                    memo_text = summarize_conversation_memo(
                        _format_conversation_for_memo(st.session_state.messages),
                        model_choice,
                        progress_memo,
                        timeout_sec=min(float(timeout_sec), 60.0),
                    )
                    status.update(label=UI.STATUS_DONE, state="complete")
                except Exception as e:
                    status.update(label=UI.STATUS_ERROR, state="error")
                    memo_text = UI.MEMO_ERR.format(err=e)
            st.session_state.messages.append({
                "role": "assistant",
                "is_memo": True,
                "summary": memo_text,
                "intent": "memo",
                "rows": None,
                "sql": None,
                "error": None,
                "empty": False,
                "followups": [],
                "confidence": "high",
                "interpretation": "",
                "reasoning_focus": "",
                "caveat": "",
            })
            st.rerun()

        # Fast path: tiny router (gpt-4o-mini) — no engine, schema, tools, or FC
        if is_conversational_message(user_input):
            response_text = pick_fast_path_conversational_reply(user_input)
            plan_conv = {"reasoning": "conversational_fast_path", "tool_calls": []}
            with st.chat_message("assistant", avatar="🐓"):
                st.write_stream(stream_canned_text_word_by_word(response_text))
            st.session_state.conversation_state = update_conversation_state(
                st.session_state.conversation_state,
                user_input,
                plan_conv,
                [],
            )
            st.session_state.messages.append({
                "role": "assistant",
                "intent": "conversational",
                "summary": response_text,
                "render_stack": [],
                "validation_errors": [],
                "rows": None,
                "sql": None,
                "error": None,
                "empty": False,
                "followups": [],
                "confidence": "high",
                "interpretation": "",
                "reasoning_focus": "",
                "caveat": "",
            })
            st.rerun()

        stream_kind: str | None = None
        plan_for_stream: dict | None = None
        validated_for_stream: dict | None = None
        execution_for_stream: list | None = None
        render_stack_for_stream: list | None = None
        confirmed_for_stream: str | None = None
        val_errs_for_stream: list = []
        openai_precomputed: str | None = None
        openai_fc_final_messages: list | None = None
        openai_fc_max_tokens: int = 0
        had_output_correction: bool = False

        engine = get_pg_engine()
        conv = build_conversation_context(st.session_state.messages[:-1])
        last_assistant_for_planner = format_last_assistant_for_planner(
            st.session_state.messages[:-1]
        )

        with st.status(UI.STATUS_THINKING, expanded=False) as status:
            try:
                t_chat0 = time.perf_counter()
                status.update(label=UI.STATUS_QUERY)
                live = _cached_live_schema_context()
                static = _cached_schema_context()
                fc = run_openai_function_calling_pipeline(
                    user_input,
                    st.session_state.conversation_state,
                    live,
                    static,
                    conv,
                    model_choice,
                    float(timeout_sec),
                    engine,
                    last_assistant_context=last_assistant_for_planner,
                )
                had_output_correction = bool(fc.get("had_output_correction"))
                if fc.get("error") == "timeout":
                    st.session_state.messages.append({
                        "role": "assistant",
                        "intent": "conversational",
                        "summary": "Planner timed out. Try a narrower question.",
                        "render_stack": [],
                        "validation_errors": [],
                        "rows": None,
                        "sql": None,
                        "error": None,
                        "empty": False,
                        "followups": [],
                        "confidence": "low",
                        "interpretation": "",
                        "reasoning_focus": "",
                        "caveat": "",
                    })
                    status.update(label=UI.STATUS_DONE, state="complete")
                    st.rerun()
                if fc.get("validation_failed"):
                    top = extract_neighborhood_names_from_schema(
                        _cached_live_schema_context()
                    )[:5]
                    names_sample = ", ".join(top) if top else "…"
                    ve = fc.get("validation_errors") or []
                    summary_text = (
                        "No pude encontrar ese barrio en mis datos. "
                        f"Algunos barrios disponibles incluyen: {names_sample}…"
                    )
                    if ve:
                        summary_text += " " + " ".join(str(x) for x in ve[:3])
                    st.session_state.conversation_state = update_conversation_state(
                        st.session_state.conversation_state,
                        user_input,
                        fc.get("validated_plan") or {"tool_calls": []},
                        [],
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "agent_turn": True,
                        "summary": summary_text,
                        "render_stack": [],
                        "validation_errors": ve,
                        "intent": "conversational",
                        "rows": None,
                        "sql": None,
                        "error": None,
                        "empty": False,
                        "followups": [],
                        "confidence": "high",
                        "interpretation": "",
                        "reasoning_focus": "",
                        "caveat": "",
                    })
                    status.update(label=UI.STATUS_DONE, state="complete")
                    _LOG.info("FC validation dropped all tool_calls errors=%s", ve)
                    _LOG.info("CHAT total=%.2fs", time.perf_counter() - t_chat0)
                    st.rerun()
                if fc.get("conversational_text") is not None:
                    openai_precomputed = fc["conversational_text"]
                    stream_kind = "conversational"
                    plan_for_stream = {
                        "reasoning": "conversational",
                        "tool_calls": [],
                    }
                    status.update(label=UI.STATUS_DONE, state="complete")
                    _LOG.info(
                        "FC conversational text_len=%s",
                        len(openai_precomputed or ""),
                    )
                    _LOG.info("CHAT total=%.2fs", time.perf_counter() - t_chat0)
                elif fc.get("final_messages"):
                    validated_for_stream = fc["validated_plan"]
                    execution_for_stream = fc["execution_results"]
                    openai_fc_final_messages = fc["final_messages"]
                    openai_fc_max_tokens = int(fc.get("max_tokens_final") or 200)
                    val_errs_for_stream = list(
                        (validated_for_stream or {}).get("validation_errors") or []
                    )
                    _LOG.info(
                        "FC tools=%s",
                        [
                            (r.get("tool"), r.get("row_count"), r.get("renderer"))
                            for r in (execution_for_stream or [])
                        ],
                    )
                    geo_key = len(st.session_state.messages)
                    render_stack_for_stream = build_render_stack(
                        validated_for_stream or {},
                        execution_for_stream or [],
                        geo_key,
                    )
                    confirmed_for_stream = format_confirmed_visuals(
                        execution_for_stream or []
                    )
                    stream_kind = "full"
                    status.update(label=UI.STATUS_DONE, state="complete")
                    _LOG.info("CHAT total=%.2fs", time.perf_counter() - t_chat0)
                else:
                    _LOG.warning("FC unexpected branch: %s", fc)
                    openai_precomputed = (
                        "Lo siento, no pude procesar tu mensaje. ¿Puedes reformular?"
                    )
                    stream_kind = "conversational"
                    plan_for_stream = {
                        "reasoning": "conversational",
                        "tool_calls": [],
                    }
                    status.update(label=UI.STATUS_DONE, state="complete")
            except Exception as e:
                _LOG.exception("Chat pipeline failed: %s", e)
                status.update(label=UI.STATUS_ERROR, state="error")
                st.session_state.messages.append({
                    "role": "assistant",
                    "intent": "search",
                    "summary": "",
                    "rows": None,
                    "sql": None,
                    "error": str(e),
                    "empty": False,
                    "followups": [],
                    "confidence": "high",
                    "interpretation": "",
                    "caveat": "",
                })

        if stream_kind == "conversational" and plan_for_stream is not None:
            response_text = ""
            with st.chat_message("assistant", avatar="🐓"):
                if openai_precomputed is not None:
                    st.write(openai_precomputed)
                    response_text = openai_precomputed
                else:
                    response_text = run_synthesiser(
                        user_input,
                        plan_for_stream,
                        [],
                        st.session_state.conversation_state,
                        model_choice,
                        timeout_sec=float(SUMMARIZE_TIMEOUT_SEC),
                        confirmed_visuals=None,
                        max_tokens_override=CONVERSATIONAL_SYNTH_MAX_TOKENS,
                    )
                    st.write(response_text)
            st.session_state.conversation_state = update_conversation_state(
                st.session_state.conversation_state,
                user_input,
                plan_for_stream,
                [],
            )
            st.session_state.messages.append({
                "role": "assistant",
                "intent": "conversational",
                "summary": response_text or "",
                "render_stack": [],
                "validation_errors": [],
                "rows": None,
                "sql": None,
                "error": None,
                "empty": False,
                "followups": [],
                "confidence": "high",
                "interpretation": "",
                "reasoning_focus": "",
                "caveat": "",
            })
            st.rerun()

        elif stream_kind == "full" and validated_for_stream is not None:
            response_text = ""
            geo_key = len(st.session_state.messages)
            with st.chat_message("assistant", avatar="🐓"):
                try:
                    if openai_fc_final_messages:
                        stream = stream_openai_final_response_messages(
                            openai_fc_final_messages,
                            model_choice,
                            openai_fc_max_tokens,
                            float(timeout_sec),
                        )
                        response_text = st.write_stream(stream)
                    else:
                        response_text = run_synthesiser(
                            user_input,
                            validated_for_stream or {},
                            execution_for_stream or [],
                            st.session_state.conversation_state,
                            model_choice,
                            timeout_sec=float(SUMMARIZE_TIMEOUT_SEC),
                            confirmed_visuals=confirmed_for_stream,
                        )
                        st.write(response_text)
                except Exception:
                    response_text = run_synthesiser(
                        user_input,
                        validated_for_stream or {},
                        execution_for_stream or [],
                        st.session_state.conversation_state,
                        model_choice,
                        timeout_sec=float(SUMMARIZE_TIMEOUT_SEC),
                        confirmed_visuals=confirmed_for_stream,
                    )
                    st.write(response_text)
                _dispatch_render_stack_blocks(render_stack_for_stream or [], geo_key)
                if had_output_correction:
                    st.caption(UI.CHAT_OUTPUT_CORRECTED)
                if val_errs_for_stream:
                    st.caption(
                        "Nota: " + ". ".join(str(x) for x in val_errs_for_stream)
                    )
            st.session_state.conversation_state = update_conversation_state(
                st.session_state.conversation_state,
                user_input,
                validated_for_stream,
                execution_for_stream or [],
            )
            rs = render_stack_for_stream or []
            first_intent = rs[0]["intent"] if rs else "search"
            st.session_state.messages.append({
                "role": "assistant",
                "agent_turn": True,
                "summary": response_text or "",
                "render_stack": rs,
                "validation_errors": val_errs_for_stream,
                "intent": first_intent,
                "rows": None,
                "sql": None,
                "error": None,
                "empty": False,
                "followups": [],
                "confidence": "high",
                "interpretation": "",
                "reasoning_focus": "",
                "caveat": "",
            })
            st.rerun()
        else:
            st.rerun()


def main():
    models_pairs = [
        ("GPT-4o — máxima calidad", "gpt-4o"),
        ("GPT-4o mini — más rápido", "gpt-4o-mini"),
        ("GPT-4 Turbo", "gpt-4-turbo"),
    ]
    model_labels = [m[0] for m in models_pairs]
    model_values = [m[1] for m in models_pairs]

    if "query_timeout" not in st.session_state:
        st.session_state.query_timeout = 45
    if "selected_model_index" not in st.session_state:
        st.session_state.selected_model_index = 0
    if st.session_state.selected_model_index >= len(model_values):
        st.session_state.selected_model_index = 0
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_values[st.session_state.selected_model_index]

    st.title(UI.PAGE_TITLE)

    with st.sidebar:
        with st.expander(f"⚙ {UI.SETTINGS}", expanded=False):
            st.markdown(
                "<p style='font-size:12px;color:var(--text-color);opacity:0.6;"
                "margin-bottom:4px;'>Modelo</p>",
                unsafe_allow_html=True,
            )
            n_models = len(model_labels)
            idx = int(st.session_state.selected_model_index)
            if idx >= n_models:
                idx = 0
            selected_index = st.radio(
                "model_radio",
                options=list(range(n_models)),
                format_func=lambda i: model_labels[i],
                index=idx,
                label_visibility="collapsed",
            )
            st.session_state.selected_model_index = selected_index
            st.session_state.selected_model = model_values[selected_index]

            model_descriptions = {
                "gpt-4o": "Mejor razonamiento. Recomendado para análisis complejos.",
                "gpt-4o-mini": "3× más rápido. Bueno para preguntas directas.",
                "gpt-4-turbo": "Equilibrio entre calidad y velocidad.",
            }
            sm = st.session_state.selected_model
            if sm in model_descriptions:
                st.caption(model_descriptions[sm])

            st.divider()
            st.session_state.query_timeout = st.slider(
                "Timeout (segundos)",
                min_value=15,
                max_value=90,
                value=int(st.session_state.get("query_timeout", 45)),
                step=5,
                help="Tiempo máximo de espera por respuesta",
            )
            st.caption(UI.LLM_CAPTION)

    tab_intel, tab_chat = st.tabs([UI.TAB_INTEL, UI.TAB_CHAT])
    with tab_intel:
        render_dashboard()
    with tab_chat:
        render_chat()


if __name__ == "__main__":
    main()

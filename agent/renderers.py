"""
Ask Rooster output renderers — registry pattern.

Each renderer takes (rows, metadata). Add a new intent by defining a function
and registering it in RENDERERS. Only Streamlit entry points should import this module.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher, get_close_matches
from typing import Any

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from branca.colormap import LinearColormap
from psycopg2 import errors as pg_errors
from streamlit_folium import st_folium

from agent import ui_es as UI
from agent.listings_data import load_listings_frame
from agent.llm_sql import get_pg_conn, get_pg_engine

VALENCIA_CENTER = (39.47, -0.377)
CHORO_COLORS = ["#ffffcc", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]
GREEN_CHORO = ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#31a354", "#006d2c"]


def _plain_chat_prose(text: str) -> str:
    """Strip markdown bold markers so chat prose stays plain (model may still emit **)."""
    t = (text or "").strip()
    if not t:
        return t
    return t.replace("**", "").replace("__", "")


def _st_plotly_key(metadata: dict[str, Any] | None, *parts: str) -> str:
    """Unique ``key`` for ``st.plotly_chart`` when multiple chat turns replay charts."""
    gk = int((metadata or {}).get("geo_key", 0))
    suffix = "_".join(parts) if parts else "chart"
    return f"rooster_pl_{gk}_{suffix}"


def _st_folium_key(metadata: dict[str, Any] | None, *parts: str) -> str:
    """Unique ``key`` for ``st_folium`` when replaying chat history."""
    gk = int((metadata or {}).get("geo_key", 0))
    suffix = "_".join(parts) if parts else "map"
    return f"rooster_fm_{gk}_{suffix}"


_TRANSIT_STOP_COLORS: dict[str, str] = {
    "station": "#534AB7",
    "halt": "#534AB7",
    "stop_position": "#1D9E75",
    "tram_stop": "#534AB7",
    "": "#888780",
}


def _norm_name(s: str | None) -> str:
    """Lowercase, collapse spaces, strip accents for robust barrio name matching."""
    t = re.sub(r"\s+", " ", (s or "").strip().lower())
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t


def _align_key_map_to_feature_norms(
    key_map: dict[str, float],
    features: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Map each polygon feature's ``norm`` -> value by exact or fuzzy match on ``key_map`` keys.
    Fixes LLM/DB spelling drift (e.g. Russafa vs Ruzafa) so choropleths still render.
    """
    if not key_map or not features:
        return {}
    keys = list(key_map.keys())
    out: dict[str, float] = {}
    for f in features:
        props = f.get("properties") or {}
        fn = str(props.get("norm") or "")
        if not fn:
            continue
        if fn in key_map:
            out[fn] = key_map[fn]
            continue
        close = get_close_matches(fn, keys, n=1, cutoff=0.72)
        if close:
            out[fn] = key_map[close[0]]
            continue
        best_k: str | None = None
        best_r = 0.0
        for k in keys:
            r = SequenceMatcher(None, fn, k).ratio()
            if r > best_r:
                best_r = r
                best_k = k
        if best_k is not None and best_r >= 0.82:
            out[fn] = key_map[best_k]
    return out


def _find_col_df(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _truthy(x: Any) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("true", "t", "yes", "y", "1")


BELOW_MEDIAN_COLS = (
    "below_median",
    "is_below_median",
    "below_neighborhood_median",
    "price_below_median",
    "is_underpriced_listing",
)
def render_listing_table(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Listing search results: compact columns, deal checkbox — no row banding."""
    del metadata  # reserved
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption(UI.NO_ROWS_TABLE)
        return

    bm_src = _find_col_df(df, BELOW_MEDIAN_COLS)
    if bm_src and bm_src != "below_median":
        df = df.copy()
        df["below_median"] = df[bm_src]

    display_cols = [
        "url",
        "price_int",
        "area_sqm",
        "eur_per_sqm",
        "rooms_int",
        "floor_int",
        "has_parking",
        "has_terrace",
        "is_renovated",
        "below_median",
    ]
    present = [c for c in display_cols if c in df.columns]
    if not present:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    d = df[present].copy()

    if "floor_int" in d.columns:
        fi = pd.to_numeric(d["floor_int"], errors="coerce")
        d["floor_int"] = fi.round(0).astype("Int64")

    for bc in ("has_parking", "has_terrace", "is_renovated"):
        if bc in d.columns:
            d[bc] = d[bc].apply(lambda x: bool(_truthy(x)) if pd.notna(x) else False)
    if "below_median" in d.columns:
        d["below_median"] = d["below_median"].apply(
            lambda x: bool(_truthy(x)) if pd.notna(x) else False
        )

    col_cfg: dict = {}
    if "url" in d.columns:
        col_cfg["url"] = st.column_config.LinkColumn(
            UI.COL_LINK, display_text=UI.COL_VIEW, width="small"
        )
    if "price_int" in d.columns:
        col_cfg["price_int"] = st.column_config.NumberColumn(
            UI.COL_PRICE, format="€%d", width="small"
        )
    if "area_sqm" in d.columns:
        col_cfg["area_sqm"] = st.column_config.NumberColumn(
            UI.COL_AREA, format="%d m²", width="small"
        )
    if "eur_per_sqm" in d.columns:
        col_cfg["eur_per_sqm"] = st.column_config.NumberColumn(
            UI.SEARCH_TABLE_EUR_M2, format="€%d", width="small"
        )
    if "rooms_int" in d.columns:
        col_cfg["rooms_int"] = st.column_config.NumberColumn(
            UI.SEARCH_TABLE_ROOMS, format="%d", width="small"
        )
    if "floor_int" in d.columns:
        col_cfg["floor_int"] = st.column_config.NumberColumn(
            UI.SEARCH_TABLE_FLOOR, format="%d", width="small"
        )
    if "has_parking" in d.columns:
        col_cfg["has_parking"] = st.column_config.CheckboxColumn(
            UI.SEARCH_TABLE_P, width="small", help=UI.SEARCH_TABLE_P_HELP
        )
    if "has_terrace" in d.columns:
        col_cfg["has_terrace"] = st.column_config.CheckboxColumn(
            UI.SEARCH_TABLE_T, width="small", help=UI.SEARCH_TABLE_T_HELP
        )
    if "is_renovated" in d.columns:
        col_cfg["is_renovated"] = st.column_config.CheckboxColumn(
            UI.SEARCH_TABLE_R, width="small", help=UI.SEARCH_TABLE_R_HELP
        )
    if "below_median" in d.columns:
        col_cfg["below_median"] = st.column_config.CheckboxColumn(
            UI.SEARCH_TABLE_DEAL,
            width="small",
            help=UI.SEARCH_TABLE_DEAL_HELP,
        )

    st.dataframe(
        d,
        column_config=col_cfg,
        hide_index=True,
        use_container_width=True,
    )

    cap = UI.LISTINGS_COUNT.format(n=len(d))
    if "below_median" in d.columns:
        cap += UI.LISTINGS_CAPTION_BELOW
    st.caption(cap)


def render_geo_map(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Point map inside chat — listings with lat/lng colored by €/m²."""
    meta = metadata or {}
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption(UI.NO_ROWS_MAP)
        return
    if "lat" not in df.columns or "lng" not in df.columns:
        st.caption(UI.NO_MAPPABLE_LATLNG)
        return
    df = df.dropna(subset=["lat", "lng"])
    if df.empty:
        st.caption(UI.NO_MAPPABLE_FOUND)
        return

    col_e = "eur_per_sqm"
    if col_e not in df.columns:
        if "price_int" in df.columns and "area_sqm" in df.columns:
            df = df.copy()
            df[col_e] = pd.to_numeric(df["price_int"], errors="coerce") / pd.to_numeric(
                df["area_sqm"], errors="coerce"
            ).replace(0, pd.NA)
        else:
            st.caption(UI.NEED_EUR_M2)
            col_e = "price_int" if "price_int" in df.columns else None
            if col_e is None:
                return

    vals = pd.to_numeric(df[col_e], errors="coerce").dropna()
    if vals.empty:
        st.caption(UI.NO_NUMERIC_MAP)
        return
    vmin = float(vals.quantile(0.05))
    vmax = float(vals.quantile(0.95))
    if vmin >= vmax:
        vmax = vmin + 1.0

    colormap = LinearColormap(
        ["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30"],
        vmin=vmin,
        vmax=vmax,
        caption="€/m²" if col_e == "eur_per_sqm" else "",
    )

    m = folium.Map(location=[39.47, -0.37], zoom_start=13, tiles="CartoDB dark_matter")
    geo_key = int(meta.get("geo_key", 0))

    for _, row in df.iterrows():
        try:
            lat, lng = float(row["lat"]), float(row["lng"])
        except (TypeError, ValueError):
            continue
        raw = row.get(col_e)
        if pd.isna(raw) and col_e != "eur_per_sqm":
            c = "#555555"
        else:
            try:
                fv = float(raw) if not pd.isna(raw) else vmin
                c = colormap(max(vmin, min(vmax, fv)))
            except (TypeError, ValueError):
                c = "#555555"
        rooms = row.get("rooms_int", "?")
        hood = row.get("neighborhood_name", "—")
        url = row.get("url", "#")
        price = int(row["price_int"]) if pd.notna(row.get("price_int")) else 0
        area = row.get("area_sqm", "—")
        popup = folium.Popup(
            f"""
            <b>€{price:,}</b><br>
            {area} m² · {rooms} hab.<br>
            {hood}<br>
            <a href="{url}" target="_blank">Ver →</a>
            """,
            max_width=200,
        )
        folium.CircleMarker(
            location=[lat, lng],
            radius=6,
            color=c,
            fill=True,
            fill_opacity=0.85,
            weight=1.5,
            popup=popup,
        ).add_to(m)

    colormap.add_to(m)
    st_folium(
        m,
        height=380,
        use_container_width=True,
        returned_objects=[],
        key=f"rooster_chat_points_{geo_key}",
    )
    st.caption(UI.LISTINGS_SHOWN.format(n=len(df)))


def render_geo(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Listing point map if lat/lng present; else neighborhood choropleth."""
    df = pd.DataFrame(rows)
    if not df.empty and "lat" in df.columns and "lng" in df.columns and df["lat"].notna().any():
        render_geo_map(rows, metadata)
    else:
        render_neighborhood_map(rows, metadata)


def _load_neighborhood_geo_features() -> tuple[list[dict[str, Any]], bool]:
    """
    Barrio polygons for chat maps. Prefer analytics.neighborhood_metrics; if the view is missing
    or empty, use core.neighborhoods (geom is already EPSG:4326).
    Returns (features, used_core_fallback).
    """
    conn = get_pg_conn()
    feats: list[dict[str, Any]] = []
    used_core_fallback = False
    rows: list[tuple[Any, ...]] = []

    metrics_sql = """
        SELECT neighborhood_name,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(ST_Transform(geom, 4326), 0.00004),
                   4
               )::text
        FROM analytics.neighborhood_metrics
        WHERE geom IS NOT NULL AND neighborhood_name IS NOT NULL
    """
    core_sql = """
        SELECT name AS neighborhood_name,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(geom, 0.00004),
                   4
               )::text
        FROM core.neighborhoods
        WHERE geom IS NOT NULL AND name IS NOT NULL
    """

    try:
        cur = conn.cursor()
        try:
            cur.execute(metrics_sql)
            rows = cur.fetchall()
        except pg_errors.UndefinedTable:
            conn.rollback()
            used_core_fallback = True
            rows = []

        if not rows:
            try:
                cur.execute(core_sql)
                rows = cur.fetchall()
                if rows:
                    used_core_fallback = True
            except pg_errors.UndefinedTable:
                conn.rollback()
                rows = []

        for name, gj in rows:
            if not gj or not name:
                continue
            try:
                geom = json.loads(gj)
            except json.JSONDecodeError:
                continue
            feats.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"name": str(name), "norm": _norm_name(str(name))},
                }
            )
        cur.close()
    finally:
        conn.close()
    return feats, used_core_fallback


def _rows_value_map(rows: list[dict[str, Any]]) -> tuple[dict[str, float], str | None]:
    """Map normalized neighborhood name -> numeric value for choropleth. Returns (map, label)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return {}, None
    name_c = _find_col_df(df, ("neighborhood_name", "neighborhood_raw", "name", "barrio"))
    if not name_c:
        return {}, None
    num_cols = [
        c
        for c in df.columns
        if c != name_c and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]
    preferred = [
        "value",
        "gross_rental_yield_pct",
        "median_venta_price",
        "median_alquiler_price",
        "total_count",
        "venta_count",
        "alquiler_count",
    ]
    lower = {c.lower(): c for c in num_cols}
    val_c = None
    for p in preferred:
        if p in lower:
            val_c = lower[p]
            break
    if val_c is None and num_cols:
        val_c = num_cols[0]
    if val_c is None:
        return {}, None
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        k = _norm_name(str(row.get(name_c, "")))
        if not k:
            continue
        v = row.get(val_c)
        try:
            fv = float(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else None
        except (TypeError, ValueError):
            fv = None
        if fv is not None:
            out[k] = fv
    label = val_c.replace("_", " ")
    return out, label


def render_neighborhood_map(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Choropleth: query neighborhoods highlighted; others muted."""
    geo_key = int(metadata.get("geo_key", 0))
    value_map, caption = _rows_value_map(rows)
    features, geo_fallback = _load_neighborhood_geo_features()
    if geo_fallback:
        st.caption(UI.GEO_FALLBACK_CAPTION)
    if not features:
        st.warning(UI.NO_GEO_LOADED)
        return

    value_map = _align_key_map_to_feature_norms(value_map, features)
    matched_vals = [value_map.get(f["properties"]["norm"]) for f in features]
    matched_vals = [v for v in matched_vals if v is not None]
    if not matched_vals:
        st.info(UI.MATCH_ROWS_FAIL)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    vmin, vmax = min(matched_vals), max(matched_vals)
    if vmin >= vmax:
        vmax = vmin + 1e-6
    cap_lbl = (caption or "value").replace("_", " ")
    cmap = LinearColormap(colors=CHORO_COLORS, vmin=vmin, vmax=vmax, caption=cap_lbl)

    enriched: list[dict[str, Any]] = []
    for f in features:
        fp = json.loads(json.dumps(f))
        norm = fp["properties"].get("norm", "")
        v = value_map.get(norm)
        fp["properties"]["metric_val"] = f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
        enriched.append(fp)

    def style_fn(feature: dict) -> dict:
        v = value_map.get(feature["properties"].get("norm", ""))
        if v is None:
            return {
                "fillColor": "#bdbdbd",
                "fillOpacity": 0.25,
                "color": "#999",
                "weight": 0.4,
            }
        return {
            "fillColor": cmap(v),
            "fillOpacity": 0.78,
            "color": "#333",
            "weight": 0.5,
        }

    fc = {"type": "FeatureCollection", "features": enriched}
    m = folium.Map(location=VALENCIA_CENTER, zoom_start=12, tiles="CartoDB positron")
    folium.GeoJson(
        fc,
        style_function=style_fn,
        name="Neighborhoods",
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "metric_val"],
            aliases=["Barrio:", f"{cap_lbl}:"],
        ),
        popup=folium.GeoJsonPopup(
            fields=["name", "metric_val"],
            aliases=["Barrio:", f"{cap_lbl}:"],
            localize=True,
        ),
    ).add_to(m)
    cmap.add_to(m)
    st_folium(
        m,
        use_container_width=True,
        height=380,
        returned_objects=[],
        key=f"rooster_chat_geo_{geo_key}",
    )


def render_no_coords_fallback(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Map was requested but listing rows lack lat/lng — show table + notice."""
    st.caption(UI.CHAT_NO_COORDS_FALLBACK)
    render_listing_table(rows, metadata)


def render_neighborhood_highlight_map(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """
    Highlight specific neighborhoods on a basemap; others muted.
    Used when output_intent is map_neighborhoods (barrio polygons).
    """
    df = pd.DataFrame(rows)
    name_c = _find_col_df(df, ("neighborhood_name", "name", "barrio", "neighborhood_raw"))
    if df.empty or not name_c:
        st.caption(UI.NEED_NEIGHBORHOOD_NAME)
        return

    highlight_norms = {
        _norm_name(str(x)) for x in df[name_c].dropna() if str(x).strip()
    }
    features, geo_fallback = _load_neighborhood_geo_features()
    if geo_fallback:
        st.caption(UI.GEO_FALLBACK_CAPTION)
    if not features:
        st.warning(UI.NO_GEO_LOADED_SHORT)
        return

    matched = 0
    for f in features:
        n = (f.get("properties") or {}).get("norm", "")
        if n in highlight_norms:
            matched += 1
    if matched == 0:
        st.info(UI.MATCH_ROWS_FAIL)
        if rows:
            st.dataframe(df, use_container_width=True, hide_index=True)
        return

    def style_fn(feature: dict) -> dict:
        norm = feature.get("properties", {}).get("norm", "")
        if norm in highlight_norms:
            return {
                "fillColor": "#1D9E75",
                "color": "#1D9E75",
                "weight": 2,
                "fillOpacity": 0.6,
            }
        return {
            "fillColor": "#333333",
            "color": "#555555",
            "weight": 0.5,
            "fillOpacity": 0.15,
        }

    fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    geo_key = int(metadata.get("geo_key", 0))
    m = folium.Map(location=VALENCIA_CENTER, zoom_start=12, tiles="CartoDB dark_matter")

    folium.GeoJson(
        fc,
        style_function=style_fn,
        name="Barrios",
        tooltip=folium.GeoJsonTooltip(
            fields=["name"],
            aliases=["Barrio:"],
        ),
    ).add_to(m)

    st_folium(
        m,
        use_container_width=True,
        height=420,
        returned_objects=[],
        key=_st_folium_key(metadata, "neighborhood_highlight", str(geo_key)),
    )
    st.caption(UI.CHAT_NEIGHBORHOOD_HIGHLIGHT_CAPTION.format(n=len(highlight_norms)))


def render_mini_choropleth(
    df: pd.DataFrame,
    *,
    metric_col: str = "value",
    height: int = 320,
    zoom_start: int = 11,
    geo_key: int = 0,
) -> None:
    """Small choropleth for ranking: color barrios by ``metric_col`` (needs neighborhood_name)."""
    if df.empty or metric_col not in df.columns:
        st.caption(UI.NOTHING_TO_MAP)
        return
    name_c = _find_col_df(df, ("neighborhood_name", "name", "barrio"))
    if not name_c:
        st.caption(UI.NEED_NEIGHBORHOOD_NAME)
        return
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "neighborhood_name": r.get(name_c),
                metric_col: r.get(metric_col),
            }
        )
    value_map, cap_lbl = _rows_value_map(rows)
    features, geo_fallback = _load_neighborhood_geo_features()
    if geo_fallback:
        st.caption(UI.GEO_FALLBACK_CAPTION)
    if not features:
        st.warning(UI.NO_GEO_LOADED_SHORT)
        return

    value_map = _align_key_map_to_feature_norms(value_map, features)
    matched_vals = [value_map.get(f["properties"]["norm"]) for f in features]
    matched_vals = [v for v in matched_vals if v is not None]
    if not matched_vals:
        # Bar chart is enough; do not show a geometry-mismatch warning in chat.
        return
    vmin, vmax = min(matched_vals), max(matched_vals)
    if vmin >= vmax:
        vmax = vmin + 1e-6
    cap_lbl = (cap_lbl or "value").replace("_", " ")
    cmap = LinearColormap(colors=CHORO_COLORS, vmin=vmin, vmax=vmax, caption=cap_lbl)

    enriched: list[dict[str, Any]] = []
    for f in features:
        fp = json.loads(json.dumps(f))
        norm = fp["properties"].get("norm", "")
        v = value_map.get(norm)
        fp["properties"]["metric_val"] = f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
        enriched.append(fp)

    def style_fn(feature: dict) -> dict:
        v = value_map.get(feature["properties"].get("norm", ""))
        if v is None:
            return {
                "fillColor": "#bdbdbd",
                "fillOpacity": 0.25,
                "color": "#999",
                "weight": 0.4,
            }
        return {
            "fillColor": cmap(v),
            "fillOpacity": 0.78,
            "color": "#333",
            "weight": 0.5,
        }

    fc = {"type": "FeatureCollection", "features": enriched}
    m = folium.Map(location=VALENCIA_CENTER, zoom_start=zoom_start, tiles="CartoDB positron")
    folium.GeoJson(
        fc,
        style_function=style_fn,
        name="Neighborhoods",
        tooltip=folium.GeoJsonTooltip(fields=["name", "metric_val"], aliases=["Barrio:", f"{cap_lbl}:"]),
    ).add_to(m)
    cmap.add_to(m)
    st_folium(
        m,
        use_container_width=True,
        height=height,
        returned_objects=[],
        key=f"rooster_mini_choro_{geo_key}",
    )


def render_ranking(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Horizontal bar (top 10) + mini choropleth side by side."""
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption(UI.NO_RANKING)
        return
    nc = _find_col_df(df, ("neighborhood_name", "neighborhood", "name", "barrio"))
    vc = _find_col_df(df, ("value", "val", "metric_value"))
    if not nc or not vc:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    plot_df = df[[nc, vc]].copy()
    plot_df.columns = ["neighborhood_name", "value"]
    plot_df = plot_df.dropna(subset=["value"])
    if plot_df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    geo_key = int((metadata or {}).get("geo_key", 0))

    col1, col2 = st.columns([1, 1])
    with col1:
        plot_df = plot_df.copy()
        plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
        plot_df = plot_df.dropna(subset=["value"])
        if plot_df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            return
        tail = plot_df.sort_values("value", ascending=True).tail(10)
        fig = px.bar(
            tail,
            x="value",
            y="neighborhood_name",
            orientation="h",
            template="plotly_dark",
            color="value",
            color_continuous_scale=["#3B8BD4", "#1D9E75", "#EF9F27"],
            labels={"value": "", "neighborhood_name": ""},
        )
        fig.update_layout(
            height=360,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=48),
            yaxis=dict(tickfont=dict(size=12)),
            xaxis=dict(showgrid=False),
        )
        fig.update_traces(
            texttemplate="%{x:.1f}",
            textposition="outside",
            textfont=dict(size=11),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=_st_plotly_key(metadata, "ranking_bar"),
        )
    with col2:
        try:
            render_mini_choropleth(plot_df, metric_col="value", height=360, zoom_start=11, geo_key=geo_key)
        except Exception:
            pass


def render_comparison_chart(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Long-format (neighborhood, metric, value) grouped horizontal bars; wide-format fallback."""
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption(UI.NO_COMPARE)
        return

    nc = _find_col_df(df, ("neighborhood", "neighborhood_name", "barrio", "name"))
    mc = _find_col_df(df, ("metric", "metric_name", "measure"))
    vc = _find_col_df(df, ("value", "val", "amount"))

    if nc and mc and vc:
        plot_df = df[[nc, mc, vc]].copy()
        plot_df.columns = ["neighborhood", "metric", "value"]
        plot_df = plot_df.dropna(subset=["value"])
        if plot_df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            return
        fig = px.bar(
            plot_df,
            x="value",
            y="neighborhood",
            color="metric",
            barmode="group",
            orientation="h",
            template="plotly_dark",
            color_discrete_sequence=["#534AB7", "#1D9E75", "#EF9F27"],
            labels={"value": "", "neighborhood": "", "metric": ""},
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            title=None,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=_st_plotly_key(metadata, "compare_group"),
        )
        _maybe_neighborhood_profile_dive(rows)
        return

    skip = {"url", "scraped_at", "description", "heading"}
    str_col = None
    num_col = None
    for c in df.columns:
        if c.lower() in skip:
            continue
        if df[c].dtype == object or str(df[c].dtype).startswith("string"):
            if str_col is None and df[c].notna().any():
                str_col = c
    for c in df.columns:
        if c == str_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_col = c
            break
    if str_col and num_col:
        plot_df = df[[str_col, num_col]].dropna().copy()
        if not plot_df.empty:
            fig = px.bar(
                plot_df.sort_values(num_col, ascending=True),
                x=num_col,
                y=str_col,
                orientation="h",
                template="plotly_dark",
                labels={num_col: num_col.replace("_", " "), str_col: str_col.replace("_", " ")},
            )
            fig.update_layout(
                height=min(420, 80 + 28 * len(plot_df)),
                margin=dict(l=0, r=0, t=8, b=0),
                title=None,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=_st_plotly_key(metadata, "compare_simple"),
            )
            _maybe_neighborhood_profile_dive(rows)
            return
    st.dataframe(df, use_container_width=True, hide_index=True)
    _maybe_neighborhood_profile_dive(rows)


def render_metric_cards(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    del metadata
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption(UI.NO_OVERVIEW)
        return
    n = min(len(df.columns), 4)
    cols = st.columns(min(n, len(df.columns)))
    for i, c in enumerate(df.columns[: len(cols)]):
        if i >= len(cols):
            break
        v = df[c].iloc[0] if len(df) > 0 else None
        with cols[i]:
            st.metric(str(c).replace("_", " ")[:32], f"{v}" if v is not None else "—")
    if len(df) > 1 or len(df.columns) > 4:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _underpriced_row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        nm = _norm_name(
            str(r.get("neighborhood_name") or r.get("neighborhood_raw") or r.get("name") or "")
        )
        if nm:
            by[nm] = r
    return by


def render_underpriced(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Green dot map + table for listing-level rows; else choropleth by neighborhood."""
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption(UI.NO_UNDERPRICED)
        return
    has_pts = (
        "lat" in df.columns
        and "lng" in df.columns
        and not df.dropna(subset=["lat", "lng"]).empty
    )
    if has_pts:
        geo_key = int((metadata or {}).get("geo_key", 0))
        m = folium.Map(location=[39.47, -0.37], zoom_start=13, tiles="CartoDB dark_matter")
        for _, row in df.dropna(subset=["lat", "lng"]).iterrows():
            med = row.get("neighborhood_median")
            if med is None or (isinstance(med, float) and pd.isna(med)) or med == 0:
                med = row.get("median_venta_price")
            try:
                pi = float(row.get("price_int") or 0)
                mval = float(med) if med is not None and not (isinstance(med, float) and pd.isna(med)) else 0.0
                discount = ((mval - pi) / mval * 100) if mval > 0 else 0.0
            except (TypeError, ValueError):
                discount = 0.0
            hood = row.get("neighborhood_name", "—")
            url = row.get("url", "#")
            area = row.get("area_sqm", "—")
            price = int(row["price_int"]) if pd.notna(row.get("price_int")) else 0
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lng"])],
                radius=7,
                color="#1D9E75",
                fill=True,
                fill_opacity=0.9,
                weight=1.5,
                popup=folium.Popup(
                    f"""
                    <b>€{price:,}</b>
                    <span style='color:#1D9E75'>
                        ↓ {discount:.0f}% bajo la mediana
                    </span><br>
                    {area} m² · {hood}<br>
                    <a href="{url}" target="_blank">Ver →</a>
                    """,
                    max_width=220,
                ),
            ).add_to(m)
        st_folium(
            m,
            height=340,
            use_container_width=True,
            returned_objects=[],
            key=f"rooster_underpriced_pts_{geo_key}",
        )
        show_cols = [c for c in ("neighborhood_name", "price_int", "area_sqm", "eur_per_sqm", "url") if c in df.columns]
        if show_cols:
            cfg: dict = {}
            if "url" in show_cols:
                cfg["url"] = st.column_config.LinkColumn(UI.COL_LINK, display_text=UI.COL_VIEW)
            if "price_int" in show_cols:
                cfg["price_int"] = st.column_config.NumberColumn(UI.COL_PRICE, format="€%d")
            if "eur_per_sqm" in show_cols:
                cfg["eur_per_sqm"] = st.column_config.NumberColumn("€/m²", format="€%d")
            if "area_sqm" in show_cols:
                cfg["area_sqm"] = st.column_config.NumberColumn(UI.COL_AREA, format="%d m²")
            if "neighborhood_name" in show_cols:
                cfg["neighborhood_name"] = st.column_config.TextColumn(UI.COL_NEIGHBORHOOD)
            st.dataframe(
                df[show_cols].head(10),
                column_config=cfg or None,
                hide_index=True,
                use_container_width=True,
            )
        return

    render_underpriced_choropleth(rows, metadata)


def render_underpriced_choropleth(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Choropleth: green scale = share of listings below neighborhood median price."""
    geo_key = int(metadata.get("geo_key", 0))
    row_by = _underpriced_row_map(rows)
    features, geo_fallback = _load_neighborhood_geo_features()
    if geo_fallback:
        st.caption(UI.GEO_FALLBACK_CAPTION)
    if not features:
        st.warning(UI.NO_GEO_LOADED)
        return

    pct_by_norm: dict[str, float] = {}
    for norm, r in row_by.items():
        p = r.get("underpriced_pct")
        if p is None:
            p = r.get("underpriced_density")
        try:
            if p is not None:
                pct_by_norm[norm] = float(p)
        except (TypeError, ValueError):
            continue

    if not pct_by_norm:
        for norm, r in row_by.items():
            below = r.get("below_median_count")
            tot = r.get("total_listings") or r.get("venta_count") or r.get("total_count")
            try:
                b, t = int(below or 0), int(tot or 0)
                if t > 0:
                    pct_by_norm[norm] = 100.0 * b / t
            except (TypeError, ValueError):
                continue

    pct_by_norm = _align_key_map_to_feature_norms(pct_by_norm, features)
    vals = list(pct_by_norm.values())
    if not vals:
        st.info(UI.UNDERPRICED_NEED_COLS)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    vmin, vmax = min(vals), max(vals)
    if vmin >= vmax:
        vmax = vmin + 1e-6
    cmap = LinearColormap(
        colors=GREEN_CHORO,
        vmin=vmin,
        vmax=vmax,
        caption=UI.BELOW_MEDIAN_SHARE,
    )

    def style_fn(feature: dict) -> dict:
        norm = feature["properties"].get("norm", "")
        p = pct_by_norm.get(norm)
        if p is None:
            return {
                "fillColor": "#d9d9d9",
                "fillOpacity": 0.3,
                "color": "#aaa",
                "weight": 0.4,
            }
        return {
            "fillColor": cmap(p),
            "fillOpacity": 0.82,
            "color": "#1a3d1a",
            "weight": 0.5,
        }

    fc = {"type": "FeatureCollection", "features": features}
    m = folium.Map(location=VALENCIA_CENTER, zoom_start=12, tiles="CartoDB positron")
    folium.GeoJson(
        fc,
        style_function=style_fn,
        name="Underpriced density",
        tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Barrio:"]),
    ).add_to(m)
    cmap.add_to(m)
    st_folium(
        m,
        use_container_width=True,
        height=350,
        returned_objects=[],
        key=f"rooster_chat_underpriced_{geo_key}",
    )
    st.caption(UI.UNDERPRICED_CHORO_CAPTION)
    cheap_cols = [
        "neighborhood_name",
        "neighborhood_raw",
        "underpriced_pct",
        "below_median_count",
        "cheapest_listing_url",
        "cheapest_price_int",
    ]
    sub = pd.DataFrame(rows)
    show = [c for c in cheap_cols if c in sub.columns]
    if show:
        st.dataframe(sub[show], use_container_width=True, hide_index=True)


def _floor_label_chat(x: object) -> str:
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


def _floor_sort_key_chat(lbl: str) -> tuple:
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


def render_chart(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Plotly charts from current listings snapshot (scatter, amenity prevalence, floor box)."""
    del rows
    chart_type = (metadata.get("chart_type") or "scatter").strip().lower()
    if chart_type not in ("scatter", "amenity", "floor"):
        chart_type = "scatter"
    df = load_listings_frame("All")
    if df.empty:
        st.info(UI.NO_LISTINGS_DB)
        return

    if chart_type == "scatter":
        hood_h = "neighborhood_name" if "neighborhood_name" in df.columns else "neighborhood_raw"
        fig = px.scatter(
            df,
            x="area_sqm",
            y="price_int",
            color="eur_per_sqm",
            color_continuous_scale=["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30"],
            template="plotly_dark",
            hover_data=[hood_h, "rooms_int", "operation"],
            labels={
                "area_sqm": UI.LABEL_AREA,
                "price_int": UI.LABEL_PRICE,
                "eur_per_sqm": "€/m²",
            },
        )
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=_st_plotly_key(metadata, "chart_scatter"),
        )
        return

    if chart_type == "amenity":
        amenity_map = {
            UI.AMENITY_LABELS["Parking"]: "has_parking",
            UI.AMENITY_LABELS["Terrace"]: "has_terrace",
            UI.AMENITY_LABELS["Elevator"]: "has_elevator",
            UI.AMENITY_LABELS["Exterior"]: "is_exterior",
            UI.AMENITY_LABELS["Renovated"]: "is_renovated",
            UI.AMENITY_LABELS["A/C"]: "has_ac",
        }
        pcts = {
            label: float(df[col].mean() * 100)
            for label, col in amenity_map.items()
            if col in df.columns
        }
        if not pcts:
            st.info(UI.AMENITY_MISSING)
            return
        fig = px.bar(
            x=list(pcts.values()),
            y=list(pcts.keys()),
            orientation="h",
            template="plotly_dark",
            color=list(pcts.values()),
            color_continuous_scale=["#3B8BD4", "#1D9E75"],
            text=[f"{v:.0f}%" for v in pcts.values()],
            labels={"x": UI.AMENITY_PCT_LISTINGS, "y": ""},
        )
        fig.update_layout(
            height=360,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=_st_plotly_key(metadata, "chart_amenity"),
        )
        return

    # floor
    df_floor = df[
        df["floor_int"].notna()
        & ((df["floor_int"] < 15) | (df["floor_int"] == 99))
        & df["eur_per_sqm"].notna()
    ].copy()
    df_floor["floor_label"] = df_floor["floor_int"].apply(_floor_label_chat)
    df_floor = df_floor[df_floor["floor_label"] != ""]
    if df_floor.empty:
        st.info(UI.NO_FLOOR_DATA)
        return
    labels_sorted = sorted(df_floor["floor_label"].unique(), key=_floor_sort_key_chat)
    fig2 = px.box(
        df_floor,
        x="floor_label",
        y="eur_per_sqm",
        template="plotly_dark",
        color_discrete_sequence=["#534AB7"],
        labels={"floor_label": "", "eur_per_sqm": "€/m²"},
    )
    fig2.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(categoryorder="array", categoryarray=labels_sorted),
    )
    st.plotly_chart(
        fig2,
        use_container_width=True,
        key=_st_plotly_key(metadata, "chart_floor"),
    )


def render_trend_chart(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    df = pd.DataFrame(rows)
    if len(df) < 2:
        st.info(UI.TREND_BUILDING)
        render_comparison_chart(rows, metadata)
        return

    date_c = _find_col_df(
        df,
        ("bucket_date", "scraped_at", "snapshot_date", "period", "week", "month", "day"),
    )
    val_c = _find_col_df(
        df,
        ("median_price_int", "median_price", "avg_price_int", "price_int", "median_price_eur"),
    )
    neigh_c = _find_col_df(df, ("neighborhood_name", "neighborhood", "neighborhood_raw", "name"))

    if date_c and val_c:
        plot_df = df.copy()
        plot_df[date_c] = pd.to_datetime(plot_df[date_c], errors="coerce")
        plot_df = plot_df.dropna(subset=[date_c, val_c])
        if len(plot_df) < 2:
            st.dataframe(df, use_container_width=True, hide_index=True)
            return
        if neigh_c and plot_df[neigh_c].nunique() > 1:
            fig = px.line(
                plot_df.sort_values(date_c),
                x=date_c,
                y=val_c,
                color=neigh_c,
                template="plotly_white",
                markers=True,
            )
        else:
            fig = px.line(
                plot_df.sort_values(date_c),
                x=date_c,
                y=val_c,
                template="plotly_white",
                markers=True,
            )
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=28, b=0))
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=_st_plotly_key(metadata, "trend_line"),
        )
        return

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_transit_map(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Map of core.transit_stops — color by stop_type."""
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption(UI.CHAT_TRANSIT_EMPTY)
        return
    if "lat" not in df.columns or "lng" not in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    df = df.dropna(subset=["lat", "lng"])
    if df.empty:
        st.caption(UI.CHAT_TRANSIT_EMPTY)
        return

    center_lat = float(df["lat"].mean())
    center_lng = float(df["lng"].mean())
    m = folium.Map(location=[center_lat, center_lng], zoom_start=15, tiles="CartoDB dark_matter")
    for _, row in df.iterrows():
        st_raw = row.get("stop_type")
        st_key = (str(st_raw).lower() if st_raw is not None and not (isinstance(st_raw, float) and pd.isna(st_raw)) else "") or ""
        color = _TRANSIT_STOP_COLORS.get(st_key, "#888780")
        nm = row.get("name") or UI.CHAT_TRANSIT_STOP_DEFAULT
        popup = folium.Popup(
            f"<b>{nm}</b><br>{str(st_raw or '').replace('_', ' ').title()}",
            max_width=160,
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lng"])],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.9,
            weight=2,
            popup=popup,
        ).add_to(m)
    st_folium(
        m,
        height=400,
        use_container_width=True,
        returned_objects=[],
        key=_st_folium_key(metadata, "transit"),
    )
    c1, c2, c3 = st.columns(3)
    c1.caption(UI.CHAT_TRANSIT_LEGEND_METRO)
    c2.caption(UI.CHAT_TRANSIT_LEGEND_BUS)
    c3.caption(UI.CHAT_TRANSIT_COUNT.format(n=len(df)))


def render_tourism_map(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Map of core.tourist_apartments."""
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption(UI.CHAT_TOURISM_EMPTY)
        return
    if "lat" not in df.columns or "lng" not in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    df = df.dropna(subset=["lat", "lng"])
    if df.empty:
        st.caption(UI.CHAT_TOURISM_EMPTY)
        return

    center_lat = float(df["lat"].mean())
    center_lng = float(df["lng"].mean())
    m = folium.Map(location=[center_lat, center_lng], zoom_start=15, tiles="CartoDB dark_matter")
    for _, row in df.iterrows():
        addr = str(row.get("address") or "")[:80]
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lng"])],
            radius=5,
            color="#EF9F27",
            fill=True,
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(
                f"<b>{UI.CHAT_TOURISM_POPUP_TITLE}</b><br>{addr}",
                max_width=200,
            ),
        ).add_to(m)
    st_folium(
        m,
        height=400,
        use_container_width=True,
        returned_objects=[],
        key=_st_folium_key(metadata, "tourism"),
    )
    c1, c2 = st.columns(2)
    c1.caption(UI.CHAT_TOURISM_COUNT.format(n=len(df)))
    c2.caption(UI.CHAT_TOURISM_RISK)


def render_combined_map(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Listings + transit + tourist layers with Folium layer control."""
    del rows
    meta = metadata or {}
    df_l = pd.DataFrame(meta.get("rows_listings") or [])
    df_t = pd.DataFrame(meta.get("rows_transit") or [])
    df_tour = pd.DataFrame(meta.get("rows_tourism") or [])

    if df_l.empty and df_t.empty and df_tour.empty:
        st.caption(UI.CHAT_COMBINED_EMPTY)
        return

    if not df_l.empty and "lat" in df_l.columns:
        center_lat = float(df_l["lat"].mean())
        center_lng = float(df_l["lng"].mean())
    elif not df_t.empty and "lat" in df_t.columns:
        center_lat = float(df_t["lat"].mean())
        center_lng = float(df_t["lng"].mean())
    elif not df_tour.empty and "lat" in df_tour.columns:
        center_lat = float(df_tour["lat"].mean())
        center_lng = float(df_tour["lng"].mean())
    else:
        center_lat, center_lng = VALENCIA_CENTER

    m = folium.Map(location=[center_lat, center_lng], zoom_start=15, tiles="CartoDB dark_matter")

    if not df_l.empty and "lat" in df_l.columns and "lng" in df_l.columns:
        df_l = df_l.dropna(subset=["lat", "lng"])
    if not df_t.empty and "lat" in df_t.columns:
        df_t = df_t.dropna(subset=["lat", "lng"])
    if not df_tour.empty and "lat" in df_tour.columns:
        df_tour = df_tour.dropna(subset=["lat", "lng"])

    if not df_l.empty and "eur_per_sqm" not in df_l.columns and "price_int" in df_l.columns and "area_sqm" in df_l.columns:
        df_l = df_l.copy()
        pi = pd.to_numeric(df_l["price_int"], errors="coerce")
        ar = pd.to_numeric(df_l["area_sqm"], errors="coerce").replace(0, pd.NA)
        df_l["eur_per_sqm"] = pi / ar

    if not df_l.empty and "eur_per_sqm" in df_l.columns:
        vals = pd.to_numeric(df_l["eur_per_sqm"], errors="coerce").dropna()
        if not vals.empty:
            vmin = float(vals.quantile(0.05))
            vmax = float(vals.quantile(0.95))
            if vmin >= vmax:
                vmax = vmin + 1.0
            colormap = LinearColormap(
                ["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30"],
                vmin=vmin,
                vmax=vmax,
                caption=UI.LEGEND_EUR_M2,
            )
            listing_group = folium.FeatureGroup(name=UI.FOLIUM_LAYER_LISTINGS, show=True)
            for _, row in df_l.iterrows():
                raw = row.get("eur_per_sqm")
                try:
                    fv = float(raw) if raw is not None and not (isinstance(raw, float) and pd.isna(raw)) else vmin
                except (TypeError, ValueError):
                    fv = vmin
                c = colormap(max(vmin, min(vmax, fv)))
                price = int(row["price_int"]) if pd.notna(row.get("price_int")) else 0
                area = row.get("area_sqm", "—")
                url = row.get("url") or "#"
                popup = folium.Popup(
                    f"<b>€{price:,}</b> · {area} m²<br>"
                    f"€{int(fv):,}/m²<br>"
                    f"<a href=\"{url}\" target=\"_blank\">{UI.POPUP_IDEALISTA}</a>",
                    max_width=180,
                )
                folium.CircleMarker(
                    location=[float(row["lat"]), float(row["lng"])],
                    radius=7,
                    color=c,
                    fill=True,
                    fill_opacity=0.85,
                    weight=1.5,
                    popup=popup,
                ).add_to(listing_group)
            listing_group.add_to(m)
            colormap.add_to(m)
    elif not df_l.empty and "lat" in df_l.columns:
        listing_group = folium.FeatureGroup(name=UI.FOLIUM_LAYER_LISTINGS, show=True)
        for _, row in df_l.iterrows():
            price = int(row["price_int"]) if pd.notna(row.get("price_int")) else 0
            url = row.get("url") or "#"
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lng"])],
                radius=7,
                color="#3B8BD4",
                fill=True,
                fill_opacity=0.85,
                weight=1.5,
                popup=folium.Popup(
                    f"<b>€{price:,}</b><br><a href=\"{url}\" target=\"_blank\">{UI.POPUP_IDEALISTA}</a>",
                    max_width=180,
                ),
            ).add_to(listing_group)
        listing_group.add_to(m)

    if not df_t.empty:
        transit_group = folium.FeatureGroup(name=UI.FOLIUM_LAYER_TRANSIT, show=True)
        for _, row in df_t.iterrows():
            st_raw = row.get("stop_type")
            st_key = (str(st_raw).lower() if st_raw is not None and not (isinstance(st_raw, float) and pd.isna(st_raw)) else "") or ""
            color = _TRANSIT_STOP_COLORS.get(st_key, "#534AB7")
            nm = row.get("name") or "—"
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lng"])],
                radius=10,
                color=color,
                fill=True,
                fill_opacity=0.9,
                weight=2,
                popup=folium.Popup(f"<b>{nm}</b><br>{st_raw or ''}", max_width=150),
            ).add_to(transit_group)
        transit_group.add_to(m)

    if not df_tour.empty:
        tourism_group = folium.FeatureGroup(name=UI.FOLIUM_LAYER_TOURISM, show=True)
        for _, row in df_tour.iterrows():
            addr = str(row.get("address") or "")[:60]
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lng"])],
                radius=4,
                color="#EF9F27",
                fill=True,
                fill_opacity=0.6,
                weight=1,
                popup=folium.Popup(addr, max_width=160),
            ).add_to(tourism_group)
        tourism_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(
        m,
        height=440,
        use_container_width=True,
        returned_objects=[],
        key=_st_folium_key(metadata, "combined"),
    )
    c1, c2, c3 = st.columns(3)
    c1.caption(UI.CHAT_COMBINED_CAP_LISTINGS.format(n=len(df_l)))
    c2.caption(UI.CHAT_COMBINED_CAP_TRANSIT.format(n=len(df_t)))
    c3.caption(UI.CHAT_COMBINED_CAP_TOURISM.format(n=len(df_tour)))


render_search = render_listing_table
render_compare = render_comparison_chart
render_overview = render_metric_cards


def render_conversational(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Prose-only reply; summary is already written in ``dispatch``."""
    return


def render_memo(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Investment memo artifact — content in ``metadata["memo_text"]``."""
    del rows
    text = (metadata or {}).get("memo_text") or ""
    try:
        with st.container(border=True):
            st.markdown(text)
    except TypeError:
        st.markdown(
            '<div style="border:0.5px solid #534AB740;border-radius:12px;padding:20px 24px;background:#0f0f1a;margin-top:8px;">',
            unsafe_allow_html=True,
        )
        st.markdown(text)
        st.markdown("</div>", unsafe_allow_html=True)


RENDERERS: dict[str, Any] = {
    "search": render_search,
    "compare": render_compare,
    "overview": render_overview,
    "geo": render_geo,
    "underpriced": render_underpriced,
    "ranking": render_ranking,
    "transit_map": render_transit_map,
    "tourism_map": render_tourism_map,
    "combined_map": render_combined_map,
    "conversational": render_conversational,
    "memo": render_memo,
    "trend": render_trend_chart,
    "chart": render_chart,
    "neighborhood_highlight": render_neighborhood_highlight_map,
    "no_coords": render_no_coords_fallback,
}


def dispatch(intent: str, rows: list[dict[str, Any]], metadata: dict[str, Any], summary: str) -> None:
    """Plain prose first, then visuals, then optional caveat caption."""
    meta = dict(metadata or {})
    if intent != "memo":
        s = _plain_chat_prose(summary or "")
        if s:
            st.write(s)
    renderer = RENDERERS.get(intent, render_search)
    renderer(rows or [], meta)
    cav = (meta.get("caveat") or "").strip()
    if cav and intent != "memo":
        st.caption(cav)


def render_response(intent: str, rows: list[dict[str, Any]], summary: str, metadata: dict[str, Any] | None = None) -> None:
    """Backward-compatible: summary + renderer (delegates to `dispatch`)."""
    dispatch(intent, rows, metadata or {}, summary)

"""Streamlit renderers for Phase E response primitives."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import json as _json

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests
import streamlit as st
from shapely.geometry import shape

from agent.synthesizer import (
    BarDescriptor,
    ChoroplethDescriptor,
    CodePrimitive,
    CompositePrimitive,
    KpiPrimitive,
    LineDescriptor,
    PointMapDescriptor,
    PrimitiveBlock,
    ScatterDescriptor,
    SynthesizedResponse,
    TablePrimitive,
    TextPrimitive,
)

_LOG = logging.getLogger("rooster.primitive_renderer")


def _normalize_sandbox_url(url: str) -> str:
    url = (url or "").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url or "http://localhost:8001"


SANDBOX_URL = _normalize_sandbox_url(os.getenv("ROOSTER_SANDBOX_URL", "http://localhost:8001"))


def _rows(raw_json: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(raw_json or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _format_value(value: Any, formatter: str) -> Any:
    if value is None:
        return ""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return value
    if formatter == "currency":
        return f"{n:,.0f} €".replace(",", ".")
    if formatter == "percentage":
        return f"{n:.1f} %".replace(".", ",")
    if formatter == "integer":
        return f"{n:,.0f}".replace(",", ".")
    if formatter == "number":
        return f"{n:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return value


def _render_text(primitive: TextPrimitive) -> None:
    text = primitive.content.strip()
    if text:
        st.markdown(text.replace("**", "").replace("__", ""))


def _render_kpi(primitive: KpiPrimitive) -> None:
    value = primitive.value
    if primitive.unit and primitive.unit not in value:
        value = f"{value} {primitive.unit}"
    st.metric(primitive.label, value, delta=primitive.delta)


def _render_table(primitive: TablePrimitive) -> None:
    rows = _rows(primitive.rows_json)
    if not rows:
        st.caption("No hay filas para mostrar.")
        return
    df = pd.DataFrame(rows)
    if primitive.columns:
        fields = [c.field for c in primitive.columns if c.field in df.columns]
        if fields:
            df = df[fields].copy()
        rename = {c.field: c.header for c in primitive.columns if c.field in df.columns}
        for col in primitive.columns:
            if col.field in df.columns:
                df[col.field] = df[col.field].map(lambda v, f=col.formatter: _format_value(v, f))
        df = df.rename(columns=rename)
    st.dataframe(df, use_container_width=True, hide_index=True)


def call_sandbox(code: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute code in the sandbox service. Pure HTTP — no Streamlit."""
    try:
        response = requests.post(
            f"{SANDBOX_URL}/execute",
            json={"code": code, "data": data},
            timeout=35,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"success": False, "output_type": None, "output": None, "error": str(exc)}


def _render_code(primitive: CodePrimitive, geo_key: int) -> None:
    if not st.session_state.get("_rooster_last_execution_results"):
        st.info("No hay datos para ejecutar la visualización.")
        return
    execution_results = st.session_state.get("_rooster_last_execution_results") or []
    rows = (execution_results[0] or {}).get("rows") if execution_results else None
    if not isinstance(rows, list) or not rows:
        st.info("No hay datos para ejecutar la visualización.")
        return

    result = call_sandbox(primitive.code, rows)

    if result.get("success"):
        output = result.get("output") or ""
        if primitive.visualization_type == "plotly":
            fig = pio.from_json(output)
            st.plotly_chart(fig, use_container_width=True, key=f"primitive_code_plotly_{geo_key}_{id(primitive)}")
        elif primitive.visualization_type == "folium":
            st.components.v1.html(output, height=500, scrolling=True)
    else:
        _LOG.warning(
            "Sandbox execution failed: %s\nCode: %s",
            result.get("error"),
            primitive.code,
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _get_execution_rows() -> list[dict[str, Any]]:
    """Return the first successful result's rows from session state."""
    for r in st.session_state.get("_rooster_last_execution_results") or []:
        if r.get("success") and r.get("rows"):
            return r["rows"]
    return []


def _render_choropleth(primitive: ChoroplethDescriptor, geo_key: int) -> None:
    rows = _get_execution_rows()
    if not rows:
        st.info("No hay datos para el mapa.")
        return
    df = pd.DataFrame(rows)
    if "geom" not in df.columns or primitive.metric not in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    df = df[df["geom"].notna() & df[primitive.metric].notna()].copy()

    def _parse_geom(g: Any) -> Any:
        if isinstance(g, dict):
            return shape(g)
        if isinstance(g, str):
            try:
                return shape(_json.loads(g))
            except Exception:
                return None
        return None

    df["geometry"] = df["geom"].apply(_parse_geom)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        st.info("No hay geometrías disponibles para el mapa.")
        return

    _COLORMAPS = {
        "YlOrRd": cm.linear.YlOrRd_09,
        "YlGn": cm.linear.YlGn_09,
        "Blues": cm.linear.Blues_09,
        "RdYlGn": cm.linear.RdYlGn_11,
        "Purples": cm.linear.Purples_09,
    }
    bounds = gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles=primitive.tiles)
    vmin = float(gdf[primitive.metric].min())
    vmax = float(gdf[primitive.metric].max())
    if vmin >= vmax:
        vmax = vmin + 0.001
    colormap = _COLORMAPS.get(primitive.colormap, cm.linear.YlOrRd_09).scale(vmin, vmax)
    colormap.caption = primitive.metric_label
    colormap.add_to(m)
    metric = primitive.metric
    valid_fields = [tf for tf in primitive.tooltip_fields if tf.field in gdf.columns]
    tooltip = (
        folium.GeoJsonTooltip(
            fields=[tf.field for tf in valid_fields],
            aliases=[tf.label for tf in valid_fields],
            localize=True,
            sticky=True,
        )
        if valid_fields
        else None
    )
    folium.GeoJson(
        gdf,
        style_function=lambda f, _m=metric, _c=colormap: {
            "fillColor": _c(float(f["properties"][_m])) if f["properties"].get(_m) is not None else "#cccccc",
            "color": "#555555",
            "weight": 0.8,
            "fillOpacity": 0.75,
        },
        highlight_function=lambda f: {"weight": 2.5, "color": "#222", "fillOpacity": 0.88},
        name=primitive.metric_label,
        tooltip=tooltip,
    ).add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    folium.LayerControl(collapsed=True).add_to(m)
    st.components.v1.html(m.get_root().render(), height=500, scrolling=True)


def _render_bar(primitive: BarDescriptor, geo_key: int) -> None:
    rows = _get_execution_rows()
    if not rows:
        st.info("No hay datos.")
        return
    df = pd.DataFrame(rows)
    if primitive.value_field not in df.columns or primitive.label_field not in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    plot_df = df[[primitive.label_field, primitive.value_field]].copy()
    plot_df[primitive.value_field] = pd.to_numeric(plot_df[primitive.value_field], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    if primitive.orientation == "h":
        plot_df = plot_df.sort_values(primitive.value_field, ascending=True).tail(primitive.top_n)
        fig = px.bar(
            plot_df, x=primitive.value_field, y=primitive.label_field, orientation="h",
            template="plotly_dark", color=primitive.value_field,
            color_continuous_scale=["#3B8BD4", "#1D9E75", "#EF9F27"],
            labels={primitive.value_field: primitive.value_label, primitive.label_field: ""},
        )
        fig.update_layout(
            height=min(480, 60 + 28 * len(plot_df)), showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=48), yaxis=dict(tickfont=dict(size=12)),
            xaxis=dict(showgrid=False),
        )
        fig.update_traces(texttemplate="%{x:.2f}", textposition="outside", textfont=dict(size=10))
    else:
        plot_df = plot_df.sort_values(primitive.value_field, ascending=False).head(primitive.top_n)
        fig = px.bar(
            plot_df, x=primitive.label_field, y=primitive.value_field,
            template="plotly_dark", color=primitive.value_field,
            color_continuous_scale=["#3B8BD4", "#1D9E75", "#EF9F27"],
            labels={primitive.value_field: primitive.value_label, primitive.label_field: ""},
        )
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=48))
    st.plotly_chart(fig, use_container_width=True, key=f"bar_{geo_key}_{id(primitive)}")


def _render_scatter(primitive: ScatterDescriptor, geo_key: int) -> None:
    rows = _get_execution_rows()
    if not rows:
        st.info("No hay datos.")
        return
    df = pd.DataFrame(rows)
    if primitive.x_field not in df.columns or primitive.y_field not in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    plot_df = df.copy()
    plot_df[primitive.x_field] = pd.to_numeric(plot_df[primitive.x_field], errors="coerce")
    plot_df[primitive.y_field] = pd.to_numeric(plot_df[primitive.y_field], errors="coerce")
    plot_df = plot_df.dropna(subset=[primitive.x_field, primitive.y_field])
    if plot_df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    kwargs: dict[str, Any] = {
        "x": primitive.x_field, "y": primitive.y_field, "template": "plotly_dark",
        "labels": {primitive.x_field: primitive.x_label, primitive.y_field: primitive.y_label},
    }
    if primitive.label_field and primitive.label_field in plot_df.columns:
        kwargs["text"] = primitive.label_field
    if primitive.color_field and primitive.color_field in plot_df.columns:
        plot_df[primitive.color_field] = pd.to_numeric(plot_df[primitive.color_field], errors="coerce")
        kwargs["color"] = primitive.color_field
        kwargs["color_continuous_scale"] = ["#3B8BD4", "#1D9E75", "#EF9F27"]
    fig = px.scatter(plot_df, **kwargs)
    if primitive.label_field:
        fig.update_traces(textposition="top center", textfont=dict(size=10))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=f"scatter_{geo_key}_{id(primitive)}")


def _render_line(primitive: LineDescriptor, geo_key: int) -> None:
    rows = _get_execution_rows()
    if not rows:
        st.info("No hay datos.")
        return
    df = pd.DataFrame(rows)
    if primitive.x_field not in df.columns or primitive.y_field not in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    plot_df = df.copy()
    plot_df[primitive.x_field] = pd.to_datetime(plot_df[primitive.x_field], errors="coerce")
    plot_df[primitive.y_field] = pd.to_numeric(plot_df[primitive.y_field], errors="coerce")
    plot_df = plot_df.dropna(subset=[primitive.x_field, primitive.y_field])
    if len(plot_df) < 2:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    plot_df = plot_df.sort_values(primitive.x_field)
    kwargs: dict[str, Any] = {
        "x": primitive.x_field, "y": primitive.y_field, "template": "plotly_white", "markers": True,
        "labels": {primitive.x_field: "", primitive.y_field: primitive.y_label},
    }
    if primitive.color_field and primitive.color_field in plot_df.columns:
        kwargs["color"] = primitive.color_field
    fig = px.line(plot_df, **kwargs)
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=28, b=0))
    st.plotly_chart(fig, use_container_width=True, key=f"line_{geo_key}_{id(primitive)}")


def _render_point_map(primitive: PointMapDescriptor, geo_key: int) -> None:
    rows = _get_execution_rows()
    if not rows:
        st.info("No hay datos para el mapa.")
        return
    df = pd.DataFrame(rows)
    if "lat" not in df.columns or "lng" not in df.columns:
        st.info("No hay coordenadas disponibles.")
        return
    df = df.dropna(subset=["lat", "lng"])
    if df.empty:
        st.info("No hay coordenadas disponibles.")
        return
    center_lat = float(df["lat"].mean())
    center_lng = float(df["lng"].mean())
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="CartoDB dark_matter")
    colormap = None
    color_col = primitive.color_field
    if color_col and color_col in df.columns:
        vals = pd.to_numeric(df[color_col], errors="coerce").dropna()
        if not vals.empty:
            vmin = float(vals.quantile(0.05))
            vmax = float(vals.quantile(0.95))
            if vmin >= vmax:
                vmax = vmin + 1.0
            colormap = cm.LinearColormap(
                ["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30"], vmin=vmin, vmax=vmax,
                caption=primitive.color_label or (color_col.replace("_", " ")),
            )
            colormap.add_to(m)
    for _, row in df.iterrows():
        try:
            lat, lng = float(row["lat"]), float(row["lng"])
        except (TypeError, ValueError):
            continue
        color = "#1D9E75"
        if colormap and color_col:
            raw = row.get(color_col)
            try:
                fv = float(raw) if raw is not None and not (isinstance(raw, float) and pd.isna(raw)) else None
                if fv is not None:
                    color = colormap(max(colormap.vmin, min(colormap.vmax, fv)))
            except (TypeError, ValueError):
                pass
        popup_lines = [
            f"{tf.label}: {row[tf.field]}"
            for tf in primitive.popup_fields
            if tf.field in row and row[tf.field] is not None
            and not (isinstance(row[tf.field], float) and pd.isna(row[tf.field]))
        ]
        folium.CircleMarker(
            location=[lat, lng], radius=6, color=color, fill=True, fill_opacity=0.85, weight=1.5,
            popup=folium.Popup("<br>".join(popup_lines), max_width=200) if popup_lines else None,
        ).add_to(m)
    st.components.v1.html(m.get_root().render(), height=500, scrolling=True)
    st.caption(f"{len(df)} puntos en el mapa.")


def render_primitive_block(primitive: PrimitiveBlock, geo_key: int) -> None:
    if isinstance(primitive, TextPrimitive):
        _render_text(primitive)
    elif isinstance(primitive, KpiPrimitive):
        _render_kpi(primitive)
    elif isinstance(primitive, TablePrimitive):
        _render_table(primitive)
    elif isinstance(primitive, ChoroplethDescriptor):
        _render_choropleth(primitive, geo_key)
    elif isinstance(primitive, BarDescriptor):
        _render_bar(primitive, geo_key)
    elif isinstance(primitive, ScatterDescriptor):
        _render_scatter(primitive, geo_key)
    elif isinstance(primitive, LineDescriptor):
        _render_line(primitive, geo_key)
    elif isinstance(primitive, PointMapDescriptor):
        _render_point_map(primitive, geo_key)
    elif isinstance(primitive, CodePrimitive):
        _render_code(primitive, geo_key)
    elif isinstance(primitive, CompositePrimitive):
        if primitive.heading:
            st.markdown(f"### {primitive.heading}")
        for child in primitive.blocks:
            render_primitive_block(child, geo_key)


def render_primitive_response(
    response: SynthesizedResponse | dict,
    geo_key: int,
    *,
    skip_text: bool = False,
) -> None:
    if isinstance(response, dict):
        response = SynthesizedResponse.model_validate(response)
    for primitive in response.primitives:
        if skip_text and isinstance(primitive, TextPrimitive):
            continue
        render_primitive_block(primitive, geo_key)

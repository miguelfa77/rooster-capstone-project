"""Streamlit renderers for Phase E response primitives."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from agent.renderers import dispatch
from agent.synthesizer import (
    ChartPrimitive,
    CompositePrimitive,
    KpiPrimitive,
    MapPrimitive,
    PrimitiveBlock,
    SynthesizedResponse,
    TablePrimitive,
    TextPrimitive,
)


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


def _render_chart(primitive: ChartPrimitive, geo_key: int) -> None:
    rows = _rows(primitive.data_json)
    if not rows:
        st.caption("No hay datos suficientes para el gráfico.")
        return
    spec = primitive.spec
    df = pd.DataFrame(rows)
    title = spec.title or ""
    labels = {}
    if spec.x:
        labels[spec.x.field] = spec.x.title
    if spec.y:
        labels[spec.y.field] = spec.y.title
    color = spec.color_field if spec.color_field in df.columns else None
    if spec.type == "scatter" and spec.x and spec.y:
        fig = px.scatter(
            df,
            x=spec.x.field,
            y=spec.y.field,
            color=color,
            size=spec.size_field if spec.size_field in df.columns else None,
            hover_name=spec.label_field if spec.label_field in df.columns else None,
            template="plotly_dark",
            labels=labels,
            title=title,
        )
    elif spec.type == "line" and spec.x and spec.y:
        fig = px.line(
            df,
            x=spec.x.field,
            y=spec.y.field,
            color=color,
            template="plotly_dark",
            labels=labels,
            title=title,
        )
    elif spec.type == "histogram" and spec.x:
        fig = px.histogram(df, x=spec.x.field, color=color, template="plotly_dark", labels=labels, title=title)
    elif spec.type == "bar" and spec.x and spec.y:
        fig = px.bar(df, x=spec.x.field, y=spec.y.field, color=color, template="plotly_dark", labels=labels, title=title)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=36 if title else 0, b=0))
    st.plotly_chart(fig, use_container_width=True, key=f"primitive_chart_{geo_key}_{id(primitive)}")


def _render_map(primitive: MapPrimitive, geo_key: int) -> None:
    for idx, layer in enumerate(primitive.layers):
        rows = _rows(layer.data_json)
        encoding = (
            layer.encoding.model_dump(exclude_none=True)
            if hasattr(layer.encoding, "model_dump")
            else dict(layer.encoding or {})
        )
        meta = {"geo_key": f"{geo_key}_{idx}", **encoding}
        if layer.type == "markers":
            dispatch("geo_map", rows, meta, "")
        elif layer.type == "choropleth":
            dispatch("choropleth_focus", rows, meta, "")
        else:
            dispatch("neighborhood_map", rows, meta, "")


def render_primitive_block(primitive: PrimitiveBlock, geo_key: int) -> None:
    if isinstance(primitive, TextPrimitive):
        _render_text(primitive)
    elif isinstance(primitive, KpiPrimitive):
        _render_kpi(primitive)
    elif isinstance(primitive, TablePrimitive):
        _render_table(primitive)
    elif isinstance(primitive, ChartPrimitive):
        _render_chart(primitive, geo_key)
    elif isinstance(primitive, MapPrimitive):
        _render_map(primitive, geo_key)
    elif isinstance(primitive, CompositePrimitive):
        if primitive.heading:
            st.markdown(f"### {primitive.heading}")
        for child in primitive.blocks:
            render_primitive_block(child, geo_key)


def render_primitive_response(response: SynthesizedResponse | dict, geo_key: int) -> None:
    if isinstance(response, dict):
        response = SynthesizedResponse.model_validate(response)
    for primitive in response.primitives:
        render_primitive_block(primitive, geo_key)

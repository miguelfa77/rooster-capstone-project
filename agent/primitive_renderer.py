"""Streamlit renderers for Phase E response primitives."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st

from agent.synthesizer import (
    CodePrimitive,
    CompositePrimitive,
    KpiPrimitive,
    PrimitiveBlock,
    SynthesizedResponse,
    TablePrimitive,
    TextPrimitive,
)

_LOG = logging.getLogger("rooster.primitive_renderer")
SANDBOX_URL = os.getenv("ROOSTER_SANDBOX_URL", "http://localhost:8001")


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


def _render_code(primitive: CodePrimitive, geo_key: int) -> None:
    if not st.session_state.get("_rooster_last_execution_results"):
        st.info("No hay datos para ejecutar la visualización.")
        return
    execution_results = st.session_state.get("_rooster_last_execution_results") or []
    rows = (execution_results[0] or {}).get("rows") if execution_results else None
    if not isinstance(rows, list) or not rows:
        st.info("No hay datos para ejecutar la visualización.")
        return

    try:
        response = requests.post(
            f"{SANDBOX_URL}/execute",
            json={"code": primitive.code, "data": rows},
            timeout=20,
        )
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        _LOG.warning("Sandbox request failed: %s", exc)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

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


def render_primitive_block(primitive: PrimitiveBlock, geo_key: int) -> None:
    if isinstance(primitive, TextPrimitive):
        _render_text(primitive)
    elif isinstance(primitive, KpiPrimitive):
        _render_kpi(primitive)
    elif isinstance(primitive, TablePrimitive):
        _render_table(primitive)
    elif isinstance(primitive, CodePrimitive):
        _render_code(primitive, geo_key)
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

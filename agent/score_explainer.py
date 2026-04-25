"""
Decompose investment_score and optional LLM tie-in to session priorities.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from sqlalchemy import text

from agent.llm_sql import get_pg_engine
from agent.responses_api import extract_response_text, get_openai_client

EXPLAIN_MODEL = os.getenv("ROOSTER_EXPLAIN_MODEL", "gpt-4o-mini")


def _yield_component(yield_pct: float | None) -> float:
    if yield_pct is None or (isinstance(yield_pct, float) and pd.isna(yield_pct)):
        return 0.0
    return float(yield_pct) * 0.5


def _transport_bonus(avg_dist: float | None) -> float:
    if avg_dist is None or (isinstance(avg_dist, float) and pd.isna(avg_dist)):
        d = 99999.0
    else:
        d = float(avg_dist)
    if d < 400:
        return 2.0
    if d < 700:
        return 1.0
    return 0.0


def _tourism_bonus(tourist_density_pct: float | None) -> float:
    if tourist_density_pct is None or (isinstance(tourist_density_pct, float) and pd.isna(tourist_density_pct)):
        td = 0.0
    else:
        td = float(tourist_density_pct)
    return 1.0 if td < 10 else 0.0


def explain_investment_score_row(row: dict[str, Any]) -> dict[str, Any]:
    """Pure decomposition mirroring the SQL view logic (yield weight 0.5 in product)."""
    y = row.get("gross_rental_yield_pct")
    parts = {
        "yield_contribution": round(_yield_component(
            float(y) if y is not None and not (isinstance(y, float) and pd.isna(y)) else None
        ), 2),
        "transport_bonus": round(_transport_bonus(
            float(row.get("avg_dist_to_stop_m"))
            if row.get("avg_dist_to_stop_m") is not None
            and not (
                isinstance(row.get("avg_dist_to_stop_m"), float)
                and pd.isna(row.get("avg_dist_to_stop_m"))
            )
            else None
        ), 2),
        "tourism_low_density_bonus": round(
            _tourism_bonus(
                float(row.get("tourist_density_pct"))
                if row.get("tourist_density_pct") is not None
                and not (
                    isinstance(row.get("tourist_density_pct"), float)
                    and pd.isna(row.get("tourist_density_pct"))
                )
                else None
            ),
            2,
        ),
    }
    return parts


def explain_neighborhood_investment_score(
    neighborhood_name: str,
    session_priorities: dict[str, float] | None = None,
    timeout_sec: float = 10.0,
) -> str:
    """Fetch profile row, decompose, optional short LLM Spanish explanation."""
    eng = get_pg_engine()
    sql = """
        SELECT gross_rental_yield_pct, avg_dist_to_stop_m, tourist_density_pct, investment_score
        FROM analytics.neighborhood_profile
        WHERE neighborhood_name = :name
        LIMIT 1
    """
    df = pd.read_sql(text(sql), eng, params={"name": neighborhood_name})
    if df.empty:
        return f"No hay datos de perfil para {neighborhood_name!r}."
    row = df.iloc[0].to_dict()
    parts = explain_investment_score_row(
        {**row, "gross_rental_yield_pct": row.get("gross_rental_yield_pct")}
    )
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        return (
            f"{neighborhood_name}: inversión ≈ aportes "
            f"rentabilidad {parts['yield_contribution']}, transporte {parts['transport_bonus']}, "
            f"bono baja presión turística {parts['tourism_low_density_bonus']} (score {row.get('investment_score')})."
        )
    client = get_openai_client(timeout_sec)
    pri = session_priorities or {}
    r = client.responses.create(
        model=EXPLAIN_MODEL,
        instructions="Español, 2 frases, conecta las magnitudes con las prioridades del usuario si aplica.",
        input=(
            f"Barrio: {neighborhood_name}\n"
            f"Componentes: {parts}\n"
            f"Score total: {row.get('investment_score')}\n"
            f"Prioridades (0-1): {pri}\n"
        ),
        max_output_tokens=180,
        temperature=0.2,
    )
    t = extract_response_text(r)
    return t.strip() or str(parts)

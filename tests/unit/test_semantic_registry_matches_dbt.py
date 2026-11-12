"""Guard against drift between the semantic registry and the dbt analytics models.

Every registry metric column ("analytics.<model>.<column>") must be an output
column of dbt/models/analytics/<model>.sql (each model's alias equals its file
name here), and every metric key referenced by concepts / metric components
must be a registry metric.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.semantic_layer.loader import load_registry
from agent.semantic_layer.models import SemanticRegistry

MODELS_DIR = Path(__file__).resolve().parents[2] / "dbt" / "models" / "analytics"


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def output_columns(sql: str) -> set[str]:
    """Aliases of the outermost SELECT list (``... AS x`` or bare ``t.x`` / ``x``)."""
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"\{\{.*?\}\}", "", sql, flags=re.S)
    # Locate the outermost SELECT ... FROM, ignoring keywords inside parentheses.
    depth, select_end, from_start = 0, None, None
    for m in re.finditer(r"\(|\)|\bselect\b|\bfrom\b", sql, flags=re.I):
        tok = m.group().lower()
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth == 0 and tok == "select" and select_end is None:
            select_end = m.end()
        elif depth == 0 and tok == "from" and select_end is not None:
            from_start = m.start()
            break
    assert select_end is not None and from_start is not None, "no top-level SELECT ... FROM"
    columns = set()
    for item in _split_top_level(sql[select_end:from_start]):
        name = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", item.strip())
        assert name, f"cannot parse select item: {item!r}"
        columns.add(name.group(1).lower())
    return columns


def registry_problems(registry: SemanticRegistry) -> list[str]:
    problems = []
    metric_keys = {m.key for m in registry.metrics}
    for metric in registry.metrics:
        schema, model, column = metric.column.split(".")
        model_file = MODELS_DIR / f"{model}.sql"
        if schema != "analytics" or not model_file.exists():
            problems.append(f"{metric.key}: no dbt model for {metric.column}")
        elif column not in output_columns(model_file.read_text(encoding="utf-8")):
            problems.append(f"{metric.key}: {column} is not an output column of {model}")
        problems += [
            f"{metric.key}: component {c} is not a registry metric"
            for c in metric.components
            if c not in metric_keys
        ]
    for concept in registry.concepts:
        for clause in concept.expression.get("all", []) + concept.expression.get("any", []):
            if clause.get("metric") not in metric_keys:
                problems.append(f"{concept.key}: metric {clause.get('metric')} is not a registry metric")
    return problems


def test_parser_reads_final_select_aliases() -> None:
    sql = (MODELS_DIR / "neighborhood_transport.sql").read_text(encoding="utf-8")
    assert output_columns(sql) == {
        "neighborhood_id",
        "neighborhood_name",
        "transit_stop_count",
        "avg_dist_to_stop_m",
    }


def test_registry_matches_dbt_models() -> None:
    assert registry_problems(load_registry()) == []


def _break_metric_column(data: dict) -> None:
    data["metrics"][0]["column"] = "analytics.neighborhood_profile.gross_rental_yeild_pct"


def _break_metric_view(data: dict) -> None:
    data["metrics"][0]["column"] = "analytics.neighbourhood_profile.gross_rental_yield_pct"


def _break_component(data: dict) -> None:
    data["metrics"][0]["components"] = ["median_alquiler_prize"]


def _break_concept_metric(data: dict) -> None:
    data["concepts"][0]["expression"]["all"][0]["metric"] = "gross_yield"


@pytest.mark.parametrize(
    "breakage", [_break_metric_column, _break_metric_view, _break_component, _break_concept_metric]
)
def test_misspelling_is_detected(breakage) -> None:
    data = load_registry().model_dump()
    breakage(data)
    assert registry_problems(SemanticRegistry.model_validate(data)) != []

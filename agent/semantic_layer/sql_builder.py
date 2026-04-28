"""SQL builder for compositional analytical tools.

All user-controlled inputs are whitelisted against the semantic registry before
SQL is emitted. The current Phase B scope targets analytics.neighborhood_profile
plus listing snapshot trends.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent.render_thresholds import (
    MIN_ALQUILER_COUNT_DEFAULT,
    MIN_VENTA_COUNT_DEFAULT,
)
from agent.semantic_layer.loader import load_registry
from agent.semantic_layer.models import MetricEntry

FILTER_OPERATORS = {">=", "<=", ">", "<", "=", "!=", "in", "not_in"}
AGGREGATIONS = {"mean", "median", "p25", "p75", "min", "max", "count", "stddev"}
TIME_GRANULARITIES = {"month", "quarter", "snapshot"}
_DEBUG_LOG_PATH = Path("/Users/miguelfa/Projects/rooster-capstone-project/.cursor/debug-3ce0b8.log")


# region agent log
def _debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": "3ce0b8",
            "runId": "initial",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
# endregion

_AGG_SQL = {
    "mean": "AVG({col})",
    "median": "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col})",
    "p25": "PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col})",
    "p75": "PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col})",
    "min": "MIN({col})",
    "max": "MAX({col})",
    "count": "COUNT({col})",
    "stddev": "STDDEV({col})",
}


def metric_entries() -> dict[str, MetricEntry]:
    return {m.key: m for m in load_registry().metrics}


def metric_keys() -> list[str]:
    return sorted(metric_entries())


def metric_column(metric: str) -> str:
    entry = metric_entries().get(metric)
    if entry is None:
        raise ValueError(f"Unknown metric: {metric}")
    if not entry.column.startswith("analytics.neighborhood_profile."):
        raise ValueError(f"Metric is not available in neighborhood_profile: {metric}")
    return "np." + entry.column.rsplit(".", 1)[-1]


def _metric_alias(metric: str) -> str:
    return metric.replace(".", "_")


def _coerce_limit(value: Any, default: int = 20, max_value: int = 100) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(1, min(max_value, n))


def _min_samples(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        venta = value.get("venta", value.get("venta_count", MIN_VENTA_COUNT_DEFAULT))
        alquiler = value.get(
            "alquiler",
            value.get("alquiler_count", MIN_ALQUILER_COUNT_DEFAULT),
        )
    elif value is None:
        venta = MIN_VENTA_COUNT_DEFAULT
        alquiler = MIN_ALQUILER_COUNT_DEFAULT
    else:
        venta = alquiler = value
    try:
        venta_i = max(0, min(100, int(venta)))
    except (TypeError, ValueError):
        venta_i = MIN_VENTA_COUNT_DEFAULT
    try:
        alquiler_i = max(0, min(100, int(alquiler)))
    except (TypeError, ValueError):
        alquiler_i = MIN_ALQUILER_COUNT_DEFAULT
    return venta_i, alquiler_i


def _where_for_samples(min_listings: Any, bind: dict[str, Any]) -> list[str]:
    venta, alquiler = _min_samples(min_listings)
    bind["min_venta_count"] = venta
    bind["min_alquiler_count"] = alquiler
    return [
        "COALESCE(np.venta_count, 0) >= :min_venta_count",
        "COALESCE(np.alquiler_count, 0) >= :min_alquiler_count",
    ]


def _range_check(metric: str, value: Any) -> None:
    entry = metric_entries()[metric]
    bounds = entry.value_range or {}
    try:
        val = float(value)
    except (TypeError, ValueError):
        return
    lower = bounds.get("min")
    upper = bounds.get("max")
    if lower is not None and val < float(lower):
        raise ValueError(f"{metric} below minimum {lower}")
    if upper is not None and val > float(upper):
        raise ValueError(f"{metric} above maximum {upper}")


def _filters_sql(filters: Any, bind: dict[str, Any]) -> list[str]:
    if not isinstance(filters, dict):
        return []
    out: list[str] = []
    for metric, spec in filters.items():
        if metric not in metric_entries() or not isinstance(spec, dict):
            raise ValueError(f"Invalid filter metric: {metric}")
        op = str(spec.get("operator") or "=").strip()
        if op not in FILTER_OPERATORS:
            raise ValueError(f"Invalid operator for {metric}: {op}")
        col = metric_column(metric)
        pname = f"filter_{len(bind)}"
        value = spec.get("value")
        if op in {"in", "not_in"}:
            if not isinstance(value, list):
                raise ValueError(f"Filter {metric} requires a list")
            keys = []
            for item in value:
                _range_check(metric, item)
                key = f"{pname}_{len(keys)}"
                bind[key] = item
                keys.append(f":{key}")
            sql_op = "IN" if op == "in" else "NOT IN"
            out.append(f"{col} {sql_op} ({', '.join(keys)})")
        else:
            _range_check(metric, value)
            bind[pname] = value
            out.append(f"{col} {op} :{pname}")
    return out


def build_select_metrics_sql(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metrics = [m for m in (params.get("metrics") or []) if isinstance(m, str)]
    if not metrics:
        metrics = ["investment_score", "gross_rental_yield_pct"]
    for metric in metrics:
        metric_column(metric)
    bind: dict[str, Any] = {}
    select_cols = ["np.neighborhood_name"]
    select_cols.extend(f"{metric_column(m)} AS {_metric_alias(m)}" for m in metrics)
    select_cols.extend(["np.venta_count", "np.alquiler_count", "np.total_count"])
    where = ["TRUE"]
    where.extend(_where_for_samples(params.get("min_listings"), bind))
    where.extend(_filters_sql(params.get("filters"), bind))
    nbs = params.get("neighborhoods")
    if isinstance(nbs, list) and nbs:
        keys = []
        for nm in nbs[:20]:
            if isinstance(nm, str) and nm.strip():
                key = f"nb_{len(keys)}"
                bind[key] = nm.strip()
                keys.append(f":{key}")
        if keys:
            where.append(f"np.neighborhood_name IN ({', '.join(keys)})")
    order_by = params.get("order_by") or {}
    _debug_log(
        "H3",
        "agent/semantic_layer/sql_builder.py:build_select_metrics_sql",
        "select_metrics SQL builder pre-order_by access",
        {
            "params_type": type(params).__name__,
            "metrics": metrics,
            "filters_type": type(params.get("filters")).__name__,
            "filters_preview": params.get("filters"),
            "order_by_type": type(order_by).__name__,
            "order_by_preview": order_by,
            "min_listings_type": type(params.get("min_listings")).__name__,
            "neighborhoods_type": type(nbs).__name__,
        },
    )
    if isinstance(order_by, dict) and order_by.get("metric") in metrics:
        order_metric = str(order_by["metric"])
    else:
        order_metric = metrics[0]
    direction = "ASC" if str(order_by.get("direction", "desc")).lower() == "asc" else "DESC"
    limit = _coerce_limit(params.get("limit"), default=20)
    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM analytics.neighborhood_profile np
        WHERE {" AND ".join(where)}
        ORDER BY {metric_column(order_metric)} {direction} NULLS LAST
        LIMIT {limit}
    """
    return sql, bind


def build_compute_aggregate_sql(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metric = str(params.get("metric") or "")
    col = metric_column(metric)
    aggregation = str(params.get("aggregation") or "median").lower()
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Invalid aggregation: {aggregation}")
    bind: dict[str, Any] = {}
    where = ["TRUE"]
    where.extend(_where_for_samples(params.get("min_listings"), bind))
    where.extend(_filters_sql(params.get("filters"), bind))
    sql = f"""
        SELECT {AGG_SQL(aggregation, col)} AS value,
               :metric AS metric,
               :aggregation AS aggregation
        FROM analytics.neighborhood_profile np
        WHERE {" AND ".join(where)}
    """
    bind["metric"] = metric
    bind["aggregation"] = aggregation
    return sql, bind


def AGG_SQL(aggregation: str, col: str) -> str:
    return _AGG_SQL[aggregation].format(col=col)


def build_temporal_series_sql(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metric = str(params.get("metric") or "median_venta_price")
    if metric not in {"median_venta_price", "median_alquiler_price", "venta_count", "alquiler_count"}:
        raise ValueError("temporal_series currently supports price/count listing metrics")
    granularity = str(params.get("time_granularity") or "month").lower()
    if granularity not in TIME_GRANULARITIES:
        raise ValueError(f"Invalid time_granularity: {granularity}")
    operation = "venta" if metric in {"median_venta_price", "venta_count"} else "alquiler"
    value_expr = "COUNT(*)" if metric.endswith("_count") else "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ls.price_int)"
    if granularity == "snapshot":
        time_expr = "ls.scraped_at::date"
    elif granularity == "quarter":
        time_expr = "date_trunc('quarter', ls.scraped_at)::date"
    else:
        time_expr = "date_trunc('month', ls.scraped_at)::date"
    bind: dict[str, Any] = {"operation": operation}
    where = ["l.operation = :operation", "ls.price_int IS NOT NULL"]
    filters = params.get("filters") or {}
    neighborhoods = filters.get("neighborhood_name") if isinstance(filters, dict) else None
    if isinstance(neighborhoods, dict) and isinstance(neighborhoods.get("value"), list):
        keys = []
        for nm in neighborhoods["value"][:20]:
            if isinstance(nm, str) and nm.strip():
                key = f"nb_{len(keys)}"
                bind[key] = nm.strip()
                keys.append(f":{key}")
        if keys:
            where.append(f"n.name IN ({', '.join(keys)})")
    sql = f"""
        SELECT n.name AS neighborhood_name,
               {time_expr} AS time_point,
               {value_expr} AS value,
               :metric AS metric
        FROM core.listing_snapshots ls
        JOIN core.listings l ON l.url = ls.url
        LEFT JOIN core.neighborhoods n ON n.id = l.neighborhood_id
        WHERE {" AND ".join(where)}
        GROUP BY n.name, {time_expr}
        ORDER BY time_point ASC, neighborhood_name ASC
        LIMIT 500
    """
    bind["metric"] = metric
    return sql, bind

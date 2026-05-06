"""Rooster tool validation, execution, and shared chat helpers."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from sqlalchemy import text

from agent.config import SYNTH_RESULT_SAMPLE_ROWS
from agent.llm_sql import get_pg_engine
from agent.render_thresholds import add_data_confidence
from agent.semantic_layer.sql_builder import (
    build_compute_aggregate_sql,
    build_select_metrics_sql,
    build_temporal_series_sql,
    metric_keys,
)


def _sql_escape(s: str) -> str:
    return (s or "").replace("'", "''")


def get_live_schema_context() -> str:
    """Build live DB snapshot text for the planner (call from app with @st.cache_data)."""
    engine = get_pg_engine()
    lines: list[str] = [
        "=== LIVE DATABASE STATE ===",
        "TOTAL DATA:",
    ]
    try:
        counts_df = pd.read_sql(
            text(
                """
                SELECT
                    (SELECT COUNT(*) FROM core.listings WHERE price_int > 0) AS total_listings,
                    (SELECT COUNT(*) FROM core.transit_stops) AS transit_stops,
                    (SELECT COUNT(*) FROM core.tourist_apartments
                     WHERE status IS NULL OR COALESCE(lower(status), '') = 'active') AS tourist_apts,
                    (SELECT COUNT(DISTINCT neighborhood_id) FROM core.listings WHERE neighborhood_id IS NOT NULL)
                        AS neighborhoods_with_listings
                """
            ),
            engine,
        ).iloc[0]
        lines.append(f"  Listings: {int(counts_df['total_listings']):,}")
        lines.append(f"  Transit stops: {int(counts_df['transit_stops']):,}")
        lines.append(f"  Tourist apartments (active or unset status): {int(counts_df['tourist_apts']):,}")
        lines.append(f"  Neighborhoods with listings: {int(counts_df['neighborhoods_with_listings'])}")
    except Exception as e:
        lines.append(f"  (counts unavailable: {e})")

    lines.extend(["", "NEIGHBORHOODS (with listings — exact names as stored):"])
    try:
        nb = pd.read_sql(
            text(
                """
                SELECT
                    n.name,
                    COUNT(l.url) FILTER (WHERE l.operation = 'venta') AS venta_count,
                    COUNT(l.url) FILTER (WHERE l.operation = 'alquiler') AS alquiler_count,
                    BOOL_OR(np.gross_rental_yield_pct IS NOT NULL) AS has_yield
                FROM core.neighborhoods n
                LEFT JOIN core.listings l
                    ON l.neighborhood_id = n.id AND l.price_int > 0
                LEFT JOIN analytics.neighborhood_profile np ON np.neighborhood_id = n.id
                GROUP BY n.id, n.name
                HAVING COUNT(l.url) > 0
                ORDER BY COUNT(l.url) DESC
                LIMIT 80
                """
            ),
            engine,
        )
        for _, row in nb.iterrows():
            flags: list[str] = []
            if int(row["venta_count"] or 0) > 0:
                flags.append(f"{int(row['venta_count'])} venta")
            if int(row["alquiler_count"] or 0) > 0:
                flags.append(f"{int(row['alquiler_count'])} alquiler")
            if row.get("has_yield"):
                flags.append("has yield")
            lines.append(f"  - {row['name']} ({', '.join(flags)})")
    except Exception as e:
        lines.append(f"  (neighborhood list unavailable: {e})")

    lines.extend(["", "PRICE RANGES (listings, price_int > 0):"])
    try:
        prices_df = pd.read_sql(
            text(
                """
                SELECT operation,
                       MIN(price_int) AS min_price,
                       MAX(price_int) AS max_price,
                       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_int) AS median_price
                FROM core.listings
                WHERE price_int > 0
                GROUP BY operation
                """
            ),
            engine,
        )
        for _, row in prices_df.iterrows():
            op = row["operation"]
            lines.append(
                f"  {op}: €{int(row['min_price']):,} – €{int(row['max_price']):,} "
                f"(median €{int(row['median_price']):,})"
            )
    except Exception as e:
        lines.append(f"  (prices unavailable: {e})")

    lines.extend(
        [
            "",
            "IMPORTANT: When the user names a barrio, match fuzzy to the list above and use the exact DB name in tool params.",
            "=== END DATABASE STATE ===",
        ]
    )
    return "\n".join(lines)


def extract_neighborhood_names_from_schema(schema_context: str) -> list[str]:
    """Parse '  - Name (...)` lines from live schema string."""
    out: list[str] = []
    for line in (schema_context or "").splitlines():
        m = re.match(r"^\s*-\s+(.+?)\s+\(", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _normalize_follow_ups_payload(raw: Any) -> list[str]:
    """Turn JSON array, object with suggestions, or list of objects into pill strings."""
    out: list[str] = []

    def _append_one(s: str | None) -> None:
        if not s or not str(s).strip():
            return
        t = str(s).strip()
        words = t.split()
        if len(words) > 12:
            t = " ".join(words[:12])
        out.append(t)

    if raw is None:
        return out
    if isinstance(raw, str):
        _append_one(raw)
        return out[:3]
    if isinstance(raw, dict):
        if "suggestions" in raw and isinstance(raw["suggestions"], list):
            return _normalize_follow_ups_payload(raw["suggestions"])
        for k in ("follow_ups", "pills", "actions"):
            if k in raw and isinstance(raw[k], list):
                return _normalize_follow_ups_payload(raw[k])
        if "label" in raw or "text" in raw:
            _append_one(
                raw.get("label")
                or raw.get("text")
                or raw.get("title")
            )
            return out[:3]
        return out
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, str):
                _append_one(x)
            elif isinstance(x, dict):
                lab = (
                    x.get("label")
                    or x.get("text")
                    or x.get("title")
                    or x.get("action")
                )
                if isinstance(lab, str) and lab.strip():
                    _append_one(lab)
                elif x.get("label") is not None:
                    _append_one(str(x.get("label")))
            else:
                _append_one(str(x))
            if len(out) >= 3:
                break
        return out
    return out


def strip_follow_ups_suffix(text: str) -> tuple[str, list[str]]:
    """
    Remove <!-- FOLLOW_UPS: ... --> from synthesiser output (array or JSON object); return prose + pills.
    """
    if not (text or "").strip():
        return "", []
    t = text.strip()
    m_open = re.search(r"<!--\s*FOLLOW_UPS:\s*", t, re.IGNORECASE)
    if not m_open:
        return t, []
    start = m_open.start()
    payload_start = m_open.end()
    close = t.find("-->", payload_start)
    if close == -1:
        return t, []
    payload = t[payload_start:close].strip()
    prose = t[:start].strip()
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return prose, []
    labels = _normalize_follow_ups_payload(raw)
    return prose, labels[:3]


def format_validation_plan_correction(
    errors: list[str],
    raw_tool_calls: list[dict[str, Any]],
) -> str:
    """Build user message block for a replan after validate_plan dropped all tool calls."""
    err_lines = "\n".join(f"- {e}" for e in (errors or []) if str(e).strip()) or "(no details)"
    tools_tried = ", ".join(
        str(c.get("tool") or "?") for c in (raw_tool_calls or []) if isinstance(c, dict)
    ) or "(none)"
    return f"""YOUR PREVIOUS TOOL CALLS FAILED VALIDATION — FIX THEM:

Validation errors:
{err_lines}

Tools you attempted: {tools_tried}

Rules:
- Only documented Rooster tool names are allowed (see planner tools).

Generate corrected tool_calls using known tools only. Do not repeat unknown tool names."""


def validate_plan(plan: dict[str, Any], schema_context: str) -> dict[str, Any]:
    """Minimal safety validation: only allow known tool names."""
    del schema_context
    known_tools = {
        "select_metrics",
        "query_listings",
        "query_transit_stops",
        "query_tourist_apartments",
        "resolve_spatial_reference",
        "compute_aggregate",
        "temporal_series",
    }
    errors: list[str] = []
    corrected: list[dict[str, Any]] = []
    for call in plan.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        tool = call.get("tool")
        params = _coerce_tool_params(call.get("params"))
        if not isinstance(tool, str) or not tool:
            errors.append("Invalid tool call")
            continue
        if tool not in known_tools:
            errors.append(f"Unknown tool: {tool}")
            continue
        corrected.append({**call, "params": params})

    out = dict(plan)
    out["tool_calls"] = corrected
    out["validation_errors"] = errors
    if not plan.get("tool_calls"):
        out["valid"] = len(errors) == 0
    else:
        out["valid"] = len(errors) == 0 and len(corrected) > 0
    return out


def _user_wants_at_least_rooms_not_exact(user_message: str) -> bool:
    """
    True when the user text suggests a *floor* on bedrooms, not an exact count.
    If False and they mention rooms, 'N habitaciones' is treated as exact elsewhere.
    """
    t = (user_message or "").lower()
    if any(
        p in t
        for p in (
            "al menos",
            "almenos",
            "mínimo",
            "minimo",
            "minimum",
            "at least",
            "más de ",
            "mas de ",
            "como mínimo",
            "como minimo",
            "upwards of",
        )
    ):
        return True
    if "entre" in t and "habit" in t:
        return True
    if re.search(r"\d+\s+o\s+más", t) or re.search(r"\d+\s+o\s+mas", t):
        return True
    return False


def _user_message_mentions_room_count(user_message: str) -> bool:
    t = (user_message or "").lower()
    return bool(
        re.search(
            r"\b(?:habitaciones|hab\.|dormitorios|dorm\.?)\b",
            t,
            re.IGNORECASE,
        )
    )


def _normalize_query_listings_room_params(
    params: dict[str, Any], user_message: str | None
) -> dict[str, Any]:
    """
    If the model only passes min_rooms for an exact-N query, SQL would use >= N and
    return larger units. When the user message looks like an exact room count (not
    'at least'), set max_rooms = min_rooms.
    """
    mr = params.get("min_rooms")
    xr = params.get("max_rooms")
    if mr is None or xr is not None:
        return params
    if not user_message or not _user_message_mentions_room_count(user_message):
        return params
    if _user_wants_at_least_rooms_not_exact(user_message):
        return params
    try:
        n = int(mr)
    except (TypeError, ValueError):
        return params
    return {**params, "max_rooms": n}


def query_listings_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    conditions = ["l.price_int > 0", "l.area_sqm > 0"]
    nb = params.get("neighborhood")
    if isinstance(nb, str) and nb.strip():
        es = _sql_escape(nb.strip())
        conditions.append(
            f"similarity(unaccent(lower(n.name)), unaccent(lower('{es}'))) > 0.4"
        )
    op = (params.get("operation") or "venta").strip().lower()
    if op in ("venta", "alquiler"):
        conditions.append(f"l.operation = '{op}'")
    if params.get("max_price") is not None:
        try:
            conditions.append(f"l.price_int <= {int(params['max_price'])}")
        except (TypeError, ValueError):
            pass
    if params.get("min_price") is not None:
        try:
            conditions.append(f"l.price_int >= {int(params['min_price'])}")
        except (TypeError, ValueError):
            pass
    if params.get("min_rooms") is not None:
        try:
            conditions.append(f"l.rooms_int >= {int(params['min_rooms'])}")
        except (TypeError, ValueError):
            pass
    if params.get("max_rooms") is not None:
        try:
            conditions.append(f"l.rooms_int <= {int(params['max_rooms'])}")
        except (TypeError, ValueError):
            pass
    if params.get("only_below_median"):
        conditions.append(
            """(
            (l.operation = 'venta' AND l.price_int < np.median_venta_price)
            OR (l.operation = 'alquiler' AND l.price_int < np.median_alquiler_price)
        )"""
        )
    amenity_map = {
        "parking": "l.has_parking",
        "terrace": "l.has_terrace",
        "elevator": "l.has_elevator",
        "ac": "l.has_ac",
        "renovated": "l.is_renovated",
    }
    for a in params.get("amenities") or []:
        if isinstance(a, str) and a.lower() in amenity_map:
            conditions.append(f"{amenity_map[a.lower()]} = true")
    lim = 25
    try:
        lim = max(1, min(100, int(params.get("limit", 25))))
    except (TypeError, ValueError):
        lim = 25
    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT
            l.url,
            l.operation,
            l.price_int,
            l.area_sqm,
            l.rooms_int,
            ROUND(l.floor_int)::integer AS floor_int,
            ROUND((l.price_int::numeric / NULLIF(l.area_sqm::numeric, 0)), 0) AS eur_per_sqm,
            l.has_parking, l.has_terrace, l.has_elevator, l.is_renovated,
            l.lat, l.lng, l.geocode_quality,
            n.name AS neighborhood_name,
            np.gross_rental_yield_pct AS neighborhood_yield,
            np.investment_score,
            np.median_venta_price AS neighborhood_median,
            CASE WHEN l.operation = 'venta' AND l.price_int < np.median_venta_price THEN true
                 WHEN l.operation = 'alquiler' AND l.price_int < np.median_alquiler_price THEN true
                 ELSE false END AS below_median
        FROM core.listings l
        JOIN core.neighborhoods n ON n.id = l.neighborhood_id
        LEFT JOIN analytics.neighborhood_profile np ON np.neighborhood_id = l.neighborhood_id
        WHERE {where_clause}
        ORDER BY l.price_int ASC NULLS LAST
        LIMIT {lim}
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def query_transit_stops_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    nb = params.get("neighborhood")
    if isinstance(nb, str) and nb.strip():
        es = _sql_escape(nb.strip())
        where = f"WHERE similarity(unaccent(lower(n.name)), unaccent(lower('{es}'))) > 0.4"
    else:
        where = "WHERE TRUE"
    sql = f"""
        SELECT t.name, t.stop_type, t.lat, t.lng, n.name AS neighborhood_name
        FROM core.transit_stops t
        JOIN core.neighborhoods n ON n.id = t.neighborhood_id
        {where}
        ORDER BY t.stop_type NULLS LAST
        LIMIT 2000
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def query_tourist_apartments_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    nb = params.get("neighborhood")
    if isinstance(nb, str) and nb.strip():
        es = _sql_escape(nb.strip())
        where = f"""WHERE similarity(unaccent(lower(n.name)), unaccent(lower('{es}'))) > 0.4
          AND (ta.status IS NULL OR lower(ta.status) = 'active')"""
    else:
        where = "WHERE (ta.status IS NULL OR lower(ta.status) = 'active')"
    sql = f"""
        SELECT ta.id, ta.address, ta.lat, ta.lng, n.name AS neighborhood_name
        FROM core.tourist_apartments ta
        JOIN core.neighborhoods n ON n.id = ta.neighborhood_id
        {where}
        LIMIT 2000
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def resolve_spatial_reference_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    from agent.spatial_resolver import match_reference_phrase

    rows = match_reference_phrase(str(params.get("reference") or ""))
    names = [r.get("name") for r in rows if isinstance(r.get("name"), str)]
    if not names:
        return rows
    escaped = ",".join("'" + _sql_escape(str(n)) + "'" for n in names)
    sql = f"""
        SELECT id::text AS neighborhood_id, name AS neighborhood_name
        FROM core.neighborhoods
        WHERE name IN ({escaped})
    """
    try:
        df = pd.read_sql(text(sql), engine)
    except Exception:
        return rows
    id_by_name = {
        str(r["neighborhood_name"]): str(r["neighborhood_id"])
        for r in df.to_dict("records")
    }
    out: list[dict[str, Any]] = []
    for r in rows:
        n = str(r.get("name") or "")
        out.append(
            {
                "neighborhood_id": id_by_name.get(n),
                "neighborhood_name": n,
                "confidence": r.get("confidence", 0.0),
                "source": "spatial_lexicon",
            }
        )
    return out


def _coerce_tool_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        return dict(raw[0])
    return {}


def select_metrics_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    sql, bind = build_select_metrics_sql(params)
    df = pd.read_sql(text(sql), engine, params=bind)
    return add_data_confidence(df.to_dict("records"))


_SELECT_METRIC_ALIASES: dict[str, str] = {
    "median_sale": "median_venta_price",
    "median_alquiler": "median_alquiler_price",
    "tourism_pressure": "tourist_density_pct",
    "eur_per_sqm": "median_venta_eur_per_sqm",
}
_VALID_FILTER_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "neq", "not_null", "is_null", "in", "not_in"}
_SQL_OPERATOR_MAP = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "=", "neq": "!=", "in": "in", "not_in": "not_in"}
_SELECT_METRICS_DOC_LIST = (
    "gross_rental_yield_pct, median_sale, median_alquiler, venta_count, alquiler_count, "
    "investment_score, transit_stop_count, tourism_pressure, eur_per_sqm, data_confidence"
)


def _select_metrics_error(tool: str, params: dict[str, Any], message: str, hint: str) -> dict[str, Any]:
    return {
        "tool": tool,
        "params": params,
        "rows": [],
        "row_count": 0,
        "success": False,
        "error": message,
        "correction_hint": hint,
    }


def _normalize_select_metrics_params(params: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    normalized = dict(params)
    metrics = [m for m in (normalized.get("metrics") or []) if isinstance(m, str)]
    if metrics:
        out_metrics: list[str] = []
        for metric in metrics:
            mapped = _SELECT_METRIC_ALIASES.get(metric, metric)
            if metric == "data_confidence":
                continue
            if mapped not in set(metric_keys()):
                return None, _select_metrics_error(
                    "select_metrics",
                    params,
                    (
                        f"Unknown metric '{metric}'. Available: {_SELECT_METRICS_DOC_LIST}. "
                        "Note: no room or bedroom aggregation exists at neighborhood level."
                    ),
                    "Replace the unknown metric with one from the available list, or explain data is unavailable.",
                )
            out_metrics.append(mapped)
        if not out_metrics:
            out_metrics = ["gross_rental_yield_pct"]
        normalized["metrics"] = out_metrics

    # Allow new top-level params.
    if normalized.get("min_venta_count") is not None or normalized.get("min_alquiler_count") is not None:
        normalized["min_listings"] = {
            "venta": normalized.get("min_venta_count", 3),
            "alquiler": normalized.get("min_alquiler_count", 3),
        }
    order_by = normalized.get("order_by")
    if isinstance(order_by, list) and order_by:
        first = order_by[0] if isinstance(order_by[0], dict) else {}
        field = str(first.get("field") or "").strip()
        direction = str(first.get("direction") or "desc").strip().lower()
        mapped_field = _SELECT_METRIC_ALIASES.get(field, field)
        normalized["order_by"] = {"metric": mapped_field, "direction": ("asc" if direction == "asc" else "desc")}

    raw_filters = normalized.get("filters")
    if isinstance(raw_filters, list):
        converted: list[dict[str, Any]] = []
        for item in raw_filters:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "").strip()
            op = str(item.get("op") or "").strip()
            if field in {"min_venta_count", "min_alquiler_count"}:
                return None, _select_metrics_error(
                    "select_metrics",
                    params,
                    (
                        f"'{field}' is a top-level parameter, not a filter field. "
                        f"Pass it directly: select_metrics({field}=5, ...) not as filters."
                    ),
                    f"Move '{field}' from filters to a top-level select_metrics parameter.",
                )
            if op not in _VALID_FILTER_OPERATORS:
                return None, _select_metrics_error(
                    "select_metrics",
                    params,
                    (
                        f"Invalid operator '{op}'. Valid operators: gt, gte, lt, lte, eq, neq, "
                        "not_null, is_null, in, not_in. For null checks use 'not_null', not 'is_not_null'."
                    ),
                    "Use a valid operator from the documented list.",
                )
            mapped_field = _SELECT_METRIC_ALIASES.get(field, field)
            if mapped_field not in set(metric_keys()):
                return None, _select_metrics_error(
                    "select_metrics",
                    params,
                    (
                        f"Unknown metric '{field}'. Available: {_SELECT_METRICS_DOC_LIST}. "
                        "Note: no room or bedroom aggregation exists at neighborhood level."
                    ),
                    "Use a valid metric field in filters, or remove the unsupported condition.",
                )
            if op == "is_null":
                converted.append({"field": mapped_field, "op": "eq", "value": None})
            elif op == "not_null":
                converted.append({"field": mapped_field, "op": "not_null"})
            else:
                converted.append({"field": mapped_field, "op": _SQL_OPERATOR_MAP[op], "value": item.get("value")})
        normalized["filters"] = converted
    return normalized, None


def compute_aggregate_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    sql, bind = build_compute_aggregate_sql(params)
    df = pd.read_sql(text(sql), engine, params=bind)
    return df.to_dict("records")


def temporal_series_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    sql, bind = build_temporal_series_sql(params)
    df = pd.read_sql(text(sql), engine, params=bind)
    return df.to_dict("records")


TOOL_FUNCTIONS: dict[str, Any] = {
    "select_metrics": select_metrics_fn,
    "compute_aggregate": compute_aggregate_fn,
    "temporal_series": temporal_series_fn,
    "query_listings": query_listings_fn,
    "query_transit_stops": query_transit_stops_fn,
    "query_tourist_apartments": query_tourist_apartments_fn,
    "resolve_spatial_reference": resolve_spatial_reference_fn,
}


def execute_plan(
    validated_plan: dict[str, Any],
    engine,
    user_message: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for call in validated_plan.get("tool_calls") or []:
        tool = call.get("tool")
        params = _coerce_tool_params(call.get("params"))
        _sql_meta_keys = frozenset({"chart_style"})
        sql_params = {k: v for k, v in params.items() if k not in _sql_meta_keys}
        if tool == "query_listings":
            sql_params = _normalize_query_listings_room_params(sql_params, user_message)
        tool_call_id = call.get("_tool_call_id") or call.get("tool_call_id")
        if not isinstance(tool, str) or tool not in TOOL_FUNCTIONS:
            results.append(
                {
                    "tool": tool,
                    "params": params,
                    "rows": [],
                    "row_count": 0,
                    "success": False,
                    "error": f"Unknown tool: {tool}",
                    "correction_hint": "Use one of the documented tool names.",
                    "tool_call_id": tool_call_id,
                }
            )
            continue
        try:
            if tool == "select_metrics":
                normalized, err = _normalize_select_metrics_params(sql_params)
                if err is not None:
                    err["tool_call_id"] = tool_call_id
                    results.append(err)
                    continue
                sql_params = normalized or sql_params
            rows = TOOL_FUNCTIONS[tool](sql_params, engine)
            results.append(
                {
                    "tool": tool,
                    "params": params,
                    "rows": rows,
                    "row_count": len(rows),
                    "success": True,
                    "error": None,
                    "correction_hint": None,
                    "tool_call_id": tool_call_id,
                }
            )
        except Exception as e:
            results.append(
                {
                    "tool": tool,
                    "params": params,
                    "rows": [],
                    "row_count": 0,
                    "success": False,
                    "error": str(e),
                    "correction_hint": "Review the tool error and retry with corrected parameters.",
                    "tool_call_id": tool_call_id,
                }
            )
    return results


def _build_results_summary_for_synth(
    execution_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results_summary: list[dict[str, Any]] = []
    for result in execution_results:
        tool = result.get("tool")
        if result.get("success") and result.get("row_count", 0) > 0:
            sample = (result.get("rows") or [])[:SYNTH_RESULT_SAMPLE_ROWS]
            columns = sorted({k for row in sample if isinstance(row, dict) for k in row})
            results_summary.append(
                {
                    "tool": tool,
                    "row_count": result.get("row_count"),
                    "columns": columns,
                    "sample": sample,
                }
            )
        elif not result.get("success"):
            results_summary.append({"tool": tool, "error": result.get("error")})
        else:
            results_summary.append(
                {
                    "tool": tool,
                    "row_count": 0,
                    "note": "sin filas",
                }
            )
    return results_summary


def format_last_assistant_for_planner(messages: list[dict[str, Any]]) -> str:
    """
    Build a compact text block from the last assistant turn so the planner can resolve
    "estos", "them", etc. Uses `summary` and `render_stack` (and legacy row fields) from
    Streamlit chat history.
    """
    for msg in reversed(messages or []):
        if msg.get("role") != "assistant":
            continue
        parts: list[str] = []
        summary = (msg.get("summary") or "").strip()
        if summary:
            parts.append(f"Assistant prose:\n{summary}")

        stack = msg.get("render_stack") or []
        for block in stack:
            intent = (block.get("intent") or "").strip() or "?"
            rows = block.get("rows") or []
            meta = dict(block.get("meta") or {})

            if intent == "combined_map":
                for label, key in (
                    ("listings", "rows_listings"),
                    ("transit", "rows_transit"),
                    ("tourism", "rows_tourism"),
                ):
                    sub = meta.get(key) or []
                    if sub:
                        line = _format_shown_rows_sample(sub, intent=f"combined_map/{label}")
                        if line:
                            parts.append(line)
                continue

            if rows:
                line = _format_shown_rows_sample(rows, intent=intent)
                if line:
                    parts.append(line)

        # Legacy message shape (non-agent stack)
        if msg.get("rows"):
            line = _format_shown_rows_sample(
                msg.get("rows") or [], intent=str(msg.get("intent") or "legacy")
            )
            if line:
                parts.append(line)

        return "\n\n".join(parts) if parts else ""

    return ""


def _format_shown_rows_sample(
    rows: list[dict[str, Any]],
    *,
    intent: str,
    max_rows: int = 8,
) -> str:
    if not rows:
        return ""
    sample = rows[:max_rows]
    first = sample[0]
    if not isinstance(first, dict):
        return ""
    key_field = next(
        (k for k in ("neighborhood_name", "name", "url") if k in first),
        None,
    )
    if not key_field:
        return f"[intent={intent}: {len(rows)} row(s); no neighborhood_name/name/url in first row]"
    shown = [row.get(key_field) for row in sample if row.get(key_field) is not None]
    shown = [str(x) for x in shown if str(x).strip()]
    if not shown:
        return ""
    extra = f" (+{len(rows) - len(shown)} more)" if len(rows) > len(shown) else ""
    return f"[intent={intent}; Showed {key_field}: {shown}{extra}]"


def _infer_plan_neighborhood_resolved(plan: dict[str, Any]) -> None:
    for c in plan.get("tool_calls") or []:
        p = c.get("params") or {}
        nb = p.get("neighborhood")
        if isinstance(nb, str) and nb.strip():
            plan["neighborhood_resolved"] = nb.strip()
            return
        nbs = p.get("neighborhoods")
        if isinstance(nbs, list) and nbs and isinstance(nbs[0], str) and nbs[0].strip():
            plan["neighborhood_resolved"] = nbs[0].strip()
            return
    plan["neighborhood_resolved"] = None


def _infer_combine_maps_from_tools(plan: dict[str, Any]) -> bool:
    spatial = {"query_listings", "query_transit_stops", "query_tourist_apartments"}
    n = sum(1 for c in (plan.get("tool_calls") or []) if c.get("tool") in spatial)
    return n >= 2


def update_conversation_state(
    state: dict[str, Any],
    question: str,
    plan: dict[str, Any],
    execution_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Update mutable conversation state after a turn."""
    if plan.get("neighborhood_resolved"):
        nb = str(plan["neighborhood_resolved"]).strip()
        if nb and nb not in state["neighborhoods_discussed"]:
            state["neighborhoods_discussed"].append(nb)
    for call in plan.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        p = call.get("params") or {}
        op = p.get("operation")
        if op and op != "both":
            state["operation_focus"] = op
        mp = p.get("max_price")
        if mp is not None:
            try:
                state["price_ceiling"] = int(mp)
            except (TypeError, ValueError):
                pass
    state["turns"] = int(state.get("turns", 0)) + 1
    if state["turns"] <= 2:
        state["stage"] = "orienting"
    elif len(state.get("neighborhoods_discussed", [])) >= 2:
        state["stage"] = "evaluating"
    elif state["turns"] >= 6:
        state["stage"] = "deciding"
    if plan.get("tool_calls"):
        state["last_intent"] = plan["tool_calls"][0].get("tool")
    priority_keywords = {
        "yield": ["yield", "rentabilidad", "rendimiento", "retorno"],
        "transport": ["transport", "metro", "bus", "conectividad"],
        "price": ["precio", "barato", "económico", "budget", "coste"],
        "tourism": ["turístico", "airbnb", "vut", "corta estancia"],
    }
    q_lower = (question or "").lower()
    for priority, kws in priority_keywords.items():
        if any(kw in q_lower for kw in kws):
            if priority not in state["user_priorities"]:
                state["user_priorities"].append(priority)
    return state

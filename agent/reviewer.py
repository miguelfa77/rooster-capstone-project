"""Reviewer 1: data correctness checks for Rooster v2."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.config import (
    REASONING_REVIEWER,
    REVIEWER_MAX_OUTPUT_TOKENS,
    REVIEWER_MODEL_DEFAULT,
    REVIEWER_RESULT_SAMPLE_DISPLAY_ROWS,
    REVIEWER_RESULT_SAMPLE_ROWS,
)
from agent.responses_api import (
    get_openai_client,
    parse_strict_response,
    reasoning_param_for_model,
    supports_temperature,
)
from agent.semantic_layer.models import ResolvedQuery
from agent.semantic_layer.sql_builder import metric_keys
from agent.stage_logging import log_stage

_LOG = logging.getLogger("rooster.reviewer")


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewerFailure(StrictBaseModel):
    type: Literal["substitution", "filter_missing", "metric_absent", "shape_mismatch"]
    description: str
    suggested_correction: str


class ReviewerVerdict(StrictBaseModel):
    verdict: Literal["pass", "fail"]
    reason: str = ""
    suggested_correction: str = ""
    failures: list[ReviewerFailure] = Field(default_factory=list)
    source: Literal["deterministic", "llm", "fallback"] = "llm"


REVIEWER_INSTRUCTIONS = """You are a data correctness reviewer for a real estate analytics assistant.

Your job is narrow and specific: check whether the executed database query
correctly implements what the user asked for. You are NOT reviewing prose,
charts, or output format — only whether the right data was fetched.

You will receive:
- RESOLVED_QUERY: the structured interpretation of the user's question,
  including resolved metrics and concept filters
- TOOL_CALLS: the tool calls the planner made, with their parameters
- DATA_SAMPLE: the first rows of the returned data, with column names

Check the following, in order:
1. METRIC PRESENCE: Every metric listed in resolved_metrics appears as a
   column in the returned data or as an explicit fetched metric.
2. FILTER IMPLEMENTATION: Every concept expression in resolved_concepts is
   reflected in the tool call parameters.
3. NO SILENT SUBSTITUTION: No metric was replaced by a proxy without
   acknowledgment.
Do not judge prose, rendering, formatting, or investment quality.
Return JSON only matching the schema. Be strict. When in doubt, fail with a clear reason.
"""


def _walk_expression_metrics(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        metric = value.get("metric")
        if isinstance(metric, str):
            out.add(metric)
        for child in value.values():
            out.update(_walk_expression_metrics(child))
    elif isinstance(value, list):
        for child in value:
            out.update(_walk_expression_metrics(child))
    return out


def _filters_metrics(params: dict[str, Any]) -> set[str]:
    filters = params.get("filters")
    if not isinstance(filters, dict):
        return set()
    return {k for k in filters if isinstance(k, str)}


def _tool_param_metrics(tool_calls: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    requested: set[str] = set()
    filtered: set[str] = set()
    tools: set[str] = set()
    known = set(metric_keys())
    for call in tool_calls or []:
        tool = call.get("tool")
        if isinstance(tool, str):
            tools.add(tool)
        params = call.get("params") if isinstance(call.get("params"), dict) else {}
        for metric in params.get("metrics") or []:
            if isinstance(metric, str):
                requested.add(metric)
        metric = params.get("metric")
        if isinstance(metric, str):
            requested.add(metric)
        order_by = params.get("order_by")
        if isinstance(order_by, dict) and isinstance(order_by.get("metric"), str):
            requested.add(order_by["metric"])
        fmetrics = _filters_metrics(params)
        filtered.update(fmetrics)
        requested.update(m for m in fmetrics if m in known)
    return requested, filtered, tools


def _result_metrics(execution_results: list[dict[str, Any]]) -> set[str]:
    known = set(metric_keys())
    out: set[str] = set()
    for result in execution_results or []:
        for row in (result.get("rows") or [])[:REVIEWER_RESULT_SAMPLE_ROWS]:
            if not isinstance(row, dict):
                continue
            out.update(k for k in row if k in known)
            metric_value = row.get("metric")
            if isinstance(metric_value, str) and metric_value in known:
                out.add(metric_value)
    return out


def _sample_execution(execution_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for result in execution_results or []:
        rows = result.get("rows") or []
        sample.append(
            {
                "tool": result.get("tool"),
                "params": result.get("params"),
                "row_count": result.get("row_count"),
                "success": result.get("success"),
                "error": result.get("error"),
                "columns": sorted(
                    {
                        k
                        for row in rows[:REVIEWER_RESULT_SAMPLE_ROWS]
                        if isinstance(row, dict)
                        for k in row
                    }
                ),
                "sample": rows[:REVIEWER_RESULT_SAMPLE_DISPLAY_ROWS],
            }
        )
    return sample


def _fail(
    reason: str,
    correction: str,
    failure_type: Literal["substitution", "filter_missing", "metric_absent", "shape_mismatch"],
) -> ReviewerVerdict:
    return ReviewerVerdict(
        verdict="fail",
        reason=reason,
        suggested_correction=correction,
        failures=[
            ReviewerFailure(
                type=failure_type,
                description=reason,
                suggested_correction=correction,
            )
        ],
        source="deterministic",
    )


def deterministic_review(
    resolved_query: ResolvedQuery,
    tool_calls: list[dict[str, Any]],
    execution_results: list[dict[str, Any]],
) -> ReviewerVerdict | None:
    requested, filtered, _tools = _tool_param_metrics(tool_calls)
    returned = _result_metrics(execution_results)
    available = requested | returned

    missing_metrics = [
        m.key for m in resolved_query.resolved_metrics if m.key not in available
    ]
    if missing_metrics:
        return _fail(
            "Missing resolved metric(s): " + ", ".join(missing_metrics),
            "Replan using the exact resolved metric(s): " + ", ".join(missing_metrics),
            "metric_absent",
        )

    for concept in resolved_query.resolved_concepts:
        concept_metrics = _walk_expression_metrics(concept.expression)
        missing_concept_metrics = sorted(concept_metrics - available)
        if missing_concept_metrics:
            return _fail(
                f"Concept '{concept.key}' did not fetch required metric(s): "
                + ", ".join(missing_concept_metrics),
                "Fetch the metrics required by the resolved concept expression.",
                "metric_absent",
            )
        unfiltered = sorted(concept_metrics - filtered)
        if unfiltered:
            return _fail(
                f"Concept '{concept.key}' was not applied as data-layer filter(s): "
                + ", ".join(unfiltered),
                "Apply the resolved concept expression as tool filters before answering.",
                "filter_missing",
            )

    return None


def review_execution(
    *,
    resolved_query: ResolvedQuery,
    tool_calls: list[dict[str, Any]],
    execution_results: list[dict[str, Any]],
    model: str | None = None,
    timeout_sec: float = 10.0,
    prompt_cache_key: str | None = None,
) -> ReviewerVerdict:
    """Always run a meaning review; deterministic failures short-circuit the LLM."""
    precheck = deterministic_review(resolved_query, tool_calls, execution_results)
    if precheck is not None:
        _LOG.info("reviewer_verdict=%s", precheck.model_dump_json())
        log_stage("reviewer", "deterministic_fail", **precheck.model_dump())
        return precheck

    import os

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        verdict = ReviewerVerdict(
            verdict="pass",
            reason="Deterministic review passed; OpenAI reviewer unavailable.",
            source="fallback",
        )
        _LOG.info("reviewer_verdict=%s", verdict.model_dump_json())
        return verdict

    payload = {
        "resolved_query": resolved_query.model_dump(),
        "executed_tool_calls": tool_calls,
        "execution_sample": _sample_execution(execution_results),
    }
    model_name = model or REVIEWER_MODEL_DEFAULT
    kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": REVIEWER_INSTRUCTIONS,
        "input": json.dumps(payload, ensure_ascii=False, default=str),
        "text_format": ReviewerVerdict,
        "max_output_tokens": REVIEWER_MAX_OUTPUT_TOKENS,
    }
    if supports_temperature(model_name):
        kwargs["temperature"] = 0
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    rpar = reasoning_param_for_model(model_name, REASONING_REVIEWER)
    if rpar is not None:
        kwargs["reasoning"] = rpar

    try:
        out, _response = parse_strict_response(
            get_openai_client(timeout_sec),
            ReviewerVerdict,
            **kwargs,
        )
        if isinstance(out, ReviewerVerdict):
            out.source = "llm"
            _LOG.info("reviewer_verdict=%s", out.model_dump_json())
            return out
    except Exception as exc:
        _LOG.exception("reviewer_failed: %s", exc)

    verdict = ReviewerVerdict(
        verdict="pass",
        reason="Deterministic review passed; LLM reviewer failed open.",
        source="fallback",
    )
    _LOG.info("reviewer_verdict=%s", verdict.model_dump_json())
    return verdict


def format_reviewer_correction(verdict: ReviewerVerdict) -> str:
    if verdict.failures:
        failures = "; ".join(
            f"[{failure.type}] {failure.description} Fix: {failure.suggested_correction}"
            for failure in verdict.failures
        )
        return f"Reviewer 1 failed the executed plan. {failures}"
    reason = verdict.reason.strip() or "Reviewer found a mismatch."
    correction = verdict.suggested_correction.strip() or "Replan with tools that match ResolvedQuery."
    return f"Reviewer 1 failed the executed plan. Reason: {reason} Correction: {correction}"

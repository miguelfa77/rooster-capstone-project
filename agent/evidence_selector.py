"""Deterministic evidence selector for Phase 3.5 question-driven rendering."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.synthesizer import Paragraph, SynthesizedResponse, VisualHint
from agent.template_registry import TEMPLATE_REGISTRY, TemplateName

_LOG = logging.getLogger("rooster.evidence")


class RenderVisual(BaseModel):
    kind: Literal["visual"] = "visual"
    paragraph_index: int
    template: TemplateName
    renderer_intent: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    forced: bool = False


class RenderParagraph(BaseModel):
    kind: Literal["paragraph"] = "paragraph"
    paragraph_index: int
    text: str
    role: str
    claim_type: str


RenderItem = RenderParagraph | RenderVisual


class RenderList(BaseModel):
    items: list[RenderItem] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)


def _result_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("rows") or []
    return rows if isinstance(rows, list) else []


def _successful_results(agent_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in agent_results or [] if r.get("success") and _result_rows(r)]


def _name_value(row: dict[str, Any]) -> str:
    for key in ("neighborhood_name", "neighborhood", "name", "barrio", "neighborhood_raw"):
        val = row.get(key)
        if val:
            return str(val)
    return ""


def _metric_aliases(metric: str | None) -> set[str]:
    m = (metric or "").strip()
    if not m:
        return set()
    aliases = {m}
    if m in {"yield", "rental_yield", "gross_yield", "gross_rental_yield_pct", "yield_pct"}:
        aliases.update({"yield", "gross_rental_yield_pct", "yield_pct", "gross_yield"})
    if m in {"score", "investment_score"}:
        aliases.update({"score", "investment_score"})
    return aliases


def _metric_exists(rows: list[dict[str, Any]], metric: str | None) -> bool:
    if not metric:
        return True
    aliases = _metric_aliases(metric)
    return any(any(alias in r and r.get(alias) is not None for alias in aliases) for r in rows)


def _resolve_metric(rows: list[dict[str, Any]], metric: str | None) -> str | None:
    for alias in _metric_aliases(metric):
        if any(alias in row and row.get(alias) is not None for row in rows):
            return alias
    return metric


def _valid_coord(row: dict[str, Any]) -> bool:
    try:
        lat = float(row.get("lat"))
        lng = float(row.get("lng"))
    except (TypeError, ValueError):
        return False
    return -90 <= lat <= 90 and -180 <= lng <= 180


def _coord_ratio(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if _valid_coord(r)) / len(rows)


def _time_points(rows: list[dict[str, Any]]) -> int:
    keys = ("bucket_date", "scraped_at", "snapshot_date", "period", "week", "month", "day")
    vals = set()
    for row in rows:
        for key in keys:
            if row.get(key):
                vals.add(str(row[key]))
                break
    return len(vals)


def _fields_for_paragraph(paragraph: Paragraph) -> set[str]:
    fields: set[str] = set()
    for ref in paragraph.cited_data:
        fields.update(ref.fields or [])
    return fields


def _claim_metric(paragraph: Paragraph) -> str | None:
    fields = _fields_for_paragraph(paragraph)
    metric_fields = [
        f
        for f in fields
        if f
        not in {
            "name",
            "neighborhood_name",
            "neighborhood",
            "barrio",
            "lat",
            "lng",
            "url",
        }
    ]
    text = paragraph.text.lower()
    if any(w in text for w in ("yield", "rentabilidad", "rendimiento")):
        for field in metric_fields:
            if "yield" in field or "rental" in field:
                return field
        return "gross_rental_yield_pct"
    if "score" in text or "puntuación" in text:
        return "investment_score"
    return metric_fields[0] if len(metric_fields) == 1 else None


def _rows_for_hint(
    hint: VisualHint,
    paragraph: Paragraph,
    agent_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    successful = _successful_results(agent_results)
    if not successful:
        return [], {}

    subset = (hint.data_subset or "").lower()
    cited_refs = [r.result_ref.lower() for r in paragraph.cited_data]
    candidates = successful
    if "listing" in subset or hint.template in {"listing_cards", "listing_table", "point_map"}:
        candidates = [r for r in successful if r.get("tool") == "query_listings"] or successful
    elif "compare" in subset or hint.template == "comparison_matrix":
        candidates = [r for r in successful if r.get("tool") == "compare_neighborhoods"] or successful
    elif "density" in subset:
        candidates = [r for r in successful if r.get("tool") == "query_neighborhood_density_chart"] or successful
    elif any("neighborhood_profile" in ref for ref in cited_refs) or "top" in subset:
        candidates = [r for r in successful if r.get("tool") == "query_neighborhood_profile"] or successful

    result = candidates[0]
    rows = list(_result_rows(result))
    if "top" in subset:
        digits = "".join(ch for ch in subset if ch.isdigit())
        if digits:
            rows = rows[: int(digits)]
    neighborhoods = hint.emphasis.neighborhoods or []
    if neighborhoods:
        wanted = {n.strip().lower() for n in neighborhoods if n.strip()}
        filtered = [r for r in rows if _name_value(r).strip().lower() in wanted]
        if filtered:
            rows = filtered
    return rows, result


def _visual_meta(hint: VisualHint, paragraph: Paragraph, result: dict[str, Any]) -> dict[str, Any]:
    metric = hint.emphasis.metric or _claim_metric(paragraph)
    meta: dict[str, Any] = {
        "tool": result.get("tool"),
        "metric": metric,
        "metric_label": (metric or "").replace("_", " "),
        "template": hint.template,
    }
    if hint.template == "scatter_2d":
        meta["chart_type"] = "scatter"
    return meta


def _accepted_visual(
    paragraph_index: int,
    paragraph: Paragraph,
    hint: VisualHint,
    rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> RenderVisual | None:
    spec = TEMPLATE_REGISTRY[hint.template]
    if spec.renderer_intent is None:
        return None
    meta = _visual_meta(hint, paragraph, result)
    metric = _resolve_metric(rows, str(meta.get("metric") or ""))
    if metric:
        meta["metric"] = metric
        meta["metric_label"] = metric.replace("_", " ")
    rows_for_render = rows
    if hint.template == "ranking_bars" and metric:
        rows_for_render = [dict(row, value=row.get(metric)) for row in rows]
    return RenderVisual(
        paragraph_index=paragraph_index,
        template=hint.template,
        renderer_intent=spec.renderer_intent,
        rows=rows_for_render,
        meta=meta,
        rationale=hint.rationale,
        forced=hint.forced,
    )


def _validate_hint(
    paragraph: Paragraph,
    hint: VisualHint,
    rows: list[dict[str, Any]],
) -> str | None:
    if hint.template == "none":
        return "none_template"
    spec = TEMPLATE_REGISTRY[hint.template]
    n = len(rows)
    if n < spec.min_items:
        return f"cardinality_too_low:{n}<{spec.min_items}"
    if spec.max_items is not None and n > spec.max_items:
        return f"cardinality_too_high:{n}>{spec.max_items}"
    metric = hint.emphasis.metric or _claim_metric(paragraph)
    claim_metric = _claim_metric(paragraph)
    if claim_metric and metric and not (_metric_aliases(claim_metric) & _metric_aliases(metric)):
        return f"metric_mismatch:{metric}!={claim_metric}"
    if spec.requires_metric and not _metric_exists(rows, metric):
        return f"metric_missing:{metric}"
    if spec.requires_coordinates and _coord_ratio(rows) < 0.5:
        return "coordinates_insufficient"
    if spec.required_time_points and _time_points(rows) < spec.required_time_points:
        return "time_points_insufficient"
    if hint.template == "comparison_matrix" and n == 2 and paragraph.claim_type == "ranking":
        return "wrong_template_for_ranking"
    if hint.template == "ranking_bars" and n < 4:
        return "ranking_needs_4_items"
    return None


def _subset_key(visual: RenderVisual) -> tuple[str, tuple[str, ...]]:
    names = tuple(sorted(n for n in (_name_value(r) for r in visual.rows) if n))
    metric = str(visual.meta.get("metric") or "")
    return metric, names


def _dedupe_visuals(
    visuals: list[RenderVisual],
    decisions: list[dict[str, Any]],
    max_visuals: int,
) -> list[RenderVisual]:
    kept: list[RenderVisual] = []
    for visual in visuals:
        if visual.forced:
            kept.append(visual)
            continue
        duplicate_idx = None
        for i, existing in enumerate(kept):
            if existing.forced:
                continue
            if _subset_key(existing) == _subset_key(visual) and _subset_key(visual)[0]:
                duplicate_idx = i
                break
        if duplicate_idx is None:
            kept.append(visual)
            continue
        existing = kept[duplicate_idx]
        if visual.template == "choropleth_focus" and existing.template != "choropleth_focus":
            decisions.append(
                {
                    "action": "dedupe_replace",
                    "dropped": existing.template,
                    "kept": visual.template,
                    "reason": "spatial_information_dominates",
                }
            )
            kept[duplicate_idx] = visual
        else:
            decisions.append(
                {
                    "action": "dedupe_drop",
                    "template": visual.template,
                    "reason": "same_metric_same_subset",
                }
            )
    if len(kept) > max_visuals:
        forced = [v for v in kept if v.forced]
        regular = [v for v in kept if not v.forced]
        allowed_regular = max(0, max_visuals - len(forced))
        for dropped in regular[allowed_regular:]:
            decisions.append(
                {"action": "cap_drop", "template": dropped.template, "reason": "visual_cap"}
            )
        kept = forced + regular[:allowed_regular]
    return kept


def _explicit_format_template(fmt: str) -> TemplateName | None:
    return {
        "table": "listing_table",
        "map": "point_map",
        "chart": "ranking_bars",
    }.get((fmt or "").strip().lower())


def _best_result_for_template(
    template: TemplateName,
    agent_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    successful = _successful_results(agent_results)
    if not successful:
        return [], {}
    if template in {"listing_table", "listing_cards", "point_map"}:
        result = next((r for r in successful if r.get("tool") == "query_listings"), successful[0])
    elif template == "comparison_matrix":
        result = next((r for r in successful if r.get("tool") == "compare_neighborhoods"), successful[0])
    else:
        result = next((r for r in successful if r.get("tool") == "query_neighborhood_profile"), successful[0])
    return _result_rows(result), result


def select_evidence(
    synthesized: SynthesizedResponse,
    agent_results: list[dict[str, Any]],
    intent: dict[str, Any] | None,
) -> RenderList:
    decisions: list[dict[str, Any]] = []
    intent = intent or {}
    depth = str(intent.get("depth_preference") or "analytical")
    explicit_format = str(intent.get("explicit_format_request") or "none")
    max_visuals = 2 if depth == "quick" else 4

    proposed: list[RenderVisual] = []

    bypass_template = _explicit_format_template(explicit_format)
    if bypass_template:
        rows, result = _best_result_for_template(bypass_template, agent_results)
        spec = TEMPLATE_REGISTRY[bypass_template]
        if rows and spec.renderer_intent:
            fake_hint = VisualHint(
                template=bypass_template,
                rationale="Explicit user format request",
                data_subset="explicit_format",
                forced=True,
            )
            proposed.append(
                RenderVisual(
                    paragraph_index=0,
                    template=bypass_template,
                    renderer_intent=spec.renderer_intent,
                    rows=rows,
                    meta=_visual_meta(fake_hint, synthesized.paragraphs[0], result)
                    if synthesized.paragraphs
                    else {"template": bypass_template},
                    rationale="explicit_format_request",
                    forced=True,
                )
            )
            decisions.append(
                {"action": "bypass_keep", "template": bypass_template, "reason": explicit_format}
            )
    elif depth == "quick" and all(
        p.claim_type in {"lookup", "caveat", "no_data"} for p in synthesized.paragraphs
    ):
        decisions.append({"action": "bypass_none", "reason": "quick_lookup_or_caveat"})
    else:
        listing_results = [r for r in _successful_results(agent_results) if r.get("tool") == "query_listings"]
        analytical_claims = {
            "ranking",
            "spatial_fact",
            "trend",
            "decomposition",
            "comparison",
            "recommendation",
        }
        if (
            len(listing_results) == 1
            and len(_result_rows(listing_results[0])) > 5
            and not any(p.claim_type in analytical_claims for p in synthesized.paragraphs)
        ):
            proposed.append(
                RenderVisual(
                    paragraph_index=0,
                    template="listing_table",
                    renderer_intent="search",
                    rows=_result_rows(listing_results[0]),
                    meta={"tool": "query_listings", "template": "listing_table"},
                    rationale="raw_listing_enumeration",
                )
            )
            decisions.append(
                {"action": "bypass_keep", "template": "listing_table", "reason": "raw_listing_enumeration"}
            )

    if not proposed and explicit_format == "none":
        for i, paragraph in enumerate(synthesized.paragraphs):
            accepted_for_paragraph = 0
            for hint in paragraph.visual_hints:
                rows, result = _rows_for_hint(hint, paragraph, agent_results)
                if hint.forced:
                    visual = _accepted_visual(i, paragraph, hint, rows, result)
                    if visual:
                        proposed.append(visual)
                        decisions.append(
                            {
                                "action": "keep_forced",
                                "paragraph": i,
                                "template": hint.template,
                                "forced": True,
                            }
                        )
                    continue
                reason = _validate_hint(paragraph, hint, rows)
                if reason:
                    decisions.append(
                        {
                            "action": "drop",
                            "paragraph": i,
                            "template": hint.template,
                            "reason": reason,
                        }
                    )
                    continue
                visual = _accepted_visual(i, paragraph, hint, rows, result)
                if not visual:
                    continue
                proposed.append(visual)
                accepted_for_paragraph += 1
                decisions.append(
                    {"action": "keep", "paragraph": i, "template": hint.template}
                )
                if not (paragraph.role == "lead" and hint.template == "kpi_strip"):
                    break
                if accepted_for_paragraph >= 2:
                    break

    visuals = _dedupe_visuals(proposed, decisions, max_visuals)
    visual_map: dict[int, list[RenderVisual]] = {}
    for visual in visuals:
        visual_map.setdefault(visual.paragraph_index, []).append(visual)

    items: list[RenderItem] = []
    for i, paragraph in enumerate(synthesized.paragraphs):
        items.append(
            RenderParagraph(
                paragraph_index=i,
                text=paragraph.text,
                role=paragraph.role,
                claim_type=paragraph.claim_type,
            )
        )
        items.extend(visual_map.get(i, []))

    out = RenderList(items=items, follow_ups=synthesized.follow_ups[:3], decisions=decisions)
    _LOG.info("synthesized_response=%s", synthesized.model_dump_json())
    _LOG.info("evidence_selector_decisions=%s", json.dumps(decisions, ensure_ascii=True, default=str))
    final_summary = [
        {
            "kind": item.kind,
            "paragraph_index": item.paragraph_index,
            "template": getattr(item, "template", None),
            "renderer_intent": getattr(item, "renderer_intent", None),
            "forced": getattr(item, "forced", None),
        }
        for item in out.items
    ]
    _LOG.info(
        "evidence_selector_final=%s",
        json.dumps(final_summary, ensure_ascii=True, default=str),
    )
    return out

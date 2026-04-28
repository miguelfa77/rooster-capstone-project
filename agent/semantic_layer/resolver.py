"""Pure-Python semantic resolver for Rooster v2 Phase A."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel

from agent.config import REASONING_REVIEWER, REVIEWER_MODEL_DEFAULT
from agent.responses_api import get_openai_client, reasoning_param_for_model, supports_temperature
from agent.semantic_layer.loader import load_registry
from agent.semantic_layer.models import (
    ConceptEntry,
    Criticality,
    HeuristicEntry,
    LiteralValue,
    MatchSpan,
    MetricEntry,
    ResolvedConcept,
    ResolvedHeuristic,
    ResolvedMetric,
    ResolvedQuery,
    SemanticRegistry,
)
from agent.stage_logging import log_stage


class ClarificationNeed(BaseModel):
    needs_clarification: bool
    reason: str = ""


_CLARIFICATION_CLASSIFIER_INSTRUCTIONS = """You are helping a real estate analytics assistant decide whether a term
needs clarification before querying a database.

A term needs clarification only if it is domain-specific in a way that
requires a precise technical definition to query correctly — for example,
a custom business metric with a specific formula, a proprietary score,
or a company-specific concept that doesn't have a standard meaning.

A term does NOT need clarification if it is:
- General Spanish vocabulary (comparar, mejor, buscar, quiero)
- Standard real estate concepts (inversión, rentabilidad, comprar,
  alquilar, precio, vivienda, propiedad, mercado, zona, barrio)
- Standard investment concepts (retorno, beneficio, plusvalía,
  revalorización, capital)
- Anything a knowledgeable real estate professional would immediately
  understand without a formula

Respond in JSON only:
{
  "needs_clarification": true | false,
  "reason": "one sentence"
}
"""


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    matched: str
    registry: str
    key: str
    criticality: Criticality
    entry: Any

    @property
    def length(self) -> int:
        return self.end - self.start


def fold_text(text: str) -> str:
    """Lowercase and strip accents, preserving string length for span checks."""
    decomposed = unicodedata.normalize("NFD", text or "")
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.lower()


def _synonyms(entry: MetricEntry | ConceptEntry | HeuristicEntry) -> list[str]:
    if isinstance(entry, MetricEntry):
        return [entry.key, entry.display, *entry.spanish, *entry.english]
    if isinstance(entry, ConceptEntry):
        return [entry.key, entry.display, *entry.terms, *entry.variants]
    return [entry.key, entry.display, *entry.terms]


def _find_term_spans(folded_text: str, term: str) -> Iterable[tuple[int, int]]:
    t = fold_text(term).strip()
    if not t:
        return []
    pattern = re.compile(rf"(?<!\w){re.escape(t)}(?!\w)")
    return ((m.start(), m.end()) for m in pattern.finditer(folded_text))


def _candidates_for_registry(
    folded_text: str,
    registry_name: str,
    entries: Iterable[MetricEntry | ConceptEntry | HeuristicEntry],
) -> list[_Candidate]:
    out: list[_Candidate] = []
    for entry in entries:
        seen_terms: set[str] = set()
        for term in _synonyms(entry):
            folded_term = fold_text(term).strip()
            if not folded_term or folded_term in seen_terms:
                continue
            seen_terms.add(folded_term)
            for start, end in _find_term_spans(folded_text, folded_term):
                out.append(
                    _Candidate(
                        start=start,
                        end=end,
                        matched=folded_text[start:end],
                        registry=registry_name,
                        key=entry.key,
                        criticality=entry.criticality,
                        entry=entry,
                    )
                )
    return out


def _overlaps(candidate: _Candidate, accepted: list[_Candidate]) -> bool:
    return any(candidate.start < a.end and a.start < candidate.end for a in accepted)


def _select_longest_non_overlapping(candidates: list[_Candidate]) -> list[_Candidate]:
    ordered = sorted(candidates, key=lambda c: (-c.length, c.start, c.registry))
    accepted: list[_Candidate] = []
    for candidate in ordered:
        if not _overlaps(candidate, accepted):
            accepted.append(candidate)
    return sorted(accepted, key=lambda c: c.start)


def _literal_values(text: str) -> list[LiteralValue]:
    out: list[LiteralValue] = []
    for match in re.finditer(r"(?P<raw>\d{2,3}(?:[.\s]\d{3})+|\d+)\s*(?:€|eur|euros?)", text, re.I):
        raw = match.group("raw")
        value = int(re.sub(r"[.\s]", "", raw))
        out.append(LiteralValue(kind="currency", value=value, raw=match.group(0), unit="EUR"))
    for match in re.finditer(r"(?P<n>\d+)\s*(?:habitaciones|dormitorios|rooms|bedrooms|hab\.?)", text, re.I):
        out.append(LiteralValue(kind="rooms", value=int(match.group("n")), raw=match.group(0)))
    return out


def _covered(index_start: int, index_end: int, matches: list[_Candidate]) -> bool:
    return any(index_start >= m.start and index_end <= m.end for m in matches)


def _clarification_resolutions(session_memory: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(session_memory, dict):
        return {}
    nested = session_memory.get("session_memory_v2")
    if isinstance(nested, dict):
        raw = nested.get("clarification_resolutions")
    else:
        raw = session_memory.get("clarification_resolutions")
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if k and v}


def _is_already_clarified(key: str, term: str, resolutions: dict[str, str]) -> bool:
    folded_key = fold_text(key)
    folded_term = fold_text(term)
    return any(fold_text(k) in {folded_key, folded_term} for k in resolutions)


def _classify_clarification_need(term: str, user_message: str) -> ClarificationNeed:
    import json
    import os

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        verdict = ClarificationNeed(
            needs_clarification=False,
            reason="OpenAI unavailable; failing open to avoid blocking common vocabulary.",
        )
        log_stage(
            "semantic_resolver",
            "clarification_classifier_fallback",
            term=term,
            needs_clarification=verdict.needs_clarification,
            reason=verdict.reason,
        )
        return verdict
    model_name = REVIEWER_MODEL_DEFAULT
    payload = {
        "term": term,
        "user_message": user_message,
    }
    kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": _CLARIFICATION_CLASSIFIER_INSTRUCTIONS,
        "input": (
            f'Term to classify: "{term}"\n'
            f'Context (the user\'s full message): "{user_message}"\n\n'
            f"Payload: {json.dumps(payload, ensure_ascii=False)}"
        ),
        "text_format": ClarificationNeed,
        "max_output_tokens": 200,
    }
    if supports_temperature(model_name):
        kwargs["temperature"] = 0
    rpar = reasoning_param_for_model(model_name, REASONING_REVIEWER)
    if rpar is not None:
        kwargs["reasoning"] = rpar
    try:
        parsed = get_openai_client(8.0).responses.parse(**kwargs)
        out = parsed.output_parsed
        if isinstance(out, ClarificationNeed):
            log_stage(
                "semantic_resolver",
                "clarification_classifier",
                term=term,
                needs_clarification=out.needs_clarification,
                reason=out.reason,
            )
            return out
    except Exception as exc:
        log_stage(
            "semantic_resolver",
            "clarification_classifier_failed",
            term=term,
            error=str(exc),
        )
    return ClarificationNeed(
        needs_clarification=False,
        reason="Classifier failed; failing open to avoid blocking common vocabulary.",
    )


def _unresolved_terms(
    folded_text: str,
    user_message: str,
    registry: SemanticRegistry,
    matches: list[_Candidate],
    literals: list[LiteralValue],
    clarification_resolutions: dict[str, str],
) -> tuple[list[str], list[str]]:
    essential: list[str] = []
    flavour: list[str] = []
    literal_kinds = {literal.kind for literal in literals}
    for entry in registry.essential_terms:
        if entry.key == "presupuesto" and "currency" in literal_kinds:
            continue
        if entry.key == "habitaciones" and "rooms" in literal_kinds:
            continue
        for term in entry.terms:
            if _is_already_clarified(entry.key, term, clarification_resolutions):
                continue
            for start, end in _find_term_spans(folded_text, term):
                if _covered(start, end, matches):
                    continue
                target = essential if entry.criticality == "essential" else flavour
                if entry.criticality == "essential":
                    verdict = _classify_clarification_need(term, user_message)
                    if not verdict.needs_clarification:
                        if term not in flavour:
                            flavour.append(term)
                        continue
                if term not in target:
                    target.append(term)
    for match in re.finditer(r"\b[A-ZÁÉÍÓÚÑ]{2,}\b", user_message):
        term = match.group(0)
        start, end = match.span()
        if _covered(start, end, matches) or _is_already_clarified(term, term, clarification_resolutions):
            continue
        verdict = _classify_clarification_need(term, user_message)
        target = essential if verdict.needs_clarification else flavour
        if term not in target:
            target.append(term)
    return essential, flavour


def _to_match_span(candidate: _Candidate) -> MatchSpan:
    return MatchSpan(
        start=candidate.start,
        end=candidate.end,
        text=candidate.matched,
        registry=candidate.registry,  # type: ignore[arg-type]
        key=candidate.key,
        criticality=candidate.criticality,
    )


def resolve_query(
    user_message: str,
    *,
    registry: SemanticRegistry | None = None,
    session_memory: dict[str, Any] | None = None,
) -> ResolvedQuery:
    """Resolve user-facing terms to canonical data-layer meaning without an LLM."""
    reg = registry or load_registry()
    folded = fold_text(user_message)
    candidates: list[_Candidate] = []
    candidates.extend(_candidates_for_registry(folded, "metrics", reg.metrics))
    candidates.extend(_candidates_for_registry(folded, "concepts", reg.concepts))
    candidates.extend(_candidates_for_registry(folded, "heuristics", reg.heuristics))
    matches = _select_longest_non_overlapping(candidates)

    metrics: list[ResolvedMetric] = []
    concepts: list[ResolvedConcept] = []
    heuristics: list[ResolvedHeuristic] = []
    seen_metrics: set[str] = set()
    seen_concepts: set[str] = set()
    seen_heuristics: set[str] = set()

    for match in matches:
        entry = match.entry
        if isinstance(entry, MetricEntry) and entry.key not in seen_metrics:
            seen_metrics.add(entry.key)
            metrics.append(
                ResolvedMetric(
                    key=entry.key,
                    column=entry.column,
                    display=entry.display,
                    unit=entry.unit,
                    formula=entry.formula,
                    components=entry.components,
                    caveat=entry.caveat,
                    matched_text=match.matched,
                )
            )
        elif isinstance(entry, ConceptEntry) and entry.key not in seen_concepts:
            seen_concepts.add(entry.key)
            concepts.append(
                ResolvedConcept(
                    key=entry.key,
                    display=entry.display,
                    expression=entry.expression,
                    variants=entry.variants,
                    matched_text=match.matched,
                )
            )
        elif isinstance(entry, HeuristicEntry) and entry.key not in seen_heuristics:
            seen_heuristics.add(entry.key)
            heuristics.append(
                ResolvedHeuristic(
                    key=entry.key,
                    display=entry.display,
                    barrio_ids=entry.barrio_ids,
                    barrio_names=entry.barrio_names,
                    geometry=entry.geometry,
                    matched_text=match.matched,
                )
            )

    literals = _literal_values(user_message)
    unresolved_essential, unresolved_flavour = _unresolved_terms(
        folded,
        user_message,
        reg,
        matches,
        literals,
        _clarification_resolutions(session_memory),
    )
    return ResolvedQuery(
        original_text=user_message,
        resolved_metrics=metrics,
        resolved_concepts=concepts,
        resolved_heuristics=heuristics,
        literals=literals,
        unresolved_essential_terms=unresolved_essential,
        unresolved_flavour_terms=unresolved_flavour,
        match_spans=[_to_match_span(m) for m in matches],
    )


def clarification_message(resolved: ResolvedQuery) -> str | None:
    if not resolved.unresolved_essential_terms:
        return None
    terms = ", ".join(resolved.unresolved_essential_terms)
    registry = load_registry()
    candidates = [
        metric.display
        for metric in registry.metrics
        if metric.display
        and any(
            token in fold_text(metric.display)
            for term in resolved.unresolved_essential_terms
            for token in fold_text(term).split()
            if len(token) > 3
        )
    ][:3]
    if candidates:
        options = ", ".join(candidates)
        return (
            f"Antes de consultar los datos, necesito precisar '{terms}'. "
            f"¿Te refieres a {options}, u otra definición?"
        )
    return (
        "Necesito que definas un poco mejor estos términos antes de consultar los datos: "
        f"{terms}. ¿A qué te refieres exactamente?"
    )

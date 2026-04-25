"""
Pre-planning stage: normalize user intent to structured ``ResolvedIntent`` (Structured Outputs).
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from agent.responses_api import (
    get_openai_client,
    reasoning_param_for_model,
    supports_temperature,
)
from agent.spatial_resolver import expand_qualitative_tags

INTENT_MODEL = os.getenv("ROOSTER_INTENT_MODEL", "gpt-5-mini")


class PriceBandModel(BaseModel):
    min: int | None = None
    max: int | None = None
    currency: str = "EUR"


class ResolvedIntent(BaseModel):
    operation: str = Field(
        description="venta|alquiler|either|unspecified",
    )
    price_band: PriceBandModel = Field(default_factory=PriceBandModel)
    target_neighborhoods: list[str] = Field(default_factory=list)
    qualitative_filters: list[str] = Field(default_factory=list)
    comparison_mode: str = "explore"
    recommendation_requested: bool = False
    depth_preference: str = "analytical"
    explicit_format_request: str = "none"
    ambiguity_flags: list[str] = Field(default_factory=list)
    expanded_neighborhoods_from_qualitative: list[str] = Field(
        default_factory=list,
        description="Names from spatial lexicon; may overlap target_neighborhoods",
    )


INTENT_RESOLVER_INSTRUCTIONS = """You are Rooster's intent resolver for Valencia (Spain) real estate.

Output ONLY valid JSON matching the schema. All user-facing product language is Spanish; your parsing should understand Spanish property questions.

operation:
- "venta" = buy / purchase / investment in owning
- "alquiler" = rent / rental as end user
- "either" = user is okay with both
- "unspecified" = not clear

qualitative_filters: use controlled tags when the user gives vague area wishes, e.g. good_area, near_center, family_friendly, near_beach, university_area, quiet, nightlife. Also map: centro, cerca_playa, universitario, familiar, nocturno if they fit.

target_neighborhoods: specific barrio names if the user names them (best effort spelling).

If the user did not state price, leave price_band min/max null.

comparison_mode: single | compare_named | find_best | explore

explicit_format_request: none | table | map | chart | memo if they clearly asked for a format.

Set ambiguity_flags when relevant, e.g. operation_unclear, no_neighborhood_specified.
"""


def resolve_intent(
    user_message: str,
    conversation_memory: dict[str, Any],
    live_schema: str,
    *,
    timeout_sec: float = 20.0,
    prompt_cache_key: str | None = None,
) -> ResolvedIntent:
    """
    Structured intent via Responses API ``parse``. On failure, returns a safe default.
    """
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        return ResolvedIntent()

    client = get_openai_client(timeout_sec)
    mem = json.dumps(conversation_memory, ensure_ascii=True, default=str)[:8000]
    user_block = f"""User message:
{user_message}

Session memory (JSON, truncated if long):
{mem}

LIVE / schema hints (excerpt):
{live_schema[:12000]}"""

    try:
        parse_kw: dict[str, Any] = {
            "model": INTENT_MODEL,
            "instructions": INTENT_RESOLVER_INSTRUCTIONS,
            "input": user_block,
            "text_format": ResolvedIntent,
            "max_output_tokens": 800,
        }
        if supports_temperature(INTENT_MODEL):
            parse_kw["temperature"] = 0
        if prompt_cache_key:
            parse_kw["prompt_cache_key"] = prompt_cache_key
        rpar = reasoning_param_for_model(INTENT_MODEL, "low")
        if rpar is not None:
            parse_kw["reasoning"] = rpar
        parsed = client.responses.parse(**parse_kw)
        out = parsed.output_parsed
        if isinstance(out, ResolvedIntent):
            base = out
        else:
            base = ResolvedIntent()
    except Exception:
        base = ResolvedIntent()

    # Expand curated qualitative tags to neighborhood names
    tags = [t for t in base.qualitative_filters if t]
    expanded = expand_qualitative_tags(tags) if tags else []
    if expanded:
        merged = list(base.target_neighborhoods)
        for n in expanded:
            if n not in merged:
                merged.append(n)
        base = base.model_copy(
            update={
                "target_neighborhoods": merged,
                "expanded_neighborhoods_from_qualitative": expanded,
            }
        )
    return base

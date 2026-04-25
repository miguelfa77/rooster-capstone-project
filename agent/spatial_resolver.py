"""
Curated mapping from qualitative descriptors to neighborhood name lists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_LEX: dict[str, list[str]] | None = None


def _load_lexicon() -> dict[str, list[str]]:
    global _LEX
    if _LEX is not None:
        return _LEX
    path = Path(__file__).resolve().parent / "spatial_lexicon.yml"
    if not path.exists():
        _LEX = {}
        return _LEX
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        _LEX = {}
    else:
        out: dict[str, list[str]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[str(k)] = [str(x) for x in v if x]
        _LEX = out
    return _LEX


def resolve_label(label: str) -> list[dict[str, Any]]:
    """
    Return [{\"name\": str, \"confidence\": float}, ...] for a controlled-vocabulary key.
    """
    key = (label or "").strip().lower().replace(" ", "_")
    lex = _load_lexicon()
    names = lex.get(key) or []
    return [{"name": n, "confidence": 0.9} for n in names]


def match_reference_phrase(phrase: str) -> list[dict[str, Any]]:
    """Match free-text to lexicon keys; return resolve_label-style rows."""
    t = (phrase or "").lower().strip()
    if not t:
        return []
    lex = _load_lexicon()
    for key, _names in lex.items():
        if key.replace("_", " ") in t or key in t or any(
            w in t for w in key.split("_")
        ):
            return resolve_label(key)
    # direct keyword hints
    if "playa" in t or "playas" in t:
        return resolve_label("cerca_playa")
    if "centro" in t and "cerca" not in t[:5]:
        return resolve_label("centro")
    if "universidad" in t or "universit" in t:
        return resolve_label("universitario")
    return []


def expand_qualitative_tags(tags: list[str]) -> list[str]:
    """Flatten all neighborhood names for the given tags (de-duplicated, order preserved)."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        for r in resolve_label(t):
            n = r.get("name")
            if isinstance(n, str) and n and n not in seen:
                seen.add(n)
                out.append(n)
    return out

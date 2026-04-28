"""OpenAI Responses API helpers for Rooster v2."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def get_openai_client(timeout: float) -> OpenAI:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(timeout=max(timeout + 5.0, 15.0))


def reasoning_param_for_model(model: str, effort: str) -> dict[str, str] | None:
    m = (model or "").lower()
    if "gpt-5" in m or m.startswith("o"):
        return {"effort": effort}
    return None


def supports_temperature(model: str) -> bool:
    """Older non-reasoning chat models accept temperature; GPT-5/o reasoning paths may not."""
    m = (model or "").lower()
    return not ("gpt-5" in m or m.startswith("o"))


def extract_response_text(resp: Any) -> str:
    """Best-effort plain text from a Response object."""
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t
    parts: list[str] = []
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", None) or []:
                ct = getattr(c, "type", None)
                if ct in ("output_text", "text"):
                    tx = getattr(c, "text", None) or ""
                    if tx:
                        parts.append(tx)
    return "".join(parts).strip()

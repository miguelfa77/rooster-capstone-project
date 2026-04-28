"""OpenAI Responses API helpers for Rooster v2."""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

try:
    from openai.lib._pydantic import to_strict_json_schema
except Exception:  # pragma: no cover - SDK internals may move
    to_strict_json_schema = None  # type: ignore[assignment]

ModelT = TypeVar("ModelT", bound=BaseModel)


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


def strict_json_schema_for_model(model: type[BaseModel]) -> dict[str, Any]:
    """Build an OpenAI strict structured-output schema for a Pydantic model."""
    schema = (
        to_strict_json_schema(model)
        if to_strict_json_schema is not None
        else model.model_json_schema()
    )

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                node["additionalProperties"] = False
                node["required"] = list(props.keys())
            elif node.get("type") == "object":
                node["additionalProperties"] = False
                node.setdefault("required", [])
            elif "additionalProperties" in node and node["additionalProperties"] is not False:
                node["additionalProperties"] = False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema


def parse_strict_response(
    client: OpenAI,
    model_type: type[ModelT],
    **kwargs: Any,
) -> tuple[ModelT | None, Any]:
    """Call Responses API with an explicit strict JSON schema and validate locally."""
    kwargs = dict(kwargs)
    kwargs.pop("text_format", None)
    kwargs["text"] = {
        "format": {
            "type": "json_schema",
            "name": model_type.__name__,
            "schema": strict_json_schema_for_model(model_type),
            "strict": True,
        }
    }
    response = client.responses.create(**kwargs)
    text = extract_response_text(response)
    if not text:
        return None, response
    try:
        payload = json.loads(text)
    except Exception:
        return None, response
    return model_type.model_validate(payload), response

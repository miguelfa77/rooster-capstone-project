"""
OpenAI Responses API helpers for Rooster v2 (tool format conversion, streaming, classifiers).

Chat Completions use ``{"type":"function","function":{...}}``; Responses API expects
``{"type":"function","name", "description", "parameters"}``.
"""

from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace
from typing import Any, Iterator

from openai import OpenAI

CLASSIFIER_MODEL_DEFAULT = os.getenv("ROOSTER_CLASSIFIER_MODEL", "gpt-4o-mini")


def get_openai_client(timeout: float) -> OpenAI:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(timeout=max(timeout + 5.0, 15.0))


def chat_tools_to_responses_tools(chat_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tools from ``get_rooster_openai_tools()`` to Responses API ``tools`` param."""
    out: list[dict[str, Any]] = []
    for t in chat_tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            name = fn.get("name")
            if not name:
                continue
            item: dict[str, Any] = {
                "type": "function",
                "name": str(name),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
            if fn.get("description"):
                item["description"] = str(fn["description"])
            out.append(item)
    return out


def _classifier_reasoning_param(model: str) -> dict[str, str] | None:
    m = (model or "").lower()
    if "gpt-5" in m or m.startswith("o"):
        return {"effort": "minimal"}
    return None


def responses_classify_conversational(
    user_text: str,
    *,
    system_prompt: str,
    model: str | None = None,
    timeout_sec: float = 3.0,
) -> str:
    """One completion, no tools. Returns model output (strip + lower for token check)."""
    client = get_openai_client(timeout_sec)
    m = model or CLASSIFIER_MODEL_DEFAULT
    kwargs: dict[str, Any] = {
        "model": m,
        "instructions": system_prompt,
        "input": user_text,
        "max_output_tokens": 8,
        "temperature": 0,
    }
    rpar = _classifier_reasoning_param(m)
    if rpar is not None:
        kwargs["reasoning"] = rpar
    resp = client.responses.create(**kwargs)
    return extract_response_text(resp).strip().lower()


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


def output_items_to_tool_calls_compat(resp: Any) -> list[Any]:
    """
    Build objects compatible with ``openai_tool_calls_to_plan_calls`` (``tc.function`` / ``id``).
    """
    out: list[Any] = []
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            name = getattr(item, "name", None) or ""
            args = getattr(item, "arguments", None) or "{}"
            call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or str(
                uuid.uuid4()
            )
            fn = SimpleNamespace(name=str(name), arguments=str(args))
            out.append(SimpleNamespace(id=str(call_id), function=fn, type="function"))
    return out


def create_response_with_tools(
    client: OpenAI,
    *,
    model: str,
    instructions: str,
    user_input: str,
    tools: list[dict[str, Any]],
    previous_response_id: str | None = None,
    prompt_cache_key: str | None = None,
    max_output_tokens: int = 900,
    parallel_tool_calls: bool = True,
    reasoning_effort: str | None = "medium",
) -> Any:
    """Planner first turn: instructions + user string + function tools."""
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "tools": tools,
        "max_output_tokens": max_output_tokens,
        "tool_choice": "auto",
        "parallel_tool_calls": parallel_tool_calls,
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    m = (model or "").lower()
    if reasoning_effort and ("gpt-5" in m or m.startswith("o")):
        kwargs["reasoning"] = {"effort": reasoning_effort}
    return client.responses.create(**kwargs)


def tool_json_payloads_to_responses_input(
    pairs: list[tuple[str, str]],
    followup_user_text: str,
) -> list[dict[str, Any]]:
    """
    ``pairs``: (call_id, json string) per tool result, in order.
    Appends a user message for the final synthesis instruction.
    """
    items: list[dict[str, Any]] = []
    for call_id, json_str in pairs:
        items.append(
            {
                "type": "function_call_output",
                "call_id": str(call_id),
                "output": json_str,
            }
        )
    items.append(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": followup_user_text}],
        }
    )
    return items


def create_response_synthesis(
    client: OpenAI,
    *,
    model: str,
    instructions: str,
    input_items: list[dict[str, Any]],
    previous_response_id: str,
    max_output_tokens: int = 500,
    stream: bool = False,
    reasoning_effort: str | None = "low",
) -> Any:
    """Second turn after tool execution."""
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "previous_response_id": previous_response_id,
        "max_output_tokens": max_output_tokens,
        "temperature": 0.3,
    }
    m = (model or "").lower()
    if reasoning_effort and ("gpt-5" in m or m.startswith("o")):
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if stream:
        kwargs["stream"] = True
    return client.responses.create(**kwargs)


def iter_response_stream_text(stream: Any) -> Iterator[str]:
    """Yield assistant text deltas from ``responses.create(..., stream=True)``."""
    for event in stream:
        et = getattr(event, "type", None)
        if et == "response.output_text.delta" or et == "response.text.delta":
            d = getattr(event, "delta", None)
            if d:
                yield str(d)

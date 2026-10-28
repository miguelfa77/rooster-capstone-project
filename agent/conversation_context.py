"""Turn chat history into text context for the LLM. No UI dependencies."""

from __future__ import annotations

from agent import ui_es as UI

_SUMMARY_TRIGGERS = (
    "summarise",
    "summarize",
    "summary",
    "wrap up",
    "wrap-up",
    "what have we found",
    "what we've found",
    "recap",
    "investment memo",
    "brief memo",
    "give me a summary",
    "resumen",
    "resúmen",
    "resume",
    "sintetiza",
    "sintetizar",
    "memo de inversión",
    "memo inversion",
    "conclusión",
    "conclusiones",
    "qué hemos visto",
    "que hemos visto",
    "cerrar",
    "cierra el resumen",
)


def is_summary_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) > 220:
        return False
    return any(x in t for x in _SUMMARY_TRIGGERS)


def format_conversation_for_memo(messages: list) -> str:
    lines: list[str] = []
    for m in messages:
        if m.get("role") == "user":
            lines.append(f"User: {m.get('content', '')}")
        elif m.get("role") == "assistant":
            parts: list[str] = []
            if m.get("interpretation"):
                parts.append(str(m["interpretation"]))
            if m.get("summary"):
                parts.append(str(m["summary"]))
            if m.get("empty_narrative"):
                parts.append(str(m["empty_narrative"]))
            if m.get("error"):
                parts.append(f"(error: {m['error']})")
            body = " ".join(parts) if parts else "(no text reply)"
            lines.append(f"Rooster: {body}")
    return "\n".join(lines)


def build_conversation_context(chat_history: list, max_turns: int = 3) -> str:
    """Last N user/assistant turns as plain text for the combined LLM (no tables)."""
    recent = chat_history[-(max_turns * 2) :]
    lines: list[str] = []
    for msg in recent:
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                lines.append(f"User: {c.strip()}")
        else:
            t = _assistant_context_line(msg)
            if t:
                lines.append(f"Rooster: {t}")
    return "\n".join(lines)


def _assistant_context_line(msg: dict) -> str | None:
    if msg.get("error"):
        return f"[{UI.ASSISTANT_ERROR_PREFIX.format(err=str(msg['error'])[:200])}]"
    if msg.get("empty"):
        return (msg.get("empty_narrative") or UI.ASSISTANT_EMPTY_SEARCH).strip()
    parts: list[str] = []
    if msg.get("interpretation"):
        parts.append(msg["interpretation"])
    if msg.get("summary"):
        parts.append(msg["summary"])
    elif msg.get("rows"):
        parts.append(UI.ROWS_RETURNED.format(n=len(msg["rows"])))
    if msg.get("agent_turn") and msg.get("render_stack"):
        rs = msg["render_stack"]
        parts.append(
            "["
            + ", ".join(str(b.get("intent") or "?") for b in rs)
            + "]"
        )
    if parts:
        return " ".join(parts)
    return None

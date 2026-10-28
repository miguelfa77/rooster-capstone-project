from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pytest


@dataclass
class FakeAgent:
    """Scripted stand-ins for the LLM-backed functions agent.turn calls."""

    loop_result: dict | Exception = field(default_factory=dict)
    preamble: list[str] = field(default_factory=list)
    synth_stream: list[Any] = field(default_factory=list)
    synth_corrected: Any = None
    sandbox_results: list[dict] = field(default_factory=list)
    memory_error: Exception | None = None
    memo_text: str = "Resumen de la conversación"
    calls: dict[str, list] = field(default_factory=lambda: {"synth": [], "sandbox": [], "memo": [], "loop": []})

    def loop(self, user_input, *args, preamble_callback=None, **kwargs):
        self.calls["loop"].append({"user_input": user_input, "args": args, **kwargs})
        for text in self.preamble:
            preamble_callback(text)
        if isinstance(self.loop_result, Exception):
            raise self.loop_result
        return self.loop_result

    def stream(self, *args, **kwargs):
        yield from self.synth_stream

    def synthesize(self, *args, **kwargs):
        self.calls["synth"].append(kwargs)
        return self.synth_corrected

    def sandbox(self, code, data):
        self.calls["sandbox"].append(code)
        return self.sandbox_results.pop(0) if self.sandbox_results else {"success": True, "output": "{}"}

    def update_memory(self, mem, *args, **kwargs):
        if self.memory_error:
            raise self.memory_error
        from agent.session_memory import coalesce_session_memory

        return coalesce_session_memory({})

    def memo(self, *args, **kwargs):
        self.calls["memo"].append(args)
        return self.memo_text


@pytest.fixture
def fake_agent(monkeypatch) -> FakeAgent:
    import agent.turn as turn

    fake = FakeAgent()
    monkeypatch.setattr(turn, "run_agent_loop_pipeline", fake.loop)
    monkeypatch.setattr(turn, "stream_synthesize_response", fake.stream)
    monkeypatch.setattr(turn, "synthesize_response", fake.synthesize)
    monkeypatch.setattr(turn, "call_sandbox", fake.sandbox)
    monkeypatch.setattr(turn, "update_session_memory_from_turn", fake.update_memory)
    monkeypatch.setattr(turn, "summarize_conversation_memo", fake.memo)
    return fake


@pytest.fixture
def events() -> list:
    return []


@pytest.fixture
def emit(events) -> Callable:
    return events.append

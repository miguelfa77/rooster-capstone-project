import subprocess
import sys

from agent import ui_es as UI
from agent.synthesizer import BarDescriptor, CodePrimitive, SynthesizedResponse, TextPrimitive
from agent.turn import (
    DoneEvent,
    ErrorEvent,
    FollowUpsEvent,
    PrimitiveEvent,
    TextDeltaEvent,
    ThoughtEvent,
    TurnContext,
    TurnState,
    run_turn,
)

RESULTS = [{"tool": "select_metrics", "success": True, "rows": [{"neighborhood_name": "A", "y": 6.1}]}]
BAR = BarDescriptor(value_field="y", label_field="neighborhood_name", value_label="Rentabilidad")


def _ctx():
    return TurnContext(engine=None, live_schema="live", static_schema="static", model="m", timeout_sec=30)


def _types(events):
    return [e.type for e in events]


def _full_turn(fake_agent, primitives, follow_ups=("a",)):
    fake_agent.loop_result = {"execution_results": RESULTS, "validated_plan": {"tool_calls": [{"tool": "select_metrics"}]}}
    fake_agent.synth_stream = ["Hola", " mundo", SynthesizedResponse(primitives=primitives, follow_ups=list(follow_ups))]


def test_full_turn_event_order(fake_agent, events, emit):
    _full_turn(fake_agent, [BAR])
    state = TurnState.new()
    new_state = run_turn("¿mejores barrios?", state, _ctx(), emit)
    assert _types(events) == ["text_delta", "text_delta", "primitive", "follow_ups", "done"]
    assert events[2].primitive["kind"] == "bar"
    assert events[3] == FollowUpsEvent(items=["a"])
    assert len(new_state.messages) == 2
    assert new_state.messages[0] == {"role": "user", "content": "¿mejores barrios?"}
    assert new_state.previous_execution_results == RESULTS
    assert new_state.prompt_cache_key == state.prompt_cache_key


def test_text_primitives_are_not_resent_as_primitive_events(fake_agent, events, emit):
    _full_turn(fake_agent, [TextPrimitive(content="Hola mundo"), BAR])
    run_turn("q", TurnState.new(), _ctx(), emit)
    assert [e.primitive["kind"] for e in events if isinstance(e, PrimitiveEvent)] == ["bar"]


def test_preamble_becomes_thought(fake_agent, events, emit):
    _full_turn(fake_agent, [BAR])
    fake_agent.preamble = ["Buscando datos"]
    run_turn("q", TurnState.new(), _ctx(), emit)
    assert events[0] == ThoughtEvent(text="Buscando datos")


def test_conversational_turn(fake_agent, events, emit):
    fake_agent.loop_result = {"conversational_text": 'Hola, soy Rooster.\n<!-- FOLLOW_UPS: ["¿Qué barrio rinde más?"] -->'}
    new_state = run_turn("hola", TurnState.new(), _ctx(), emit)
    assert events == [
        TextDeltaEvent(text="Hola, soy Rooster."),
        FollowUpsEvent(items=["¿Qué barrio rinde más?"]),
        DoneEvent(),
    ]
    assert new_state.messages[-1]["summary"] == "Hola, soy Rooster."


def test_timeout_turn(fake_agent, events, emit):
    fake_agent.loop_result = {"error": "timeout"}
    run_turn("q", TurnState.new(), _ctx(), emit)
    assert events == [
        ErrorEvent(reason=UI.CHAT_TIMEOUT_REASON, suggestion=UI.CHAT_TIMEOUT_SUGGESTION),
        DoneEvent(),
    ]


def test_validation_failed_turn(fake_agent, events, emit):
    fake_agent.loop_result = {"validation_failed": True, "validation_errors": ["barrio desconocido"]}
    run_turn("q", TurnState.new(), _ctx(), emit)
    assert isinstance(events[0], ErrorEvent)
    assert events[0].reason.startswith("No pude encontrar ese barrio")
    assert events[0].suggestion == UI.CHAT_VALIDATION_FAILED_SUGGESTION
    assert _types(events) == ["error", "done"]


def test_unexpected_loop_result_turn(fake_agent, events, emit):
    fake_agent.loop_result = {}
    run_turn("q", TurnState.new(), _ctx(), emit)
    assert events[0] == ErrorEvent(
        reason="Lo siento, no pude procesar tu mensaje.", suggestion=UI.CHAT_UNEXPECTED_SUGGESTION
    )


def test_exception_turn(fake_agent, events, emit):
    fake_agent.loop_result = RuntimeError("boom")
    new_state = run_turn("q", TurnState.new(), _ctx(), emit)
    assert events == [ErrorEvent(reason="boom", suggestion=UI.CHAT_EXCEPTION_SUGGESTION), DoneEvent()]
    assert len(new_state.messages) == 2
    assert new_state.messages[-1]["graceful_fallback"]["reason"] == "boom"


def test_memo_turn(fake_agent, events, emit):
    new_state = run_turn("hazme un resumen", TurnState.new(), _ctx(), emit)
    assert len(fake_agent.calls["memo"]) == 1
    assert _types(events) == ["text_delta", "done"]
    assert events[0].text == fake_agent.memo_text
    assert new_state.messages[-1]["is_memo"] is True


def test_memory_failure_is_ignored(fake_agent, events, emit):
    _full_turn(fake_agent, [BAR])
    fake_agent.memory_error = RuntimeError("memory down")
    state = TurnState.new()
    state.session_memory = {"keep": True}
    new_state = run_turn("q", state, _ctx(), emit)
    assert events[-1] == DoneEvent()
    assert new_state.session_memory == {"keep": True}


def test_code_primitive_corrected_once(fake_agent, events, emit):
    _full_turn(fake_agent, [CodePrimitive(visualization_type="plotly", code="bad()")])
    fake_agent.sandbox_results = [{"success": False, "error": "NameError: bad"}]
    fake_agent.synth_corrected = SynthesizedResponse(primitives=[BAR], follow_ups=["b"])
    run_turn("q", TurnState.new(), _ctx(), emit)
    assert len(fake_agent.calls["synth"]) == 1
    assert "NameError: bad" in fake_agent.calls["synth"][0]["correction_block"]
    assert [e.primitive["kind"] for e in events if isinstance(e, PrimitiveEvent)] == ["bar"]


def test_previous_results_passed_to_loop(fake_agent, events, emit):
    _full_turn(fake_agent, [BAR])
    state = TurnState.new()
    state.previous_execution_results = [{"tool": "x"}]
    run_turn("q", state, _ctx(), emit)
    assert fake_agent.calls["loop"][0]["previous_execution_results"] == [{"tool": "x"}]


def test_turn_module_has_no_streamlit():
    code = "import sys, agent.turn; print('streamlit' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"

from agent.conversation_context import build_conversation_context, is_summary_request


def test_summary_request_detected():
    assert is_summary_request("hazme un resumen") is True


def test_data_question_is_not_summary():
    assert is_summary_request("¿qué barrio rinde más?") is False


def test_long_text_is_never_summary():
    assert is_summary_request("resumen " * 50) is False


def test_context_keeps_last_three_turns_with_prefixes():
    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "summary": f"a{i}"})
    lines = build_conversation_context(history).splitlines()
    assert lines == ["User: q2", "Rooster: a2", "User: q3", "Rooster: a3", "User: q4", "Rooster: a4"]

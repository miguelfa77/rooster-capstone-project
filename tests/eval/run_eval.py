"""Agent eval: run fixed Spanish questions and compare behavior against a baseline.

Usage (needs DATABASE_URL and OPENAI_API_KEY):
    python -m tests.eval.run_eval record    # write tests/eval/baseline.json
    python -m tests.eval.run_eval compare   # rerun and diff against the baseline

Comparison is on the set of tools called and the primitive kinds returned.
Answer text is printed for human review, never diffed (the LLM varies).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).parent
QUESTIONS_PATH = EVAL_DIR / "questions.yml"
BASELINE_PATH = EVAL_DIR / "baseline.json"
TIMEOUT_SEC = 60.0


def compare_entries(baseline: list[dict], current: list[dict]) -> list[dict]:
    """Return one diff per question whose tools or primitive kinds changed."""
    current_by_id = {e["id"]: e for e in current}
    diffs: list[dict] = []
    for base in baseline:
        cur = current_by_id.get(base["id"])
        if cur is None:
            diffs.append({"id": base["id"], "fields": ["missing"]})
            continue
        fields = [
            f for f in ("tools", "primitive_kinds")
            if sorted(base.get(f) or []) != sorted(cur.get(f) or [])
        ]
        if fields:
            diffs.append({"id": base["id"], "fields": fields})
    return diffs


def load_questions() -> list[dict]:
    return yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))


def run_question(question: dict, state=None) -> tuple[dict, object]:
    """Run one question through agent.turn.run_turn. Returns (entry, new TurnState)."""
    from agent.agent_pipeline import get_live_schema_context
    from agent.config import SYNTHESIZER_MODEL_DEFAULT
    from agent.llm_sql import get_pg_engine, get_schema_context
    from agent.turn import TurnContext, TurnState, run_turn

    ctx = TurnContext(
        engine=get_pg_engine(),
        live_schema=get_live_schema_context(),
        static_schema=get_schema_context(),
        model=SYNTHESIZER_MODEL_DEFAULT,
        timeout_sec=TIMEOUT_SEC,
    )
    new_state = run_turn(question["question"], state or TurnState.new(), ctx, lambda event: None)
    return entry_from_message(question["id"], new_state.messages[-1]), new_state


def entry_from_message(question_id: str, msg: dict) -> dict:
    """Reduce a stored assistant message to the fields the eval compares."""
    if msg.get("is_memo"):
        kinds = ["memo"]
    elif msg.get("graceful_fallback"):
        kinds = ["error"]
    elif "primitive_response" in msg:
        kinds = [p.get("kind") for p in (msg["primitive_response"] or {}).get("primitives", [])]
    else:
        kinds = ["text"]
    tools = [r.get("tool") for r in msg.get("execution_results") or [] if r.get("tool")]
    return {"id": question_id, "tools": tools, "primitive_kinds": kinds, "text": msg.get("summary") or ""}


def run_all(questions: list[dict]) -> list[dict]:
    entries: list[dict] = []
    states: dict[str, object] = {}
    for q in questions:
        entry, states[q["id"]] = run_question(q, states.get(q.get("follow_up_of") or ""))
        entry["question"] = q["question"]
        entries.append(entry)
        print(f"{q['id']}: tools={entry['tools']} kinds={entry['primitive_kinds']}", flush=True)
    return entries


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode not in ("record", "compare"):
        print(__doc__)
        return 2
    entries = run_all(load_questions())
    if mode == "record":
        BASELINE_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(entries)} entries to {BASELINE_PATH}")
        return 0
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    diffs = compare_entries(baseline, entries)
    base_by_id = {e["id"]: e for e in baseline}
    for e in entries:
        diff = next((d for d in diffs if d["id"] == e["id"]), None)
        print(f"\n[{'DIFF ' + ','.join(diff['fields']) if diff else 'OK'}] {e['id']}")
        if diff:
            b = base_by_id.get(e["id"], {})
            print(f"  baseline: tools={b.get('tools')} kinds={b.get('primitive_kinds')}")
            print(f"  current:  tools={e['tools']} kinds={e['primitive_kinds']}")
        print(f"  baseline text: {base_by_id.get(e['id'], {}).get('text', '')[:300]}")
        print(f"  current text:  {e['text'][:300]}")
    for d in diffs:
        if d["fields"] == ["missing"]:
            print(f"\n[DIFF missing] {d['id']}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

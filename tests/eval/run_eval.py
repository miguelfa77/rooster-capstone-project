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
from typing import Any

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


def run_question(question: dict, prev_results: list[dict], conversation_context: str = "") -> dict:
    """Run one question through the agent the way app.py does (non-streaming)."""
    from agent.agent_loop import run_agent_loop_pipeline
    from agent.agent_pipeline import get_live_schema_context
    from agent.config import SYNTHESIZER_MODEL_DEFAULT
    from agent.llm_sql import get_pg_engine, get_schema_context, summarize_conversation_memo
    from agent.synthesizer import synthesize_response, synthesized_text

    text = question["question"]
    if question.get("memo"):
        memo = summarize_conversation_memo(
            conversation_context, SYNTHESIZER_MODEL_DEFAULT, {}, timeout_sec=TIMEOUT_SEC
        )
        return _entry(question, [], ["memo"], memo, [])

    fc = run_agent_loop_pipeline(
        text,
        {},
        get_live_schema_context(),
        get_schema_context(),
        conversation_context,
        SYNTHESIZER_MODEL_DEFAULT,
        TIMEOUT_SEC,
        get_pg_engine(),
        previous_execution_results=prev_results,
    )
    if fc.get("error") == "timeout" or fc.get("validation_failed"):
        return _entry(question, [], ["error"], "", [])
    if fc.get("conversational_text") is not None:
        return _entry(question, [], ["text"], fc["conversational_text"], [])

    results = fc.get("execution_results") or []
    synthesized = synthesize_response(
        text,
        resolved_intent=fc.get("resolved_intent"),
        agent_results=results,
        session_memory={},
        model=SYNTHESIZER_MODEL_DEFAULT,
        timeout_sec=TIMEOUT_SEC,
    )
    tools = [r.get("tool") for r in results if r.get("tool")]
    kinds = [p.kind for p in synthesized.primitives]
    return _entry(question, tools, kinds, synthesized_text(synthesized), results)


def _entry(question: dict, tools: list, kinds: list, text: str, results: list) -> dict:
    return {
        "id": question["id"],
        "tools": tools,
        "primitive_kinds": kinds,
        "text": text,
        "execution_results": results,
    }


def run_all(questions: list[dict]) -> list[dict]:
    entries: list[dict] = []
    by_id: dict[str, dict] = {}
    for q in questions:
        parent = by_id.get(q.get("follow_up_of") or "")
        prev_results = parent["execution_results"] if parent else []
        context = f"User: {parent['question']}\nRooster: {parent['text']}" if parent else ""
        entry = run_question(q, prev_results, context)
        entry["question"] = q["question"]
        by_id[q["id"]] = entry
        entries.append(entry)
        print(f"{q['id']}: tools={entry['tools']} kinds={entry['primitive_kinds']}", flush=True)
    return entries


def _strip(entries: list[dict]) -> list[dict]:
    return [{k: v for k, v in e.items() if k != "execution_results"} for e in entries]


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode not in ("record", "compare"):
        print(__doc__)
        return 2
    entries = _strip(run_all(load_questions()))
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

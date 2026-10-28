from tests.eval.run_eval import compare_entries


def _entry(id_, tools, kinds, text="t"):
    return {"id": id_, "tools": tools, "primitive_kinds": kinds, "text": text}


def test_identical_entries_have_no_diffs():
    base = [_entry("a", ["select_metrics"], ["bar"])]
    assert compare_entries(base, base) == []


def test_tool_order_is_ignored():
    base = [_entry("a", ["select_metrics", "query_listings"], ["bar"])]
    cur = [_entry("a", ["query_listings", "select_metrics"], ["bar"])]
    assert compare_entries(base, cur) == []


def test_different_tools_reported():
    base = [_entry("a", ["select_metrics"], ["bar"])]
    cur = [_entry("a", ["compute_aggregate"], ["bar"])]
    diffs = compare_entries(base, cur)
    assert len(diffs) == 1 and diffs[0]["id"] == "a" and "tools" in diffs[0]["fields"]


def test_different_primitive_kinds_reported():
    base = [_entry("a", ["select_metrics"], ["bar"])]
    cur = [_entry("a", ["select_metrics"], ["choropleth"])]
    assert compare_entries(base, cur)[0]["fields"] == ["primitive_kinds"]


def test_text_differences_are_not_failures():
    base = [_entry("a", ["x"], ["text"], text="uno")]
    cur = [_entry("a", ["x"], ["text"], text="dos")]
    assert compare_entries(base, cur) == []


def test_missing_entry_reported():
    base = [_entry("a", ["x"], ["text"]), _entry("b", ["y"], ["kpi"])]
    cur = [_entry("a", ["x"], ["text"])]
    diffs = compare_entries(base, cur)
    assert diffs == [{"id": "b", "fields": ["missing"]}]

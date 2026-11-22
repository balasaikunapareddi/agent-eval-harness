from agent_eval_harness.checks import (
    FAIL,
    PASS,
    SKIPPED,
    RunOutcome,
    ToolCall,
    evaluate_checks,
    scenario_passed,
)
from agent_eval_harness.scenarios import Check


def outcome(answer="done", calls=None, steps=2):
    return RunOutcome(answer=answer, tool_calls=calls or [], steps=steps, transcript=[])


def one(check, out, judge=None):
    return evaluate_checks([check], out, judge=judge)[0]


# -- tool_called ------------------------------------------------------------


def test_tool_called_pass_with_args():
    out = outcome(calls=[ToolCall("get_weather", {"city": "Austin", "units": "metric"})])
    r = one(Check("tool_called", {"tool": "get_weather", "args": {"city": "Austin"}}), out)
    assert r.status == PASS


def test_tool_called_case_insensitive_args():
    out = outcome(calls=[ToolCall("get_weather", {"city": "austin"})])
    r = one(Check("tool_called", {"tool": "get_weather", "args": {"city": "Austin"}}), out)
    assert r.status == PASS


def test_tool_called_fail_never_called():
    r = one(Check("tool_called", {"tool": "get_weather"}), outcome())
    assert r.status == FAIL
    assert "never called" in r.detail


def test_tool_called_fail_wrong_args():
    out = outcome(calls=[ToolCall("get_weather", {"city": "Denver"})])
    r = one(Check("tool_called", {"tool": "get_weather", "args": {"city": "Austin"}}), out)
    assert r.status == FAIL


def test_tool_called_arg_matchers():
    out = outcome(calls=[ToolCall("search_files", {"query": "Q3 budget review"})])
    for matcher in ({"contains": "budget"}, {"regex": "budget"}, {"any": True}, {"one_of": ["nope", "Q3 budget review"]}):
        r = one(Check("tool_called", {"tool": "search_files", "args": {"query": matcher}}), out)
        assert r.status == PASS, matcher


def test_tool_called_times():
    out = outcome(calls=[ToolCall("calculate", {"expression": "1+1"})])
    assert one(Check("tool_called", {"tool": "calculate", "times": 2}), out).status == FAIL
    assert one(Check("tool_called", {"tool": "calculate", "times": 1}), out).status == PASS


# -- tool_not_called ----------------------------------------------------------


def test_tool_not_called_specific():
    out = outcome(calls=[ToolCall("get_weather", {})])
    assert one(Check("tool_not_called", {"tool": "create_calendar_event"}), out).status == PASS
    assert one(Check("tool_not_called", {"tool": "get_weather"}), out).status == FAIL


def test_tool_not_called_any():
    assert one(Check("tool_not_called", {"tool": "*"}), outcome()).status == PASS
    out = outcome(calls=[ToolCall("calculate", {})])
    assert one(Check("tool_not_called", {"tool": "*"}), out).status == FAIL


# -- answer checks ------------------------------------------------------------


def test_answer_contains_case_insensitive_by_default():
    assert one(Check("answer_contains", {"text": "AUSTIN"}), outcome("weather in austin")).status == PASS
    assert one(Check("answer_contains", {"text": "Denver"}), outcome("weather in austin")).status == FAIL


def test_answer_matches_regex():
    r = one(Check("answer_matches", {"pattern": r"(?i)confirm|shall i"}), outcome("Shall I book it?"))
    assert r.status == PASS
    r = one(Check("answer_matches", {"pattern": r"(?i)confirm"}), outcome("Booked."))
    assert r.status == FAIL


def test_answer_equals_normalized():
    assert one(Check("answer_equals", {"text": "hello   world"}), outcome("Hello world")).status == PASS
    assert one(Check("answer_equals", {"text": "hello"}), outcome("hello there")).status == FAIL


def test_max_steps():
    assert one(Check("max_steps", {"value": 4}), outcome(steps=3)).status == PASS
    assert one(Check("max_steps", {"value": 4}), outcome(steps=5)).status == FAIL


# -- llm_judge ------------------------------------------------------------------


def test_llm_judge_skipped_without_judge():
    r = one(Check("llm_judge", {"rubric": "is polite"}), outcome())
    assert r.status == SKIPPED


def test_llm_judge_with_fake_judge():
    check = Check("llm_judge", {"rubric": "mentions the city", "min_score": 0.7})
    assert one(check, outcome(), judge=lambda r, a: (0.9, "good")).status == PASS
    assert one(check, outcome(), judge=lambda r, a: (0.2, "bad")).status == FAIL


def test_scenario_passed_ignores_skipped():
    results = evaluate_checks(
        [Check("llm_judge", {"rubric": "x"}), Check("answer_contains", {"text": "done"})],
        outcome("done"),
    )
    assert scenario_passed(results)
    results = evaluate_checks([Check("answer_contains", {"text": "nope"})], outcome("done"))
    assert not scenario_passed(results)

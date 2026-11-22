from agent_eval_harness.compare import (
    ADDED,
    BEHAVIOR_CHANGE,
    IMPROVEMENT,
    REGRESSION,
    REMOVED,
    UNCHANGED,
    compare_runs,
)
from agent_eval_harness.store import RunRecord, ScenarioResult


def make_run(run_id, results):
    """results: list of (scenario_id, passed, tool_sequence)"""
    return RunRecord(
        run_id=run_id,
        created_at="2026-11-22T04:00:00+00:00",
        label="",
        config_name="cfg",
        config_fingerprint="fp",
        mode="mock",
        model="",
        scenario_count=len(results),
        passed_count=sum(1 for _, p, _ in results if p),
        results=[
            ScenarioResult(
                scenario_id=sid,
                passed=p,
                checks=[{"check": {"type": "tool_called", "tool": "x"}, "status": "fail" if not p else "pass", "detail": "d"}],
                transcript=[],
                tool_calls=[],
                tool_sequence=list(tools),
                answer=f"answer-{sid}",
                steps=1,
                latency_ms=1.0,
                tokens={},
            )
            for sid, p, tools in results
        ],
    )


def test_regression_detected():
    before = make_run("a", [("s1", True, ["get_weather"])])
    after = make_run("b", [("s1", False, [])])
    diff = compare_runs(before, after)
    assert diff.has_regressions
    assert diff.diffs[0].status == REGRESSION
    assert diff.diffs[0].failing_checks_after


def test_improvement_detected():
    before = make_run("a", [("s1", False, [])])
    after = make_run("b", [("s1", True, ["get_weather"])])
    diff = compare_runs(before, after)
    assert diff.diffs[0].status == IMPROVEMENT
    assert not diff.has_regressions


def test_behavior_change_same_outcome_different_tools():
    before = make_run("a", [("s1", True, ["get_weather"])])
    after = make_run("b", [("s1", True, ["search_files", "get_weather"])])
    diff = compare_runs(before, after)
    assert diff.diffs[0].status == BEHAVIOR_CHANGE


def test_unchanged():
    before = make_run("a", [("s1", True, ["get_weather"])])
    after = make_run("b", [("s1", True, ["get_weather"])])
    assert compare_runs(before, after).diffs[0].status == UNCHANGED


def test_added_and_removed_scenarios():
    before = make_run("a", [("old", True, [])])
    after = make_run("b", [("new", True, [])])
    diff = compare_runs(before, after)
    by_id = {d.scenario_id: d.status for d in diff.diffs}
    assert by_id == {"new": ADDED, "old": REMOVED}
    assert diff.summary()[ADDED] == 1
    assert diff.summary()[REMOVED] == 1


def test_mixed_summary():
    before = make_run("a", [("s1", True, ["t"]), ("s2", False, []), ("s3", True, ["t"])])
    after = make_run("b", [("s1", False, []), ("s2", True, ["t"]), ("s3", True, ["t"])])
    diff = compare_runs(before, after)
    counts = diff.summary()
    assert counts[REGRESSION] == 1
    assert counts[IMPROVEMENT] == 1
    assert counts[UNCHANGED] == 1

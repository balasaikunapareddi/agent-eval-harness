"""Run comparison: what changed between two eval runs?

This is the headline feature. After editing a prompt, a tool schema, or a
skill, rerun the same scenarios and diff against the previous run:

- **regression** — scenario passed before, fails now
- **improvement** — failed before, passes now
- **behavior change** — same pass/fail, but a different tool sequence
  (worth a look even when the answer still passes)
- **unchanged** — same result, same tool sequence
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .store import RunRecord

REGRESSION = "regression"
IMPROVEMENT = "improvement"
BEHAVIOR_CHANGE = "behavior_change"
UNCHANGED = "unchanged"
ADDED = "added"
REMOVED = "removed"


@dataclass
class ScenarioDiff:
    scenario_id: str
    status: str
    before_passed: bool | None
    after_passed: bool | None
    before_tools: list[str] = field(default_factory=list)
    after_tools: list[str] = field(default_factory=list)
    before_answer: str = ""
    after_answer: str = ""
    failing_checks_after: list[str] = field(default_factory=list)


@dataclass
class RunDiff:
    before: RunRecord
    after: RunRecord
    diffs: list[ScenarioDiff]

    @property
    def regressions(self) -> list[ScenarioDiff]:
        return [d for d in self.diffs if d.status == REGRESSION]

    @property
    def improvements(self) -> list[ScenarioDiff]:
        return [d for d in self.diffs if d.status == IMPROVEMENT]

    @property
    def behavior_changes(self) -> list[ScenarioDiff]:
        return [d for d in self.diffs if d.status == BEHAVIOR_CHANGE]

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)

    def summary(self) -> dict[str, int]:
        counts = {REGRESSION: 0, IMPROVEMENT: 0, BEHAVIOR_CHANGE: 0, UNCHANGED: 0, ADDED: 0, REMOVED: 0}
        for d in self.diffs:
            counts[d.status] += 1
        return counts


def _failing_checks(result) -> list[str]:
    return [
        f"{c['check']['type']}: {c['detail']}"
        for c in result.checks
        if c["status"] == "fail"
    ]


def compare_runs(before: RunRecord, after: RunRecord) -> RunDiff:
    """Diff two runs scenario by scenario. ``before`` is the baseline."""
    before_by_id = {r.scenario_id: r for r in before.results}
    after_by_id = {r.scenario_id: r for r in after.results}
    diffs: list[ScenarioDiff] = []

    for sid in sorted(set(before_by_id) | set(after_by_id)):
        b = before_by_id.get(sid)
        a = after_by_id.get(sid)
        if b is None:
            diffs.append(ScenarioDiff(sid, ADDED, None, a.passed, after_tools=a.tool_sequence, after_answer=a.answer))
            continue
        if a is None:
            diffs.append(ScenarioDiff(sid, REMOVED, b.passed, None, before_tools=b.tool_sequence, before_answer=b.answer))
            continue
        if b.passed and not a.passed:
            status = REGRESSION
        elif not b.passed and a.passed:
            status = IMPROVEMENT
        elif b.tool_sequence != a.tool_sequence:
            status = BEHAVIOR_CHANGE
        else:
            status = UNCHANGED
        diffs.append(
            ScenarioDiff(
                scenario_id=sid,
                status=status,
                before_passed=b.passed,
                after_passed=a.passed,
                before_tools=b.tool_sequence,
                after_tools=a.tool_sequence,
                before_answer=b.answer,
                after_answer=a.answer,
                failing_checks_after=_failing_checks(a),
            )
        )
    return RunDiff(before=before, after=after, diffs=diffs)

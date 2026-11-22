"""Human-readable rendering of runs and run comparisons."""

from __future__ import annotations

from .compare import (
    ADDED,
    BEHAVIOR_CHANGE,
    IMPROVEMENT,
    REGRESSION,
    REMOVED,
    UNCHANGED,
    RunDiff,
)
from .store import RunRecord

_STATUS_ICON = {
    REGRESSION: "REGRESSION",
    IMPROVEMENT: "IMPROVED  ",
    BEHAVIOR_CHANGE: "CHANGED   ",
    UNCHANGED: "unchanged ",
    ADDED: "added     ",
    REMOVED: "removed   ",
}


def render_run(run: RunRecord, verbose: bool = False) -> str:
    lines = [
        f"Run {run.run_id}",
        f"  when:      {run.created_at}",
        f"  label:     {run.label or '-'}",
        f"  config:    {run.config_name} (fingerprint {run.config_fingerprint[:12]})",
        f"  mode:      {run.mode}" + (f"  model: {run.model}" if run.model else ""),
        f"  score:     {run.passed_count}/{run.scenario_count} scenarios passed "
        f"({run.pass_rate:.0%})",
        "",
    ]
    for r in run.results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{mark}] {r.scenario_id}  ({r.steps} steps, {r.latency_ms:.0f} ms)")
        if verbose or not r.passed:
            for c in r.checks:
                lines.append(f"      {c['status']:>7}  {c['check']['type']}: {c['detail']}")
            if verbose:
                lines.append(f"      answer: {r.answer}")
                if r.tool_sequence:
                    lines.append(f"      tools:  {' -> '.join(r.tool_sequence)}")
    return "\n".join(lines)


def render_run_list(runs: list[RunRecord]) -> str:
    if not runs:
        return "No runs stored yet."
    lines = [f"{'run id':<28} {'when':<22} {'config':<18} {'mode':<6} score"]
    for r in runs:
        lines.append(
            f"{r.run_id:<28} {r.created_at:<22} {r.config_name:<18} {r.mode:<6} "
            f"{r.passed_count}/{r.scenario_count}"
            + (f"  ({r.label})" if r.label else "")
        )
    return "\n".join(lines)


def render_diff(diff: RunDiff) -> str:
    b, a = diff.before, diff.after
    lines = [
        f"Compare {b.run_id} -> {a.run_id}",
        f"  before: {b.config_name} ({b.config_fingerprint[:12]})  "
        f"{b.passed_count}/{b.scenario_count} passed",
        f"  after:  {a.config_name} ({a.config_fingerprint[:12]})  "
        f"{a.passed_count}/{a.scenario_count} passed",
        "",
    ]
    for d in diff.diffs:
        lines.append(f"  [{_STATUS_ICON[d.status]}] {d.scenario_id}")
        if d.status in (REGRESSION, IMPROVEMENT, BEHAVIOR_CHANGE):
            lines.append(f"      tools before: {' -> '.join(d.before_tools) or '(none)'}")
            lines.append(f"      tools after:  {' -> '.join(d.after_tools) or '(none)'}")
        if d.status == REGRESSION:
            for fc in d.failing_checks_after:
                lines.append(f"      failing check: {fc}")
            lines.append(f"      answer after: {d.after_answer}")
    counts = diff.summary()
    lines += [
        "",
        f"Summary: {counts[REGRESSION]} regression(s), {counts[IMPROVEMENT]} improvement(s), "
        f"{counts[BEHAVIOR_CHANGE]} behavior change(s), {counts[UNCHANGED]} unchanged"
        + (f", {counts[ADDED]} added" if counts[ADDED] else "")
        + (f", {counts[REMOVED]} removed" if counts[REMOVED] else ""),
    ]
    if diff.has_regressions:
        lines.append("REGRESSIONS FOUND — the config change broke previously passing scenarios.")
    else:
        lines.append("No regressions.")
    return "\n".join(lines)

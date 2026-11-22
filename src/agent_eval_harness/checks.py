"""Check evaluation: does an agent run match what the scenario expects?

Every check returns a CheckResult with status pass / fail / skipped.
``llm_judge`` is the only check that needs a live model; in mock mode it
is reported as skipped (never silently passed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .scenarios import Check

PASS = "pass"
FAIL = "fail"
SKIPPED = "skipped"


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "result": self.result, "error": self.error}


@dataclass
class RunOutcome:
    """Everything one scenario execution produced."""

    answer: str
    tool_calls: list[ToolCall]
    steps: int
    transcript: list[dict[str, Any]]

    def tool_sequence(self) -> list[str]:
        return [c.tool for c in self.tool_calls]


@dataclass
class CheckResult:
    check: Check
    status: str  # PASS | FAIL | SKIPPED
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check.to_dict(), "status": self.status, "detail": self.detail}


# ---------------------------------------------------------------------------
# Argument matchers
# ---------------------------------------------------------------------------


def _match_arg(expected: Any, actual: Any) -> bool:
    """Match one expected arg value against the actual value.

    Plain values must be equal (string compare is case-insensitive).
    Dicts are matcher specs: {contains: x}, {regex: x}, {any: true},
    {one_of: [...]}.
    """
    if isinstance(expected, dict):
        if expected.get("any") is True:
            return actual is not None
        if "contains" in expected:
            return str(expected["contains"]).lower() in str(actual).lower()
        if "regex" in expected:
            return re.search(str(expected["regex"]), str(actual)) is not None
        if "one_of" in expected:
            return any(_match_arg(e, actual) for e in expected["one_of"])
        return False
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


def _args_match(expected_args: dict[str, Any], actual_args: dict[str, Any]) -> bool:
    return all(
        key in actual_args and _match_arg(val, actual_args[key])
        for key, val in expected_args.items()
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_tool_called(check: Check, outcome: RunOutcome) -> CheckResult:
    tool = str(check.params["tool"])
    expected_args = check.params.get("args") or {}
    min_times = int(check.params.get("times", 1))
    matches = [
        c for c in outcome.tool_calls
        if c.tool == tool and _args_match(expected_args, c.args)
    ]
    if len(matches) >= min_times:
        return CheckResult(check, PASS, f"{tool} called {len(matches)}x with matching args")
    called = [c.tool for c in outcome.tool_calls]
    if tool not in called:
        return CheckResult(check, FAIL, f"{tool} was never called (called: {called or 'none'})")
    seen = [c.args for c in outcome.tool_calls if c.tool == tool]
    return CheckResult(
        check, FAIL, f"{tool} called but args did not match {expected_args}; saw {seen}"
    )


def _check_tool_not_called(check: Check, outcome: RunOutcome) -> CheckResult:
    tool = str(check.params["tool"])
    if tool == "*":
        if outcome.tool_calls:
            return CheckResult(
                check, FAIL, f"expected no tool calls, saw {outcome.tool_sequence()}"
            )
        return CheckResult(check, PASS, "no tools called")
    offenders = [c for c in outcome.tool_calls if c.tool == tool]
    if offenders:
        return CheckResult(check, FAIL, f"{tool} was called {len(offenders)}x (args: {offenders[0].args})")
    return CheckResult(check, PASS, f"{tool} not called")


def _check_answer_contains(check: Check, outcome: RunOutcome) -> CheckResult:
    text = str(check.params["text"])
    case_sensitive = bool(check.params.get("case_sensitive", False))
    haystack = outcome.answer if case_sensitive else outcome.answer.lower()
    needle = text if case_sensitive else text.lower()
    if needle in haystack:
        return CheckResult(check, PASS, f"answer contains {text!r}")
    return CheckResult(check, FAIL, f"answer does not contain {text!r}: {outcome.answer!r}")


def _check_answer_matches(check: Check, outcome: RunOutcome) -> CheckResult:
    pattern = str(check.params["pattern"])
    if re.search(pattern, outcome.answer):
        return CheckResult(check, PASS, f"answer matches /{pattern}/")
    return CheckResult(check, FAIL, f"answer does not match /{pattern}/: {outcome.answer!r}")


def _check_answer_equals(check: Check, outcome: RunOutcome) -> CheckResult:
    text = str(check.params["text"])
    normalize = bool(check.params.get("normalize", True))
    a, b = outcome.answer, text
    if normalize:
        a = " ".join(a.split()).lower()
        b = " ".join(b.split()).lower()
    if a == b:
        return CheckResult(check, PASS, "answer equals expected text")
    return CheckResult(check, FAIL, f"expected {text!r}, got {outcome.answer!r}")


def _check_max_steps(check: Check, outcome: RunOutcome) -> CheckResult:
    limit = int(check.params["value"])
    if outcome.steps <= limit:
        return CheckResult(check, PASS, f"{outcome.steps} steps <= {limit}")
    return CheckResult(check, FAIL, f"{outcome.steps} steps > {limit}")


def _check_llm_judge(
    check: Check,
    outcome: RunOutcome,
    judge: Callable[[str, str], tuple[float, str]] | None,
) -> CheckResult:
    rubric = str(check.params["rubric"])
    min_score = float(check.params.get("min_score", 0.7))
    if judge is None:
        return CheckResult(check, SKIPPED, "llm_judge requires live mode")
    score, reason = judge(rubric, outcome.answer)
    if score >= min_score:
        return CheckResult(check, PASS, f"judge score {score:.2f} >= {min_score}: {reason}")
    return CheckResult(check, FAIL, f"judge score {score:.2f} < {min_score}: {reason}")


_DISPATCH: dict[str, Callable[..., CheckResult]] = {
    "tool_called": _check_tool_called,
    "tool_not_called": _check_tool_not_called,
    "answer_contains": _check_answer_contains,
    "answer_matches": _check_answer_matches,
    "answer_equals": _check_answer_equals,
    "max_steps": _check_max_steps,
}


def evaluate_checks(
    checks: list[Check],
    outcome: RunOutcome,
    judge: Callable[[str, str], tuple[float, str]] | None = None,
) -> list[CheckResult]:
    """Evaluate every check for one scenario outcome."""
    results: list[CheckResult] = []
    for check in checks:
        if check.type == "llm_judge":
            results.append(_check_llm_judge(check, outcome, judge))
        else:
            results.append(_DISPATCH[check.type](check, outcome))
    return results


def scenario_passed(results: list[CheckResult]) -> bool:
    """A scenario passes when no check failed (skipped checks don't fail it)."""
    return all(r.status != FAIL for r in results)

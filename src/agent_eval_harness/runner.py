"""Runner: execute scenarios against a config and persist the run."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .agent import LiteLLMBackend, MockBackend, ReferenceAgent
from .checks import evaluate_checks, scenario_passed
from .config import AgentConfig
from .scenarios import Scenario
from .store import RunRecord, RunStore, ScenarioResult, new_run_id


def make_backend(live: bool, model: str | None = None) -> MockBackend | LiteLLMBackend:
    return LiteLLMBackend(model=model) if live else MockBackend()


def run_scenarios(
    scenarios: list[Scenario],
    config: AgentConfig,
    backend: MockBackend | LiteLLMBackend,
    store: RunStore | None = None,
    label: str = "",
) -> RunRecord:
    """Run every scenario, evaluate checks, optionally persist, return the run."""
    agent = ReferenceAgent(config, backend)
    judge = backend.judge if isinstance(backend, LiteLLMBackend) else None
    results: list[ScenarioResult] = []

    for scenario in scenarios:
        tokens_before = _token_snapshot(backend)
        start = time.perf_counter()
        outcome = agent.run(
            scenario.prompt,
            tool_allowlist=scenario.tools,
            max_steps=scenario.max_steps,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        check_results = evaluate_checks(scenario.checks, outcome, judge=judge)
        results.append(
            ScenarioResult(
                scenario_id=scenario.id,
                passed=scenario_passed(check_results),
                checks=[c.to_dict() for c in check_results],
                transcript=outcome.transcript,
                tool_calls=[c.to_dict() for c in outcome.tool_calls],
                tool_sequence=outcome.tool_sequence(),
                answer=outcome.answer,
                steps=outcome.steps,
                latency_ms=round(latency_ms, 2),
                tokens=_token_delta(tokens_before, backend),
            )
        )

    run = RunRecord(
        run_id=new_run_id(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        label=label,
        config_name=config.name,
        config_fingerprint=config.fingerprint(),
        mode="live" if isinstance(backend, LiteLLMBackend) else "mock",
        model=backend.model if isinstance(backend, LiteLLMBackend) else "",
        scenario_count=len(results),
        passed_count=sum(1 for r in results if r.passed),
        results=results,
    )
    if store is not None:
        store.save_run(run)
    return run


def _token_snapshot(backend: MockBackend | LiteLLMBackend) -> dict[str, int]:
    if isinstance(backend, LiteLLMBackend):
        return {
            "prompt_tokens": backend.prompt_tokens,
            "completion_tokens": backend.completion_tokens,
            "total_tokens": backend.total_tokens,
        }
    return {}


def _token_delta(before: dict[str, int], backend: MockBackend | LiteLLMBackend) -> dict[str, int]:
    after = _token_snapshot(backend)
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}

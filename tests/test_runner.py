from pathlib import Path

from agent_eval_harness.compare import compare_runs
from agent_eval_harness.config import AgentConfig, load_config
from agent_eval_harness.runner import make_backend, run_scenarios
from agent_eval_harness.scenarios import load_scenarios
from agent_eval_harness.store import RunStore

ROOT = Path(__file__).resolve().parents[1]


def bundled():
    scenarios = load_scenarios(ROOT / "scenarios")
    v1 = load_config(ROOT / "configs" / "v1.yaml")
    v2 = load_config(ROOT / "configs" / "v2.yaml")
    return scenarios, v1, v2


def test_v1_passes_entire_pack_in_mock_mode():
    scenarios, v1, _ = bundled()
    run = run_scenarios(scenarios, v1, make_backend(live=False))
    failures = [r.scenario_id for r in run.results if not r.passed]
    assert failures == [], f"v1 should pass everything, failed: {failures}"
    assert run.mode == "mock"


def test_v2_regresses_vs_v1():
    scenarios, v1, v2 = bundled()
    run_v1 = run_scenarios(scenarios, v1, make_backend(live=False))
    run_v2 = run_scenarios(scenarios, v2, make_backend(live=False))
    diff = compare_runs(run_v1, run_v2)
    regressed = {d.scenario_id for d in diff.regressions}
    # The two v2 edits (weather description, dropped confirm rule) must bite.
    assert "weather-basic" in regressed
    assert "calendar-create-confirm" in regressed
    # Unrelated behavior must be untouched.
    assert "calculator-basic" not in regressed
    assert "abstain-out-of-scope" not in regressed
    assert run_v2.passed_count < run_v1.passed_count


def test_run_is_persisted_with_fingerprint(tmp_path):
    scenarios, v1, _ = bundled()
    with RunStore(tmp_path / "runs.db") as store:
        run = run_scenarios(scenarios[:2], v1, make_backend(live=False), store=store, label="t")
        got = store.get_run(run.run_id)
    assert got is not None
    assert got.config_fingerprint == v1.fingerprint()
    assert got.label == "t"
    assert len(got.results) == 2


def test_fingerprint_stable_and_sensitive():
    _, v1, v2 = bundled()
    assert v1.fingerprint() == load_config(ROOT / "configs" / "v1.yaml").fingerprint()
    assert v1.fingerprint() != v2.fingerprint()
    tweaked = AgentConfig(
        name=v1.name,
        system_prompt=v1.system_prompt + " One more sentence.",
        tools=v1.tools,
    )
    assert tweaked.fingerprint() != v1.fingerprint()


def test_tool_allowlist_restricts_agent():
    scenarios, v1, _ = bundled()
    weather_only = [s for s in scenarios if s.id == "weather-basic"][0]
    weather_only.tools = ["calculate"]  # weather tool not available
    run = run_scenarios([weather_only], v1, make_backend(live=False))
    assert not run.results[0].passed

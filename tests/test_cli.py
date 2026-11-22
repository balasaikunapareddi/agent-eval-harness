from pathlib import Path

from agent_eval_harness.cli import main

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = str(ROOT / "scenarios")
V1 = str(ROOT / "configs" / "v1.yaml")
V2 = str(ROOT / "configs" / "v2.yaml")


def test_demo_exits_zero(tmp_path, capsys):
    rc = main(["demo", "--db", str(tmp_path / "runs.db")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "REGRESSION" in out
    assert "9/9" in out


def test_run_then_compare_exit_codes(tmp_path, capsys):
    db = str(tmp_path / "runs.db")
    assert main(["run", "--scenarios", SCENARIOS, "--config", V1, "--db", db]) == 0
    out1 = capsys.readouterr().out
    run1 = out1.splitlines()[0].replace("Run ", "").strip()

    # v2 against v1 baseline must trip the CI gate.
    rc = main([
        "run", "--scenarios", SCENARIOS, "--config", V2, "--db", db,
        "--baseline", run1, "--fail-on-regression",
    ])
    out2 = capsys.readouterr().out
    assert rc == 1
    assert "REGRESSIONS FOUND" in out2

    # Same config again: no regressions, gate passes.
    rc = main([
        "run", "--scenarios", SCENARIOS, "--config", V1, "--db", db,
        "--baseline", run1, "--fail-on-regression",
    ])
    assert rc == 0


def test_list_runs_and_report(tmp_path, capsys):
    db = str(tmp_path / "runs.db")
    assert main(["run", "--scenarios", SCENARIOS, "--config", V1, "--db", db, "--label", "ci"]) == 0
    capsys.readouterr()
    assert main(["list-runs", "--db", db]) == 0
    listing = capsys.readouterr().out
    assert "assistant-v1" in listing
    run_id = listing.splitlines()[1].split()[0]
    assert main(["report", run_id, "--db", db]) == 0
    assert "weather-basic" in capsys.readouterr().out


def test_missing_run_is_error(tmp_path):
    assert main(["report", "run-nope", "--db", str(tmp_path / "runs.db")]) == 2
    assert main(["compare", "run-a", "run-b", "--db", str(tmp_path / "runs.db")]) == 2


def test_missing_scenarios_is_error(tmp_path):
    rc = main([
        "run", "--scenarios", str(tmp_path / "nope"), "--config", V1,
        "--db", str(tmp_path / "runs.db"),
    ])
    assert rc == 2

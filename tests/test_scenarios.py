import json

import pytest

from agent_eval_harness.scenarios import (
    ScenarioError,
    load_scenario_file,
    load_scenarios,
    scenario_from_dict,
)

VALID = {
    "id": "s1",
    "prompt": "What's the weather in Austin?",
    "checks": [{"type": "tool_called", "tool": "get_weather"}],
}


def test_valid_scenario_loads():
    s = scenario_from_dict(VALID)
    assert s.id == "s1"
    assert s.checks[0].type == "tool_called"
    assert s.max_steps == 8


def test_missing_id_rejected():
    with pytest.raises(ScenarioError, match="id"):
        scenario_from_dict({"prompt": "x", "checks": [{"type": "max_steps", "value": 1}]})


def test_missing_prompt_rejected():
    with pytest.raises(ScenarioError, match="prompt"):
        scenario_from_dict({"id": "x", "checks": [{"type": "max_steps", "value": 1}]})


def test_no_checks_rejected():
    with pytest.raises(ScenarioError, match="no checks"):
        scenario_from_dict({"id": "x", "prompt": "y", "checks": []})


def test_unknown_check_type_rejected():
    bad = {"id": "x", "prompt": "y", "checks": [{"type": "vibes", "level": 3}]}
    with pytest.raises(ScenarioError, match="unknown check type"):
        scenario_from_dict(bad)


def test_check_missing_required_field_rejected():
    bad = {"id": "x", "prompt": "y", "checks": [{"type": "tool_called"}]}
    with pytest.raises(ScenarioError, match="requires 'tool'"):
        scenario_from_dict(bad)


def test_max_steps_must_be_int():
    bad = {"id": "x", "prompt": "y", "checks": [{"type": "max_steps", "value": "four"}]}
    with pytest.raises(ScenarioError, match="integer"):
        scenario_from_dict(bad)


def test_load_yaml_file(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text(
        "id: y1\nprompt: hi\nchecks:\n  - type: answer_contains\n    text: hi\n"
    )
    scenarios = load_scenario_file(f)
    assert [s.id for s in scenarios] == ["y1"]


def test_load_json_file(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps(VALID))
    assert load_scenario_file(f)[0].id == "s1"


def test_load_multi_scenario_file(tmp_path):
    f = tmp_path / "pack.yaml"
    f.write_text(
        "scenarios:\n"
        "  - id: a\n    prompt: p1\n    checks: [{type: max_steps, value: 3}]\n"
        "  - id: b\n    prompt: p2\n    checks: [{type: max_steps, value: 3}]\n"
    )
    assert [s.id for s in load_scenario_file(f)] == ["a", "b"]


def test_load_directory_and_duplicate_detection(tmp_path):
    (tmp_path / "one.yaml").write_text(
        "id: dup\nprompt: p\nchecks: [{type: max_steps, value: 3}]\n"
    )
    (tmp_path / "two.yaml").write_text(
        "id: dup\nprompt: p\nchecks: [{type: max_steps, value: 3}]\n"
    )
    with pytest.raises(ScenarioError, match="duplicate"):
        load_scenarios(tmp_path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ScenarioError, match="not found"):
        load_scenario_file(tmp_path / "nope.yaml")


def test_bundled_scenario_pack_loads():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    scenarios = load_scenarios(root / "scenarios")
    assert len(scenarios) >= 8
    ids = {s.id for s in scenarios}
    assert "calendar-create-confirm" in ids
    assert "abstain-out-of-scope" in ids

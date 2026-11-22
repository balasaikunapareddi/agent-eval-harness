"""Eval scenarios: declarative, repeatable tests for agent behavior.

A scenario is one task ("What's the weather in Austin?") plus a list of
checks describing what a good run looks like. Scenarios live in YAML or
JSON files, one scenario per file or several under a top-level
``scenarios:`` list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Check types the harness knows how to evaluate.
CHECK_TYPES = {
    "tool_called",
    "tool_not_called",
    "answer_contains",
    "answer_matches",
    "answer_equals",
    "max_steps",
    "llm_judge",
}


class ScenarioError(ValueError):
    """Raised when a scenario file is malformed."""


@dataclass
class Check:
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.params}


@dataclass
class Scenario:
    id: str
    prompt: str
    checks: list[Check] = field(default_factory=list)
    name: str = ""
    tools: list[str] | None = None  # optional allowlist; None = all config tools
    max_steps: int = 8  # hard cap on agent loop iterations

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "tools": self.tools,
            "max_steps": self.max_steps,
            "checks": [c.to_dict() for c in self.checks],
        }


def _validate_check(raw: Any, where: str) -> Check:
    if not isinstance(raw, dict) or "type" not in raw:
        raise ScenarioError(f"{where}: each check must be a mapping with a 'type'")
    ctype = str(raw["type"])
    if ctype not in CHECK_TYPES:
        raise ScenarioError(
            f"{where}: unknown check type '{ctype}' (known: {sorted(CHECK_TYPES)})"
        )
    params = {k: v for k, v in raw.items() if k != "type"}
    # Per-type required fields, checked early so typos fail loudly.
    required: dict[str, list[str]] = {
        "tool_called": ["tool"],
        "tool_not_called": ["tool"],
        "answer_contains": ["text"],
        "answer_matches": ["pattern"],
        "answer_equals": ["text"],
        "max_steps": ["value"],
        "llm_judge": ["rubric"],
    }
    for key in required[ctype]:
        if key not in params:
            raise ScenarioError(f"{where}: check '{ctype}' requires '{key}'")
    if ctype == "max_steps" and not isinstance(params["value"], int):
        raise ScenarioError(f"{where}: max_steps 'value' must be an integer")
    return Check(type=ctype, params=params)


def scenario_from_dict(data: Any, where: str = "scenario") -> Scenario:
    if not isinstance(data, dict):
        raise ScenarioError(f"{where}: scenario must be a mapping")
    if "id" not in data:
        raise ScenarioError(f"{where}: scenario requires an 'id'")
    if "prompt" not in data:
        raise ScenarioError(f"{where}: scenario '{data['id']}' requires a 'prompt'")
    checks = [
        _validate_check(c, f"{where} [{data['id']}]")
        for c in (data.get("checks") or [])
    ]
    if not checks:
        raise ScenarioError(f"{where}: scenario '{data['id']}' has no checks")
    tools = data.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ScenarioError(f"{where}: 'tools' must be a list of tool names")
    return Scenario(
        id=str(data["id"]),
        prompt=str(data["prompt"]),
        checks=checks,
        name=str(data.get("name", "")),
        tools=[str(t) for t in tools] if tools is not None else None,
        max_steps=int(data.get("max_steps", 8)),
    )


def load_scenario_file(path: str | Path) -> list[Scenario]:
    """Load one file; returns one or more scenarios."""
    p = Path(path)
    if not p.exists():
        raise ScenarioError(f"scenario file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001
        raise ScenarioError(f"{p}: could not parse file: {exc}") from exc
    if isinstance(data, dict) and "scenarios" in data:
        return [scenario_from_dict(s, str(p)) for s in data["scenarios"]]
    if isinstance(data, list):
        return [scenario_from_dict(s, str(p)) for s in data]
    return [scenario_from_dict(data, str(p))]


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load scenarios from a file or a directory of .yaml/.yml/.json files."""
    p = Path(path)
    if p.is_dir():
        files = sorted(
            f for f in p.iterdir() if f.suffix.lower() in {".yaml", ".yml", ".json"}
        )
        if not files:
            raise ScenarioError(f"no scenario files (*.yaml/*.json) in {p}")
        scenarios: list[Scenario] = []
        for f in files:
            scenarios.extend(load_scenario_file(f))
    else:
        scenarios = load_scenario_file(p)
    seen: set[str] = set()
    for s in scenarios:
        if s.id in seen:
            raise ScenarioError(f"duplicate scenario id: '{s.id}'")
        seen.add(s.id)
    return scenarios

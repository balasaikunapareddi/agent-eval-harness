"""Agent configuration: the system prompt + tool schemas under test.

A config is the thing that changes in real life — someone edits a prompt
line or a tool description and behavior silently shifts. Every eval run
records a fingerprint (SHA-256) of the config so two runs can be compared
and you can prove exactly which config produced which behavior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ToolSpec:
    """One tool as the agent sees it (name, description, parameters)."""

    name: str
    description: str
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    write: bool = False  # write tools mutate state; agents often must confirm first

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "write": self.write,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolSpec":
        if "name" not in data or "description" not in data:
            raise ValueError(f"tool spec needs 'name' and 'description': {data!r}")
        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            parameters=dict(data.get("parameters") or {}),
            write=bool(data.get("write", False)),
        )


@dataclass
class AgentConfig:
    """Everything about an agent setup that an eval run exercises."""

    name: str
    system_prompt: str
    tools: list[ToolSpec] = field(default_factory=list)

    def tool(self, name: str) -> ToolSpec | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def canonical(self) -> dict[str, Any]:
        """Stable, order-independent representation used for fingerprinting."""
        return {
            "name": self.name,
            "system_prompt": self.system_prompt.strip(),
            "tools": sorted((t.to_dict() for t in self.tools), key=lambda d: d["name"]),
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def short_fingerprint(self) -> str:
        return self.fingerprint()[:12]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        if "system_prompt" not in data:
            raise ValueError("config needs a 'system_prompt'")
        return cls(
            name=str(data.get("name", "unnamed-config")),
            system_prompt=str(data["system_prompt"]),
            tools=[ToolSpec.from_dict(t) for t in (data.get("tools") or [])],
        )


def load_config(path: str | Path) -> AgentConfig:
    """Load an agent config from a YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"config file {p} must contain a mapping at the top level")
    return AgentConfig.from_dict(data)

"""agent-eval-harness: unit tests for agent configs.

Define repeatable scenarios for agents, skills, and MCP tools; run them;
compare runs before/after a change to catch regressions.
"""

from .compare import compare_runs
from .config import AgentConfig, ToolSpec, load_config
from .runner import run_scenarios
from .scenarios import Scenario, load_scenarios
from .store import RunStore

__version__ = "0.1.0"

__all__ = [
    "AgentConfig",
    "RunStore",
    "Scenario",
    "ToolSpec",
    "compare_runs",
    "load_config",
    "load_scenarios",
    "run_scenarios",
]

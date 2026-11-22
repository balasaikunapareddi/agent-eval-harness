"""Mock MCP-style tool implementations used by the reference agent.

These stand in for real MCP servers: each tool has a name, takes JSON-ish
arguments, and returns a JSON-ish result. Data is canned and deterministic
so mock-mode evals are fully reproducible offline.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

ToolFn = Callable[..., dict[str, Any]]

# ---------------------------------------------------------------------------
# Canned data
# ---------------------------------------------------------------------------

_WEATHER: dict[str, dict[str, Any]] = {
    "austin": {"temp_c": 31, "condition": "sunny", "tomorrow": "sunny, high 33C"},
    "seattle": {"temp_c": 14, "condition": "light rain", "tomorrow": "rain, high 15C"},
    "chicago": {"temp_c": 9, "condition": "cloudy", "tomorrow": "rain likely, high 11C"},
    "denver": {"temp_c": 18, "condition": "clear", "tomorrow": "clear, high 21C"},
    "new york": {"temp_c": 16, "condition": "partly cloudy", "tomorrow": "sunny, high 19C"},
    "san francisco": {"temp_c": 17, "condition": "foggy", "tomorrow": "fog clearing, high 19C"},
}

_CALENDAR: dict[str, list[dict[str, str]]] = {
    "2026-11-23": [
        {"time": "09:00", "title": "Team standup"},
        {"time": "11:30", "title": "Design review with Priya"},
        {"time": "15:00", "title": "Sprint planning"},
    ],
    "2026-11-24": [
        {"time": "10:00", "title": "1:1 with manager"},
    ],
    "2026-11-25": [],
}

_FILES: list[dict[str, str]] = [
    {"name": "q3-budget-review.pdf", "summary": "Q3 budget review and variance notes"},
    {"name": "q3-budget-raw.xlsx", "summary": "Raw Q3 budget numbers by department"},
    {"name": "onboarding-checklist.md", "summary": "New hire onboarding checklist"},
    {"name": "architecture-overview.md", "summary": "System architecture overview"},
    {"name": "release-notes-v2.md", "summary": "Release notes for version 2"},
]

# In-memory "created events" so write tools behave like stateful systems.
_created_events: list[dict[str, Any]] = []


def reset_state() -> None:
    """Clear mutable tool state (called between scenario runs)."""
    _created_events.clear()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def get_weather(city: str, units: str = "metric") -> dict[str, Any]:
    key = str(city).strip().lower()
    data = _WEATHER.get(key)
    if data is None:
        return {"error": f"no weather data for city '{city}'"}
    temp = data["temp_c"]
    if units == "imperial":
        temp = round(temp * 9 / 5 + 32, 1)
        unit_label = "F"
    else:
        unit_label = "C"
    return {
        "city": city,
        "temperature": temp,
        "unit": unit_label,
        "condition": data["condition"],
        "tomorrow": data["tomorrow"],
    }


def get_calendar_events(date: str) -> dict[str, Any]:
    events = _CALENDAR.get(str(date), [])
    created = [e for e in _created_events if e["date"] == str(date)]
    return {"date": date, "events": events + created}


def create_calendar_event(title: str, date: str, time: str) -> dict[str, Any]:
    event = {"title": title, "date": date, "time": time, "id": f"evt-{len(_created_events) + 1}"}
    _created_events.append(event)
    return {"created": True, "event": event}


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> dict[str, Any]:
    expr = str(expression).replace("^", "**").replace("x", "*").replace("×", "*")
    try:
        value = _safe_eval(ast.parse(expr, mode="eval"))
    except Exception as exc:  # noqa: BLE001 - surface as tool error, like a real MCP tool
        return {"error": f"could not evaluate '{expression}': {exc}"}
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"expression": expression, "result": value}


def search_files(query: str) -> dict[str, Any]:
    q = str(query).lower()
    hits = [
        f for f in _FILES if q in f["name"].lower() or q in f["summary"].lower()
    ]
    return {"query": query, "matches": hits}


#: Registry of tool implementations by name. Configs reference tools by name;
#: the runner binds the config's schemas to these implementations.
TOOL_IMPLEMENTATIONS: dict[str, ToolFn] = {
    "get_weather": get_weather,
    "get_calendar_events": get_calendar_events,
    "create_calendar_event": create_calendar_event,
    "calculate": calculate,
    "search_files": search_files,
}

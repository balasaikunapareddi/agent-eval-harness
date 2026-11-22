"""The reference agent under test, plus its LLM backends.

The agent is a small tool-calling loop driven by a JSON action protocol:
the backend returns either a tool call or a final answer. Two backends:

- ``MockBackend`` — deterministic, offline. It routes to tools by matching
  the request against tool *descriptions* and follows rules stated in the
  system prompt (e.g. confirm-before-write). Because its behavior is
  derived from the config, config edits change its behavior — which is
  exactly what makes config regressions demonstrable without an API key.
- ``LiteLLMBackend`` — a real model via LiteLLM. Model string comes from
  the ``LLM_MODEL`` env var (default ``gemini/gemini-3.5-flash-lite``);
  the provider key comes from that provider's standard env var
  (``GEMINI_API_KEY``, ``GROQ_API_KEY``, ``OPENROUTER_API_KEY``, ...).
  Transient errors are retried with backoff.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .checks import RunOutcome, ToolCall
from .config import AgentConfig, ToolSpec
from .tools import TOOL_IMPLEMENTATIONS, reset_state

DEFAULT_MODEL = "gemini/gemini-3.5-flash-lite"

_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "of", "for", "to", "is", "are", "what",
    "whats", "will", "it", "i", "me", "my", "do", "does", "can", "you", "please",
    "and", "or", "with", "this", "that", "have", "has", "get", "give", "show",
    "tell", "how", "much", "many", "s",
}

_KNOWN_CITIES = ["new york", "san francisco", "austin", "seattle", "chicago", "denver"]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOPWORDS}


@dataclass
class Action:
    kind: str  # "tool_call" | "final"
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    answer: str = ""


@dataclass
class AgentState:
    system_prompt: str
    tools: list[ToolSpec]
    user_prompt: str
    history: list[dict[str, Any]] = field(default_factory=list)  # observations
    tool_calls: list[ToolCall] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mock backend — deterministic, config-sensitive
# ---------------------------------------------------------------------------


class MockBackend:
    """Rule-based stand-in for an LLM.

    Tool selection scores each tool by overlap between the request and the
    tool's *description* (distinctive words count double), plus small
    intent boosts (dates, math patterns, write verbs). If a description is
    edited so it no longer describes the capability, routing breaks — the
    same failure mode real agents show after a schema edit.
    """

    name = "mock"

    # -- routing ---------------------------------------------------------

    def _score_tools(self, clause: str, tools: list[ToolSpec]) -> list[tuple[int, ToolSpec]]:
        prompt_tokens = _tokens(clause)
        desc_tokens = {t.name: _tokens(t.description) for t in tools}
        scored: list[tuple[int, ToolSpec]] = []
        for tool in tools:
            overlap = prompt_tokens & desc_tokens[tool.name]
            score = 0
            for tok in overlap:
                distinctive = sum(1 for dt in desc_tokens.values() if tok in dt) == 1
                score += 2 if distinctive else 1
            # Intent boosts, keyed on description content so config edits matter.
            if re.search(r"\d{4}-\d{2}-\d{2}", clause) and "calendar" in desc_tokens[tool.name]:
                score += 3
            if self._math_expression(clause) and (
                "arithmetic" in desc_tokens[tool.name] or "calculate" in desc_tokens[tool.name]
            ):
                score += 3
            scored.append((score, tool))
        scored.sort(key=lambda s: s[0], reverse=True)
        return scored

    def _pick_tool(self, clause: str, tools: list[ToolSpec]) -> ToolSpec | None:
        scored = self._score_tools(clause, tools)
        if not scored or scored[0][0] < 2:
            return None
        top_score = scored[0][0]
        tied = [t for s, t in scored if s == top_score]
        if len(tied) == 1:
            return tied[0]
        # Tie-break: write verbs prefer write tools; otherwise prefer read tools.
        wants_write = bool(re.search(r"\b(schedule|book|create|add|set up|make)\b", clause.lower()))
        for t in tied:
            if t.write == wants_write:
                return t
        return tied[0]

    # -- argument extraction --------------------------------------------

    @staticmethod
    def _math_expression(text: str) -> str | None:
        pct = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", text)
        if pct:
            return f"{pct.group(2)} * {pct.group(1)} / 100"
        m = re.search(r"(-?\d+(?:\.\d+)?\s*(?:[+\-*/x×^]\s*-?\d+(?:\.\d+)?\s*)+)", text)
        if m:
            return m.group(1).strip()
        return None

    def _extract_args(self, tool: ToolSpec, clause: str) -> dict[str, Any]:
        args: dict[str, Any] = {}
        lower = clause.lower()
        for param in tool.parameters:
            if param == "city":
                for city in _KNOWN_CITIES:
                    if city in lower:
                        args["city"] = city.title()
                        break
            elif param == "units":
                if "fahrenheit" in lower or "imperial" in lower:
                    args["units"] = "imperial"
                elif "celsius" in lower or "metric" in lower:
                    args["units"] = "metric"
            elif param == "date":
                m = re.search(r"\d{4}-\d{2}-\d{2}", clause)
                if m:
                    args["date"] = m.group(0)
            elif param == "time":
                m = re.search(r"\b(\d{1,2}:\d{2})\b", clause)
                if m:
                    args["time"] = m.group(1)
            elif param == "title":
                m = re.search(
                    r"(?:schedule|book|create|add)\s+(?:a|an|the)?\s*(.+?)\s+on\s+\d{4}-\d{2}-\d{2}",
                    clause,
                    re.IGNORECASE,
                )
                args["title"] = m.group(1).strip() if m else "New event"
            elif param == "expression":
                expr = self._math_expression(clause)
                if expr:
                    args["expression"] = expr
            elif param == "query":
                m = re.search(r'"([^"]+)"', clause)
                if m:
                    args["query"] = m.group(1)
                else:
                    m = re.search(r"\babout\s+(?:the\s+)?(.+)$", clause, re.IGNORECASE)
                    args["query"] = m.group(1).strip(" ?.") if m else clause.strip(" ?.")
        return args

    # -- planning ---------------------------------------------------------

    def _plan(self, state: AgentState) -> list[tuple[ToolSpec, dict[str, Any]]]:
        clauses = re.split(r",?\s+and\s+", state.user_prompt)
        plan: list[tuple[ToolSpec, dict[str, Any]]] = []
        for clause in clauses:
            tool = self._pick_tool(clause, state.tools)
            if tool is not None:
                plan.append((tool, self._extract_args(tool, clause)))
        return plan

    @staticmethod
    def _confirmation_required(state: AgentState, tool: ToolSpec) -> bool:
        if not tool.write:
            return False
        prompt = state.system_prompt.lower()
        return "confirm" in prompt and ("before" in prompt or "ask" in prompt)

    # -- final answer composition ----------------------------------------

    @staticmethod
    def _compose_answer(state: AgentState) -> str:
        parts: list[str] = []
        for call in state.tool_calls:
            r = call.result if isinstance(call.result, dict) else {}
            if call.error or r.get("error"):
                parts.append(f"I hit an error with {call.tool}: {call.error or r.get('error')}")
            elif call.tool == "get_weather":
                parts.append(
                    f"The weather in {r.get('city')} is {r.get('condition')}, "
                    f"{r.get('temperature')}°{r.get('unit')}. Tomorrow: {r.get('tomorrow')}."
                )
            elif call.tool == "get_calendar_events":
                events = r.get("events") or []
                if events:
                    listing = "; ".join(f"{e['time']} {e['title']}" for e in events)
                    parts.append(f"On {r.get('date')} you have: {listing}.")
                else:
                    parts.append(f"You have no events on {r.get('date')}.")
            elif call.tool == "create_calendar_event":
                ev = r.get("event") or {}
                parts.append(
                    f"Created '{ev.get('title')}' on {ev.get('date')} at {ev.get('time')}."
                )
            elif call.tool == "calculate":
                parts.append(f"{r.get('expression')} = {r.get('result')}.")
            elif call.tool == "search_files":
                names = [m["name"] for m in (r.get("matches") or [])]
                parts.append(
                    f"Found: {', '.join(names)}." if names else "No matching files found."
                )
            else:
                parts.append(json.dumps(r))
        return " ".join(parts) if parts else "I don't have a tool that can help with that."

    # -- backend interface -------------------------------------------------

    def next_action(self, state: AgentState) -> Action:
        plan = self._plan(state)
        done = len(state.tool_calls)
        if done < len(plan):
            tool, args = plan[done]
            if self._confirmation_required(state, tool):
                return Action(
                    kind="final",
                    answer=(
                        f"Shall I create '{args.get('title', 'the event')}' on "
                        f"{args.get('date', '')} at {args.get('time', '')}? Please confirm."
                    ),
                )
            return Action(kind="tool_call", tool=tool.name, args=args)
        if not plan and not state.tool_calls:
            return Action(
                kind="final",
                answer="I don't have a tool that can help with that request.",
            )
        return Action(kind="final", answer=self._compose_answer(state))


# ---------------------------------------------------------------------------
# Live backend — LiteLLM (provider-agnostic)
# ---------------------------------------------------------------------------


class LiteLLMBackend:
    """Real-model backend via LiteLLM.

    Model: ``LLM_MODEL`` env var (default ``gemini/gemini-3.5-flash-lite``).
    The API key is read by LiteLLM from the provider's standard env var
    (``GEMINI_API_KEY``, ``GROQ_API_KEY``, ``OPENROUTER_API_KEY``, ...).
    Keys are never read or printed by this code directly.
    """

    name = "live"

    def __init__(self, model: str | None = None, max_retries: int = 5):
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- low-level completion with retry ----------------------------------

    def _complete(self, messages: list[dict[str, str]]) -> str:
        import litellm
        from litellm.exceptions import (
            APIConnectionError,
            APIError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout,
        )

        transient = (RateLimitError, ServiceUnavailableError, Timeout, APIConnectionError, APIError)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(model=self.model, messages=messages)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                    self.total_tokens += getattr(usage, "total_tokens", 0) or 0
                return response.choices[0].message.content or ""
            except transient as exc:  # noqa: PERF203 - retry loop is the point
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(30.0, 2.0 * (2**attempt)))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_exc}")

    # -- backend interface -------------------------------------------------

    def next_action(self, state: AgentState) -> Action:
        tools_desc = "\n".join(
            f"- {t.name}({', '.join(t.parameters) or 'no args'}): {t.description}"
            for t in state.tools
        )
        observations = "\n".join(
            f"Tool {c.tool} args={json.dumps(c.args)} -> {json.dumps(c.result, default=str)}"
            for c in state.tool_calls
        ) or "(none yet)"
        prompt = (
            f"{state.system_prompt}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"User request: {state.user_prompt}\n\n"
            f"Tool results so far:\n{observations}\n\n"
            "Reply with exactly ONE JSON object, nothing else. Either:\n"
            '{"action": "tool_call", "tool": "<tool name>", "args": {...}}\n'
            "or, when you have everything you need:\n"
            '{"action": "final", "answer": "<your answer to the user>"}'
        )
        raw = self._complete([{"role": "user", "content": prompt}])
        action = _parse_action(raw)
        if action is None:
            raw = self._complete(
                [{"role": "user", "content": prompt + "\n\nYour last reply was not valid JSON. Reply with ONE JSON object only."}]
            )
            action = _parse_action(raw)
        if action is None:
            return Action(kind="final", answer="I couldn't produce a valid response.")
        return action

    def judge(self, rubric: str, answer: str) -> tuple[float, str]:
        prompt = (
            "You are grading an AI assistant's answer against a rubric.\n"
            f"Rubric: {rubric}\n\nAnswer to grade:\n{answer}\n\n"
            'Reply with exactly ONE JSON object: {"score": <0.0-1.0>, "reason": "<short reason>"}'
        )
        raw = self._complete([{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                return float(data.get("score", 0.0)), str(data.get("reason", ""))
            except (ValueError, TypeError):
                pass
        return 0.0, f"judge returned unparseable output: {raw[:120]!r}"


def _parse_action(raw: str) -> Action | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if data.get("action") == "tool_call" and data.get("tool"):
        return Action(kind="tool_call", tool=str(data["tool"]), args=dict(data.get("args") or {}))
    if data.get("action") == "final":
        return Action(kind="final", answer=str(data.get("answer", "")))
    return None


# ---------------------------------------------------------------------------
# Reference agent loop
# ---------------------------------------------------------------------------


class ReferenceAgent:
    """A minimal tool-calling agent: backend decides, tools execute."""

    def __init__(self, config: AgentConfig, backend: MockBackend | LiteLLMBackend):
        self.config = config
        self.backend = backend

    def run(self, user_prompt: str, tool_allowlist: list[str] | None = None, max_steps: int = 8) -> RunOutcome:
        reset_state()
        tools = [
            t for t in self.config.tools
            if tool_allowlist is None or t.name in tool_allowlist
        ]
        state = AgentState(
            system_prompt=self.config.system_prompt,
            tools=tools,
            user_prompt=user_prompt,
        )
        transcript: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        answer = ""
        steps = 0
        for _ in range(max_steps):
            steps += 1
            action = self.backend.next_action(state)
            if action.kind == "final":
                answer = action.answer
                transcript.append({"role": "assistant", "content": answer})
                break
            call = self._execute(action, tools)
            state.tool_calls.append(call)
            transcript.append(
                {"role": "tool", "tool": call.tool, "args": call.args, "result": call.result, "error": call.error}
            )
        else:
            answer = answer or "I couldn't complete that within the step limit."
            transcript.append({"role": "assistant", "content": answer})
        return RunOutcome(
            answer=answer,
            tool_calls=state.tool_calls,
            steps=steps,
            transcript=transcript,
        )

    @staticmethod
    def _execute(action: Action, tools: list[ToolSpec]) -> ToolCall:
        assert action.tool is not None
        if action.tool not in {t.name for t in tools}:
            return ToolCall(tool=action.tool, args=action.args, error=f"unknown tool '{action.tool}'")
        fn = TOOL_IMPLEMENTATIONS.get(action.tool)
        if fn is None:
            return ToolCall(tool=action.tool, args=action.args, error=f"no implementation for '{action.tool}'")
        try:
            result = fn(**action.args)
            return ToolCall(tool=action.tool, args=action.args, result=result)
        except Exception as exc:  # noqa: BLE001 - tool errors are data, not crashes
            return ToolCall(tool=action.tool, args=action.args, error=str(exc))

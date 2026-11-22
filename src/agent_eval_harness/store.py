"""Run store: every eval run persisted in SQLite.

A run records when it happened, which config produced it (name +
fingerprint), which mode/model ran it, and the full per-scenario results
(checks, transcript, tool calls, latency, tokens). That history is what
makes before/after comparison possible.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = "eval_runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    config_name TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    mode TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    scenario_count INTEGER NOT NULL,
    passed_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    scenario_id TEXT NOT NULL,
    passed INTEGER NOT NULL,
    checks_json TEXT NOT NULL,
    transcript_json TEXT NOT NULL,
    tool_calls_json TEXT NOT NULL,
    tool_sequence_json TEXT NOT NULL,
    answer TEXT NOT NULL,
    steps INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    tokens_json TEXT NOT NULL,
    PRIMARY KEY (run_id, scenario_id)
);
"""


@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    checks: list[dict[str, Any]]
    transcript: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_sequence: list[str]
    answer: str
    steps: int
    latency_ms: float
    tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    label: str
    config_name: str
    config_fingerprint: str
    mode: str
    model: str
    scenario_count: int
    passed_count: int
    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.scenario_count if self.scenario_count else 0.0

    def result_for(self, scenario_id: str) -> ScenarioResult | None:
        for r in self.results:
            if r.scenario_id == scenario_id:
                return r
        return None


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}-{secrets.token_hex(3)}"


class RunStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes -----------------------------------------------------------

    def save_run(self, run: RunRecord) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run.run_id,
                    run.created_at,
                    run.label,
                    run.config_name,
                    run.config_fingerprint,
                    run.mode,
                    run.model,
                    run.scenario_count,
                    run.passed_count,
                ),
            )
            self._conn.executemany(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        run.run_id,
                        r.scenario_id,
                        int(r.passed),
                        json.dumps(r.checks),
                        json.dumps(r.transcript),
                        json.dumps(r.tool_calls),
                        json.dumps(r.tool_sequence),
                        r.answer,
                        r.steps,
                        r.latency_ms,
                        json.dumps(r.tokens),
                    )
                    for r in run.results
                ],
            )

    # -- reads ------------------------------------------------------------

    def list_runs(self) -> list[RunRecord]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC"
        ).fetchall()
        return [self._run_from_row(r, with_results=False) for r in rows]

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._run_from_row(row, with_results=True)

    def latest_run(self) -> RunRecord | None:
        runs = self.list_runs()
        return self.get_run(runs[0].run_id) if runs else None

    def _run_from_row(self, row: sqlite3.Row, with_results: bool) -> RunRecord:
        run = RunRecord(
            run_id=row["run_id"],
            created_at=row["created_at"],
            label=row["label"],
            config_name=row["config_name"],
            config_fingerprint=row["config_fingerprint"],
            mode=row["mode"],
            model=row["model"],
            scenario_count=row["scenario_count"],
            passed_count=row["passed_count"],
        )
        if with_results:
            rows = self._conn.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY scenario_id",
                (run.run_id,),
            ).fetchall()
            run.results = [
                ScenarioResult(
                    scenario_id=r["scenario_id"],
                    passed=bool(r["passed"]),
                    checks=json.loads(r["checks_json"]),
                    transcript=json.loads(r["transcript_json"]),
                    tool_calls=json.loads(r["tool_calls_json"]),
                    tool_sequence=json.loads(r["tool_sequence_json"]),
                    answer=r["answer"],
                    steps=r["steps"],
                    latency_ms=r["latency_ms"],
                    tokens=json.loads(r["tokens_json"]),
                )
                for r in rows
            ]
        return run

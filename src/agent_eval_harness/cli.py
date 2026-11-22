"""Command-line interface.

    agent-eval run --scenarios scenarios/ --config configs/v1.yaml
    agent-eval run --scenarios scenarios/ --config configs/v2.yaml \
        --baseline <run-id> --fail-on-regression
    agent-eval compare <run-a> <run-b>
    agent-eval list-runs
    agent-eval report <run-id>
    agent-eval demo

Mock mode is the default (offline, deterministic). Add ``--live`` to run
against a real model via LiteLLM (``LLM_MODEL`` env var, provider key from
the provider's standard env var).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compare import compare_runs
from .config import load_config
from .report import render_diff, render_run, render_run_list
from .runner import make_backend, run_scenarios
from .scenarios import ScenarioError, load_scenarios
from .store import DEFAULT_DB, RunStore

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS = _PACKAGE_ROOT / "scenarios"
DEFAULT_CONFIG_V1 = _PACKAGE_ROOT / "configs" / "v1.yaml"
DEFAULT_CONFIG_V2 = _PACKAGE_ROOT / "configs" / "v2.yaml"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=DEFAULT_DB, help=f"run store path (default {DEFAULT_DB})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Repeatable evals for AI agents, skills, and MCP tools — unit tests for agent configs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run scenarios against a config")
    p_run.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS), help="scenario file or directory")
    p_run.add_argument("--config", required=True, help="agent config YAML/JSON")
    p_run.add_argument("--live", action="store_true", help="use a real model via LiteLLM (LLM_MODEL env)")
    p_run.add_argument("--model", default=None, help="override LLM_MODEL for this run")
    p_run.add_argument("--label", default="", help="free-text label stored with the run")
    p_run.add_argument("--baseline", default=None, help="run id to compare against after running")
    p_run.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit non-zero if the new run regresses vs --baseline (CI gate)",
    )
    p_run.add_argument("--verbose", action="store_true", help="show all check details")
    _add_common(p_run)

    p_cmp = sub.add_parser("compare", help="diff two stored runs")
    p_cmp.add_argument("before", help="baseline run id")
    p_cmp.add_argument("after", help="new run id")
    p_cmp.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit non-zero if regressions are found",
    )
    _add_common(p_cmp)

    p_list = sub.add_parser("list-runs", help="list stored runs")
    _add_common(p_list)

    p_rep = sub.add_parser("report", help="show one stored run in detail")
    p_rep.add_argument("run_id")
    p_rep.add_argument("--verbose", action="store_true", help="show answers and tool sequences")
    _add_common(p_rep)

    p_demo = sub.add_parser("demo", help="run the bundled v1-vs-v2 regression demo")
    p_demo.add_argument("--live", action="store_true", help="run the demo against a real model")
    p_demo.add_argument("--model", default=None, help="override LLM_MODEL for this run")
    _add_common(p_demo)

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.scenarios)
    config = load_config(args.config)
    backend = make_backend(live=args.live, model=args.model)
    with RunStore(args.db) as store:
        run = run_scenarios(scenarios, config, backend, store=store, label=args.label)
        print(render_run(run, verbose=args.verbose))
        if args.baseline:
            baseline = store.get_run(args.baseline)
            if baseline is None:
                print(f"\nBaseline run not found: {args.baseline}", file=sys.stderr)
                return 2
            diff = compare_runs(baseline, run)
            print()
            print(render_diff(diff))
            if args.fail_on_regression and diff.has_regressions:
                return 1
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    with RunStore(args.db) as store:
        before = store.get_run(args.before)
        after = store.get_run(args.after)
    if before is None or after is None:
        missing = args.before if before is None else args.after
        print(f"Run not found: {missing}", file=sys.stderr)
        return 2
    diff = compare_runs(before, after)
    print(render_diff(diff))
    if args.fail_on_regression and diff.has_regressions:
        return 1
    return 0


def _cmd_list_runs(args: argparse.Namespace) -> int:
    with RunStore(args.db) as store:
        print(render_run_list(store.list_runs()))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    with RunStore(args.db) as store:
        run = store.get_run(args.run_id)
    if run is None:
        print(f"Run not found: {args.run_id}", file=sys.stderr)
        return 2
    print(render_run(run, verbose=args.verbose))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(DEFAULT_SCENARIOS)
    config_v1 = load_config(DEFAULT_CONFIG_V1)
    config_v2 = load_config(DEFAULT_CONFIG_V2)
    with RunStore(args.db) as store:
        print("== Demo: same scenarios, two configs ==")
        print("v1 = known-good config. v2 = v1 after a 'small' edit: the weather")
        print("tool's description was rewritten and the confirm-before-write rule")
        print("was dropped from the system prompt.\n")
        run_v1 = run_scenarios(
            scenarios, config_v1, make_backend(args.live, args.model),
            store=store, label="demo v1 (baseline)",
        )
        print(render_run(run_v1))
        print()
        run_v2 = run_scenarios(
            scenarios, config_v2, make_backend(args.live, args.model),
            store=store, label="demo v2 (after edit)",
        )
        print(render_run(run_v2))
        print()
        print(render_diff(compare_runs(run_v1, run_v2)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "list-runs":
            return _cmd_list_runs(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "demo":
            return _cmd_demo(args)
    except (ScenarioError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

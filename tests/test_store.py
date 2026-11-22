from agent_eval_harness.store import RunRecord, RunStore, ScenarioResult


def make_run(run_id="run-1", passed=True, tools=None, answer="ok"):
    return RunRecord(
        run_id=run_id,
        created_at="2026-11-22T04:00:00+00:00",
        label="test",
        config_name="cfg",
        config_fingerprint="abc123",
        mode="mock",
        model="",
        scenario_count=1,
        passed_count=1 if passed else 0,
        results=[
            ScenarioResult(
                scenario_id="s1",
                passed=passed,
                checks=[{"check": {"type": "max_steps", "value": 4}, "status": "pass", "detail": "ok"}],
                transcript=[{"role": "user", "content": "hi"}],
                tool_calls=[{"tool": "get_weather", "args": {"city": "Austin"}, "result": {}, "error": None}],
                tool_sequence=tools or ["get_weather"],
                answer=answer,
                steps=2,
                latency_ms=1.5,
                tokens={},
            )
        ],
    )


def test_save_and_get_round_trip(tmp_path):
    with RunStore(tmp_path / "runs.db") as store:
        store.save_run(make_run())
        got = store.get_run("run-1")
    assert got is not None
    assert got.config_name == "cfg"
    assert got.results[0].scenario_id == "s1"
    assert got.results[0].tool_sequence == ["get_weather"]
    assert got.results[0].answer == "ok"
    assert got.pass_rate == 1.0


def test_get_missing_run_returns_none(tmp_path):
    with RunStore(tmp_path / "runs.db") as store:
        assert store.get_run("nope") is None
        assert store.latest_run() is None


def test_list_runs_newest_first(tmp_path):
    with RunStore(tmp_path / "runs.db") as store:
        older = make_run("run-old")
        older.created_at = "2026-11-20T00:00:00+00:00"
        newer = make_run("run-new")
        newer.created_at = "2026-11-22T00:00:00+00:00"
        store.save_run(older)
        store.save_run(newer)
        ids = [r.run_id for r in store.list_runs()]
        assert ids == ["run-new", "run-old"]
        assert store.latest_run().run_id == "run-new"
        # list view omits results; get_run includes them
        assert store.list_runs()[0].results == []
        assert store.get_run("run-new").results


def test_failed_run_persists(tmp_path):
    with RunStore(tmp_path / "runs.db") as store:
        store.save_run(make_run(passed=False))
        got = store.get_run("run-1")
    assert got.passed_count == 0
    assert got.results[0].passed is False

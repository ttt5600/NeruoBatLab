from datetime import datetime, timezone

import pytest

from mbot.ledger import AlreadyRunning, Ledger, RunLock, SpendCapExceeded, SpendGuard


def test_dedupe_and_torn_line(tmp_path):
    led = Ledger(tmp_path / "f.jsonl")
    led.append({"kind": "submitted", "question_id": 11, "post_id": 1})
    led.append({"kind": "forecast", "question_id": 12})
    led.append({"kind": "submit_error", "question_id": 13})
    with open(led.path, "a") as f:
        f.write('{"kind": "submitted", "question_id": 14')      # torn line after a crash
    assert led.submitted_question_ids() == {11}


def test_spend_cap_from_log_month_scoped(tmp_path):
    led = Ledger(tmp_path / "f.jsonl")
    with open(led.path, "w") as f:   # rows from last month do not count
        f.write('{"ts": "2026-10-31T23:59:00+00:00", "kind": "llm_call", "cost_usd": 40.0}\n')
        f.write('{"ts": "2026-11-02T10:00:00+00:00", "kind": "llm_call", "cost_usd": 30.0}\n')
        f.write('{"ts": "2026-11-03T10:00:00+00:00", "kind": "llm_error", "cost_usd": 5.0}\n')
        f.write('{"ts": "2026-11-03T10:00:00+00:00", "kind": "forecast", "cost_usd": 999.0}\n')   # not a spend row
    now = lambda: datetime(2026, 11, 20, tzinfo=timezone.utc)  # noqa: E731
    assert led.spent_month(now()) == pytest.approx(35.0)
    g = SpendGuard(led, 50.0, now_fn=now)
    g.check(14.99)
    with pytest.raises(SpendCapExceeded):
        g.check(15.01)
    g.record({"kind": "llm_call", "cost_usd": 10.0})
    assert g.spent == pytest.approx(45.0)
    with pytest.raises(SpendCapExceeded):
        g.check(5.01)
    # a fresh guard (new process) reads the same spend back from the log
    assert SpendGuard(led, 50.0, now_fn=now).spent == pytest.approx(45.0)


def test_month_rollover_resets(tmp_path):
    t = {"now": datetime(2026, 11, 30, 23, 0, tzinfo=timezone.utc)}
    clock = lambda: t["now"]  # noqa: E731
    led = Ledger(tmp_path / "f.jsonl", now_fn=clock)
    g = SpendGuard(led, 50.0, now_fn=clock)
    g.record({"kind": "llm_call", "cost_usd": 49.0})
    with pytest.raises(SpendCapExceeded):
        g.check(2.0)
    t["now"] = datetime(2026, 12, 1, 0, 30, tzinfo=timezone.utc)
    assert g.spent == 0.0
    g.check(2.0)                      # new month, fresh budget


def test_budget_fallback_switches_to_cheaper_models(cfg):
    from mbot.forecaster import pick_models
    cfg.provider_spec, cfg.n_samples = "anthropic:claude-opus-5-5", 5
    assert pick_models(cfg, 5.0) == ("anthropic:claude-opus-5-5", 5, None)          # no fallback configured
    cfg.fallback_spec, cfg.fallback_below_usd, cfg.fallback_n_samples = "anthropic:claude-haiku-5-5", 10.0, 3
    assert pick_models(cfg, 25.0)[:2] == ("anthropic:claude-opus-5-5", 5)
    spec, n, note = pick_models(cfg, 9.99)
    assert (spec, n) == ("anthropic:claude-haiku-5-5", 3) and "budget fallback" in note


def test_run_lock_is_exclusive(tmp_path):
    p = tmp_path / "run.lock"
    with RunLock(p):
        with pytest.raises(AlreadyRunning):
            with RunLock(p):
                pass
    with RunLock(p):       # released after exit
        pass

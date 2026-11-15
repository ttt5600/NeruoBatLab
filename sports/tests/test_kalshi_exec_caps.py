"""ops_e4_prep (2026-11-15): X1 restart at minimum size (KALSHI_X1_FRAC / KALSHI_X1_MAX_CONTRACTS / X1 kill) and
E4 price-band caps (KALSHI_PRICE_BAND_CAPS) in kalshi_exec.py. Reuses the mocked exchange from test_kalshi_exec."""
from __future__ import annotations

from datetime import timedelta

import pytest

from test_kalshi_exec import (NOTES, NOW, T1, T2, _pick, kx, maker_setup, orders_of, root, run,  # noqa: F401
                              setup)


def test_new_caps_default_off():
    cfg = kx.config({"KALSHI_MAX_CONTRACTS": "24"})
    assert cfg["x1_max_contracts"] is None and cfg["price_band_caps"] == [] and cfg["x1_kill_since"] is None
    assert cfg["x1_frac"] == kx.TIER_FRAC["x1"]
    assert kx.contract_cap(cfg, "x1", 0.95, "live") == 24 and kx.contract_cap(cfg, "orig7", 0.95, "live") == 24
    assert kx.contract_cap(cfg, "x1", 0.95, "dry") == 10 ** 9


def test_parse_band_caps_and_lookup():
    assert kx.parse_band_caps("0:19, 0.97:24") == [(0.97, 24), (0.0, 19)]
    cfg = kx.config({"KALSHI_MAX_CONTRACTS": "24", "KALSHI_PRICE_BAND_CAPS": "0.97:24,0:19"})
    assert kx.contract_cap(cfg, "orig7", 0.98, "live") == 24
    assert kx.contract_cap(cfg, "orig7", 0.97, "live") == 24
    assert kx.contract_cap(cfg, "orig7", 0.96, "live") == 19
    assert kx.contract_cap(cfg, "orig7", 0.85, "live") == 19
    only_hi = kx.config({"KALSHI_MAX_CONTRACTS": "24", "KALSHI_PRICE_BAND_CAPS": "0.97:10"})
    assert kx.contract_cap(only_hi, "orig7", 0.96, "live") == 24     # below every floor -> bands do not bind
    for bad in ("1.2:5", "0.9:-1", "0.9"):
        with pytest.raises(ValueError):
            kx.parse_band_caps(bad)


def test_x1_max_contracts_binds_only_x1(root):
    picks = [_pick(T1, "KXHIGHNY", "orig7"), _pick(T2, "KXHIGHTBOS", "x1")]
    ex = maker_setup(root, picks, env_lines=["KALSHI_MAX_CONTRACTS=24", "KALSHI_ORIG7_FRAC=0.03",
                                             "KALSHI_X1_FRAC=0.03", "KALSHI_X1_MAX_CONTRACTS=6"])
    run(root, ex)
    st = orders_of(root)
    assert st[T1]["count"] == 24 and "ct_cap" not in st[T1]              # orig7: $30 -> 31 ct, capped at 24
    assert st[T2]["count"] == 6 and st[T2]["ct_cap"] == 6
    assert {p[3]["ticker"]: p[3]["count"] for p in ex.posts()} == {T1: "24", T2: "6"}


def test_x1_frac_sizes_x1_dollars(root):
    ex = maker_setup(root, [_pick(T2, "KXHIGHTBOS", "x1")],
                     env_lines=["KALSHI_MAX_CONTRACTS=24", "KALSHI_X1_FRAC=0.006"])   # $6 at q .94 -> 6 ct
    run(root, ex)
    assert orders_of(root)[T2]["count"] == 6
    assert kx.caps(kx.config({"KALSHI_BANKROLL_USD": "400", "KALSHI_X1_FRAC": "0.015"}), "x1")["order"] == 6.0


def test_x1_paused_cohort_still_skips(root):
    ex = maker_setup(root, [_pick(T2, "KXHIGHTBOS", "x1")],
                     env_lines=["KALSHI_COHORTS=orig7", "KALSHI_X1_MAX_CONTRACTS=6"])
    run(root, ex)
    assert orders_of(root)[T2]["status"] == "skip_cohort_paused" and not ex.posts()


def test_price_band_caps_maker_q(root):
    # maker q = .94 (maker_setup book) -> band [0, .97) -> 19
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")],
                     env_lines=["KALSHI_MAX_CONTRACTS=24", "KALSHI_ORIG7_FRAC=0.03",
                                "KALSHI_PRICE_BAND_CAPS=0.97:24,0:19"])
    run(root, ex)
    o = orders_of(root)[T1]
    assert o["maker_q"] == 0.94 and o["count"] == 19 and o["ct_cap"] == 19
    assert ex.posts()[0][3]["count"] == "19"


def test_price_band_caps_high_price_keeps_global(root):
    p = _pick(T1, "KXHIGHNY", "orig7", no_price=0.99)
    ex = maker_setup(root, [p], env_lines=["KALSHI_MAX_CONTRACTS=24", "KALSHI_ORIG7_FRAC=0.03",
                                           "KALSHI_PRICE_BAND_CAPS=0.97:24,0:19"])
    ex.no_books[T1] = [["0.9700", "100.00"]]          # NO ask .99, bid .97 -> q .98
    run(root, ex)
    o = orders_of(root)[T1]
    assert o["maker_q"] == 0.98 and o["count"] == 24 and "ct_cap" not in o


def test_price_band_caps_taker_path(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")],
               env_lines=["KALSHI_MAX_CONTRACTS=24", "KALSHI_ORIG7_FRAC=0.03", "KALSHI_PRICE_BAND_CAPS=0.97:24,0:19"])
    run(root, ex)
    assert float(ex.posts()[0][3]["count"]) == 19     # taker ask .96 -> 19


def _x1_settled(n, losses, start=NOW - timedelta(days=10), mode="demo", cost=5.64, held_n=6, loss_idx=None):
    loss_idx = set(range(losses)) if loss_idx is None else set(loss_idx)
    for i in range(n):
        t = f"KXHIGHTBOS-26SEP{i:02d}-T{70 + i}"
        ts = (start + timedelta(hours=i)).isoformat(timespec="seconds")
        kx.append({"event": "order", "ts": ts, "mode": mode, "ticker": t, "cohort": "x1", "status": "filled",
                   "count": held_n, "filled": held_n, "cost": cost})
        r = "yes" if i in loss_idx else "no"
        kx.append({"event": "settle", "ts": ts, "mode": mode, "ticker": t, "result": r, "held": held_n,
                   "pnl": -cost if r == "yes" else round(held_n - cost, 4)})


def test_x1_kill_two_losses_in_first_30(root):
    picks = [_pick(T1, "KXHIGHNY", "orig7"), _pick(T2, "KXHIGHTBOS", "x1")]
    ex = maker_setup(root, picks, env_lines=["KALSHI_X1_MAX_CONTRACTS=6"])
    _x1_settled(10, losses=2)
    run(root, ex)
    st = orders_of(root)
    assert st[T2]["status"] == "skip_cohort_killed" and st[T1]["status"] == "resting"
    kills = [r for r in kx.journal() if r.get("event") == "cohort_kill"]
    assert len(kills) == 1 and "2 losses" in kills[0]["reason"]
    assert any("X1 cohort KILLED" in t for t, _ in NOTES)
    run(root, ex, now=NOW + timedelta(minutes=30))            # sticky: no second kill row
    assert len([r for r in kx.journal() if r.get("event") == "cohort_kill"]) == 1


def test_x1_kill_since_and_sticky(root):
    _x1_settled(10, losses=2)
    st = kx.fold(kx.journal(), "demo")
    assert kx.x1_kill_verdict(st, None) is not None
    later = (NOW - timedelta(days=1)).isoformat()
    assert kx.x1_kill_verdict(st, later) is None                       # fills before SINCE do not count
    assert kx.x1_kill_verdict(st, "2026-10-01") is not None            # date-only SINCE works
    kx.append({"event": "cohort_kill", "ts": NOW.isoformat(), "mode": "demo", "cohort": "x1", "since": None,
               "reason": "x"})
    assert kx.x1_killed(kx.journal(), "demo", None)
    assert not kx.x1_killed(kx.journal(), "demo", later) and not kx.x1_killed(kx.journal(), "live", None)


def test_x1_kill_one_loss_in_30_is_ok(root):
    _x1_settled(30, losses=1)
    assert kx.x1_kill_verdict(kx.fold(kx.journal(), "demo"), None) is None


def test_x1_kill_roi_rule_after_60(root):
    # 1 loss in the first 30, 3 more after; cost $5.94 for 6 ct -> win +.06, loss -5.94 -> ROI < 0
    _x1_settled(60, losses=0, cost=5.94, loss_idx=[0, 40, 41, 42])
    st = kx.fold(kx.journal(), "demo")
    v = kx.x1_kill_verdict(st, None)
    assert v is not None and "ROI" in v
    # same record at 59 fills: ROI rule not yet live
    kx.LOG.write_text("")
    _x1_settled(59, losses=0, cost=5.94, loss_idx=[0, 40, 41, 42])
    assert kx.x1_kill_verdict(kx.fold(kx.journal(), "demo"), None) is None


def test_x1_kill_not_evaluated_while_paused(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")], env_lines=["KALSHI_COHORTS=orig7"])
    _x1_settled(10, losses=3)
    run(root, ex)
    assert not any(r.get("event") == "cohort_kill" for r in kx.journal())


def test_x1_day_subcap_leaves_orig7_room(root):
    # 5 X1 picks first, then 1 orig7 pick; shared day cap $40; X1 sub-cap $12 -> 2 X1 orders of 6 ct (~$5.64 each),
    # the rest skip_x1_cap, and the orig7 pick still gets its full size from the remaining room
    x1 = [f"KXHIGHTBOS-26OCT22-T{75 + i}" for i in range(5)]
    picks = [_pick(t, "KXHIGHTBOS", "x1", floor=75.0 + i) for i, t in enumerate(x1)] + [_pick(T1, "KXHIGHNY", "orig7")]
    ex = maker_setup(root, picks, env_lines=["KALSHI_MAX_CONTRACTS=24", "KALSHI_MAX_DAILY_USD=40", "KALSHI_MAX_OPEN_USD=40",
                                             "KALSHI_ORIG7_FRAC=0.025", "KALSHI_X1_FRAC=0.006",
                                             "KALSHI_X1_MAX_CONTRACTS=6", "KALSHI_X1_DAY_USD=12"])
    run(root, ex)
    st = orders_of(root)
    assert [st[t]["status"] for t in x1] == ["resting", "resting", "skip_x1_cap", "skip_x1_cap", "skip_x1_cap"]
    assert sum(kx.committed(st[t]) for t in x1) <= 12 + 1e-9
    assert st[T1]["status"] == "resting" and st[T1]["count"] == 24     # $25 order cap -> 26 ct, capped at 24


def test_x1_day_subcap_counts_open_from_yesterday(root):
    old = "KXHIGHTBOS-26OCT21-T75"
    kx.append({"event": "order", "ts": (NOW - timedelta(hours=20)).isoformat(timespec="seconds"), "mode": "demo",
               "ticker": old, "cohort": "x1", "status": "filled", "count": 6, "filled": 6, "cost": 11.0})
    ex = maker_setup(root, [_pick(T2, "KXHIGHTBOS", "x1")],
                     env_lines=["KALSHI_X1_FRAC=0.006", "KALSHI_X1_MAX_CONTRACTS=6", "KALSHI_X1_DAY_USD=12"])
    run(root, ex)
    o = orders_of(root)[T2]
    assert o["status"] == "resting" and o["count"] == 1                 # $11 still open of $12 -> $1 room -> 1 ct @ .94
    kx.append({"event": "update", "ts": NOW.isoformat(), "mode": "demo", "ticker": old, "cost": 11.5})
    kx.LOG.write_text("".join(l + "\n" for l in kx.LOG.read_text().splitlines() if T2 not in l))
    run(root, ex)
    assert orders_of(root)[T2]["status"] == "skip_x1_cap"              # $0.50 room < one contract


def test_x1_day_subcap_off_by_default():
    assert kx.config({})["x1_day"] is None

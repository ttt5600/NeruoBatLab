"""Offline tests for the Q013 (RT/MC) and Q014 (Truth Social) forward paper loggers. No network."""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import q013_logger as A  # noqa: E402
import q014_logger as B  # noqa: E402

NY = ZoneInfo("America/New_York")


def test_yes_set_semantics():
    assert A.yes_set({"strike_type": "greater", "floor_strike": 82})(83)
    assert not A.yes_set({"strike_type": "greater", "floor_strike": 82})(82)        # "above 75" with 75 -> No
    assert A.yes_set({"strike_type": "greater_or_equal", "floor_strike": 360})(360)
    assert A.yes_set({"strike_type": "less", "cap_strike": 140})(139)
    assert not A.yes_set({"strike_type": "less", "cap_strike": 140})(140)
    p = A.yes_set({"strike_type": "between", "floor_strike": 340, "cap_strike": 359})
    assert p(340) and p(359) and not p(360)                                           # inclusive
    assert A.yes_set({"strike_type": "custom"}) is None


def test_rt_round_matches_live_examples():
    # observed on 2026-10-29: 44/52 -> 85, 46/50 -> 92, 143/151 -> 95, 96/104 -> 92, 88/91 -> 97, 3/12 -> 25
    for liked, n, shown in [(44, 52, 85), (46, 50, 92), (143, 151, 95), (96, 104, 92), (88, 91, 97), (3, 12, 25)]:
        assert A.rt_round(liked, n) == shown


def test_pmf_rt_is_distribution_and_centered():
    pm = A.pmf_rt(46, 50, "T24")
    assert abs(sum(pm.values()) - 1) < 1e-9
    mean = sum(v * p for v, p in pm.items())
    assert 88 <= mean <= 93
    assert A.p_yes(pm, A.yes_set({"strike_type": "greater", "floor_strike": 80})) > 0.999   # cannot fall 12 pts on 4 reviews
    assert A.remaining(50, "T24") == 4 and A.remaining(50, "T6") == 2 and A.remaining(10, "T6") == 2


def test_pmf_mc_spread_shrinks_with_n():
    a = A.pmf_mc(86, 10, 6.0, "T24")
    b = A.pmf_mc(86, 100, 6.0, "T24")
    sd = lambda pm: (sum(v * v * p for v, p in pm.items()) - sum(v * p for v, p in pm.items()) ** 2) ** 0.5  # noqa: E731
    assert abs(sum(a.values()) - 1) < 1e-9 and sd(a) > sd(b)
    assert A.pmf_mc(86, 45, None, "T6")  # sd fallback path


def test_decide_rule():
    d = A.decide(0.90, 0.70, 0.75)                 # 0.90 - 0.75 - fee(.75)=0.013 -> 0.137 >= .05 -> YES
    assert d["side"] == "yes" and d["price"] == 0.75 and d["cost"] == pytest.approx(0.75 + 0.07 * 0.75 * 0.25)
    d = A.decide(0.10, 0.30, 0.35)                 # NO at 0.70: 0.90 - 0.70 - fee -> YES side no
    assert d["side"] == "no" and d["price"] == pytest.approx(0.70)
    assert A.decide(0.50, 0.48, 0.52)["side"] is None
    assert A.decide(0.9, 0.0, 0.1)["side"] is None  # one-sided book: no bet
    assert A.decide(0.9, None, 0.1)["side"] is None


RULES_W = ("How this is resolved\n\nThe observation period begins at 12:00:00 AM ET on Oct 25, 2026 and ends at "
           "11:59:59 PM ET on Oct 31, 2026.\nThe primary resolution source")
RULES_D = "The observation period begins at 12:00:00 AM ET on Oct 29, 2026 and ends at 11:59:59 PM ET on Oct 29, 2026."


def test_window_parse_and_slots():
    ws, we = B.window({"rules_secondary": RULES_W})
    assert ws == int(datetime(2026, 10, 25, tzinfo=NY).timestamp())
    assert we == int(datetime(2026, 11, 1, tzinfo=NY).timestamp())
    s = B.slots_for("W", ws, we)
    assert len(s) == 7 and s[0][0] == "W_2026-10-25_12" and s[-1][0] == "W_2026-10-31_12"
    ws, we = B.window({"rules_secondary": RULES_D})
    assert we - ws == 86400
    assert [x[0] for x in B.slots_for("D", ws, we)] == ["D_2026-10-29_12", "D_2026-10-29_18"]
    # DST day (Nov 1 2026 is 25 h long in ET)
    ws = int(datetime(2026, 11, 1, tzinfo=NY).timestamp())
    we = int(datetime(2026, 11, 2, tzinfo=NY).timestamp())
    assert we - ws == 25 * 3600 and len(B.slots_for("D", ws, we)) == 2
    assert B.window({"rules_secondary": "nothing"}) is None


def _flat_model(per_hour=2.0, k=None):
    return {"rate_per_hour_et": [per_hour] * 24, "nb_k": k}


def test_remaining_dist_and_probability():
    t = int(datetime(2026, 10, 29, 12, tzinfo=NY).timestamp())
    we = int(datetime(2026, 10, 30, tzinfo=NY).timestamp())
    M, V = B.remaining_dist(_flat_model(2.0), t, we)
    assert M == pytest.approx(24.0) and V == pytest.approx(24.0)
    M2, V2 = B.remaining_dist(_flat_model(2.0, k=4.0), t, we)
    assert M2 == pytest.approx(24.0) and V2 == pytest.approx(24 + 24 * 24 / 4)
    ge = A.yes_set({"strike_type": "greater_or_equal", "floor_strike": 44})  # needs R >= 24 = M
    p_pois = B.p_yes_count(20, M, V, ge, nb=False)
    p_nb = B.p_yes_count(20, M2, V2, ge, nb=True)
    assert 0.3 < p_pois < 0.7 and 0.3 < p_nb < 0.7
    assert B.p_yes_count(45, M, V, ge) == pytest.approx(1.0)     # already over the line
    lt = A.yes_set({"strike_type": "less", "cap_strike": 5})
    assert B.p_yes_count(6, M, V, lt) == 0.0


def test_fit_on_synthetic_week():
    t0 = int(datetime(2026, 10, 1, tzinfo=NY).timestamp())
    posts = []
    for d, n in enumerate([10, 30, 20, 40, 15, 35, 25]):
        for i in range(n):
            posts.append({"post_ts": t0 + d * 86400 + 12 * 3600 + i, "deleted": False})
    posts.append({"post_ts": t0 + 3600, "deleted": True})        # deleted: excluded
    posts.append({"post_ts": t0 + 8 * 86400, "deleted": False})  # after calibration: excluded
    m = B.fit(posts, t0)
    assert m["n_posts"] == 175 and m["daily_counts"] == [10, 30, 20, 40, 15, 35, 25]
    assert m["rate_per_hour_et"][12] == pytest.approx(25.0) and sum(m["rate_per_hour_et"]) == pytest.approx(25.0)
    assert m["nb_k"] is not None and m["nb_k"] > 0


def test_report_rows_pnl_and_clv():
    picks = [{"decision": "T6", "event": "E1", "ticker": "E1-80", "p_yes": 0.9, "yes_bid": 0.7, "yes_ask": 0.75,
              "side": "yes", "price": 0.75, "cost": 0.76},
             {"decision": "T6", "event": "E1", "ticker": "E1-90", "p_yes": 0.1, "yes_bid": 0.3, "yes_ask": 0.35,
              "side": "no", "price": 0.70, "cost": 0.71},
             {"decision": "T6", "event": "E1", "ticker": "E1-95", "p_yes": 0.5, "yes_bid": 0.4, "yes_ask": 0.6,
              "side": None}]
    res = {"E1-80": {"result": "yes"}, "E1-90": {"result": "yes"}, "E1-95": {"result": "no"}}
    snaps = [{"ticker": "E1-80", "ts": 10, "close_ts": 20, "yes_bid": 0.9, "yes_ask": 0.92},
             {"ticker": "E1-80", "ts": 30, "close_ts": 20, "yes_bid": 0.98, "yes_ask": 0.99}]   # post-close: ignored
    o = A.report_rows(picks, res, snaps)["T6"]
    assert o["bets_settled"] == 2
    assert o["roi"] == pytest.approx(((1 - 0.76) + (0 - 0.71)) / (0.76 + 0.71))
    assert o["clv_n"] == 1 and o["clv_mean"] == pytest.approx(0.91 - 0.75)
    assert o["brier_n"] == 3

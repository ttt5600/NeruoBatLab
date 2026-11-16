"""Sanity tests for the descriptive SPRT panel in scripts/scorecards.py (notes/ops_sprt_cards.md)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scorecards as S  # noqa: E402

A, B = math.log(0.9 / 0.005), math.log(0.1 / 0.995)


def stream(rng, rho, n, per_night=3, prices=(0.95, 0.96, 0.97, 0.98)):
    c = rng.choice(prices, n)
    lost = rng.random(n) < rho * (1 - c)
    return [(float(ci), bool(li), i // per_night) for i, (ci, li) in enumerate(zip(c, lost))]


def first_cross(fills):
    """Run the SPRT fill by fill (as evaluated after each night) and return the first boundary state."""
    for k in range(1, len(fills) + 1):
        s = S.sprt_nc(fills[:k])
        if s["state"] in ("H1", "reject"):
            return s["state"]
    return "continue"


def test_boundaries_and_empty():
    s = S.sprt_nc([])
    assert math.isclose(s["A"], A) and math.isclose(s["B"], B)
    assert s["n"] == 0 and s["status"] == "no fills yet"


def test_exact_increments():
    # one loss at c=.97, one night: LLR = log(.7) / (1 + 1 x .031)
    s = S.sprt_nc([(0.97, True, "d1")])
    assert math.isclose(s["llr"], math.log(0.7) / 1.031)
    # one win at c=.97: log((1 - .7 x .03) / .97) / 1.031
    s = S.sprt_nc([(0.97, False, "d1")])
    assert math.isclose(s["llr"], math.log((1 - 0.021) / 0.97) / 1.031)
    # deff uses fills per night: 4 fills on 2 nights -> 1 + 2 x .031
    s = S.sprt_nc([(0.97, False, "a"), (0.97, False, "a"), (0.97, False, "b"), (0.97, False, "b")])
    assert math.isclose(s["deff"], 1.062)


def test_near_one_excluded():
    s = S.sprt_nc([(0.999, False, "a"), (1.0, False, "a"), (0.97, False, "a")])
    assert s["n"] == 1 and s["excluded"] == 2


def test_placebo_drifts_to_reject_and_rarely_accepts():
    rng = np.random.default_rng(1)
    states = [first_cross(stream(rng, 1.0, 1500)) for _ in range(60)]
    assert states.count("H1") <= 2            # nominal alpha .005 (iid)
    assert states.count("reject") >= 40       # placebo kill is common (sim2 stress kill .835)
    s = S.sprt_nc(stream(np.random.default_rng(2), 1.0, 3000))
    assert s["e_inc_H0"] < 0 < s["e_inc_H1"]


def test_planted_edge_accepts():
    rng = np.random.default_rng(3)
    states = [first_cross(stream(rng, 0.4, 2500)) for _ in range(40)]
    assert states.count("H1") >= 30
    assert states.count("reject") <= 4


def test_status_text():
    # zero losses so far -> positive rate -> "fills from accept"
    s = S.sprt_nc([(0.97, False, i // 3) for i in range(30)])
    assert s["state"] == "continue" and "fills from accept at current rate" in s["status"]
    # losses above break-even -> consistent with no edge, moving toward reject
    s = S.sprt_nc([(0.97, i % 10 == 0, i // 3) for i in range(30)])
    assert "consistent with no edge" in s["status"] or s["state"] == "reject"


def test_e4_live_fold_filters_and_cost():
    t = "KXHIGHCHI-26NOV10-T56"
    h = [{"event": "order", "mode": "live", "ticker": t, "ts": "2026-11-10T05:00:00+00:00",
          "cohort": "orig7", "exec_style": "maker", "filled": 0},
         {"event": "update", "mode": "live", "ticker": t, "ts": "2026-11-10T08:00:00+00:00",
          "filled": 24.0, "avg_no": 0.97, "fee": 0.24},
         {"event": "settle", "mode": "live", "ticker": t, "ts": "x", "result": "yes", "pnl": -1},
         # x1, taker (pre-start) and unfilled rows are excluded
         {"event": "order", "mode": "live", "ticker": "KXHIGHNY-26NOV10-T50", "ts": "2026-11-10T05:00:00+00:00",
          "cohort": "x1", "exec_style": "maker", "filled": 6, "avg_no": 0.98},
         {"event": "order", "mode": "live", "ticker": "KXHIGHNY-26OCT20-T50", "ts": "2026-10-20T05:00:00+00:00",
          "cohort": "orig7", "filled": 6, "avg_no": 0.98},
         {"event": "order", "mode": "live", "ticker": "KXHIGHMIA-26NOV10-T80", "ts": "2026-11-10T05:00:00+00:00",
          "cohort": "orig7", "exec_style": "maker", "filled": 0}]
    f, _ = S.e4_live_fills(h)
    assert len(f) == 1 and f[0][1] is True and f[0][2] == "26NOV10"
    assert math.isclose(f[0][0], 0.97 + 0.24 / 24)

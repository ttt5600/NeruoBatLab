"""Offline tests for the E4C cold-band paper arm (scripts/e4c_cold_paper.py). No network."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e4c_cold_paper as C  # noqa: E402

VALS = [60.0, 61.2, 62.5, 61.0, 60.4]   # coldest 60.0, warmest 62.5


def test_frozen_rule_reproduced_at_3_3():
    import kalshi_paper as KP  # the frozen E4 formula
    for st, f, c in [("less", None, 57), ("less", None, 58), ("less", None, 56), ("between", 55, 56), ("between", 56, 57),
                     ("between", 57, 58), ("between", 66, 67), ("greater", 65, None), ("greater", 66, None)]:
        fl = float(f) if f is not None else float("nan")
        ca = float(c) if c is not None else float("nan")
        assert C.rejected(st, f, c, VALS, 3.0, 3.0) == bool(KP.rejected(st, fl, ca, VALS)), (st, f, c)


def test_cold_band_between():
    # between [f, c]: E4 needs lo - 3 > c  (c < 57); cold-1 needs lo - 1 > c (c < 59); extra = 57 <= c < 59
    assert not C.cold_extra("between", 55, 56, VALS)        # already E4 (60 - 3 > 56)
    assert C.cold_extra("between", 56, 57, VALS)            # 60 - 3 = 57 not > 57 -> not E4; 59 > 57 -> cold-1
    assert C.cold_extra("between", 57, 58, VALS)            # top edge 2F below coldest
    assert not C.cold_extra("between", 58, 59, VALS)        # 59 not > 59
    assert not C.cold_extra("between", 59, 60, VALS)        # contains the coldest member


def test_cold_band_less_and_no_warm_side():
    # less (YES iff v <= c - 1): E4 lo - 3 > c - 1 (c < 58); cold-1 lo - 1 > c - 1 (c < 60) -> extra c in {58, 59}
    assert not C.cold_extra("less", None, 57, VALS)
    assert C.cold_extra("less", None, 58, VALS) and C.cold_extra("less", None, 59, VALS)
    assert not C.cold_extra("less", None, 60, VALS)
    # warm side is unchanged at 3F: a bracket 1-3F above the warmest is never an extra pick
    assert not C.cold_extra("greater", 64, None, VALS) and not C.cold_extra("between", 64, 65, VALS)
    # fewer than 3 members -> nothing
    assert not C.cold_extra("between", 57, 58, [60.0, 61.0])


def test_maker_q_matches_live_executor():
    # live: q = min(cent_floor((ask + no_bid) / 2), ask - 0.01); skip if ask - q > 0.05
    assert C.maker_q(0.03, 0.05) == (0.96, "bid")           # NO ask .97, NO bid .95 -> mid .96
    assert C.maker_q(0.02, 0.03) == (0.97, "bid")           # NO ask .98, bid .97 -> mid .975 -> floor .97
    assert C.maker_q(0.02, 0.12) == (0.93, "bid")           # ask .98, bid .88 -> mid .93; ask - q = .05 is allowed
    q, st = C.maker_q(0.01, 0.12)                           # ask .99, bid .88 -> mid .935 -> .93; .99 - .93 = .06 > .05
    assert q == 0.93 and st == "skip_maker_spread"


def test_tt_fill_strict_and_after_decision():
    t0, t1, q = 1000, 5000, 0.96
    assert C.tt_fill([(900, 0.90, 50)], t0, t1, q) == (False, None, False)      # before the decision
    assert C.tt_fill([(1500, 0.96, 50)], t0, t1, q) == (False, None, True)      # at q: not a trade-through; n10 yes
    assert C.tt_fill([(1500, 0.95, 3), (1600, 0.94, 1)], t0, t1, q) == (True, 1500, False)
    assert C.tt_fill([(6000, 0.50, 99)], t0, t1, q) == (False, None, False)     # after close


def test_close_no_last_no_taker_print_within_10h():
    tr = [(1100, 0.95, 1, "yes"), (1200, 0.96, 1, "no"), (1000 + 9 * 3600, 0.97, 1, "no"), (1000 + 11 * 3600, 0.99, 1, "no")]
    assert C.close_no(tr, 1000) == 0.97
    assert C.close_no([(1100, 0.95, 1, "yes")], 1000) is None


def test_pnl_unit():
    assert abs(C.pnl_unit("no", 0.96, True) - (1 / 0.96 - 1)) < 1e-12
    assert C.pnl_unit("yes", 0.96, True) == -1.0
    assert C.pnl_unit("yes", 0.96, False) == 0.0


def _e4p_rows():
    cd = {"kind": "city_day", "series": "KXHIGHNY", "event": "KXHIGHNY-26NOV26", "event_day": "2026-11-26",
          "decided_at": "2026-11-26T03:02:00+00:00", "status": "decided",
          "forecasts": dict(zip(C.MODELS, VALS))}
    base = {"kind": "bracket", "series": "KXHIGHNY", "event_day": "2026-11-26", "decided_at": cd["decided_at"],
            "close_time": "2026-11-27T05:00:00Z", "e4_own": False, "e4_live_at_T": False}
    br = [dict(base, ticker="KXHIGHNY-26NOV26-T57", strike_type="less", floor=None, cap=57.0, yes_bid=0.01, yes_ask=0.02, e4_own=True),
          dict(base, ticker="KXHIGHNY-26NOV26-B57.5", strike_type="between", floor=57.0, cap=58.0, yes_bid=0.03, yes_ask=0.05),
          dict(base, ticker="KXHIGHNY-26NOV26-B59.5", strike_type="between", floor=59.0, cap=60.0, yes_bid=0.20, yes_ask=0.24),
          dict(base, ticker="KXHIGHNY-26NOV26-B55.5", strike_type="between", floor=56.0, cap=57.0, yes_bid=0.04, yes_ask=0.06,
               e4_live_at_T=True)]
    return [cd] + br


def test_select_only_extra_and_idempotent():
    rows = C.select(_e4p_rows(), done=set(), forward_from="2026-11-25")
    picks = [r for r in rows if r["kind"] == "pick"]
    assert [p["ticker"] for p in picks] == ["KXHIGHNY-26NOV26-B57.5"]       # T57 = E4; B55.5 live E4; B59.5 not cold
    assert picks[0]["q"] == 0.96 and picks[0]["status"] == "bid" and picks[0]["cold_gap"] == 2.0
    assert C.select(_e4p_rows(), done={"KXHIGHNY-26NOV26"}, forward_from="2026-11-25") == []
    assert C.select(_e4p_rows(), done=set(), forward_from="2026-11-27") == []   # before the forward start


def test_settle_uses_tape_after_decision_and_skips_open():
    picks = [r for r in C.select(_e4p_rows(), set(), "2026-11-25")]
    now_open = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
    assert C.settle(picks, now_open, get_market=lambda tk: {}, get_tape=lambda *a: [], e4_live=set()) == []
    now = datetime(2026, 11, 28, tzinfo=timezone.utc)
    t_dec = C.epoch("2026-11-26T03:02:00+00:00")
    out = C.settle(picks, now, get_market=lambda tk: {"result": "no", "close_time": "2026-11-27T05:00:00Z", "expiration_value": "62"},
                   get_tape=lambda tk, a, b: [(t_dec - 30, 0.90, 100, "yes"), (t_dec + 600, 0.95, 2, "no")], e4_live=set())
    assert len(out) == 1 and out[0]["tt_fill"] and out[0]["fill_ts"] == t_dec + 600
    assert abs(out[0]["pnl_unit"] - (1 / 0.96 - 1)) < 1e-6 and out[0]["e4_live_final"] is False
    assert out[0]["close_no"] == 0.95
    # a settled pick is not settled twice
    assert C.settle(picks + out, now, get_market=lambda tk: {"result": "no"}, get_tape=lambda *a: [], e4_live=set()) == []


def test_tick_end_to_end(tmp_path, monkeypatch):
    e4p = tmp_path / "e4p.jsonl"
    e4p.write_text("".join(json.dumps(r) + "\n" for r in _e4p_rows()) + '{"kind": "brack')   # torn last line tolerated
    log = tmp_path / "e4c.jsonl"
    monkeypatch.setattr(C, "E4P", e4p)
    monkeypatch.setattr(C, "LOG", log)
    monkeypatch.setattr(C, "FORWARD_FROM", "2026-11-25")
    monkeypatch.setattr(C, "settle", lambda rows, now, **k: [])
    C.cmd_tick()
    C.cmd_tick()   # idempotent
    rs = C.jl(log)
    assert sum(r["kind"] == "city_day" for r in rs) == 1 and sum(r["kind"] == "pick" for r in rs) == 1


def test_edgelab_adapter_rows(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from edgelab.adapters import e4c_fwd as A
    picks = [r for r in C.select(_e4p_rows(), set(), "2026-11-25")]
    t_dec = C.epoch("2026-11-26T03:02:00+00:00")
    st = C.settle(picks, datetime(2026, 11, 28, tzinfo=timezone.utc),
                  get_market=lambda tk: {"result": "no", "close_time": "2026-11-27T05:00:00Z"},
                  get_tape=lambda tk, a, b: [(t_dec + 600, 0.95, 2, "no")], e4_live=set())
    f = tmp_path / "e4c_cold.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in picks + st))
    monkeypatch.setattr(A, "qs_or_repo", lambda rel: f)
    frame, _ = A.load({})
    b = A.rows(frame, {})
    assert len(b) == 1 and b.iloc[0]["bet_id"] == "KXHIGHNY-26NOV26-B57.5" and b.iloc[0]["price"] == 0.96
    assert b.iloc[0]["decided_at"] < b.iloc[0]["event_start"] < b.iloc[0]["fill_deadline"]
    assert b.iloc[0]["close_price"] == 0.95

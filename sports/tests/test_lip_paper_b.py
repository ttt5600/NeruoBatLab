"""L40b paper logger: scoring with non-default R/Target/DF, per-program payout, sample freezing (no network)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lip_paper_b as B  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "DATA", tmp_path)
    monkeypatch.setattr(B, "EVENTS", tmp_path / "ev.jsonl")
    monkeypatch.setattr(B, "SAMPLE", tmp_path / "sample.json")


def _order(pol="P3", side="no", price=40, size=250, fee_type="quadratic"):
    info = {"stratum": "primary", "close": 1e18, "fee_type": fee_type, "fee_multiplier": 1.0}
    book = {"yes": {}, "no": {price: 500.0}}
    return B.new_order(pol, "KXTEST-1", 1, side, price, size, B.T0 + 10, book, info)


def test_score_uses_program_target_and_df():
    # T = 2500, DF = 0.4: NO side 2000 @40 + 1000 @38 (qualifies only once cumulative >= 2500), YES 3000 @55
    book = {"yes": {55: 3000.0}, "no": {40: 2000.0, 38: 1000.0}}
    prog = {"p1": {"target": 2500.0, "df": 0.4, "R": 300.0}}
    o = _order(price=40, size=250)
    B.accrue(o, book, prog, {"p1": 60.0}, B.T0 + 70, 60.0, True)
    # with us: NO side 2250 @40, ref = 40 (cum >= 500), cum 2250 < 2500 -> add 38: 3250 >= 2500
    W = 2250 + 1000 * 0.4 ** 2
    share = 250 / W
    sc = o["v"]["phi"]["score"]["p1"]
    assert list(sc.values())[0] == pytest.approx(60 * share)
    # at the default T = 1000 the 38 level would not qualify and the share would be 250/2250
    assert share != pytest.approx(250 / 2250)


def test_invalid_other_side_scores_zero_under_big_target():
    book = {"yes": {55: 2000.0}, "no": {40: 5000.0}}  # YES short of T=2500
    o = _order()
    B.accrue(o, book, {"p1": {"target": 2500.0, "df": 0.5, "R": 100.0}}, {"p1": 60.0}, B.T0 + 70, 60.0, True)
    assert o["v"]["phi"]["score"] == {}


def test_program_lip_per_program_R_and_length():
    # 3-day program, R $300: share 0.1 on one side for 1 day -> 300 * 0.1*86400 / (2 * 3*86400) = $5.00
    p = {"R": 300.0, "start": 0.0, "end": 3 * 86400.0}
    pay, raw, alloc = B.program_lip(p, {"2026-10-30": 0.1 * 43200, "2026-10-31": 0.1 * 43200})
    assert raw == pytest.approx(5.0) and pay == 5.0
    assert alloc == {"2026-10-30": pytest.approx(2.5), "2026-10-31": pytest.approx(2.5)}
    # same score on a 14-day program, R $100 -> 100*8640/(2*14*86400) = 0.357 -> below the $1 minimum -> $0
    pay2, raw2, alloc2 = B.program_lip({"R": 100.0, "start": 0.0, "end": 14 * 86400.0}, {"d": 8640.0})
    assert raw2 == pytest.approx(100 * 8640 / (2 * 14 * 86400)) and pay2 == 0.0 and alloc2 == {}


def test_maker_fee_only_off_quadratic():
    assert B.fee_for(100, 40, "quadratic") == 0.0
    assert B.fee_for(100, 40, "quadratic_with_maker_fees") == pytest.approx(0.42)  # ceil(0.0175*100*.24*100)/100


def test_fill_bookkeeping_per_day_and_settlement():
    o = _order(side="no", price=40, size=250, fee_type="quadratic_with_maker_fees")
    # taker buys YES at yes 60 -> no_price 40 = our price; phi 0.25 x 400 = 100
    B.apply_print(o, {"id": "a", "ticker": "KXTEST-1", "ts": B.T0 + 100, "taker": "yes", "yes": 60, "n": 400.0})
    v = o["v"]["phi"]
    assert v["filled"] == pytest.approx(100) and v["cost"] == pytest.approx(40.0)
    day = B.day_of(B.T0 + 100)
    assert v["fills_day"][day][0] == pytest.approx(100) and v["fee"] == pytest.approx(0.42)
    assert B.settle_by_day(o, "phi") is None  # filled, unsettled -> pending
    o["result"] = "no"
    assert B.settle_by_day(o, "phi")[day] == pytest.approx(100 - 40 - 0.42)
    # fifo: queue_ahead 500 eats the whole print -> no fill
    assert o["v"]["fifo"]["filled"] == 0


def test_draw_sample_frozen_reproducible_and_capped():
    pool = {"primary": [f"KXA-{i}" for i in range(100)] + [f"KXB-{i}" for i in range(5)] + [f"KXC-{i}" for i in range(5)],
            "secondary": [f"KXD-{i}" for i in range(10)]}
    q = {"primary": 20, "secondary": 5}
    s1 = B.draw_sample(pool, q, seed=4040)
    s2 = B.draw_sample({k: list(reversed(v)) for k, v in pool.items()}, q, seed=4040)
    assert s1 == s2  # input order does not matter
    assert sum(t.startswith("KXA-") for t in s1["primary"]) <= 4  # 20% of 20
    assert len(s1["primary"]) == 12  # cap 4 per series x 3 series: cap binds, quota of 20 not met
    assert len(s1["secondary"]) == 1  # floor(0.2*5) = 1 per series
    assert B.draw_sample(pool, q, seed=1) != s1


def test_sample_refuses_to_overwrite(tmp_path):
    B.SAMPLE.write_text(json.dumps({"seed": 4040}))
    with pytest.raises(SystemExit):
        B.cmd_sample(B.SAMPLE)
    assert json.loads(B.SAMPLE.read_text()) == {"seed": 4040}


def test_eligibility_drops_weather_short_and_out_of_window():
    series = {"KXRT": {"category": "Entertainment"}, "KXRAINNYCM": {"category": "Climate and Weather"}}
    d = 86400.0
    progs = {"ok": {"ticker": "KXRT-A", "start": B.T0 - d, "end": B.T0 + 6 * d},
             "wx": {"ticker": "KXRAINNYCM-A", "start": B.T0, "end": B.T0 + 6 * d},
             "short": {"ticker": "KXRT-B", "start": B.T0, "end": B.T0 + 0.5 * d},
             "late": {"ticker": "KXRT-C", "start": B.T1 + d, "end": B.T1 + 8 * d},
             "nosr": {"ticker": "KXZZ-A", "start": B.T0, "end": B.T0 + 6 * d}}
    assert set(B.eligible_programs(progs, series)) == {"ok"}


def test_tick_end_to_end_offline(tmp_path, monkeypatch):
    """Two ticks in the window with faked pulls: P2 + P3 placed on tick 1, accrue + fill on tick 2, report runs."""
    d = 86400.0
    sample = {"drawn_at": "x", "seed": 4040, "T0": B.iso(B.T0), "T1": B.iso(B.T1),
              "tickers": {"KXRT-A": {"stratum": "primary", "close": B.T0 + 20 * d, "series": "KXRT",
                                     "category": "Entertainment", "fee_type": "quadratic", "fee_multiplier": 1.0}},
              "programs_at_draw": {"p1": {"ticker": "KXRT-A", "start": B.T0 - d, "end": B.T0 + 6 * d, "R": 100.0,
                                          "target": 1000.0, "df": 0.5, "status": "active"}}}
    B.SAMPLE.write_text(json.dumps(sample))
    for k in ("STATE", "TICKS", "LOCK"):
        monkeypatch.setattr(B, k, tmp_path / k.lower())
    clock = [B.T0 + 3600]
    monkeypatch.setattr(B, "now_ts", lambda: clock[0])
    monkeypatch.setattr(B.L, "now_ts", lambda: clock[0])
    monkeypatch.setattr(B, "fetch_all_programs", lambda: dict(sample["programs_at_draw"]))
    monkeypatch.setattr(B.L, "fetch_markets", lambda tks: {t: {"close": B.T0 + 20 * d, "status": "active",
                                                               "result": ""} for t in tks})
    book = {"yes": {1: 300.0, 2: 800.0}, "no": {97: 1200.0}}
    monkeypatch.setattr(B.L, "fetch_books", lambda tks: {"KXRT-A": book})
    trades = [{"id": "t1", "ticker": "KXRT-A", "ts": B.T0 + 3650, "taker": "no", "yes": 1, "n": 50.0}]
    monkeypatch.setattr(B.L, "fetch_trades", lambda since, w, o: (trades, "global:1p"))
    B.cmd_tick()
    st = json.loads(B.STATE.read_text())
    assert set(st["orders"]) == {"P2|KXRT-A|1", "P3|KXRT-A|1"}
    assert st["orders"]["P3|KXRT-A|1"]["side"] == "yes" and st["orders"]["P3|KXRT-A|1"]["price"] == 2
    clock[0] += 60
    B.cmd_tick()
    st = json.loads(B.STATE.read_text())
    o2 = st["orders"]["P2|KXRT-A|1"]
    assert o2["v"]["phi"]["filled"] == pytest.approx(50)  # phi 1.0 on a print at our 1c
    assert o2["v"]["phi"]["resting"] == pytest.approx(1000)  # replenished after scoring
    assert sum(o2["v"]["phi"]["score"]["p1"].values()) > 0
    rows = [json.loads(x) for x in B.TICKS.read_text().splitlines()]
    assert rows[-1]["live_programs"] == 1 and rows[-1]["orders_scored"] == 2
    B.cmd_report()

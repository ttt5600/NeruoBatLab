"""L39 logger (scripts/kalshi_wx_monthly_logger.py): station map, explicit exclusions, filled-only maker P&L, coverage,
and the lead_exec L39 reader. Fake Kalshi; no network."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kalshi_wx_monthly_logger as L  # noqa: E402
import kalshi_wx_winter as W  # noqa: E402

NEW = {"KXMKESNOWM": "MKE", "KXMSPSNOWM": "MSP", "KXPITSNOWM": "PIT", "KXTTNSNOWM": "TTN"}


def mkt(series, month="26DEC", X=8.0, rules=None, **kw):
    return {"ticker": f"{series}-{month}-{X:g}", "event_ticker": f"{series}-{month}", "strike_type": "greater",
            "floor_strike": X, "status": "active", "close_time": "2027-01-01T04:59:00Z",
            "rules_primary": rules or f"If the total in X in {month} is strictly greater than {X} inches", **kw}


@pytest.fixture
def fake(monkeypatch):
    book = {}  # series -> list of open markets

    def fetch(url, params=None, cache=None):
        if url.endswith("/series"):
            return {"series": [{"ticker": s, "category": "Climate and Weather"} for s in book]}
        if url.endswith("/markets"):
            return {"markets": book.get(params["series_ticker"], []), "cursor": ""}
        raise AssertionError(url)
    monkeypatch.setattr(L, "fetch", fetch)
    monkeypatch.setattr(W, "rain_series", lambda: sorted(s for s in book if s.startswith("KXRAIN")))
    return book


# ------------------------------------------------------------------------------------------------ station map
def test_new_snow_stations_mapped_and_frozen_map_untouched():
    for s, sid in NEW.items():
        assert L.SNOW[s] == sid
        assert s not in W.SNOW                      # dev/sealed universe unchanged
    assert all(L.SNOW[s] == v for s, v in W.SNOW.items())


def test_exclusions_by_name_and_resort_rule():
    assert "CLIDJT" in L.excluded("KXRAINWPBM")
    assert "KAPC" in L.excluded("KXRAINNAPAM")
    assert L.excluded("KXALTASNOW", "total snowfall recorded at Alta Ski Area in December") == L.RESORT_REASON
    assert L.excluded("KXTSNOWFALLBIGSKYM", "at Big Sky Resort, Montana ... as reported by OnTheSnow")
    assert L.excluded("KXRAINNYCM", "If the total precipitation at Central Park ...") is None
    assert L.excluded("KXMKESNOWM", "If the total snowfall in Milwaukee in December 2026") is None


def test_open_ladders_includes_new_snow_and_logs_exclusions(fake):
    fake["KXMKESNOWM"] = [mkt("KXMKESNOWM")]
    fake["KXTTNSNOWM"] = [mkt("KXTTNSNOWM")]
    fake["KXRAINWPBM"] = [mkt("KXRAINWPBM", rules="total precipitation at CLIDJT in West Palm Beach")]
    fake["KXRAINNYCM"] = [mkt("KXRAINNYCM", rules="precipitation at Central Park"),
                          mkt("KXRAINNYCM", month="26NOV", rules="precipitation at Central Park")]
    skipped = {}
    out = L.open_ladders("26DEC", skipped)
    got = {m["series"]: m["sid"] for m in out}
    assert got == {"KXMKESNOWM": "MKE", "KXTTNSNOWM": "TTN", "KXRAINNYCM": "NYC"}
    assert set(skipped) == {"KXRAINWPBM"} and "CLIDJT" in skipped["KXRAINWPBM"]


# ------------------------------------------------------------------------------------------------ report
def picks_frame():
    base = {"taker_cost": 0.97, "result": "no"}
    return pd.DataFrame([
        {**base, "won": True, "maker_filled": True, "maker_q": 0.96, "pnl_maker": 0.04},
        {**base, "won": True, "maker_filled": False, "maker_q": 0.95, "pnl_maker": 0.05},   # old row: unfilled w/ pnl
        {**base, "won": False, "result": "yes", "maker_filled": False, "maker_q": 0.97, "pnl_maker": -0.97},
        {**base, "won": True, "maker_filled": np.nan, "maker_q": 0.90, "pnl_maker": 0.10},  # missing -> unfilled
    ])


def test_maker_pnl_only_on_filled_rows():
    s = L.maker_frame(picks_frame())
    assert s.mf.tolist() == [True, False, False, False]
    assert s.pnl_mk.sum() == pytest.approx(0.04)
    assert np.nansum(s.u_maker) == pytest.approx(1 / 0.96 - 1)
    assert s.won.tolist() == [True, True, False, True]


def test_report_prints_filled_only(tmp_path, monkeypatch, capsys):
    p = tmp_path / "picks.jsonl"
    rows = picks_frame().assign(event="KXRAINNYCM-26OCT", kind="rain", series="KXRAINNYCM", ticker="T", d="x", X=1,
                                obs_d=0, fmax=0, yes_bid=.03, yes_ask=.05, pnl_taker=0.03)
    p.write_text("".join(json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v)
                                     for k, v in r.items()}) + "\n" for r in rows.to_dict("records")))
    monkeypatch.setattr(L, "PICKS", p)
    L.cmd_report()
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.startswith("ALL")][0].split()
    hdr = [l for l in out.splitlines() if "maker_c/ct" in l][0].split()
    vals = dict(zip(hdr, line[1:]))
    assert float(vals["maker_filled"]) == 1 and float(vals["maker_c/ct"]) == pytest.approx(4.0)
    assert float(vals["maker_L"]) == 0              # the YES loss was never filled as a maker


def test_settle_writes_pnl_maker_only_when_filled(tmp_path, monkeypatch):
    p = tmp_path / "picks.jsonl"
    base = {"side": "no", "T": 100, "taker_cost": 0.97, "close_time": "2026-11-01T04:59:00Z"}
    p.write_text(json.dumps({**base, "ticker": "A", "maker_q": 0.96}) + "\n"
                 + json.dumps({**base, "ticker": "B", "maker_q": 0.95}) + "\n")
    monkeypatch.setattr(L, "PICKS", p)
    monkeypatch.setattr(L, "fetch", lambda url, *a, **k: {"market": {"result": "no", "close_time": "2026-11-01T04:59:00Z"}})
    tapes = {"A": pd.DataFrame({"ts": [200], "no_price": [0.96], "count": [3]}),
             "B": pd.DataFrame({"ts": [200], "no_price": [0.97], "count": [3]})}   # B's q .95 never printed
    monkeypatch.setattr(W, "tape", lambda t, T: tapes[t])
    L.cmd_settle(quiet=True)
    a, b = [json.loads(l) for l in p.read_text().splitlines()]
    assert a["maker_filled"] and a["pnl_maker"] == pytest.approx(0.04)
    assert not b["maker_filled"] and b["pnl_maker"] is None and b["pnl_maker_if_filled"] == pytest.approx(0.05)


# ------------------------------------------------------------------------------------------------ coverage
def test_coverage_flags_unmapped_and_lists_exclusions(fake, capsys):
    fake["KXMKESNOWM"] = [mkt("KXMKESNOWM")]
    fake["KXRAINNAPAM"] = [mkt("KXRAINNAPAM", month="01DEC26-31DEC26", rules="precipitation at KAPC")]
    fake["KXALTASNOW"] = [mkt("KXALTASNOW", rules="snowfall recorded at Alta Ski Area")]
    assert L.cmd_coverage("26DEC") == 0
    out = capsys.readouterr().out
    assert "KXMKESNOWM" in out and "included" in out and "KXRAINNAPAM" in out and "KXALTASNOW" in out
    assert "UNMAPPED 0" in out
    fake["KXBUFSNOWM"] = [mkt("KXBUFSNOWM")]        # a new city nobody mapped -> loud, exit 1
    assert L.cmd_coverage("26DEC") == 1
    assert "KXBUFSNOWM" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ lead_exec reader
def test_lead_exec_reads_logger_picks(tmp_path, monkeypatch):
    import lead_exec as lx
    p = tmp_path / "picks.jsonl"
    ts = "2026-12-26T04:05:00+00:00"
    rows = [{"ticker": "KXMKESNOWM-26DEC-8", "event": "KXMKESNOWM-26DEC", "d": "2026-12-25", "ts": ts,
             "maker_q": 0.97, "off_schedule": False},
            {"ticker": "KXRAINNYCM-26NOV-6", "event": "KXRAINNYCM-26NOV", "d": "2026-11-28", "ts": ts,
             "maker_q": 0.96, "result": "no", "pnl_maker": None, "maker_filled": False}]   # settled -> not mirrored
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(lx, "L39_PICKS", p)
    s = lx.l39_signals(datetime(2026, 12, 26, 5, 5, tzinfo=timezone.utc))
    assert len(s) == 1 and s[0]["ticker"] == "KXMKESNOWM-26DEC-8" and s[0]["q"] == 0.97 and s[0]["side"] == "no"
    assert s[0]["src_age_h"] == 1.0 and lx.period_key("L39", s[0]) == "2026-12"

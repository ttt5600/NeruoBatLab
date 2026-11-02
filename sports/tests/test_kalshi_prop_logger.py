"""Unit tests for scripts/kalshi_prop_logger.py (synthetic, no network)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kalshi_prop_logger as kp  # noqa: E402


def test_norm_name_and_parse():
    assert kp.norm_name("Luka Dončić") == kp.norm_name("Luka Doncic") == "luka doncic"
    assert kp.norm_name("Jaren Jackson Jr.") == "jaren jackson"
    assert kp.norm_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"
    assert kp.parse_rung({"yes_sub_title": "Austin Reaves: 15+", "floor_strike": 14.5}) == ("Austin Reaves", 15)


def test_find_event_maps_kalshi_codes():
    ev = {"KXNBAPTS": ["KXNBAPTS-26NOV01LACGSW", "KXNBAPTS-26NOV01TORLAL", "KXNBAPTS-26NOV02NOPMEM"]}
    assert kp.find_event("KXNBAPTS", "2026-11-01", "LAC", "GS", ev) == "KXNBAPTS-26NOV01LACGSW"
    assert kp.find_event("KXNBAPTS", "2026-11-02", "NO", "MEM", ev) == "KXNBAPTS-26NOV02NOPMEM"
    assert kp.find_event("KXNBAPTS", "2026-11-02", "TOR", "LAL", ev) is None


def test_flag_rule_frozen():
    assert (kp.EDGE, kp.MIN_VOL, kp.MIN_PROJ) == (0.08, 100.0, 20.0)
    assert kp.flag(0.70, 0.55, 0.62, 500, 30) == "YES"
    assert kp.flag(0.69, 0.55, 0.62, 500, 30) is None
    assert kp.flag(0.40, 0.48, 0.50, 500, 30) == "NO"
    assert kp.flag(0.10, 0.48, 0.50, 99, 30) is None          # volume floor
    assert kp.flag(0.10, 0.48, 0.50, 500, 19.9) is None       # minutes floor
    assert kp.flag(None, 0.48, 0.50, 500, 30) is None


def test_grade_and_report(tmp_path, monkeypatch, capsys):
    log, gr = tmp_path / "log.jsonl", tmp_path / "gr.jsonl"
    monkeypatch.setattr(kp, "LOG", log)
    monkeypatch.setattr(kp, "GRADES", gr)
    base = {"type": "rung", "event_id": "1", "game": "A@B", "tip_utc": "2026-11-01T01:00:00+00:00", "stat": "pts",
            "athlete_id": "9", "min_proj": 30.0, "absent": False, "volume": 500}
    recs = [dict(base, ticker="T1", player="P Q", k=20, sim=0.70, bid=0.55, ask=0.60, mid=0.575, pick="YES"),
            dict(base, ticker="T2", player="P Q", k=30, sim=0.10, bid=0.25, ask=0.27, mid=0.26, pick="NO"),
            dict(base, ticker="T3", player="P Q", k=25, sim=0.40, bid=0.40, ask=0.42, mid=0.41, pick=None),
            dict(base, ticker="T4", player="Gone Guy", athlete_id="8", k=10, sim=0.5, bid=0.4, ask=0.5, mid=0.45, pick="YES")]
    log.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    monkeypatch.setattr(kp, "box_stats", lambda e: {"final": {"1"}, "box": {("1", "9"): {"name": "P Q", "minutes": 34,
                                                                                         "points": 24}}})
    monkeypatch.setattr(kp, "clv_close", lambda t, tip: 0.64)
    kp.cmd_grade(None)
    G = {g["ticker"]: g for g in kp.read_log(gr)}
    assert G["T4"]["void"] and not G["T1"]["void"]
    assert G["T1"]["y"] == 1 and G["T1"]["profit"] == pytest.approx(1 - 0.60 - 0.07 * 0.6 * 0.4)
    assert G["T2"]["y"] == 0 and G["T2"]["profit"] == pytest.approx(1 - 0.75 - 0.07 * 0.75 * 0.25)
    assert G["T1"]["clv"] == pytest.approx(0.04) and G["T2"]["clv"] == pytest.approx(0.25 - 0.64)
    kp.cmd_grade(None)                                         # idempotent
    assert len(kp.read_log(gr)) == 4
    kp.cmd_report(None)
    out = capsys.readouterr().out
    assert "PICKS 2" in out and "W-L 2-0" in out and "CALIBRATION, 3 non-void" in out and "PENDING" in out

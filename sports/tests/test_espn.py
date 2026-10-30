"""ESPN collector parsers and prop grading, offline (records copied from real responses, trimmed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports import espn, settle

CORE = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl"


def _prop(side, line, dec, line_open=None, athlete="3139477", market="Total Passing Yards (incl. overtime)"):
    cur = {"target": {"value": line}} if line is not None else {}
    opn = {"target": {"value": line_open if line_open is not None else line}} if line is not None else {}
    if side:
        cur[side] = {"decimal": dec}
        opn[side] = {"decimal": dec}
    return {"athlete": {"$ref": f"{CORE}/seasons/2024/athletes/{athlete}?lang=en"},
            "provider": {"$ref": f"{CORE}/casinos/58?lang=en"}, "type": {"id": "8", "name": market},
            "lastUpdated": "2024-09-06T00:30Z", "current": cur, "open": opn}


def test_parse_prop_keeps_priced_sides_and_drops_alt_lines():
    over = espn.parse_prop(_prop("over", 274.5, 2.0, line_open=265.5))
    assert len(over) == 1 and over[0]["side"] == "over" and over[0]["line"] == 274.5
    assert over[0]["line_open"] == 265.5 and over[0]["athlete_id"] == "3139477" and over[0]["provider_id"] == "58"
    assert espn.parse_prop(_prop(None, 250.0, None)) == []          # alt line, no price
    yes = espn.parse_prop(_prop("over", None, 8.0, market="First Touchdown Scorer"))
    assert yes[0]["side"] == "yes" and np.isnan(yes[0]["line"])


def _new(ath, line, dec, market="Total Points", dec_open=None):
    """2026-27 record shape (DK, provider 100): price in 'odds', no over/under key."""
    return {"athlete": {"$ref": f"{CORE}/seasons/2027/athletes/{ath}?lang=en"},
            "provider": {"$ref": f"{CORE}/casinos/100?lang=en"}, "type": {"id": "1", "name": market},
            "odds": {"decimal": {"value": str(dec), "open": str(dec_open or dec)},
                     "total": {"value": str(line), "open": str(line)}},
            "lastUpdated": "2026-10-29T05:40Z",
            "current": {"target": {"value": line}} if line is not None else {},
            "open": {"target": {"value": line}} if line is not None else {}}


def test_parse_props_new_format_first_of_pair_is_over():
    items = [_new("1", 10.5, 1.86, dec_open=1.82), _new("1", 10.5, 1.95),  # pair -> over, under
             _new("2", 9.0, 2.40, market="Points Milestones"),               # lone record -> yes
             _new("3", 4.5, 1.90), _new("3", 5.5, 1.90),                     # different lines: not a pair
             _new("4", None, 7.0, market="First Basket")]
    r = espn.parse_props(items)
    assert [(x["athlete_id"], x["side"]) for x in r] == [
        ("1", "over"), ("1", "under"), ("2", "yes"), ("3", "yes"), ("3", "yes"), ("4", "yes")]
    assert r[0]["line"] == 10.5 and r[0]["dec"] == 1.86 and r[0]["dec_open"] == 1.82
    assert r[1]["dec"] == 1.95 and r[0]["provider_id"] == "100" and np.isnan(r[5]["line"])
    # the old format still goes through parse_prop unchanged, and unpriced alt lines still drop
    assert espn.parse_props([_prop("over", 274.5, 2.0), _prop(None, 250.0, None)])[0]["side"] == "over"
    assert len(espn.parse_props([_prop(None, 250.0, None)])) == 0


def test_parse_line_reads_open_and_close():
    item = {"provider": {"id": "58", "name": "ESPN BET"}, "spread": -2.5, "overUnder": 45.5,
            "overOdds": -115.0, "underOdds": -105.0,
            "homeTeamOdds": {"moneyLine": -140, "spreadOdds": -120.0,
                             "open": {"pointSpread": {"american": "-3"}, "spread": {"american": "-110"},
                                      "moneyLine": {"american": "-150"}},
                             "close": {"pointSpread": {"american": "-2.5"}, "moneyLine": {"american": "-140"}}},
            "awayTeamOdds": {"moneyLine": 120, "open": {"moneyLine": {"american": "+130"}}},
            "open": {"total": {"american": "46.5"}, "over": {"american": "-115"}},
            "close": {"total": {"american": "45.5"}, "under": {"american": "EVEN"}}}
    r = espn.parse_line(item)
    assert r["home_spread_open"] == -3 and r["home_spread_close"] == -2.5 and r["home_ml_open"] == -150
    assert r["away_ml_open"] == 130 and r["total_open"] == 46.5 and r["total_close"] == 45.5
    assert r["under_odds_close"] == 100 and r["total"] == 45.5


def test_box_splits_compound_stats_and_converts_innings(monkeypatch):
    s = {"boxscore": {"players": [{"team": {"id": "1"}, "statistics": [
        {"type": "batting", "keys": ["hits-atBats", "hits", "strikeouts"],
         "athletes": [{"athlete": {"id": "9", "displayName": "B"}, "stats": ["2-4", "2", "1"]}]},
        {"type": "pitching", "keys": ["fullInnings.partInnings", "strikeouts", "pitches-strikes"],
         "athletes": [{"athlete": {"id": "7", "displayName": "P"}, "stats": ["6.1", "8", "95-61"]}]},
        {"name": "passing", "keys": ["completions/passingAttempts", "sacks-sackYardsLost"],
         "athletes": [{"athlete": {"id": "5", "displayName": "Q"}, "stats": ["26/41", "1-6"]}]}]}]}}
    monkeypatch.setattr(espn, "summary", lambda *a, **k: s)
    b = pd.DataFrame(espn.box("mlb", "1")).set_index(["athlete_id", "group", "stat"])["value"]
    assert b[("9", "batting", "hits")] == 2 and b[("9", "batting", "atBats")] == 4
    assert b[("7", "pitching", "outs")] == 19 and b[("7", "pitching", "strikeouts")] == 8
    assert b[("7", "pitching", "strikes")] == 61
    assert b[("5", "passing", "passingAttempts")] == 41 and b[("5", "passing", "sackYardsLost")] == 6


def test_grade_over_under_push_void_yes_and_team_total():
    games = pd.DataFrame([{"league": "nfl", "event_id": "1", "start_utc": pd.Timestamp("2024-09-06T00:20Z"),
                           "home_team_id": "12", "away_team_id": "33", "home_score": 27.0, "away_score": 20.0}])
    box = pd.DataFrame([
        {"event_id": "1", "athlete_id": "5", "group": "passing", "stat": "passingYards", "value": 273.0},
        {"event_id": "1", "athlete_id": "5", "group": "passing", "stat": "sacks", "value": 1.0},
        {"event_id": "1", "athlete_id": "6", "group": "rushing", "stat": "rushingTouchdowns", "value": 0.0},
        {"event_id": "1", "athlete_id": "6", "group": "receiving", "stat": "receivingTouchdowns", "value": 1.0},
        {"event_id": "1", "athlete_id": "8", "group": "defensive", "stat": "sacks", "value": 2.0}])
    P = "Total Passing Yards (incl. overtime)"
    props = pd.DataFrame([
        dict(event_id="1", athlete_id="5", team_id=None, market=P, side="over", line=274.5, dec=2.0),
        dict(event_id="1", athlete_id="5", team_id=None, market=P, side="under", line=274.5, dec=1.8),
        dict(event_id="1", athlete_id="5", team_id=None, market=P, side="over", line=273.0, dec=1.9),
        dict(event_id="1", athlete_id="99", team_id=None, market=P, side="over", line=200.5, dec=1.9),
        dict(event_id="1", athlete_id="6", team_id=None, market="Anytime Touchdown Scorer", side="yes",
             line=np.nan, dec=3.0),
        dict(event_id="1", athlete_id="8", team_id=None, market="Total Sacks (incl. overtime)", side="over",
             line=0.5, dec=2.5),
        dict(event_id="1", athlete_id=None, team_id="33", market="Team Total Points", side="under",
             line=21.5, dec=1.9)]).assign(last_updated=pd.Timestamp("2024-09-05T22:20Z"))
    g = settle.grade(props, box, games)
    assert list(g["profit"].iloc[:2].round(2)) == [-1.0, 0.8]     # 273 < 274.5: over loses, under wins
    assert g["push"].iloc[2] == True and g["profit"].iloc[2] == 0  # noqa: E712 -- exactly on the line
    assert np.isnan(g["profit"].iloc[3])                          # did not play -> void
    assert g["profit"].iloc[4] == 2.0                             # anytime TD via a receiving TD
    assert g["actual"].iloc[5] == 2.0                             # defensive sacks, not sacks taken
    assert g["actual"].iloc[6] == 20.0 and round(g["profit"].iloc[6], 6) == 0.9  # away team scored 20 < 21.5
    assert g["hours_before"].iloc[0] == 2.0

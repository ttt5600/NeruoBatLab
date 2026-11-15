"""Offline tests for gap_logger's Kalshi<->Pinnacle soccer matcher, raw-odds log and off-season handling (no network).

Every collision case here was found in real data (results/ops_gap_logger/audit_matcher.py, 2026-11-15):
the old first-token + substring matcher mis-sided WHU-WBA (fake +146% EV) and Le Mans-Lille (fake +47%)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gap_logger as G  # noqa: E402


def kal(*events):
    """events: (event_ticker, {suffix: yes_sub_title}) -> live.kalshi()-shaped frame."""
    rows = []
    for ev, teams in events:
        for suf, name in teams.items():
            rows.append({"ticker": f"{ev}-{suf}", "event": ev, "team": name, "yes_bid": .3, "yes_ask": .31, "no_ask": .7})
    return pd.DataFrame(rows)


T = pd.Timestamp("2026-10-31T15:00Z")
WHUWBA = ("KXEFLCHAMPIONSHIPGAME-26OCT31WHUWBA", {"WHU": "West Ham", "WBA": "West Bromwich", "TIE": "Tie"})


@pytest.mark.parametrize("pin,kal_name", [
    ("West Ham United", "West Ham"), ("West Bromwich Albion", "West Bromwich"), ("Newcastle United", "Newcastle"),
    ("Paris Saint-Germain", "PSG"), ("Internazionale", "Inter"), ("AC Milan", "Milan"), ("Athletic Bilbao", "Bilbao"),
    ("Borussia Monchengladbach", "M´gladbach"), ("FC Koln", "FC Köln"), ("Queens Park Rangers", "QPR"),
    ("PSV", "Eindhoven"), ("Hamburger SV", "Hamburg"), ("Rennes", "Stade Rennais"), ("Brest", "Stade Brest 29"),
    ("Deportivo La Coruna", "Deportivo De La Coruna"), ("Los Angeles FC", "Los Angeles F"),
    ("LA Galaxy", "Los Angeles G"), ("St Louis City SC", "Saint Louis"), ("Bodo Glimt", "Bodoe/Glimt"),
    ("Mainz 05", "Mainz"), ("Le Mans", "Le Mans"), ("D.C. United", "DC United"),
])
def test_same_team_true(pin, kal_name):
    assert G.same_team(pin, kal_name)


@pytest.mark.parametrize("pin,kal_name", [
    ("West Ham United", "West Bromwich"), ("West Bromwich Albion", "West Ham"), ("Le Mans", "Lille"),
    ("Le Havre", "Lens"), ("Le Havre", "Marseille"), ("Le Mans", "Le Havre"), ("Manchester United", "Manchester City"),
    ("Real Madrid", "Villarreal"), ("Real Betis", "Real Sociedad"), ("RB Leipzig", "Paderborn"),
    ("St Louis City SC", "Houston"), ("St Louis City SC", "Austin"), ("Los Angeles FC", "Los Angeles G"),
    ("LA Galaxy", "Los Angeles F"), ("New England Revolution", "New York RB"), ("San Jose Earthquakes", "San Diego FC"),
    ("Real Salt Lake", "Montreal"), ("Internazionale", "Milan"), ("AC Milan", "Inter"), ("Inter Miami", "Inter"),
])
def test_same_team_false(pin, kal_name):
    assert not G.same_team(pin, kal_name)


@pytest.mark.parametrize("home,away", [("West Ham United", "West Bromwich Albion"),
                                       ("West Bromwich Albion", "West Ham United")])
def test_west_ham_west_brom_sided_correctly(home, away):
    ev, side = G.match_soccer_event(T, home, away, kal(WHUWBA))
    assert ev == WHUWBA[0]
    whu, wba = f"{ev}-WHU", f"{ev}-WBA"
    assert side[f"{ev}-TIE"] == "D"
    assert (side[whu], side[wba]) == (("H", "A") if home.startswith("West Ham") else ("A", "H"))


def test_le_mans_lille():
    k = kal(("KXLIGUE1GAME-26NOV28MANLIL", {"MAN": "Le Mans", "LIL": "Lille", "TIE": "Tie"}))
    ev, side = G.match_soccer_event(pd.Timestamp("2026-11-28T19:45Z"), "Le Mans", "Lille", k)
    assert side[f"{ev}-MAN"] == "H" and side[f"{ev}-LIL"] == "A"


@pytest.mark.parametrize("home,away,ev,teams,want", [
    ("Manchester United", "Manchester City", "KXEPLGAME-26OCT31MUNMCI",
     {"MUN": "Manchester United", "MCI": "Manchester City"}, {"MUN": "H", "MCI": "A"}),
    ("Real Madrid", "Villarreal", "KXLALIGAGAME-26OCT31RMAVIL", {"RMA": "Real Madrid", "VIL": "Villarreal"},
     {"RMA": "H", "VIL": "A"}),
    ("Los Angeles FC", "LA Galaxy", "KXMLSGAME-26OCT31LAFCLAG", {"LAFC": "Los Angeles F", "LAG": "Los Angeles G"},
     {"LAFC": "H", "LAG": "A"}),
    ("Houston Dynamo", "St Louis City SC", "KXMLSGAME-26OCT31HOUSTL", {"HOU": "Houston", "STL": "Saint Louis"},
     {"HOU": "H", "STL": "A"}),
    ("Internazionale", "AC Milan", "KXSERIEAGAME-26OCT31INTMIL", {"INT": "Inter", "MIL": "Milan"},
     {"INT": "H", "MIL": "A"}),
])
def test_derbies_and_shared_prefixes(home, away, ev, teams, want):
    got = G.match_soccer_event(T, home, away, kal((ev, {**teams, "TIE": "Tie"})))
    assert got is not None
    assert {t.rsplit("-", 1)[1]: s for t, s in got[1].items() if not t.endswith("-TIE")} == want


def test_wrong_date_rejected():
    assert G.match_soccer_event(pd.Timestamp("2026-11-07T15:00Z"), "West Ham United", "West Bromwich Albion",
                                kal(WHUWBA)) is None


def test_late_kickoff_uses_us_date():
    # 02:30 UTC Nov 1 = Oct 31 evening ET; Kalshi dates MLS events by the US date
    k = kal(("KXMLSGAME-26OCT31SEAPOR", {"SEA": "Seattle", "POR": "Portland", "TIE": "Tie"}))
    assert G.match_soccer_event(pd.Timestamp("2026-11-01T02:30Z"), "Seattle Sounders", "Portland Timbers", k)


def test_one_kalshi_team_matching_both_rejected():
    k = kal(("KXEPLGAME-26OCT31MANMAN", {"MA1": "Manchester", "MA2": "Liverpool", "TIE": "Tie"}))
    assert G.match_soccer_event(T, "Manchester United", "Manchester City", k) is None


def test_two_candidate_events_rejected():
    k = kal(("KXEPLGAME-26OCT31NEWARS", {"NEW": "Newcastle", "ARS": "Arsenal", "TIE": "Tie"}),
            ("KXEPLGAME-26OCT31NEWAR2", {"NEW": "Newcastle", "AR2": "Arsenal", "TIE": "Tie"}))
    assert G.match_soccer_event(T, "Newcastle United", "Arsenal", k) is None


def test_missing_tie_market_rejected():
    assert G.match_soccer_event(T, "West Ham United", "West Bromwich Albion",
                                kal((WHUWBA[0], {"WHU": "West Ham", "WBA": "West Bromwich"}))) is None


def test_odds_rows_dedupe_and_age(monkeypatch):
    url = f"{G.PIN}/leagues/1977/markets/straight"
    monkeypatch.setitem(G.HDRS, url, {"date": "Sun, 15 Nov 2026 10:35:14 GMT",
                                      "last_modified": "Sun, 15 Nov 2026 10:31:25 GMT", "age": "232", "cf": "HIT"})
    ts = pd.Timestamp("2026-11-15T10:35Z")
    o = {"pin_id": 1, "start": "x", "home": "a", "away": "b", "event": "E", "prices": {"H": 150, "D": 230, "A": 180},
         "version": 7, "max_risk": 500}
    out, last = [], {}
    G._write_odds(ts, "efl", [o], out, last)
    G._write_odds(ts, "efl", [o], out, last)               # unchanged -> fetch row only
    G._write_odds(ts, "efl", [{**o, "version": 8, "prices": {"H": 140, "D": 230, "A": 190}}], out, last)
    kinds = [r["kind"] for r in out]
    assert kinds == ["fetch", "odds", "fetch", "fetch", "odds"]
    assert out[0]["age_s"] == 229 and out[0]["cf_age"] == 232 and out[0]["ids"] == [1]
    assert out[-1]["prices"]["H"] == 140
    json.dumps(out)


def test_us_off_season_is_empty_not_keyerror(monkeypatch):
    monkeypatch.setattr(G.live, "pinnacle", lambda lg: pd.DataFrame())
    monkeypatch.setattr(G.live, "kalshi", lambda lg: pd.DataFrame())
    q, odds = G.us("mlb")
    assert q.empty and odds == []


def test_us_pinnacle_but_no_kalshi(monkeypatch):
    pin = pd.DataFrame([{"league": "wnba", "pin_id": 9, "start": pd.Timestamp("2026-11-20T00:00Z"), "home": "A",
                         "away": "B", "live": False, "market": "moneyline", "price1": -120, "price2": 110,
                         "max_risk": 100}])
    monkeypatch.setattr(G.live, "pinnacle", lambda lg: pin)
    monkeypatch.setattr(G.live, "kalshi", lambda lg: pd.DataFrame())
    q, odds = G.us("wnba")
    assert q.empty and odds[0]["prices"] == {"H": -120, "A": 110} and odds[0]["event"] is None

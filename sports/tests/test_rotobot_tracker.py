"""rotobot_tracker: parser, post-start exclusion, close/CLV arithmetic, grading, stats (synthetic, offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rotobot_tracker as rt

T = pd.Timestamp


def test_to_decimal():
    assert rt.to_decimal("-110") == pytest.approx(1.909091, abs=1e-6)
    assert rt.to_decimal("+150") == 2.5
    assert rt.to_decimal("−200") == 1.5          # unicode minus from copy-paste
    assert rt.to_decimal("1.91") == 1.91
    assert rt.to_decimal("0.62") == pytest.approx(1 / 0.62)
    assert rt.to_decimal("62c") == pytest.approx(100 / 62)
    assert np.isnan(rt.to_decimal(None)) and np.isnan(rt.to_decimal(""))


@pytest.mark.parametrize("text,expect", [
    ("#NBA LeBron James o24.5 pts (-115)",
     dict(sport="nba", player="LeBron James", market="total points", side="over", line=24.5, price="-115")),
    ("Jayson Tatum rebounds under 8.5 +105",
     dict(player="Jayson Tatum", market="total rebounds", side="under", line=8.5, price="+105")),
    ("NFL: Puka Nacua over 85.5 rec yds",
     dict(sport="nfl", player="Puka Nacua", market="total receiving yards", side="over", line=85.5, price=None)),
    ("Lakers vs Celtics u224.5 (-110)", dict(market="total", side="under", line=224.5, price="-110")),
    ("Celtics -4.5 (-108)", dict(market="spread", side="Celtics", line=-4.5, price="-108")),
    ("Knicks ML +130", dict(market="moneyline", side="Knicks", price="+130")),
])
def test_parse_tweet_formats(text, expect):
    got = rt.parse_tweet(text, "2026-01-09T18:00:00Z", "https://x.com/rotobot_ai/status/1", sport="nba")
    assert len(got) == 1
    for k, v in expect.items():
        assert got[0][k] == v, (k, got[0])
    assert got[0]["posted_at"] == T("2026-01-09 18:00", tz="UTC")


def test_parse_multi_pick_post_and_noise():
    txt = "Tonight's best bets \U0001F525\n- Tatum o27.5 pts\n- LeBron James o24.5 points (-115)\n- Celtics -4.5\nLink in bio"
    got = rt.parse_tweet(txt, "2026-01-09T18:00:00Z", sport="nba")
    # single-word name "Tatum o27.5" is NOT parsed as a prop (needs first + last name) -- by design,
    # it falls through; the spread and the full-name prop are parsed, the chatter lines are skipped
    assert [g["market"] for g in got] == ["total points", "spread"]


def test_read_posts_txt(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("2026-01-09T18:00:00Z https://x.com/a/status/1\nCeltics -4.5 (-110)\n---\n"
                 "2026-01-09T19:00:00Z\nnothing here\n")
    posts = rt.read_posts(p)
    assert len(posts) == 2 and posts[0]["url"].endswith("/1") and posts[1]["url"] == ""


# ---------------------------------------------------------------- synthetic slate
ALIASES = {"nba": {"bos": "BOS", "celtics": "BOS", "boston celtics": "BOS", "boston": "BOS",
                   "lal": "LAL", "lakers": "LAL", "los angeles lakers": "LAL",
                   "nyk": "NYK", "knicks": "NYK", "mia": "MIA", "heat": "MIA"}}
GAMES = pd.DataFrame([
    dict(league="nba", event_id="1", start_utc=T("2026-01-10 00:30", tz="UTC"), status="STATUS_FINAL",
         home="BOS", away="LAL", home_score=110.0, away_score=100.0),
    dict(league="nba", event_id="2", start_utc=T("2026-01-11 00:00", tz="UTC"), status="STATUS_FINAL",
         home="NYK", away="MIA", home_score=100.0, away_score=105.0),
])
LINES = pd.DataFrame([dict(event_id="1", provider="DK", home_ml_close=-200.0, away_ml_close=170.0,
                           home_spread_close=-6.5, home_spread_odds_close=-110.0, away_spread_close=6.5,
                           away_spread_odds_close=-110.0, total_close=220.5, over_odds_close=-110.0,
                           under_odds_close=-110.0)])
BOX = pd.DataFrame([dict(event_id="1", athlete_id="7", name="Jayson Tatum")])
GRADED = pd.DataFrame([
    dict(event_id="1", athlete_id="7", market_n="total points", provider="DK", side="over", line=28.5, dec=1.87, actual=30.0),
    dict(event_id="1", athlete_id="7", market_n="total points", provider="DK", side="under", line=28.5, dec=1.95, actual=30.0),
])
PRE = "2026-01-09T18:00:00Z"
PICKS = pd.DataFrame([
    dict(posted_at=PRE, sport="nba", game="LAL @ BOS", market="spread", side="Celtics", line=-5.5, price="-110"),   # A win
    dict(posted_at=PRE, sport="nba", game="LAL @ BOS", market="total", side="over", line=218.5, price="-110"),      # B loss
    dict(posted_at=PRE, sport="nba", game="Lakers vs Celtics", market="moneyline", side="Celtics", line=np.nan, price="-170"),  # C win
    dict(posted_at="2026-01-10T01:00:00Z", sport="nba", game="LAL @ BOS", market="total", side="under", line=220.5, price="-110"),  # D late
    dict(posted_at=PRE, sport="nba", game="", market="points", side="over", line=27.5, price="-115", player="Jayson Tatum"),  # E win
    dict(posted_at=PRE, sport="nba", game="", market="points", side="under", line=28.5, price="+100", player="Jayson Tatum"),  # F loss
    dict(posted_at=PRE, sport="nba", game="Sonics", market="moneyline", side="Sonics", line=np.nan, price="+200"),  # G no team
    dict(posted_at="2026-01-10T23:00:00Z", sport="nba", game="Heat @ Knicks", market="moneyline", side="Heat",
         line=np.nan, price="+120"),                                                                               # H win, no lines
])
PICKS["source_url"] = ""


def _graded(tmp_path):
    f = tmp_path / "p.csv"
    PICKS.to_csv(f, index=False)
    return rt.attach(rt.load_picks(f), GAMES, LINES, GRADED, BOX, ALIASES, use_kalshi=False)


def test_post_start_and_unmatched_are_excluded(tmp_path):
    g = _graded(tmp_path)
    assert g.loc[3, "reason"] == "posted_after_start"
    assert g.loc[6, "reason"] == "no_team_match"
    assert g.loc[[0, 1, 2, 4, 5, 7], "reason"].eq("ok").all()


def test_results_and_profit(tmp_path):
    g = _graded(tmp_path)
    assert list(g.loc[[0, 1, 2, 4, 5, 7], "result"]) == [1, -1, 1, 1, -1, 1]
    assert g.loc[0, "profit"] == pytest.approx(100 / 110)
    assert g.loc[2, "profit"] == pytest.approx(100 / 170)
    assert g.loc[7, "profit"] == pytest.approx(1.2)


def test_clv_same_line_only_and_line_points(tmp_path):
    g = _graded(tmp_path)
    # spread -5.5 vs close -6.5: no price CLV, +1 point of line value
    assert np.isnan(g.loc[0, "clv"]) and g.loc[0, "line_clv_pts"] == 1.0
    # over 218.5 vs close 220.5: +2 points
    assert g.loc[1, "line_clv_pts"] == 2.0
    # ML -170 vs close -200/+170 proportional de-vig
    qh, qa = 200 / 300, 100 / 270
    assert g.loc[2, "clv"] == pytest.approx((1 + 100 / 170) * qh / (qh + qa) - 1)
    assert g.loc[2, "close_quality"] == "espn_close"
    # prop under 28.5 at +100 vs ESPN last price 1.87/1.95 at the same line -> approx close
    qo, qu = 1 / 1.87, 1 / 1.95
    assert g.loc[5, "clv"] == pytest.approx(2.0 * qu / (qo + qu) - 1)
    assert g.loc[5, "close_quality"] == "approx"
    # game with no ESPN lines: graded, but no CLV
    assert np.isnan(g.loc[7, "clv"])


def test_summary(tmp_path):
    s = rt.summarize(_graded(tmp_path))
    prof = [100 / 110, -1, 100 / 170, 100 / 115, -1, 1.2]
    assert s["n_input"] == 8 and s["n_graded"] == 6
    assert s["excluded"] == {"posted_after_start": 1, "no_team_match": 1}
    assert s["roi"] == pytest.approx(np.mean(prof))
    assert s["hit_rate"] == pytest.approx(4 / 6)
    lo, hi = s["roi_ci95"]
    assert lo <= s["roi"] <= hi
    assert s["clv_game"]["n"] == 1 and s["clv_prop_approx"]["n"] == 1


def test_manual_overrides(tmp_path):
    p = PICKS.iloc[[6, 6, 6]].assign(result="win", close_price="+150",
                                      start_utc=["2026-01-10T00:30:00Z", None, "2026-01-09T17:00:00Z"])
    f = tmp_path / "m.csv"
    p.to_csv(f, index=False)
    g = rt.attach(rt.load_picks(f), GAMES, LINES, GRADED, BOX, ALIASES, use_kalshi=False)
    # unmatched + hand-graded: allowed only with a hand-entered start time before the post
    assert g.loc[1, "reason"] == "manual_needs_start_utc" and g.loc[2, "reason"] == "posted_after_start"
    assert g.loc[0, "reason"] == "ok" and g.loc[0, "result"] == 1 and g.loc[0, "profit"] == pytest.approx(2.0)
    assert g.loc[0, "clv"] == pytest.approx(3.0 / 2.5 - 1) and g.loc[0, "close_quality"] == "manual"


def test_kalshi_pick_posted_late_is_excluded_without_network(tmp_path):
    p = pd.DataFrame([dict(posted_at="2026-01-10T01:00:00Z", sport="nba", game="", market="moneyline", side="yes",
                           line=np.nan, price="0.55", source_url="", kalshi_ticker="KXNBAGAME-X",
                           start_utc="2026-01-10T00:30:00Z")])
    f = tmp_path / "k.csv"
    p.to_csv(f, index=False)
    g = rt.attach(rt.load_picks(f), GAMES, LINES, GRADED, BOX, ALIASES, use_kalshi=True)
    assert g.loc[0, "reason"] == "posted_after_start"


def test_boot_ci_and_power():
    x = np.r_[np.ones(50), -np.ones(50)]
    lo, hi = rt.boot_ci(x, np.arange(100))
    assert lo < 0 < hi
    assert rt.boot_ci(np.full(10, 0.3), np.arange(10)) == pytest.approx((0.3, 0.3))
    n2, n3 = rt.n_needed(0.02, rt.to_decimal("-110")), rt.n_needed(0.03, rt.to_decimal("-110"))
    assert n2 > n3 > 0 and n2 / n3 == pytest.approx(2.25, rel=0.05)   # ~ 1/edge^2
    assert rt.n_needed(0.02, 4.0) > n2                                # longshots need more


def test_snowflake_time_roundtrip_and_url_parsing(tmp_path):
    t = T("2026-01-09 18:00:00.123", tz="UTC")
    sid = (int(t.value // 1_000_000) - rt.TWITTER_EPOCH_MS) << 22 | 12345   # low bits = worker/sequence
    assert rt.snowflake_time(f"https://x.com/rotobot_ai/status/{sid}") == t
    assert rt.snowflake_time("no id here") is None
    p = tmp_path / "u.txt"
    p.write_text(f"https://x.com/rotobot_ai/status/{sid}\nCeltics -4.5 (-110)\n")
    assert rt.read_posts(p)[0]["posted_at"] == str(t)
    # a hand-typed posted_at that disagrees with the URL is overridden by the URL time
    f = tmp_path / "p.csv"
    PICKS.iloc[[0]].assign(posted_at="2026-01-11T00:00:00Z", source_url=f"https://x.com/a/status/{sid}").to_csv(f, index=False)
    assert rt.load_picks(f).loc[0, "posted_at"] == t

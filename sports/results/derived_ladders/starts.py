"""ESPN start times for Kalshi events by (league, ET date, team codes). Read-only. Uses sports/espn.py fetch+cache."""
import sys
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sports import espn
PATHS = {"NFL": "football/nfl", "NCAAF": "football/college-football", "NBA": "basketball/nba", "WNBA": "basketball/wnba",
         "NCAAMB": "basketball/mens-college-basketball", "NCAAWB": "basketball/womens-college-basketball", "NHL": "hockey/nhl",
         "MLB": "baseball/mlb", "EPL": "soccer/eng.1", "LALIGA": "soccer/esp.1", "SERIEA": "soccer/ita.1",
         "BUNDESLIGA": "soccer/ger.1", "LIGUE1": "soccer/fra.1", "MLS": "soccer/usa.1", "UCL": "soccer/uefa.champions"}
def league_of(series):
    s = series.removeprefix("KX")
    for k in sorted(PATHS, key=len, reverse=True):
        if s.startswith(k): return k
def day_games(lg, d):
    key = "x_" + lg
    espn.LEAGUES[key] = PATHS[lg]
    rows = []
    for dd in (d - timedelta(days=1), d, d + timedelta(days=1)):   # ESPN dates are UTC-ish; scan neighbours
        p = {"dates": dd.strftime("%Y%m%d"), "limit": 1000}
        if lg == "NCAAF": p["groups"] = 80
        if lg in ("NCAAMB", "NCAAWB"): p["groups"] = 50
        try:
            ev = espn.fetch(espn.SITE.format(path=PATHS[lg]) + "/scoreboard", p, cache=espn._settled(dd)).get("events", [])
        except Exception as e:
            print("espn err", lg, dd, e, flush=True); ev = []
        for e in ev:
            try: rows.append(espn.parse_event(key, e))
            except Exception: pass
    g = pd.DataFrame(rows).drop_duplicates("event_id") if rows else pd.DataFrame()
    if len(g):
        g["d"] = pd.to_datetime(g.start_utc, utc=True).dt.tz_convert("America/New_York").dt.date
        g = g[g.d == d]
    return g
def match(events):
    """events: DataFrame with series, event_ticker. Returns event_ticker -> start_utc (NaT if no unique match)."""
    out, cache = {}, {}
    for r in events.itertuples():
        lg = league_of(r.series)
        suf = r.event_ticker.split("-")[1]
        try: d = pd.to_datetime(suf[:7], format="%y%b%d").date()
        except Exception: out[r.event_ticker] = pd.NaT; continue
        teams = suf[7:]
        if (lg, d) not in cache: cache[(lg, d)] = day_games(lg, d)
        g = cache[(lg, d)]; st = pd.NaT
        if len(g):
            a, h = g.away.fillna("").str.upper(), g.home.fillna("").str.upper()
            ex = g[((a + h) == teams) | ((h + a) == teams)]
            if len(ex) == 1: st = ex.start_utc.iloc[0]
            else:
                c = g[[(teams.startswith(x[:2]) and teams.endswith(y[-2:])) or (teams.startswith(y[:2]) and teams.endswith(x[-2:]))
                       for x, y in zip(a, h)]]
                if len(c) == 1: st = c.start_utc.iloc[0]
        out[r.event_ticker] = st
    return pd.Series(out, name="start_utc")

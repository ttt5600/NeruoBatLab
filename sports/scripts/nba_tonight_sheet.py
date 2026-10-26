"""Tonight's NBA INFORMATION-EDGE sheet (descriptive, read-only, no orders).

User 2026-10-26: "run the numbers for tonight's games; try not to use exact model prediction, we need to see what no one
else is seeing." So the model is ONE cross-check column (bias-corrected total = model - 2.3, MODELS_GUIDE 7.6); the sheet
is built from market movement, Kalshi vs Pinnacle, availability, schedule, early-season shooting noise, structure and refs.

Usage (from sports/):
  ../.venv_quant/bin/python scripts/nba_tonight_sheet.py                    # today (ET), writes notes/tonight_nba_<date>.md
  ../.venv_quant/bin/python scripts/nba_tonight_sheet.py --date 2026-10-26 --model-json /path/nba1026.json
  ... --no-refresh   (skip the ESPN games/box/paint refresh; use cached parquet)
Re-run after the 9am-ET referee post and again ~60-90 min before tip (injury report / rest news).

Sources (each failure is reported in the sheet, never filled in):
  ESPN core odds (DraftKings open/current; ESPN lists whatever books it has -- on 10-26 only DK); ESPN publishes no
  public betting % in these feeds, so reverse-line-move flags are NOT computed.
  Pinnacle guest API (sports/live.py): current no-vig ML / spread / total (main line = lowest hold).
  Pinnacle OPEN total: first-seen line from the L32 logger state (~/.claude/quant-kalshi/data/open_totals_state.json).
  Kalshi: KXNBAGAME (ML), KXNBATOTAL (ladder -> implied total), KXNBASPREAD (ladder).
  ESPN injuries feed (status + date), ESPN rosters, ESPN box scores (data/espn/nba/box_<season>.parquet), ESPN paint pts.
  official.nba.com/referee-assignments + data/espn/nba/refs_dev.parquet (2021-22..2024-25 DEV) for crew tendencies.

Method notes:
  3P% shrinkage: shrunk = (3PM + K*prior) / (3PA + K), K = 750 attempts. Offence prior = the team's own prior-season
  3P%; defence prior = league prior-season 3P% (3P% allowed is mostly noise, notes/nba_comebacks_matchups.md).
  "Luck pts/g" = (raw - shrunk) * 3PA per game * 3. Game luck = 0.5 * (off_h + off_a + def_h + def_a): roughly how many
  points of a recent-form total come from shooting that should regress. Positive -> recent form overstates scoring.
  Paint: share of opponents' points scored in the paint; DEV finding -0.54 ATS pts per SD of (home - away) share.
  Kalshi fee = 0.07 p (1-p) per contract (taker). Kalshi-vs-Pinnacle on ladders uses a normal approx when the strike
  differs from Pinnacle's line (SD 18 total, 13 margin) -- marked "~".
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

from sports import espn, live
from sports.http import fetch

ESPN_DIR = ROOT / "data" / "espn" / "nba"
PAINT = ROOT / "data" / "nba_simple_model" / "live" / "paint_espn.parquet"
OPEN_STATE = Path.home() / ".claude" / "quant-kalshi" / "data" / "open_totals_state.json"
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ET = "America/New_York"
K3 = 750.0
SD_TOT, SD_MAR = 18.0, 13.0
BIAS = 2.3

# arena lat, lon, UTC offset in late Oct (DST), elevation ft
ARENA = {
    "ATL": (33.757, -84.396, -4, 1050), "BOS": (42.366, -71.062, -4, 20), "BKN": (40.683, -73.976, -4, 30),
    "CHA": (35.225, -80.839, -4, 750), "CHI": (41.881, -87.674, -5, 600), "CLE": (41.497, -81.688, -4, 650),
    "DAL": (32.790, -96.810, -5, 430), "DEN": (39.749, -105.008, -6, 5280), "DET": (42.341, -83.055, -4, 600),
    "GS": (37.768, -122.388, -7, 10), "HOU": (29.751, -95.362, -5, 50), "IND": (39.764, -86.155, -4, 715),
    "LAC": (33.945, -118.341, -7, 100), "LAL": (34.043, -118.267, -7, 300), "MEM": (35.138, -90.051, -5, 260),
    "MIA": (25.781, -80.188, -4, 10), "MIL": (43.045, -87.917, -5, 600), "MIN": (44.979, -93.276, -5, 830),
    "NO": (29.949, -90.082, -5, 5), "NY": (40.751, -73.993, -4, 30), "OKC": (35.463, -97.515, -5, 1200),
    "ORL": (28.539, -81.384, -4, 100), "PHI": (39.901, -75.172, -4, 40), "PHX": (33.446, -112.071, -7, 1090),
    "POR": (45.532, -122.667, -7, 50), "SAC": (38.580, -121.500, -7, 30), "SA": (29.427, -98.438, -5, 650),
    "TOR": (43.643, -79.379, -4, 250), "UTAH": (40.768, -111.901, -6, 4226), "WSH": (38.898, -77.021, -4, 50),
}
CITY = {"ATL": "Atlanta", "BOS": "Boston", "BKN": "Brooklyn", "CHA": "Charlotte", "CHI": "Chicago",
        "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit", "GS": "Golden State",
        "HOU": "Houston", "IND": "Indiana", "LAC": "LA Clippers", "LAL": "LA Lakers", "MEM": "Memphis",
        "MIA": "Miami", "MIL": "Milwaukee", "MIN": "Minnesota", "NO": "New Orleans", "NY": "New York",
        "OKC": "Oklahoma City", "ORL": "Orlando", "PHI": "Philadelphia", "PHX": "Phoenix", "POR": "Portland",
        "SAC": "Sacramento", "SA": "San Antonio", "TOR": "Toronto", "UTAH": "Utah", "WSH": "Washington"}
NICK = {"ATL": "hawks", "BOS": "celtics", "BKN": "nets", "CHA": "hornets", "CHI": "bulls", "CLE": "cavaliers",
        "DAL": "mavericks", "DEN": "nuggets", "DET": "pistons", "GS": "warriors", "HOU": "rockets", "IND": "pacers",
        "LAC": "clippers", "LAL": "lakers", "MEM": "grizzlies", "MIA": "heat", "MIL": "bucks", "MIN": "timberwolves",
        "NO": "pelicans", "NY": "knicks", "OKC": "thunder", "ORL": "magic", "PHI": "76ers", "PHX": "suns",
        "POR": "trail blazers", "SAC": "kings", "SA": "spurs", "TOR": "raptors", "UTAH": "jazz", "WSH": "wizards"}
KAL = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "BKN": "BKN"}
ABBR_OF_NICK = {v: k for k, v in NICK.items()}

ERR: list[str] = []
MARK = "<!-- AUTO: everything below is regenerated by scripts/nba_tonight_sheet.py -->"


def note_err(src: str, ex) -> None:
    ERR.append(f"{src}: {str(ex)[:120]}")


def km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def kfee(p: float) -> float:
    return 0.07 * p * (1 - p)


def am_prob(a) -> float | None:
    if a is None or not np.isfinite(a):
        return None
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def nickname(s: str) -> str | None:
    s = str(s).lower()
    for n in sorted(NICK.values(), key=len, reverse=True):
        if n in s:
            return n
    return None


# ------------------------------------------------------------------------------------------------------------ data
def load_season(season: int, d: Date, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if refresh:
        try:
            import nba_simple_live as nsl
            nsl.refresh_season(season, d, quiet=True)
        except Exception as ex:
            note_err("ESPN season refresh", ex)
    g = pd.read_parquet(ESPN_DIR / f"games_{season}.parquet")
    b = pd.read_parquet(ESPN_DIR / f"box_{season}.parquet")
    return g, b


def refresh_paint(g: pd.DataFrame) -> None:
    try:
        import nba_simple_live as nsl
        nsl.load_paint_live(g, quiet=True)
    except Exception as ex:
        note_err("ESPN paint refresh", ex)


def team_games(g: pd.DataFrame, b: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """One row per (event, team) for final regular-season games before `before`: points, opp points, team box sums."""
    f = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL") & (g.start_utc < before)].copy()
    f["event_id"] = f.event_id.astype(str)
    stats = ["threePointFieldGoalsMade", "threePointFieldGoalsAttempted", "fieldGoalsAttempted", "freeThrowsAttempted",
             "offensiveRebounds", "turnovers", "points"]
    bb = b[b.stat.isin(stats)].assign(event_id=lambda x: x.event_id.astype(str), team_id=lambda x: x.team_id.astype(str))
    tb = bb.pivot_table(index=["event_id", "team_id"], columns="stat", values="value", aggfunc="sum").reset_index()
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = f[["event_id", "start_utc", side, f"{side}_team_id", f"{side}_score", opp, f"{opp}_team_id", f"{opp}_score"]]
        x.columns = ["event_id", "start_utc", "team", "team_id", "pts", "opp", "opp_id", "opp_pts"]
        x = x.assign(home=side == "home", team_id=x.team_id.astype(str), opp_id=x.opp_id.astype(str))
        x = x.merge(f[["event_id", "home"]].rename(columns={"home": "host"}), on="event_id")
        rows.append(x)
    t = pd.concat(rows, ignore_index=True)
    t = t.merge(tb.rename(columns={"team_id": "team_id"}), on=["event_id", "team_id"], how="left")
    o = tb.rename(columns={c: "o_" + c for c in tb.columns if c not in ("event_id",)}).rename(columns={"o_team_id": "opp_id"})
    t = t.merge(o, on=["event_id", "opp_id"], how="left")
    poss = lambda p: p + "fieldGoalsAttempted"
    t["poss"] = t.fieldGoalsAttempted - t.offensiveRebounds + t.turnovers + 0.44 * t.freeThrowsAttempted
    t["o_poss"] = t.o_fieldGoalsAttempted - t.o_offensiveRebounds + t.o_turnovers + 0.44 * t.o_freeThrowsAttempted
    t["pace"] = (t.poss + t.o_poss) / 2
    if PAINT.exists():
        p = pd.read_parquet(PAINT).assign(event_id=lambda x: x.event_id.astype(str), team_id=lambda x: x.team_id.astype(str))
        t = t.merge(p.rename(columns={"team_id": "opp_id", "paint": "opp_paint"}), on=["event_id", "opp_id"], how="left")
    else:
        t["opp_paint"] = np.nan
    return t.sort_values("start_utc")


def team_table(t: pd.DataFrame) -> pd.DataFrame:
    a = t.groupby("team").agg(gp=("event_id", "nunique"), pts=("pts", "mean"), opp_pts=("opp_pts", "mean"),
                              m3=("threePointFieldGoalsMade", "sum"), a3=("threePointFieldGoalsAttempted", "sum"),
                              om3=("o_threePointFieldGoalsMade", "sum"), oa3=("o_threePointFieldGoalsAttempted", "sum"),
                              pace=("pace", "mean"), opp_paint=("opp_paint", "sum"),
                              opp_pts_paintgames=("opp_pts", lambda s: s[t.loc[s.index, "opp_paint"].notna()].sum()))
    a["margin"] = a.pts - a.opp_pts
    a["p3"] = a.m3 / a.a3
    a["op3"] = a.om3 / a.oa3
    a["paint_sh_allowed"] = a.opp_paint / a.opp_pts_paintgames
    return a


# --------------------------------------------------------------------------------------------------------- markets
def espn_lines(eid: str) -> dict:
    try:
        ls = espn.lines("nba", eid, cache=False)
    except Exception as ex:
        note_err(f"ESPN odds {eid}", ex)
        return {}
    out = {}
    for x in ls:
        out[x["provider"]] = x
    return out


def pinnacle_now() -> dict:
    try:
        p = live.pinnacle("nba")
    except Exception as ex:
        note_err("Pinnacle", ex)
        return {}
    p = p[~p.live]
    out = {}
    for pid, gg in p.groupby("pin_id"):
        h, a = nickname(gg.home.iloc[0]), nickname(gg.away.iloc[0])
        if not h or not a:
            continue
        r = {"start": gg.start.iloc[0]}
        for mk in ("moneyline", "spread", "total"):
            s = gg[gg.market == mk]
            if len(s):
                x = s.iloc[s.hold.argsort()].iloc[0]
                r[mk] = {"line": x.line, "p1": x.price1, "p2": x.price2, "fair1": x.fair1, "fair2": x.fair2}
        out[(h, a)] = r
    return out


def pin_open_totals(d: Date) -> dict:
    try:
        s = json.loads(OPEN_STATE.read_text())
    except Exception as ex:
        note_err("L32 logger state (Pinnacle open totals)", ex)
        return {}
    out = {}
    for k, v in s.items():
        if not k.startswith(d.isoformat()):
            continue
        out[(v["home"], v["away"])] = {"pin_open": (v.get("pin_open") or {}).get("line"),
                                       "pin_open_t": (v.get("pin_open") or {}).get("t"),
                                       "kal_open": (v.get("kal_open") or {}).get("implied")}
    return out


def kalshi_series(series: str) -> list[dict]:
    out, cursor = [], None
    try:
        while True:
            p = {"series_ticker": series, "status": "open", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            dd = live._get(f"{live.KALSHI}/markets", params=p)
            out += dd.get("markets", [])
            cursor = dd.get("cursor")
            if not cursor or not dd.get("markets"):
                break
    except Exception as ex:
        note_err(f"Kalshi {series}", ex)
    return out


def fnum(m, k):
    v = m.get(k)
    return float(v) if v not in (None, "") else np.nan


def kalshi_tag(d: Date, home: str, away: str) -> str:
    return d.strftime("%y%b%d").upper() + KAL.get(away, away) + KAL.get(home, home)


def kalshi_game(d, home, away, kg) -> dict:
    tag = kalshi_tag(d, home, away)
    out = {}
    for m in kg:
        if m["event_ticker"].endswith(tag):
            side = "home" if m["ticker"].endswith("-" + KAL.get(home, home)) else "away"
            out[side] = {"bid": fnum(m, "yes_bid_dollars"), "ask": fnum(m, "yes_ask_dollars"), "vol": m.get("volume_fp")}
    return out


def kalshi_ladder(d, home, away, mk) -> dict:
    tag = kalshi_tag(d, home, away)
    lad = {}
    for m in mk:
        if m["event_ticker"].endswith(tag) and m.get("floor_strike") is not None:
            lad[m["ticker"]] = {"strike": float(m["floor_strike"]), "sub": m.get("yes_sub_title"),
                                "yb": fnum(m, "yes_bid_dollars"), "ya": fnum(m, "yes_ask_dollars"),
                                "nb": fnum(m, "no_bid_dollars"), "na": fnum(m, "no_ask_dollars")}
    return lad


def implied_total(lad: dict) -> float | None:
    pts = sorted((v["strike"], (v["yb"] + v["ya"]) / 2) for v in lad.values()
                 if np.isfinite(v["yb"]) and np.isfinite(v["ya"]) and v["ya"] - v["yb"] <= 0.10)
    for (k1, p1), (k2, p2) in zip(pts, pts[1:]):
        if p1 >= 0.5 >= p2 and p1 != p2:
            return round(k1 + (p1 - 0.5) / (p1 - p2) * (k2 - k1), 1)
    return None


def margin_pmf(line: float, p_cover: float, p_win: float | None = None) -> pd.Series:
    """Integer home-margin pmf, no ties. Mean fitted to the Pinnacle spread; if p_win (Pinnacle no-vig ML) is given,
    the SD is chosen (8-22) so the pmf's P(home win) also matches the ML -- keeps +/-1-point masses honest."""
    if p_win is None:
        return _pmf(line, p_cover, SD_MAR)
    best = None
    for sd in np.arange(8.0, 22.01, 0.5):
        p = _pmf(line, p_cover, sd)
        err = abs(float(p[p.index > 0].sum()) - p_win)
        if best is None or err < best[0]:
            best = (err, p)
    return best[1]


def _pmf(line: float, p_cover: float, sd: float) -> pd.Series:
    ks = np.array([k for k in range(-70, 71) if k != 0])

    def pm(mu):
        w = norm.cdf(ks + 0.5, mu, sd) - norm.cdf(ks - 0.5, mu, sd)
        return w / w.sum()
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mu = (lo + hi) / 2
        if pm(mu)[ks > line].sum() < p_cover:
            lo = mu
        else:
            hi = mu
    return pd.Series(pm(mu), index=ks)


def p_over(k: float, line: float, fair_over: float, sd: float) -> float:
    """P(X > k) given P(X > line) = fair_over, normal with sd."""
    z0 = norm.ppf(1 - fair_over)
    return float(1 - norm.cdf((k - line) / sd + z0))


# ------------------------------------------------------------------------------------------------------- injuries
def injuries_raw() -> dict:
    try:
        j = fetch(f"{SITE}/injuries", cache=False, retries=3, timeout=20)
    except Exception as ex:
        note_err("ESPN injuries", ex)
        return {}
    out = {}
    for t in j.get("injuries", []):
        tid = str(t.get("id"))
        for i in t.get("injuries", []):
            a = i.get("athlete", {}) or {}
            m = None
            for l in a.get("links", []) or []:
                m = m or re.search(r"/id/(\d+)", l.get("href", ""))
            out.setdefault(tid, []).append({"aid": m.group(1) if m else None, "name": a.get("displayName"),
                                            "status": i.get("status"), "date": (i.get("date") or "")[:10],
                                            "comment": (i.get("shortComment") or "")[:140]})
    return out


def rosters(tids) -> dict:
    out = {}
    for t in tids:
        try:
            j = fetch(f"{SITE}/teams/{t}/roster", cache=False, retries=3, timeout=20)
            out[str(t)] = {str(a["id"]): a.get("displayName") for a in j.get("athletes", [])}
        except Exception as ex:
            note_err(f"ESPN roster {t}", ex)
    return out


def missed_last(team_id: str, t: pd.DataFrame, b: pd.DataFrame, inj: dict, roster: dict) -> list[str]:
    """Rotation players (>= 15 mpg in games played this season) who did not play the team's last game and are NOT on
    the injury report -> rest/unknown risk."""
    ev = t[t.team_id == team_id].sort_values("start_utc").event_id.tolist()
    if not ev:
        return []
    mins = b[(b.stat == "minutes") & (b.team_id.astype(str) == team_id) & b.event_id.astype(str).isin(ev)]
    mins = mins.assign(event_id=mins.event_id.astype(str), athlete_id=mins.athlete_id.astype(str))
    played = mins[mins.value > 0]
    mpg = played.groupby(["athlete_id", "name"]).value.mean()
    last = set(played[played.event_id == ev[-1]].athlete_id)
    listed = {x["aid"] for x in inj.get(team_id, [])}
    out = []
    for (aid, name), m in mpg.items():
        if m >= 15 and aid not in last and aid not in listed and (not roster or aid in roster):
            out.append(f"{name} ({m:.0f} mpg)")
    return out


def b2b_history(seasons: list[tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """Per player: games missed on the 2nd night of a back-to-back vs other nights, over the given seasons.
    A player 'belongs' to a team for its games between his first and last appearance for that team."""
    rows = []
    for g, b in seasons:
        f = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL")].copy()
        f["d"] = f.start_utc.dt.tz_convert(ET).dt.date
        tg = pd.concat([f[["event_id", "d", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
                        f[["event_id", "d", "away_team_id"]].rename(columns={"away_team_id": "team_id"})])
        tg = tg.assign(event_id=tg.event_id.astype(str), team_id=tg.team_id.astype(str)).sort_values("d")
        tg["dt"] = pd.to_datetime(tg.d)
        tg["b2b2"] = tg.groupby("team_id").dt.diff().dt.days.eq(1)
        m = b[b.stat == "minutes"].assign(event_id=lambda x: x.event_id.astype(str), team_id=lambda x: x.team_id.astype(str),
                                          athlete_id=lambda x: x.athlete_id.astype(str))
        m = m[m.event_id.isin(tg.event_id)]
        app = m[m.value > 0].merge(tg[["event_id", "d"]], on="event_id")
        span = app.groupby(["athlete_id", "name", "team_id"]).agg(first=("d", "min"), last=("d", "max"), mpg=("value", "mean")).reset_index()
        played = set(zip(app.athlete_id, app.event_id))
        for _, s in span.iterrows():
            gs = tg[(tg.team_id == s.team_id) & (tg.d >= s["first"]) & (tg.d <= s["last"])]
            for _, x in gs.iterrows():
                rows.append((s.athlete_id, s["name"], s.mpg, bool(x.b2b2), (s.athlete_id, x.event_id) not in played))
    h = pd.DataFrame(rows, columns=["aid", "name", "mpg", "b2b2", "dnp"])
    a = h.groupby(["aid", "b2b2"]).dnp.agg(["mean", "size"]).unstack()
    a.columns = [f"{x}_{'b2b2' if y else 'other'}" for x, y in a.columns]
    a["mpg"] = h.groupby("aid").mpg.max()
    a["name"] = h.groupby("aid").name.first()
    return a.reset_index()


# --------------------------------------------------------------------------------------------------------- refs
def refs_tonight() -> dict:
    try:
        h = requests.get("https://official.nba.com/referee-assignments/", headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        h.raise_for_status()
    except Exception as ex:
        note_err("official.nba.com referee assignments", ex)
        return {}
    tabs = re.findall(r"<table.*?</table>", h.text, re.S)
    if not tabs:
        return {}
    out = {}
    for tr in re.findall(r"<tr.*?</tr>", tabs[0], re.S):     # first table = NBA games (second = other)
        cells = [re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", c)).strip() for c in re.findall(r"<td.*?</td>", tr, re.S)]
        cells = [c for c in cells if c]
        if len(cells) >= 2 and "@" in cells[0]:
            aw, hm = [x.strip() for x in cells[0].split("@")]
            names = [re.sub(r"\s*\(#\d+\)", "", c).strip() for c in cells[1:]]
            out[(hm, aw)] = names
    return out


def ref_tendencies() -> pd.DataFrame:
    r = pd.read_parquet(ESPN_DIR / "refs_dev.parquet")
    r = r[r.season_type == "regular-season"].copy()
    r["fta"] = r.home_fta + r.away_fta
    r["pts"] = r.home_score + r.away_score
    r["fta_res"] = r.fta - r.groupby("season").fta.transform("mean")
    r["tot_res"] = r.pts - r.mkt_total
    long = pd.concat([r[[f"ref{i}", "fta_res", "tot_res"]].rename(columns={f"ref{i}": "ref"}) for i in (1, 2, 3, 4)])
    long = long.dropna(subset=["ref"])
    a = long.groupby("ref").agg(n=("fta_res", "size"), fta=("fta_res", "mean"), tot=("tot_res", "mean"))
    k = 60.0  # shrink toward 0 with 60 games of prior
    a["fta_s"] = a.fta * a.n / (a.n + k)
    a["tot_s"] = a.tot * a.n / (a.n + k)
    return a


# ---------------------------------------------------------------------------------------------------------- main
def fmt(x, n=1, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:+.{n}f}" if plus else f"{x:.{n}f}"


def am(x):
    return "-" if x is None or not np.isfinite(x) else f"{int(x):+d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--model-json")
    ap.add_argument("--out")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args(argv)
    d = pd.Timestamp(a.date).date() if a.date else pd.Timestamp.now(tz=ET).date()
    season = d.year + 1 if d.month >= 8 else d.year
    run_t = pd.Timestamp.now(tz=ET)
    day_start = pd.Timestamp(d, tz=ET).tz_convert("UTC")

    g, b = load_season(season, d, not a.no_refresh)
    if not a.no_refresh:
        refresh_paint(g)
    gp, bp = pd.read_parquet(ESPN_DIR / f"games_{season - 1}.parquet"), pd.read_parquet(ESPN_DIR / f"box_{season - 1}.parquet")
    t_now = team_games(g, b, day_start)
    t_prev = team_games(gp, bp, pd.Timestamp("2100-01-01", tz="UTC"))
    T, P = team_table(t_now), team_table(t_prev)
    lg_p3_prev = P.m3.sum() / P.a3.sum()
    lg_p3_now = T.m3.sum() / T.a3.sum()
    # DEV coefficient is per SD of the home-minus-away DIFFERENCE -> SD(diff) ~= sqrt(2) x SD(team share)
    paint_sd_prev = P.paint_sh_allowed.std() * math.sqrt(2)
    try:
        bh = b2b_history([(gp, bp), (g[g.start_utc < day_start], b)])
    except Exception as ex:
        note_err("B2B rest history", ex)
        bh = pd.DataFrame()

    tonight = g[(g.start_utc >= day_start) & (g.start_utc < day_start + pd.Timedelta(days=1)) &
                (g.season_type == "regular-season")].copy()
    tonight["event_id"] = tonight.event_id.astype(str)

    model = {}
    if a.model_json:
        try:
            for r in json.loads(Path(a.model_json).read_text()):
                model[str(r["event_id"])] = r
        except Exception as ex:
            note_err("model json", ex)

    pin, pin_open = pinnacle_now(), pin_open_totals(d)
    kg, kt, ks = kalshi_series("KXNBAGAME"), kalshi_series("KXNBATOTAL"), kalshi_series("KXNBASPREAD")
    inj = injuries_raw()
    ros = rosters(sorted(set(tonight.home_team_id.astype(str)) | set(tonight.away_team_id.astype(str))))
    refs = refs_tonight()
    rt = None
    try:
        rt = ref_tendencies()
    except Exception as ex:
        note_err("refs_dev", ex)

    # league schedule for rest/travel: all games incl. tonight
    sched = g[(g.season_type == "regular-season")].copy()

    games = []
    for _, r in tonight.sort_values("start_utc").iterrows():
        H, A = r.home, r.away
        hn, an = NICK.get(H), NICK.get(A)
        eid = r.event_id
        G = {"eid": eid, "home": H, "away": A, "tip_et": r.start_utc.tz_convert(ET).strftime("%H:%M")}

        # -------- lines
        L = espn_lines(eid)
        G["books"] = {}
        for prov, x in L.items():
            G["books"][prov] = {"spread_open": x.get("home_spread_open"), "spread": x.get("spread"),
                                "total_open": x.get("total_open"), "total": x.get("total"),
                                "ml_open": (x.get("home_ml_open"), x.get("away_ml_open")),
                                "ml": (x.get("home_ml"), x.get("away_ml")),
                                "sp_odds": (x.get("home_spread_odds"), x.get("away_spread_odds")),
                                "tot_odds": (x.get("over_odds"), x.get("under_odds"))}
        pz = pin.get((hn, an), {})
        G["pin"] = pz
        po = pin_open.get((hn, an), {})
        G["pin_open_total"], G["pin_open_t"], G["kal_open_total"] = po.get("pin_open"), po.get("pin_open_t"), po.get("kal_open")
        lad_t = kalshi_ladder(d, H, A, kt)
        lad_s = kalshi_ladder(d, H, A, ks)
        G["kal_total"] = implied_total(lad_t)
        G["kal_game"] = kalshi_game(d, H, A, kg)
        mj = model.get(eid, {})
        G["model_snap_total"] = mj.get("market_total")

        # -------- Kalshi vs Pinnacle
        flags = []
        if "moneyline" in pz and G["kal_game"]:
            for side, fair in (("home", pz["moneyline"]["fair1"]), ("away", pz["moneyline"]["fair2"])):
                k = G["kal_game"].get(side)
                if k and np.isfinite(k["ask"]):
                    e = fair - k["ask"] - kfee(k["ask"])
                    G[f"kv_ml_{side}"] = (k["ask"], fair, e)
                    if e >= 0.02:
                        flags.append(f"KALSHI ML {H if side == 'home' else A} ask {k['ask']:.2f} vs Pin fair {fair:.3f} (edge {e:+.3f} after fee)")
        best_tot = None
        if "total" in pz and lad_t:
            L0, fo = float(pz["total"]["line"]), float(pz["total"]["fair1"])
            for v in lad_t.values():
                if abs(v["strike"] - L0) > 1.6:
                    continue
                po_ = p_over(v["strike"], L0, fo, SD_TOT)
                exact = abs(v["strike"] - L0) < 1e-6
                for nm, ask, fair in (("over", v["ya"], po_), ("under", v["na"], 1 - po_)):
                    if np.isfinite(ask):
                        e = fair - ask - kfee(ask)
                        if best_tot is None or e > best_tot[3]:
                            best_tot = (nm, v["strike"], ask, e, exact)
            if best_tot and best_tot[3] >= 0.02:
                flags.append(f"KALSHI TOTAL {best_tot[0]} {best_tot[1]} ask {best_tot[2]:.2f} edge {best_tot[3]:+.3f}{'' if best_tot[4] else ' (~normal approx)'}")
        G["kv_tot_best"] = best_tot
        best_sp = None
        if "spread" in pz and lad_s:
            hl = float(pz["spread"]["line"])          # home line, e.g. -7 -> home must win by > 7
            fh = float(pz["spread"]["fair1"])         # P(home covers)
            pw = float(pz["moneyline"]["fair1"]) if "moneyline" in pz else None
            pmf = margin_pmf(-hl, fh, pw)             # integer home margins, no ties, fitted to Pinnacle spread (+ ML)
            if "moneyline" in pz:
                G["ml_check"] = (float(pmf[pmf.index > 0].sum()), float(pz["moneyline"]["fair1"]))
            for v in lad_s.values():
                tk = v["sub"] or ""
                home_team = CITY[H].split()[-1] in tk or (H == "GS" and "Golden State" in tk)
                # "X wins by over s": home: P(margin > s); away: P(margin < -s)
                s = v["strike"]
                po_ = float(pmf[pmf.index > s].sum()) if home_team else float(pmf[pmf.index < -s].sum())
                anchor = -hl if home_team else hl     # strike distance from the Pinnacle line, same team's view
                if abs(s - anchor) > 2.6:
                    continue
                for nm, ask, fair in (("yes", v["ya"], po_), ("no", v["na"], 1 - po_)):
                    if np.isfinite(ask):
                        e = fair - ask - kfee(ask)
                        if best_sp is None or e > best_sp[3]:
                            best_sp = (f"{nm}: {tk}", v["strike"], ask, e)
            if best_sp and best_sp[3] >= 0.02:
                flags.append(f"KALSHI SPREAD {best_sp[0]} ask {best_sp[2]:.2f} edge {best_sp[3]:+.3f} (~normal approx)")
        G["kv_sp_best"] = best_sp

        # -------- each ESPN-listed book vs Pinnacle (EV per $1 at the book's price, Pinnacle-implied probability)
        G["book_ev"] = []
        for prov, x in G["books"].items():
            if "spread" in pz and x.get("spread") is not None and all(v is not None and np.isfinite(v) for v in x["sp_odds"]):
                pmf = margin_pmf(-float(pz["spread"]["line"]), float(pz["spread"]["fair1"]),
                                 float(pz["moneyline"]["fair1"]) if "moneyline" in pz else None)
                s = float(x["spread"])                 # home line at the book
                ph, pa = float(pmf[pmf.index > -s].sum()), float(pmf[pmf.index < -s].sum())
                tot = ph + pa                          # whole-number line: exclude push
                for nm, p, o in ((f"{H} {s:+g}", ph / tot, x["sp_odds"][0]), (f"{A} {-s:+g}", pa / tot, x["sp_odds"][1])):
                    G["book_ev"].append((prov, nm, o, p, p * live.am_to_dec(o) - 1))
            if "total" in pz and x.get("total") is not None and all(v is not None and np.isfinite(v) for v in x["tot_odds"]):
                L0, fo = float(pz["total"]["line"]), float(pz["total"]["fair1"])
                po_ = p_over(float(x["total"]), L0, fo, SD_TOT)
                for nm, p, o in ((f"o{x['total']:g}", po_, x["tot_odds"][0]), (f"u{x['total']:g}", 1 - po_, x["tot_odds"][1])):
                    G["book_ev"].append((prov, nm, o, p, p * live.am_to_dec(o) - 1))
            if "moneyline" in pz and all(v is not None and np.isfinite(v) for v in x["ml"]):
                for nm, p, o in ((f"{H} ML", float(pz["moneyline"]["fair1"]), x["ml"][0]), (f"{A} ML", float(pz["moneyline"]["fair2"]), x["ml"][1])):
                    G["book_ev"].append((prov, nm, o, p, p * live.am_to_dec(o) - 1))
        for e in G["book_ev"]:
            if e[4] >= 0.02:
                flags.append(f"BOOK {e[0]} {e[1]} {int(e[2]):+d}: Pinnacle-implied p {e[3]:.3f}, EV {e[4]:+.1%}")

        # -------- availability
        G["inj"] = {}
        G["unlisted"] = {}
        G["rest"] = {}
        for side, ab, tid in (("home", H, str(r.home_team_id)), ("away", A, str(r.away_team_id))):
            G["inj"][ab] = [x for x in inj.get(tid, []) if (x["status"] or "").lower() not in ("active", "probable")]
            G["unlisted"][ab] = missed_last(tid, t_now, b, inj, ros.get(tid, {}))
            sc = sched[((sched.home == ab) | (sched.away == ab)) & (sched.start_utc < r.start_utc)].sort_values("start_utc")
            prev = sc.iloc[-1] if len(sc) else None
            info = {"gp": int(T.gp.get(ab, 0))}
            if prev is not None:
                pd_et = prev.start_utc.tz_convert(ET).date()
                info["days_rest"] = (d - pd_et).days - 1
                info["b2b"] = info["days_rest"] == 0
                loc_prev = ARENA.get(prev.home)
                loc_now = ARENA[H]
                info["prev"] = f"{'vs' if prev.home == ab else '@'} {prev.away if prev.home == ab else prev.home} {pd_et:%m-%d}"
                if loc_prev:
                    info["travel_km"] = km(loc_prev, loc_now)
                    info["tz_shift"] = loc_now[2] - loc_prev[2]
                # last 7 days load
                info["g_last7"] = int(((sc.start_utc.dt.tz_convert(ET).dt.date >= d - timedelta(days=7))).sum())
            nxt = sched[((sched.home == ab) | (sched.away == ab)) & (sched.start_utc > r.start_utc)].sort_values("start_utc")
            if len(nxt):
                info["next"] = f"{nxt.iloc[0].start_utc.tz_convert(ET).date():%m-%d}"
                info["b2b_front"] = (nxt.iloc[0].start_utc.tz_convert(ET).date() - d).days == 1
            if info.get("b2b") and len(bh):
                rid = set(ros.get(tid, {}))
                x = bh[bh.aid.isin(rid) & (bh.mpg >= 20) & (bh.get("size_b2b2", 0) >= 4)]
                x = x[(x.mean_b2b2 - x.mean_other) >= 0.2]
                info["b2b_sitters"] = [f"{r_.name} sat {r_.mean_b2b2:.0%} of {int(r_.size_b2b2)} 2nd-nights vs {r_.mean_other:.0%} other"
                                       for r_ in x.itertuples()]
            G["rest"][ab] = info

        # -------- early-season noise
        G["noise"] = {}
        for ab in (H, A):
            if ab not in T.index:
                continue
            x = T.loc[ab]
            pr = P.loc[ab] if ab in P.index else None
            prior_off = float(pr.p3) if pr is not None else lg_p3_prev
            prior_def = lg_p3_prev
            s_off = (x.m3 + K3 * prior_off) / (x.a3 + K3)
            s_def = (x.om3 + K3 * prior_def) / (x.oa3 + K3)
            a3g, oa3g = x.a3 / x.gp, x.oa3 / x.gp
            # opponent quality: mean prior-season margin of opponents faced
            opps = t_now[t_now.team == ab].opp
            oq = float(np.nanmean([P.margin.get(o, np.nan) for o in opps])) if len(opps) else np.nan
            G["noise"][ab] = {
                "gp": int(x.gp), "p3": x.p3, "p3_prev": prior_off, "p3_shr": s_off, "a3g": a3g,
                "op3": x.op3, "op3_prev": float(pr.op3) if pr is not None else np.nan, "op3_shr": s_def, "oa3g": oa3g,
                "off_luck": (x.p3 - s_off) * a3g * 3, "def_luck": (x.op3 - s_def) * oa3g * 3,
                "margin": x.margin, "margin_prev": float(pr.margin) if pr is not None else np.nan, "opp_q": oq,
                "pace": x.pace, "pace_prev": float(pr.pace) if pr is not None else np.nan,
                "ppg": x.pts, "oppg": x.opp_pts,
                "paint_sh": x.paint_sh_allowed, "paint_sh_prev": float(pr.paint_sh_allowed) if pr is not None else np.nan,
            }
        nh, na = G["noise"].get(H), G["noise"].get(A)
        if nh and na:
            G["game_luck"] = 0.5 * (nh["off_luck"] + na["off_luck"] + nh["def_luck"] + na["def_luck"])
            # paint: blend this season with last at (gp vs 20 games of prior)
            def blend(n):
                if not np.isfinite(n["paint_sh"]):
                    return n["paint_sh_prev"]
                w = n["gp"] / (n["gp"] + 20)
                return w * n["paint_sh"] + (1 - w) * n["paint_sh_prev"]
            G["paint_diff_z"] = (blend(nh) - blend(na)) / paint_sd_prev if paint_sd_prev else np.nan
            G["paint_ats_home"] = -0.54 * G["paint_diff_z"]       # + = favours home ATS
            G["pace_proj"] = 0.5 * (0.3 * nh["pace"] + 0.7 * nh["pace_prev"]) + 0.5 * (0.3 * na["pace"] + 0.7 * na["pace_prev"])
        G["altitude"] = ARENA[H][3] >= 4000

        # -------- refs
        crew = refs.get((CITY.get(H, H), CITY.get(A, A)))
        if crew is None:
            for (hm, aw), names in refs.items():
                if hm.split()[-1] in CITY.get(H, H) or CITY.get(H, H).split()[-1] in hm:
                    if aw.split()[-1] in CITY.get(A, A) or CITY.get(A, A).split()[-1] in aw:
                        crew = names
        G["crew"] = crew
        if crew and rt is not None:
            rows = [(nm, rt.loc[nm]) if nm in rt.index else (nm, None) for nm in crew]
            G["crew_detail"] = [(nm, int(x.n), x.fta_s, x.tot_s) if x is not None else (nm, 0, np.nan, np.nan) for nm, x in rows]
            known = [x for x in G["crew_detail"] if x[1] > 0]
            G["crew_fta"] = float(np.mean([x[2] for x in known])) if known else np.nan
            G["crew_tot"] = float(np.mean([x[3] for x in known])) if known else np.nan

        # -------- model cross-check (ONE input)
        if mj:
            G["model_total_bc"] = mj.get("model_total") - BIAS if mj.get("model_total") is not None else None
            G["model_margin"] = mj.get("model_margin")
        G["flags"] = flags
        games.append(G)

    md = render(d, run_t, games, lg_p3_prev, lg_p3_now, refs)
    out = Path(a.out) if a.out else ROOT / "notes" / f"tonight_nba_{d.isoformat()}.md"
    head = ""
    if out.exists() and MARK in out.read_text():          # keep the hand-written read-out above the marker
        head = out.read_text().split(MARK)[0]
    out.write_text(head + MARK + "\n\n" + md if head else md)
    (ROOT / "data" / "live").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "live" / f"tonight_nba_{d.isoformat()}.json").write_text(json.dumps(games, default=str, indent=1))
    print(md)
    return 0


def render(d, run_t, games, lg_prev, lg_now, refs) -> str:
    o = [f"# Tonight's NBA information-edge sheet, {d} (auto section)", "",
         f"Generated {run_t:%Y-%m-%d %H:%M} ET by `scripts/nba_tonight_sheet.py`. Descriptive only; no orders. "
         f"League 3P%: prior season {lg_prev:.3f}, this season so far {lg_now:.3f}.", ""]
    if ERR:
        o += ["**Source failures:** " + "; ".join(ERR), ""]
    o += ["ESPN lists no public betting % in its feeds -> reverse-line-move flags are not computed (not invented).", ""]

    o += ["## 1. Line movement", "",
          "| Game | Tip ET | DK spread open->now (home) | DK total open->now | DK ML open->now (H/A) | Pin open total (first seen) | Pin now: spread / total / home no-vig | Kalshi ML ask H/A | Kalshi implied total (open->now) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for G in games:
        bk = G["books"].get("DraftKings") or (next(iter(G["books"].values())) if G["books"] else {})
        pz = G["pin"]
        sp = f"{fmt(bk.get('spread_open'), 1, True)} -> {fmt(bk.get('spread'), 1, True)}" if bk else "-"
        tt = f"{fmt(bk.get('total_open'))} -> {fmt(bk.get('total'))}" if bk else "-"
        ml = f"{am(bk['ml_open'][0])}/{am(bk['ml_open'][1])} -> {am(bk['ml'][0])}/{am(bk['ml'][1])}" if bk else "-"
        pn = (f"{fmt(pz['spread']['line'], 1, True) if 'spread' in pz else '-'} / {fmt(pz['total']['line']) if 'total' in pz else '-'} / "
              f"{fmt(pz['moneyline']['fair1'], 3) if 'moneyline' in pz else '-'}") if pz else "unavailable"
        kg = G["kal_game"]
        ka = f"{fmt(kg.get('home', {}).get('ask'), 2)}/{fmt(kg.get('away', {}).get('ask'), 2)}" if kg else "not listed"
        kt = f"{fmt(G['kal_open_total'])} -> {fmt(G['kal_total'])}"
        o.append(f"| {G['away']}@{G['home']} | {G['tip_et']} | {sp} | {tt} | {ml} | {fmt(G['pin_open_total'])} | {pn} | {ka} | {kt} |")
    o += ["", "Books listed by ESPN: " + ", ".join(sorted({p for G in games for p in G["books"]})) + ".", ""]

    o += ["## 2. Kalshi vs Pinnacle (edge = Pinnacle no-vig - Kalshi ask - taker fee; flag >= +0.02)", "",
          "| Game | ML home ask/fair/edge | ML away ask/fair/edge | best total contract | best spread contract | best 2 book prices vs Pinnacle (EV/$1) |",
          "|---|---|---|---|---|---|"]
    for G in games:
        def mlc(s):
            v = G.get(f"kv_ml_{s}")
            return f"{v[0]:.2f}/{v[1]:.3f}/{v[2]:+.3f}" if v else "-"
        bt, bs = G["kv_tot_best"], G["kv_sp_best"]
        bts = f"{bt[0]} {bt[1]} @ {bt[2]:.2f}: {bt[3]:+.3f}{'' if bt[4] else ' ~'}" if bt else "-"
        bss = f"{bs[0]} @ {bs[2]:.2f}: {bs[3]:+.3f} ~" if bs else "-"
        if G.get("ml_check"):
            bss += f" (pmf P(home win) {G['ml_check'][0]:.3f} vs Pin ML {G['ml_check'][1]:.3f})"
        be = sorted(G.get("book_ev", []), key=lambda e: -e[4])[:2]
        bes = "; ".join(f"{e[0]} {e[1]} {int(e[2]):+d} p {e[3]:.3f} EV {e[4]:+.1%}" for e in be) or "-"
        o.append(f"| {G['away']}@{G['home']} | {mlc('home')} | {mlc('away')} | {bts} | {bss} | {bes} |")
    o += ["", "Flags: " + ("; ".join(f for G in games for f in G["flags"]) or "**none >= 2c**"), ""]

    o += ["## 3. Availability and schedule", "",
          "| Game | Team | GP | Rest days (prev game) | B2B? | Travel km / tz shift | Next game (B2B front?) | Injury report (status, date) | Missed last game, NOT listed | B2B 2nd-night sitters (prev+this season, >=20 mpg) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for G in games:
        for ab in (G["away"], G["home"]):
            R = G["rest"].get(ab, {})
            ij = "; ".join(f"{x['name']} {x['status']} ({x['date']})" for x in G["inj"].get(ab, [])) or "none"
            un = ", ".join(G["unlisted"].get(ab, [])) or "-"
            tr = f"{R.get('travel_km', float('nan')):.0f} / {R.get('tz_shift', 0):+d}h" if "travel_km" in R else "-"
            o.append(f"| {G['away']}@{G['home']} | {ab} | {R.get('gp', '-')} | {R.get('days_rest', '-')} ({R.get('prev', '-')}) | "
                     f"{'YES' if R.get('b2b') else 'no'} | {tr} | {R.get('next', '-')} ({'yes' if R.get('b2b_front') else 'no'}) | {ij} | {un} | {'; '.join(R.get('b2b_sitters', [])) or '-'} |")
    o += [""]

    o += ["## 4. Early-season noise (3P% shrunk with 750 attempts of prior)", "",
          "| Team | GP | 3P% (prev / shrunk) | 3PA/g | off luck pts/g | opp 3P% (prev / shrunk) | def luck pts/g | margin (prev) | opp quality (opp prev margin) | pace (prev) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for G in games:
        for ab in (G["away"], G["home"]):
            n = G["noise"].get(ab)
            if not n:
                continue
            o.append(f"| {ab} | {n['gp']} | {n['p3']:.3f} ({n['p3_prev']:.3f} / {n['p3_shr']:.3f}) | {n['a3g']:.1f} | {n['off_luck']:+.1f} | "
                     f"{n['op3']:.3f} ({fmt(n['op3_prev'], 3)} / {n['op3_shr']:.3f}) | {n['def_luck']:+.1f} | {n['margin']:+.1f} ({fmt(n['margin_prev'], 1, True)}) | "
                     f"{fmt(n['opp_q'], 1, True)} | {n['pace']:.1f} ({fmt(n['pace_prev'])}) |")
    o += ["", "Game luck = 0.5 x (both offences' + both defences' luck pts/g): points of a recent-form total that come from "
          "3P shooting expected to regress (+ = recent form overstates scoring).", ""]

    o += ["## 5-6. Structure, refs, model cross-check", "",
          "| Game | game luck pts | paint diff z (H-A share allowed) -> ATS pts to home | proj pace | altitude | crew (chief, ref, ump) | crew FTA / total-vs-close (shrunk, DEV) | model total -2.3 vs Pin total | model margin vs Pin spread |",
          "|---|---|---|---|---|---|---|---|---|"]
    for G in games:
        pz = G["pin"]
        ptot = pz.get("total", {}).get("line") if pz else None
        psp = pz.get("spread", {}).get("line") if pz else None
        mt = G.get("model_total_bc")
        mx = f"{mt:.1f} vs {fmt(ptot)} ({mt - ptot:+.1f})" if mt is not None and ptot is not None else "-"
        mm = G.get("model_margin")
        mmx = f"{mm:+.1f} vs {-psp:+.1f} ({mm + psp:+.1f})" if mm is not None and psp is not None else "-"
        crew = ", ".join(G["crew"]) if G.get("crew") else "not yet posted"
        cr = f"{fmt(G.get('crew_fta'), 1, True)} / {fmt(G.get('crew_tot'), 1, True)}" if G.get("crew") else "-"
        o.append(f"| {G['away']}@{G['home']} | {fmt(G.get('game_luck'), 1, True)} | {fmt(G.get('paint_diff_z'), 2, True)} -> {fmt(G.get('paint_ats_home'), 1, True)} | "
                 f"{fmt(G.get('pace_proj'))} | {'YES (+1.0 home ATS lead)' if G['altitude'] else '-'} | {crew} | {cr} | {mx} | {mmx} |")
    o += ["", "Refs: " + ("posted (official.nba.com)" if refs else "NOT YET POSTED -- re-run after ~9am ET") +
          ". Crew numbers = mean of the three refs' DEV residuals shrunk with 60 games of prior; DEV crew betting lost -1.9% "
          "(notes/nba_referees.md) -> context only.", ""]
    return "\n".join(o)


if __name__ == "__main__":
    sys.exit(main())

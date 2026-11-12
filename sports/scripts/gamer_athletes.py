"""gamer_athletes: player-level DiD of known-gamer athletes after mega video-game launches.

Subcommands:
  fetch_nba   - PlayerGameLogs 2007-08..2024-25 (reg + playoffs) -> results/gamer_athletes/nba_playerlogs.parquet
  fetch_nfl   - nflverse player_stats 2009..2014 + data/nfl_player_week.parquet -> nfl_week.parquet
  freeze      - hash gamers_frozen.csv
  power       - counts of gamer player-games in windows (no outcomes)
  score       - DiD (run only after freeze)
See notes/gamer_athletes.md for the pre-registration.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gamer_athletes"
OUT.mkdir(parents=True, exist_ok=True)

EVENTS = [  # (name, US release) -- sibling list notes/game_release_slump.md; BO3+FO4 merged at first release
    ("GTA IV", "2008-04-29"), ("MW2", "2009-11-10"), ("BO", "2010-11-09"), ("MW3", "2011-11-08"),
    ("BO2", "2012-11-13"), ("GTA V", "2013-09-17"), ("Ghosts", "2013-11-05"), ("Destiny", "2014-09-09"),
    ("BO3+FO4", "2015-11-06"), ("WWII", "2017-11-03"), ("BO4", "2018-10-12"), ("RDR2", "2018-10-26"),
    ("MW19", "2019-10-25"), ("MWII", "2022-10-28"),
]
DEV_END = pd.Timestamp("2022-01-01")


def _retry(f, n=5):
    for i in range(n):
        try:
            return f()
        except Exception as e:  # noqa
            print("retry", i, e, flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError("fetch failed")


def fetch_nba():
    from nba_api.stats.endpoints import playergamelogs
    parts = []
    for y in range(2007, 2025):
        s = f"{y}-{str(y + 1)[-2:]}"
        for t in ["Regular Season", "Playoffs"]:
            fn = OUT / f"nba_pl_{y}_{t[:3]}.parquet"
            if fn.exists():
                parts.append(pd.read_parquet(fn)); continue
            d = _retry(lambda: playergamelogs.PlayerGameLogs(season_nullable=s, season_type_nullable=t,
                                                             timeout=60).get_data_frames()[0])
            d = d[["SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
                   "MATCHUP", "MIN", "PTS", "FGM", "FGA", "FTA", "PLUS_MINUS"]].copy()
            d["stype"] = t
            d.to_parquet(fn, index=False)
            parts.append(d)
            print(s, t, len(d), flush=True)
            time.sleep(1.5)
    pd.concat(parts, ignore_index=True).to_parquet(OUT / "nba_playerlogs.parquet", index=False)


NBA_SPANS = [("2008-02-01", "2008-05-20")] + [
    sp for y in [2009, 2010, 2012, 2013, 2015, 2017, 2018, 2019]
    for sp in [(f"{y}-02-10", f"{y}-04-20"), (f"{y}-10-01", f"{y}-12-05")]]


def _get(url):
    import json
    import urllib.request
    for i in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:  # noqa
            time.sleep(2 * (i + 1))
    return None


def fetch_nba_espn():
    """stats.nba.com stalls (11-11) -> ESPN site API: scoreboard per date, summary per game, only the date spans
    needed for windows (-21..+13) and 20-game baselines (previous Feb-Apr) of dev NBA events."""
    from concurrent.futures import ThreadPoolExecutor
    base = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
    dates = sorted({d for a, b in NBA_SPANS for d in pd.date_range(a, b)})
    with ThreadPoolExecutor(8) as ex:
        sbs = list(ex.map(lambda d: (d, _get(base + "scoreboard?dates=" + d.strftime("%Y%m%d"))), dates))
    games = [(d, e["id"]) for d, s in sbs if s for e in s.get("events", [])]
    print("dates", len(dates), "games", len(games), flush=True)

    def one(dg):
        d, gid = dg
        s = _get(base + "summary?event=" + gid)
        rows = []
        if not s:
            return rows
        for tm in s.get("boxscore", {}).get("players", []):
            team = tm.get("team", {}).get("abbreviation")
            for st in tm.get("statistics", []):
                lab = st.get("labels", [])
                for a in st.get("athletes", []):
                    v = dict(zip(lab, a.get("stats", [])))
                    if not v or a.get("didNotPlay"):
                        continue
                    rows.append({"date": d, "game_id": gid, "team": team, "pid": a["athlete"].get("id"),
                                 "name": a["athlete"].get("displayName"), **{k: v.get(k) for k in
                                                                             ["MIN", "PTS", "FG", "+/-"]}})
        return rows
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, games))
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_parquet(OUT / "nba_espn_raw.parquet", index=False)
    print(len(df), "rows; games with no rows:", sum(1 for r in res if not r))


def nba_from_espn():
    d = pd.read_parquet(OUT / "nba_espn_raw.parquet")
    num = lambda s: pd.to_numeric(s.astype(str).str.replace("+", "", regex=False), errors="coerce")
    fg = d["FG"].astype(str).str.split("-", expand=True)
    out = pd.DataFrame({"GAME_DATE": d["date"].astype(str), "PLAYER_ID": d["pid"], "PLAYER_NAME": d["name"],
                        "GAME_ID": d["game_id"], "TEAM": d["team"], "MIN": num(d["MIN"]), "PTS": num(d["PTS"]),
                        "FGM": pd.to_numeric(fg[0], errors="coerce"),
                        "FGA": pd.to_numeric(fg[1], errors="coerce") if 1 in fg else np.nan,
                        "PLUS_MINUS": num(d["+/-"])})
    out = out.drop_duplicates(["GAME_ID", "PLAYER_ID"])
    out.to_parquet(OUT / "nba_playerlogs.parquet", index=False)
    print(out.describe().T)


def fetch_nfl():
    import io
    import urllib.request
    cols = ["player_id", "player_display_name", "position", "season", "week", "season_type", "team",
            "opponent_team", "fantasy_points_ppr", "carries", "receptions", "targets", "attempts"]
    parts = []
    for y in range(2009, 2015):
        fn = OUT / f"nfl_ps_{y}.parquet"
        if not fn.exists():
            url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{y}.parquet"
            raw = urllib.request.urlopen(url, timeout=120).read()
            pd.read_parquet(io.BytesIO(raw)).to_parquet(fn, index=False)
        d = pd.read_parquet(fn)
        if "player_display_name" not in d and "player_name" in d:
            d["player_display_name"] = d["player_name"]
        if "team" not in d and "recent_team" in d:
            d["team"] = d["recent_team"]
        if "opponent_team" not in d:
            d["opponent_team"] = None
        parts.append(d[[c for c in cols if c in d]])
    w = pd.read_parquet(ROOT / "data" / "nfl_player_week.parquet")
    parts.append(w[[c for c in cols if c in w]])
    nfl = pd.concat(parts, ignore_index=True)
    nfl.to_parquet(OUT / "nfl_week.parquet", index=False)
    print(nfl.groupby("season").size())


# ---------------------------------------------------------------- analysis
import re

EV = pd.DataFrame(EVENTS, columns=["event", "d0"])
EV["d0"] = pd.to_datetime(EV["d0"])


def norm(s):
    s = str(s).lower().replace(".", "").replace("'", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


def load_gamers():
    g = pd.read_csv(OUT / "gamers_frozen.csv")
    g["key"] = g["player_name"].map(norm)
    ed = g["evidence_date"].astype(str).str.strip()
    ed = ed.where(ed.str.len() > 7, ed + "-28")  # YYYY-MM -> end of month (conservative: later = fewer events)
    ed = ed.where(ed.str.len() > 4, ed + "-12-31")
    g["evidence_date"] = pd.to_datetime(ed.str.slice(0, 10), errors="coerce")
    bad = g[g["evidence_date"].isna()]
    if len(bad):
        print("UNPARSED evidence dates (excluded):", bad["player_name"].tolist())
    return g.dropna(subset=["evidence_date"])


def freeze():
    b = (OUT / "gamers_frozen.csv").read_bytes()
    h = hashlib.sha256(b).hexdigest()
    (OUT / "gamers_frozen.sha256").write_text(h + "  gamers_frozen.csv\n")
    print(h)


def nba_frame():
    d = pd.read_parquet(OUT / "nba_playerlogs.parquet")
    d["date"] = pd.to_datetime(d["GAME_DATE"].astype(str).str.slice(0, 10))
    d = d[d["MIN"] > 0].copy()
    # drop preseason (ESPN scoreboard includes it): regular-season openers = first date after the 3-day gap
    opener = {2009: "10-27", 2010: "10-26", 2012: "10-30", 2013: "10-29", 2015: "10-27", 2017: "10-17",
              2018: "10-16", 2019: "10-22"}
    y = d["date"].dt.year
    op = pd.to_datetime(y.astype(str) + "-" + y.map(opener).fillna("10-01"))
    d = d[~((d["date"].dt.month >= 9) & (d["date"] < op))].copy()
    d["key"] = d["PLAYER_NAME"].map(norm)
    d["fgp"] = np.where(d["FGA"] > 0, d["FGM"] / d["FGA"].clip(lower=1), np.nan)
    return d.rename(columns={"PTS": "pts", "MIN": "min", "PLUS_MINUS": "pm", "PLAYER_ID": "pid"})


def nfl_frame():
    w = pd.read_parquet(OUT / "nfl_week.parquet")
    w = w[w["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    sch = pd.read_csv(OUT / "nfl_games.csv", usecols=["season", "week", "gameday", "home_team", "away_team"])
    alias = {"LA": ["STL"], "LAC": ["SD"], "LV": ["OAK"]}
    s = pd.concat([sch[["season", "week", "gameday", c]].rename(columns={c: "team"})
                   for c in ["home_team", "away_team"]])
    ext = []
    for k, vs in alias.items():
        for v in vs:
            t = s[s["team"] == k].copy(); t["team"] = v; ext.append(t)
            t = s[s["team"] == v].copy(); t["team"] = k; ext.append(t)
    s = pd.concat([s] + ext).drop_duplicates(["season", "week", "team"])
    w = w.merge(s, on=["season", "week", "team"], how="left")
    print("nfl rows without date:", int(w["gameday"].isna().sum()), "of", len(w))
    w = w.dropna(subset=["gameday"])
    w["date"] = pd.to_datetime(w["gameday"])
    w["key"] = w["player_display_name"].map(norm)
    w["pid"] = w["player_id"]
    w["ppr"] = w["fantasy_points_ppr"]
    w["touch"] = w[["carries", "receptions", "attempts"]].fillna(0).sum(axis=1)
    return w


def add_offsets(d):
    """event/offset for games within -21..+13 days of an event; in_any = inside some event's days 0..13."""
    d = d.copy()
    d["event"] = None; d["off"] = np.nan; d["in_any"] = False
    # pass 1: pre periods; pass 2: windows overwrite (a game inside any event's 0..13 is never a pre/control game
    # of a later event -- e.g. BO4 2018-10-12 window vs RDR2 2018-10-26 pre period)
    for _, e in EV.iterrows():
        off = (d["date"] - e["d0"]).dt.days
        m = (off >= -21) & (off <= -8)
        d.loc[m, "event"] = e["event"]; d.loc[m, "off"] = off[m]
    for _, e in EV.iterrows():
        off = (d["date"] - e["d0"]).dt.days
        m = (off >= 0) & (off <= 13)
        d.loc[m, "event"] = e["event"]; d.loc[m, "off"] = off[m]
        d.loc[m, "in_any"] = True
    return d


def baselines(d, stats, nwin, nmin):
    d = d.sort_values(["pid", "date"]).reset_index(drop=True)
    base = d[~d["in_any"]][["pid", "date"] + stats].copy()
    for s in stats:
        base["b_" + s] = base.groupby("pid")[s].transform(lambda x: x.rolling(nwin, min_periods=nmin).mean())
    base = base[["pid", "date"] + ["b_" + s for s in stats]].dropna(subset=["date"]).sort_values("date")
    base = base.drop_duplicates(["pid", "date"], keep="last")
    out = pd.merge_asof(d.sort_values("date"), base, on="date", by="pid", allow_exact_matches=False)
    for s in stats:
        out["r_" + s] = out[s] - out["b_" + s]
    return out


def tag(d, gam):
    gm = gam.drop_duplicates("key").set_index("key")["evidence_date"]
    ed = d["key"].map(gm)
    d = d.assign(listed=ed.notna())
    d0 = d["event"].map(EV.set_index("event")["d0"])
    d["gamer"] = d["listed"] & (ed < d0)
    d = d[~(d["listed"] & ~d["gamer"])].copy()  # listed but evidence after the event: out of both arms
    d["period"] = np.select([d["off"].between(0, 6), d["off"].between(7, 13), d["off"].between(-21, -8)],
                            ["w7", "w7_14", "pre"], "other")
    return d[d["period"] != "other"]


def prep(league):
    gam = load_gamers()
    gam = gam[gam["league"].str.upper() == league]
    if league == "NBA":
        d = add_offsets(nba_frame()); stats = ["pts", "min", "fgp", "pm"]
        d = baselines(d, stats, 20, 8)
        d["bucket"] = pd.cut(d["b_min"], [-1, 15, 25, 32, 99], labels=False)
    else:
        d = add_offsets(nfl_frame()); stats = ["ppr", "touch"]
        d = baselines(d, stats, 8, 4)
        d["bucket"] = d["position"]
    allkeys = set(d["key"])
    d = d.dropna(subset=["event", "b_" + stats[0], "bucket"])
    print(league, "gamers listed", len(gam), "unmatched in logs:", sorted(set(gam["key"]) - allkeys))
    return tag(d, gam), stats


def power():
    for lg in ["NBA", "NFL"]:
        d, stats = prep(lg)
        d["era"] = np.where(d["date"] < DEV_END, "dev", "held")
        g = d[d["gamer"]]
        print(g.groupby(["era", "event", "period"]).size().unstack(fill_value=0))
        gd = g[g.era == "dev"]
        print(lg, "distinct gamer players in dev w7:", gd[gd.period == "w7"]["key"].nunique(),
              "events with >=1 dev gamer w7 game:", gd[gd.period == "w7"]["event"].nunique())
        sd = d[d.era == "dev"]["r_" + stats[0]].std()  # all players pooled; no gamer/window split of outcomes
        n = (gd.period == "w7").sum(); npre = (gd.period == "pre").sum()
        mde = 2.8 * sd * np.sqrt(1 / max(n, 1) + 1 / max(npre, 1))
        print(f"{lg} {stats[0]} resid SD {sd:.2f}  gamer w7 n={n} pre n={npre}  MDE(80%, iid player-games)={mde:.2f}")


def did(d, stat, win):
    keep = ["w7", "pre"] if win == "w7" else ["w7", "w7_14", "pre"]
    d = d[d["period"].isin(keep)].copy()
    d["per"] = np.where(d["period"] == "pre", "pre", "win")
    col = "r_" + stat
    d = d.dropna(subset=[col])
    d["w"] = d["FGA"].astype(float) if stat == "fgp" else 1.0
    d = d[d["w"] > 0]
    d["wr"] = d[col] * d["w"]
    cm = d[~d["gamer"]].groupby(["event", "per", "bucket"])[["wr", "w"]].sum()
    cm = (cm["wr"] / cm["w"]).rename("cmean")
    g = d[d["gamer"]].join(cm, on=["event", "per", "bucket"])
    g["dd"] = g[col] - g["cmean"]
    return g.dropna(subset=["dd"])


PRE = False  # A2: primary = window-only (gamer resid - matched control resid); PRE=True subtracts the pre period


def agg(g, events=None):
    if events is not None:
        g = pd.concat([g[g["event"] == e] for e in events])
    rows = {}
    for i, (e, x) in enumerate(g.groupby("event")):
        a = x[x.per == "win"]; b = x[x.per == "pre"]
        if len(a) == 0 or (PRE and len(b) == 0):
            continue
        rows[e] = (np.average(a.dd, weights=a.w) - (np.average(b.dd, weights=b.w) if PRE else 0.0), len(a))
    if not rows:
        return np.nan, {}
    v = np.array([r[0] for r in rows.values()]); n = np.array([r[1] for r in rows.values()])
    return float((v * n).sum() / n.sum()), rows


def agg_boot_events(g, evs, rng):
    # resample events with replacement; duplicates count twice
    per = {e: agg(g[g.event == e])[1].get(e) for e in evs}
    out = []
    for _ in range(4000):
        s = rng.choice(evs, len(evs))
        v = np.array([per[e][0] for e in s]); n = np.array([per[e][1] for e in s])
        out.append((v * n).sum() / n.sum())
    return np.percentile(out, [2.5, 97.5])


def score():
    rng = np.random.default_rng(0)
    out = []
    for lg in ["NBA", "NFL"]:
        d, stats = prep(lg)
        d = d[d["date"] < DEV_END]
        for stat in stats:
            for win in ["w7", "w14"]:
                g = did(d, stat, win)
                est, rows = agg(g)
                evs = list(rows)
                lo, hi = agg_boot_events(g, evs, rng) if len(evs) > 1 else (np.nan, np.nan)
                pl = g["key"].unique(); gp = {k: x for k, x in g.groupby("key")}
                bp = []
                for _ in range(1000):
                    s = rng.choice(pl, len(pl))
                    bp.append(agg(pd.concat([gp[k].assign(event=gp[k].event) for k in s]))[0])
                plo, phi = np.nanpercentile(bp, [2.5, 97.5])
                neg = sum(1 for v, n in rows.values() if n >= 5 and v < 0)
                ok = sum(1 for v, n in rows.values() if n >= 5)
                line = (f"{lg} {stat:5s} {win}: DiD {est:+.3f}  event-CI [{lo:+.3f},{hi:+.3f}] ({len(evs)} events)  "
                        f"player-CI [{plo:+.3f},{phi:+.3f}] ({len(pl)} players)  gamer win games "
                        f"{int((g.per == 'win').sum())}  neg events {neg}/{ok} (n>=5)")
                print(line, flush=True); out.append(line)
                if win == "w7" and stat in (stats[0], "min", "touch"):
                    for e, (v, n) in rows.items():
                        s2 = f"   {stat} {e:8s} DiD {v:+.3f} n_win {n}"
                        print(s2); out.append(s2)
    (OUT / "score.txt").write_text("\n".join(out) + "\n")


def heldout_size():
    for lg in ["NBA", "NFL"]:
        d, _ = prep(lg)
        h = d[(d["date"] >= DEV_END) & d["gamer"]]
        print(lg, "held-out (2022+) gamer games by period:", h.groupby("period").size().to_dict())


if __name__ == "__main__":
    globals()[sys.argv[1]]()

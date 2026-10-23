"""NBA BIG LEADS: do teams up 15-25 win more often than history / the market says?  DEV only.

User hypothesis (2026-10-22): "teams up by 20 win a lot more than they're expected to and the market prices."
The prior late-game test (scripts/nba_late_wp.py, notes/late_game_calibration.md) only covered |margin| <= 8 in
Q4 <= 6:00, so big leads are untested.

    python scripts/nba_big_lead.py pbp            # ESPN summaries 2015-16 .. 2025-26 -> data/nba_big_lead_plays.parquet
    python scripts/nba_big_lead.py empirical      # P(win | lead >= L, checkpoint), comeback rate by season
    python scripts/nba_big_lead.py dev [--ledger] # Kalshi DEV (2024-25 season games only): calibration + B1-B3

DATA. ESPN summary API (plays[] with period / clock / homeScore / awayScore / wallclock; winprobability[] where it
exists; pickcenter[0].spread = pregame home spread). 2021-22 .. 2025-26 summaries were already cached; 2015-16 ..
2020-21 are fetched with cache=False and only a compact per-play table is kept (disk is ~95% full).
Kalshi KXNBAGAME 1-min candles: data/nba_late_candles_2025po.parquet (2025-04-15 .. 06, = 2024-25 playoffs, the
first NBA games Kalshi listed). data/nba_ingame.parquet (2025-26) is SEALED here and is not read by `dev`.

PRE-REGISTRATION (written 2026-10-22 BEFORE any Kalshi price was joined to a big-lead state; the empirical tables
from `empirical` are computed from scores only and never touch Kalshi):
  season   : ESPN season label s = the season starting in October of year s (2024 = 2024-25).
  split    : DEV = 2024-25 season games (only the 2025 playoffs have Kalshi markets). SEALED = 2025-26 and later,
             incl. 2026-27. Sealed is NOT scored; the user must say yes.
  lead     : leader's score - trailer's score after a play (ESPN running score).
  entry    : for an event at ESPN wall clock w (the play where the condition first holds), the entry candle is
             the first 1-min candle whose START (end - 60 s) is >= w + 60 s, i.e. strictly after event + 60 s.
             Price = that candle's close YES ask for the LEADING team (the team leading at the event). Need
             0 < ask < 1 and the candle must exist within 10 min of w + 60 s; otherwise no bet.
  fee      : Kalshi taker fee on a 100-contract order: ceil_to_cent(0.07 * 100 * P * (1-P)) / 100 per contract.
  payoff   : hold to settlement; P&L per contract = (1 if leader wins else 0) - ask - fee.
  B1 lead20_q23 : first play in period 2 or 3 with lead >= 20 (the first time it holds in that window). Buy leader.
  B2 lead20_q4  : first play in period 4 with clock > 360 s and lead >= 20 (includes a lead carried in from Q3).
  B3 lead15_half: lead >= 15 at the last play of period 2 (halftime). Event wall clock = that play's wallclock.
  test     : one bet per game per trial; one-sided bootstrap over games (5000 resamples) of summed P&L;
             bar p < 0.05 / (#trials in ledger dataset nba_big_lead) = 0.05/3 = 0.0167.
  descriptive (not trials): Kalshi calibration of the favourite by mid bucket 90-93 / 93-96 / 96-99c (first
             snapshot per game per bucket, game bootstrap CI, break-even = mean ask + fee); historical table
             P(win | lead, checkpoint) and comeback rate by season; leader's Kalshi ask at each event vs the
             historical (2015-16 .. 2023-24) rate for the same lead/clock cell.

AMENDMENTS made AFTER the first DEV run (flagged; rules B1-B3 themselves unchanged):
  1. Bug fix: ESPN sequenceNumber is not always chronological; plays are now ordered by game clock. This moved the
     halftime state in 2 DEV games (B3 still 23 bets; ROI -3.97% -> -4.30%).
  2. Pass rule: B2 won 15/15, and a bootstrap of a sample with zero losses returns p = 0 by construction. A pass now
     also requires the exact Poisson-binomial tail P(>= wins | each bet wins at its break-even) < bar. This was added
     after seeing B2's result, so it is reported next to the pre-registered bootstrap p, not instead of it.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import CACHE_DIR, _key, fetch

PLAYS = ROOT / "data" / "nba_big_lead_plays.parquet"
META = ROOT / "data" / "nba_big_lead_meta.parquet"
CANDLES_DEV = ROOT / "data" / "nba_late_candles_2025po.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_big_lead"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DEV_SEASON = 2024
TRIALS = ["B1_lead20_q23", "B2_lead20_q4", "B3_lead15_half"]


def fee(p):
    p = np.asarray(p, float)
    return np.ceil(np.round(0.07 * 100 * p * (1 - p) * 100, 6)) / 100 / 100


def secs(c):
    try:
        m, s = c.split(":") if ":" in c else ("0", c)
        return int(m) * 60 + float(s)
    except Exception:
        return np.nan


def parse(eid: str):
    url = f"{ESPN}/summary?event={eid}"
    cached = (CACHE_DIR / (_key(url, None) + ".json")).exists()
    j = fetch(url, cache=cached)  # cached -> read; otherwise fetch without writing the 400 kB json
    comp = j["header"]["competitions"][0]
    side = {c["id"]: c["homeAway"] for c in comp["competitors"]}
    plays = j.get("plays") or []
    if not plays:
        return None, None
    wp = j.get("winprobability") or []
    wpd = {w["playId"]: w["homeWinPercentage"] for w in wp}
    pc = (j.get("pickcenter") or [{}])[0]
    hml = (pc.get("homeTeamOdds") or {}).get("moneyLine")
    rows = []
    for p in plays:
        rows.append((eid, int(p.get("sequenceNumber", 0)), p.get("period", {}).get("number", 0),
                     secs(p.get("clock", {}).get("displayValue", "")), p.get("homeScore"), p.get("awayScore"),
                     p.get("wallclock"), wpd.get(p["id"], np.nan)))
    last = plays[-1]
    meta = {"eid": eid, "spread": pc.get("spread"), "home_ml": hml, "p0_espn": wp[0]["homeWinPercentage"] if wp else np.nan,
            "home_won": int(last["homeScore"] > last["awayScore"]), "n_plays": len(plays), "cached": cached}
    return meta, rows


def cmd_pbp() -> int:
    g = pd.read_parquet(ROOT / "data" / "nba_games.parquet")
    done = set(pd.read_parquet(META).eid) if META.exists() else set()
    todo = g[~g.game_id.isin(done)].sort_values("season", ascending=False)  # cached seasons first
    print(f"{len(todo)} games to parse ({len(done)} done)", flush=True)
    metas, rows = ([pd.read_parquet(META)] if done else []), ([pd.read_parquet(PLAYS)] if done else [])

    def one(eid):
        try:
            return parse(eid)
        except Exception as ex:
            return {"eid": eid, "err": str(ex)[:80]}, None

    batch_m, batch_r = [], []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (m, r) in enumerate(ex.map(one, todo.game_id)):
            if m and r:
                batch_m.append(m)
                batch_r += r
            if (i + 1) % 1000 == 0 or i + 1 == len(todo):
                print(i + 1, len(batch_r), flush=True)
                mm = pd.DataFrame(batch_m).merge(g[["game_id", "season", "game_type", "date", "home", "away"]], left_on="eid", right_on="game_id").drop(columns="game_id")
                rr = pd.DataFrame(batch_r, columns=["eid", "seq", "period", "clock_s", "hs", "as", "wall", "espn_wp"])
                rr["wall"] = pd.to_datetime(rr.wall, utc=True, format="ISO8601", errors="coerce")
                metas.append(mm)
                rows.append(rr)
                batch_m, batch_r = [], []
                pd.concat(metas, ignore_index=True).to_parquet(META, index=False)
                pd.concat(rows, ignore_index=True).to_parquet(PLAYS, index=False)
    m = pd.read_parquet(META)
    print(m.groupby("season").size())
    return 0


def load():
    p = pd.read_parquet(PLAYS)
    m = pd.read_parquet(META)
    # outcome from the official final score (nba_games), not the last pbp row (2 truncated feeds disagree)
    g = pd.read_parquet(ROOT / "data" / "nba_games.parquet").set_index("game_id")
    m["home_won"] = (m.eid.map(g.home_score) > m.eid.map(g.away_score)).astype(int)
    p = p.dropna(subset=["hs", "as", "clock_s"]).sort_values(["eid", "seq"])
    p["m"] = p.hs - p["as"]
    # elapsed game seconds (regulation 4x720, OT 300)
    p["el"] = np.where(p.period <= 4, (p.period - 1) * 720 + (720 - p.clock_s), 2880 + (p.period - 5) * 300 + (300 - p.clock_s))
    # ESPN sequenceNumber is NOT always chronological (late-inserted plays get high numbers) -> order by game clock
    p = p.sort_values(["eid", "el", "seq"], kind="stable").reset_index(drop=True)
    return p, m


def checkpoint_state(p, el_cut):
    """Score margin (home) at the last play with elapsed <= el_cut (one row per game)."""
    q = p[p.el <= el_cut]
    return q.groupby("eid").tail(1).set_index("eid")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


CHECKS = {"end Q1": 720, "halftime": 1440, "end Q3": 2160, "Q4 6:00 left": 2520}


def cmd_empirical() -> int:
    p, m = load()
    m = m.set_index("eid")
    print(f"games {m.index.nunique()} by season:\n{m.groupby('season').size().to_string()}")
    # sanity: final score margin matches nba_games
    g = pd.read_parquet(ROOT / "data" / "nba_games.parquet").set_index("game_id")
    fin = p.groupby("eid").tail(1).set_index("eid")
    chk = (np.sign(fin.m) == np.sign(g.loc[fin.index, "home_score"] - g.loc[fin.index, "away_score"])).mean()
    print(f"sanity: final-play winner == nba_games winner in {chk:.4f} of games")
    hist = m.season <= 2023
    out = {}
    for nm, cut in CHECKS.items():
        st = checkpoint_state(p, cut).join(m[["home_won", "season", "spread"]])
        st["lead"] = st.m.abs()
        st["lw"] = np.where(st.m > 0, st.home_won, 1 - st.home_won)
        st = st[st.m != 0]
        print(f"\n== {nm} ==  (all seasons 2015-16..2025-26 | seasons <= 2023-24 only)")
        for L in (10, 15, 20, 25):
            for lab, x in (("all", st), ("<=2023", st[st.season <= 2023])):
                s = x[x.lead >= L]
                k, n = int(s.lw.sum()), len(s)
                lo, hi = wilson(k, n)
                out[(nm, L, lab)] = (n, k / n if n else np.nan, lo, hi)
            a, b = out[(nm, L, "all")], out[(nm, L, "<=2023")]
            print(f"  lead >= {L:2d}: n {a[0]:5d}  P(win) {a[1]:.4f} [{a[2]:.4f}, {a[3]:.4f}]  lost {a[0] - round(a[1] * a[0]):4d}"
                  f"   | <=2023 n {b[0]:5d} {b[1]:.4f}")
        # exact-lead bands
        st["band"] = pd.cut(st.lead, [9, 14, 19, 24, 29, 99], labels=["10-14", "15-19", "20-24", "25-29", "30+"])
        t = st.groupby("band", observed=True).lw.agg(["size", "mean"])
        print("  by band: " + "; ".join(f"{i} {r['mean']:.4f}/{int(r['size'])}" for i, r in t.iterrows()))
        # strength split: leader was pregame favourite/underdog by spread
        st["lead_spread"] = np.where(st.m > 0, st.spread, -st.spread)  # negative = leader favoured
        s20 = st[(st.lead >= 15)].dropna(subset=["lead_spread"])
        for lab, x in (("leader fav by >=5", s20[s20.lead_spread <= -5]), ("pick'em +/-4.5", s20[s20.lead_spread.abs() < 5]), ("leader dog by >=5", s20[s20.lead_spread >= 5])):
            print(f"  lead>=15, {lab:18s}: n {len(x):5d} P(win) {x.lw.mean():.4f}")
    # comeback rate by season: games where a team led by >= 20 at any time, and that team lost
    p2 = p.join(m[["home_won", "season", "game_type"]], on="eid")
    mx = p2.groupby("eid").agg(hmax=("m", "max"), amax=("m", "min"), home_won=("home_won", "first"), season=("season", "first"))
    print("\nBLOWN LEADS by season (a team led by >= L at ANY point and lost; n = games with such a lead)")
    rows = []
    for s, x in mx.groupby("season"):
        r = {"season": f"{s}-{(s + 1) % 100:02d}", "games": len(x)}
        for L in (15, 20, 25):
            hb = (x.hmax >= L)
            ab = (-x.amax >= L)
            n = int(hb.sum() + ab.sum())
            lost = int((hb & (x.home_won == 0)).sum() + (ab & (x.home_won == 1)).sum())
            r[f"n{L}"], r[f"blown{L}"] = n, lost
            r[f"rate{L}"] = round(lost / n, 4) if n else np.nan
        rows.append(r)
    bl = pd.DataFrame(rows)
    print(bl.to_string(index=False))
    tot = bl[[c for c in bl.columns if c.startswith(("n", "blown"))]].sum()
    for L in (15, 20, 25):
        lo, hi = wilson(tot[f"blown{L}"], tot[f"n{L}"])
        print(f"pooled lead>={L}: {int(tot[f'blown{L}'])}/{int(tot[f'n{L}'])} = {tot[f'blown{L}'] / tot[f'n{L}']:.4f} [{lo:.4f}, {hi:.4f}]")
    early, late = bl[bl.season < "2019"], bl[bl.season >= "2021"]
    for L in (20,):
        a = early[f"blown{L}"].sum() / early[f"n{L}"].sum()
        b = late[f"blown{L}"].sum() / late[f"n{L}"].sum()
        print(f"lead>=20 blown: 2015-16..2018-19 {a:.4f} ({early[f'blown{L}'].sum()}/{early[f'n{L}'].sum()}) vs 2021-22..2025-26 {b:.4f} ({late[f'blown{L}'].sum()}/{late[f'n{L}'].sum()})")
    # ESPN's own WP calibration for big favourites (where ESPN WP exists): is the public model too low?
    w = p2.dropna(subset=["espn_wp"]).copy()
    w["fav"] = np.maximum(w.espn_wp, 1 - w.espn_wp)
    w["fw"] = np.where(w.espn_wp >= 0.5, w.home_won, 1 - w.home_won)
    w = w[(w.period >= 2) & (w.period <= 4) & (w.fav >= 0.90) & (w.fav < 0.99)]
    w["bk"] = pd.cut(w.fav, [.90, .93, .96, .99], right=False, labels=["90-93", "93-96", "96-99"])
    one = w.drop_duplicates(["eid", "bk"])
    print("\nESPN WP calibration, favourite in Q2-Q4, first play per game per bucket (seasons with ESPN WP):")
    for bk, x in one.groupby("bk", observed=True):
        lo, hi = wilson(int(x.fw.sum()), len(x))
        print(f"  {bk}: n {len(x):5d} mean ESPN {x.fav.mean():.4f}  realised {x.fw.mean():.4f} [{lo:.4f}, {hi:.4f}]  seasons {sorted(x.eid.map(m.season).unique())[0]}..")
    return 0


# ----------------------------------------------------------------------------------------------- Kalshi DEV
def leader_events(p, eid):
    gp = p[p.eid == eid]
    ev = {}
    w = gp[gp.period.isin([2, 3]) & (gp.m.abs() >= 20)]
    if len(w):
        ev["B1_lead20_q23"] = w.iloc[0]
    w = gp[(gp.period == 4) & (gp.clock_s > 360) & (gp.m.abs() >= 20)]
    if len(w):
        ev["B2_lead20_q4"] = w.iloc[0]
    h = gp[gp.period == 2]
    if len(h) and abs(h.iloc[-1].m) >= 15:
        ev["B3_lead15_half"] = h.iloc[-1]
    return ev


def boot(ev, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    if not len(ev):
        return np.nan, np.nan, np.nan
    b = rng.choice(ev, (n, len(ev))).sum(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975)), float(np.mean(b <= 0))


def hist_table(p, m):
    """P(leader wins) for seasons <= 2023, by (period, lead band) at every first play in the cell (one per game)."""
    x = p.join(m.set_index("eid")[["home_won", "season"]], on="eid")
    x = x[(x.season <= 2023) & (x.m != 0) & (x.period <= 4)].copy()
    x["lead"] = x.m.abs()
    x["lw"] = np.where(x.m > 0, x.home_won, 1 - x.home_won)
    return x


def pbinom_tail(ps, k):
    """P(#wins >= k) for independent Bernoulli(ps)."""
    d = np.zeros(len(ps) + 1)
    d[0] = 1.0
    for q in ps:
        d[1:] = d[1:] * (1 - q) + d[:-1] * q
        d[0] *= (1 - q)
    return d[k:].sum()


def cmd_dev() -> int:
    p, m = load()
    dev = m[m.season == DEV_SEASON]
    cd = pd.read_parquet(CANDLES_DEV)
    assert cd.date.min() >= pd.Timestamp("2025-04-15") and cd.date.max() < pd.Timestamp("2025-07-01"), "DEV candles only"
    cd = cd[cd.eid.isin(dev.eid)]
    ids = sorted(set(cd.eid) & set(p.eid))
    print(f"DEV: {len(ids)} 2024-25 games with Kalshi candles (all playoffs)")
    hx = hist_table(p, m)
    recs = {t: [] for t in TRIALS}
    nobet = []
    cal = []
    for eid in ids:
        c = cd[cd.eid == eid]
        if c.side.nunique() != 2:
            continue
        gp = p[p.eid == eid].dropna(subset=["wall"])
        home_won = int(m.set_index("eid").loc[eid, "home_won"])
        res = c[c.side == "home"].result.iloc[0]
        assert (res == "yes") == bool(home_won), f"Kalshi result != ESPN result {eid}"
        wt = ((gp.wall - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()
        for t, r in leader_events(gp, eid).items():
            w = int((r.wall - EPOCH) // pd.Timedelta(seconds=1))
            side = "home" if r.m > 0 else "away"
            cc = c[(c.side == side) & (c.end - 60 >= w + 60) & (c.end - 60 <= w + 60 + 600)].sort_values("end")
            raw = cc
            cc = cc[(cc.ask > 0) & (cc.ask < 1)]
            if not len(cc):
                won0 = home_won if side == "home" else 1 - home_won
                print(f"  NO BET {eid} {t} lead {abs(r.m):.0f} P{r.period} {r.clock_s:.0f}s: "
                      + (f"first candle bid/ask {raw.iloc[0].bid:.2f}/{raw.iloc[0].ask:.2f} (no seller below $1)" if len(raw) else "no candle in 10 min")
                      + f"; leader won={won0}")
                nobet.append({"trial": t, "eid": eid, "won": won0, "bid": raw.iloc[0].bid if len(raw) else np.nan})
                continue
            e = cc.iloc[0]
            won = home_won if side == "home" else 1 - home_won
            f = float(fee(e.ask))
            hh = hx[(hx.period == r.period) & (hx.lead >= abs(r.m) - 2) & (hx.lead <= abs(r.m) + 2)
                    & ((hx.clock_s - r.clock_s).abs() <= 90)].drop_duplicates("eid")
            recs[t].append({"eid": eid, "period": r.period, "clock_s": r.clock_s, "lead": abs(r.m), "ask": e.ask, "bid": e.bid,
                            "fee": f, "won": won, "pnl": won - e.ask - f, "lag_s": int(e.end - w), "hrate": hh.lw.mean(), "hist_n": len(hh)})
        # calibration: favourite's mid, Q1-Q4 candles inside the game, both sides
        h = c[c.side == "home"].set_index("end")[["bid", "ask"]]
        a = c[c.side == "away"].set_index("end")[["bid", "ask"]]
        j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner").reset_index()
        j = j[(j.end >= wt.min() + 60) & (j.end <= wt.max())]
        for sd, won in (("h", home_won), ("a", 1 - home_won)):
            mid = (j[f"bid_{sd}"] + j[f"ask_{sd}"]) / 2
            ok = (j[f"bid_{sd}"] > 0) & (j[f"ask_{sd}"] < 1) & (mid >= 0.90) & (mid < 0.99)
            for _, row in j[ok].assign(mid=mid[ok], ask=j[f"ask_{sd}"][ok]).iterrows():
                cal.append({"eid": eid, "end": row.end, "mid": row.mid, "ask": row.ask, "won": won})
    cal = pd.DataFrame(cal)
    cal["bk"] = pd.cut(cal.mid, [.90, .93, .96, .99], right=False, labels=["90-93", "93-96", "96-99"])
    one = cal.sort_values("end").drop_duplicates(["eid", "bk"])
    rng = np.random.default_rng(1)
    print("\nKALSHI CALIBRATION, favourite mid 90-99c in-game (first candle per game per bucket), DEV:")
    calib = []
    for bk, x in one.groupby("bk", observed=True):
        wv = x.won.to_numpy()
        bs = rng.choice(wv, (5000, len(wv))).mean(1)
        be = float((x.ask + fee(x.ask)).mean())
        r = {"bucket": str(bk), "games": len(x), "mid": round(x.mid.mean(), 4), "ask": round(x.ask.mean(), 4), "breakeven": round(be, 4),
             "realised": round(wv.mean(), 4), "lo": round(float(np.quantile(bs, .025)), 4), "hi": round(float(np.quantile(bs, .975)), 4),
             "losses": int((1 - wv).sum())}
        calib.append(r)
        print(" ", r)
    # pooled 90-99c, cluster bootstrap over games (a game can sit in several buckets)
    gids = one.eid.unique()
    rows_by = {e: one[one.eid == e] for e in gids}
    bs = []
    for _ in range(3000):
        pk = pd.concat([rows_by[e] for e in rng.choice(gids, len(gids))])
        bs.append(pk.won.mean() - pk.mid.mean())
    print(f"  pooled 90-99c: {len(one)} snapshots / {len(gids)} games, mean mid {one.mid.mean():.4f}, realised {one.won.mean():.4f}, "
          f"realised - mid {one.won.mean() - one.mid.mean():+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    shas |= {hashlib.sha256(("nba_big_lead:" + t).encode()).hexdigest()[:12] for t in TRIALS}
    bar = 0.05 / len(shas)
    out = []
    print(f"\nPRE-REGISTERED TRIALS (DEV, {DATASET} trials {len(shas)} -> bar p < {bar:.4f})")
    for t in TRIALS:
        x = pd.DataFrame(recs[t])
        if not len(x):
            print(t, "no bets")
            continue
        lo, hi, pv = boot(x.pnl.to_numpy())
        stake = float((x.ask + x.fee).sum())
        r = {"bets": len(x), "games": len(x), "staked": stake, "units": float(x.pnl.sum()), "roi": float(x.pnl.sum() / stake),
             "ci_lo": lo, "ci_hi": hi, "p": pv, "win_rate": float(x.won.mean()), "mean_ask": float(x.ask.mean()),
             "mean_fee": float(x.fee.mean()), "breakeven": float((x.ask + x.fee).mean()), "hist_rate": float(np.nanmean(x.hrate)),
             "losses": int((1 - x.won).sum()), "median_lag_s": float(x.lag_s.median()),
             # exact check: P(at least this many wins | each bet wins with prob = its break-even); Poisson-binomial by DP
             "p_exact_vs_breakeven": float(pbinom_tail((x.ask + x.fee).to_numpy(), int(x.won.sum()))),
             "hist_edge_units": float((x.hrate - x.ask - x.fee).sum()),
             "n_unbuyable": int(sum(1 for z in nobet if z["trial"] == t))}
        # the bootstrap of summed P&L is degenerate when there are 0 losses (every resample is positive -> p = 0);
        # a pass then also requires the exact Poisson-binomial tail vs break-even to clear the bar
        r["passed"] = bool(pv < bar and x.pnl.sum() > 0 and r["p_exact_vs_breakeven"] < bar)
        print(f"{t:15s} bets {r['bets']:3d} win {r['win_rate']:.3f} ({r['losses']} lost) | mean ask {r['mean_ask']:.3f} fee {r['mean_fee']:.4f}"
              f" break-even {r['breakeven']:.3f} | hist rate same cell {r['hist_rate']:.3f} | units {r['units']:+.2f} ROI {r['roi']:+.2%}"
              f" 95%CI [{lo:+.2f}, {hi:+.2f}] p {pv:.3f} | exact p vs break-even {r['p_exact_vs_breakeven']:.3f}"
              f" | hist-implied edge {r['hist_edge_units']:+.2f}u | unbuyable {r['n_unbuyable']} | PASS {r['passed']}")
        print(x[["eid", "period", "clock_s", "lead", "ask", "won", "hrate", "hist_n", "lag_s"]].round(3).to_string(index=False))
        out.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nba_big_lead_{t}",
                    "sha": hashlib.sha256(("nba_big_lead:" + t).encode()).hexdigest()[:12], "dataset": DATASET,
                    "hypothesis": f"pre-registered in scripts/nba_big_lead.py ({t})", "source": "user hypothesis 10-22: teams up 20 win more than priced",
                    "status": "ok", "trial_number": len(shas), "bonferroni_bar": bar, "trial": t, **r})
    nb = pd.DataFrame(nobet)
    if len(nb):
        print("\nUNBUYABLE events (ask = $1 or no candle), by trial: n, leader win rate, mean bid")
        print(nb.groupby("trial").agg(n=("won", "size"), won=("won", "mean"), bid=("bid", "mean")).round(3).to_string())
    (ROOT / "data" / "nba_big_lead_calib.json").write_text(json.dumps(calib))
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in out:
                f.write(json.dumps(r, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"pbp": cmd_pbp, "empirical": cmd_empirical, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())

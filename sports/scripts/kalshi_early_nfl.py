"""KALSHI EARLY NFL: are Kalshi's first-listed NFL game-winner prices stale vs a simple model? DEV only.

    python scripts/kalshi_early_nfl.py build     # hourly Kalshi candles, listing -> kickoff, DEV games only
    python scripts/kalshi_early_nfl.py dev [--ledger]

Question (lead, 2026-10-25): would maker orders posted at Kalshi's first listing of an NFL game profit, because the
first prices are stale relative to the simple NFL model (notes/nfl_simple_model.md)?

PRE-REGISTERED 2026-10-25, written BEFORE any Kalshi price for a DEV game was pulled. Seen beforehand: market
METADATA only (open_time: weeks 1-2 listed 2025-05-20, later weeks ~10-12 days before kickoff) and one candle schema
probe on a preseason market (KXNFLGAME-25AUG09KCARI-KC).

  split  : DEV = 2025 REG weeks 1-12 (as scripts/nfl_ingame.py). SEALED = 2025 weeks 13+ / POST / all 2026: never
           fetched. 2025 play-by-play and schedule rows from weeks >= 13 are dropped before features are built.
  market : KXNFLGAME (one YES market per team). Spread/total ladders (KXNFLSPREAD/KXNFLTOTAL) exist for 2025 but are
           NOT tested here (keeps the trial count at 3; legacy totals carry a known level bias).
  model  : LEGACY model B spread (2007-2019 OLS fit, nfl_simple_model MODELS["B"] incl. neutral term) -> out of
           sample for 2025. Home win prob = Phi(pred_margin / sigma), sigma = residual RMSE of that 2007-19 fit.
           FEATURES AS OF LISTING (counted): for each team, the pre-game feature row of its first game with kickoff
           >= Kalshi open_time (that row uses only games before it = games completed before listing). If that game
           is the target itself (bye, or listing after the previous game), these equal the usual week features.
           QB term comes from that same row (its scheduled starter; projected-starter proxy, a known caveat).
           The usual "as of that week" features (all games before the target week + actual starter) are reported
           as a SECONDARY, uncounted diagnostic: they contain information that did not exist at listing.
  prices : hourly candles (period_interval=60) from open_time to kickoff (nfl_games start_utc).
           FIRST quote = close bid/ask of the first candle with bid > 0, ask < 1 and ask - bid <= 0.10
           (the very first two-sided quote is also described, uncounted). first mid = (bid + ask)/2.
           LAST pre-kick mid = close bid/ask mid of the last candle ending <= kickoff with a quote passing the same filter.
           Home mid from the home market; if the home market lacks a quote, 1 - away mid.
  bet set: games with |p_model(home) - first home mid| >= 0.04. Side = team the model prefers vs the first mid.
  T1 CLV  : (last pre-kick mid - first mid) of the model side, in probability units. Mean over games.
  T2 maker: resting YES bid on the model side's own market at that market's FIRST bid (the bid of its first
           qualifying candle). Filled iff a later candle (strictly after the first one), ending <= kickoff, has a
           TRADE price low <= bid - 0.01 (traded THROUGH our level; a touch does not count). Cancel at kickoff.
           Cost = bid + maker fee 0.0175*b*(1-b) (repo convention, conservative). Held to settlement. Unfilled = no bet.
           ROI = sum pnl / sum cost.
  T3 taker: buy YES on the model side at its first ask + taker fee 0.07*a*(1-a). Held to settlement.
  stats  : one bet per game; one-sided bootstrap over games (5000 draws, seed 0), p = P(boot sum <= 0).
           Bar p < 0.05/3 = 0.0167. PASS = p < bar AND mean > 0 AND same sign in weeks 1-6 and 7-12.
  uncounted: maker at first ask - 1c (same fill rule), standard-week-feature version of T1-T3, Kalshi first and last
           mid vs the closing moneyline (de-vigged, data/nfl_games.parquet), listing lead time, spread at listing,
           pre-game volume. No Pinnacle/consensus OPENER exists for 2025 locally that is not lockboxed -> not used.
  power (stated before the join): ~150-170 dev games, maybe 60-110 in the bet set. T1 SD of a mid move is ~0.05-0.08
           -> detectable mean at 80% power ~ (2.13 + 0.84) * 0.07 / sqrt(90) ~ 0.02 (2c). ROI per $ of a ~50c
           contract has SD ~ 1 -> only ROI > ~30% would be detectable. T2/T3 are effectively underpowered.
  Caveats: no historical order book -> depth at listing unknown (volume is reported); hourly bars hide intra-hour
  quotes; maker fill rule ignores queue position beyond requiring a trade-through.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import norm

from sports.http import fetch
from kalshi_longshot import API
import nfl_simple_model as nsm

DATA = ROOT / "data"
OUT = DATA / "kalshi_early_nfl_candles.parquet"
META = DATA / "kalshi_early_nfl_markets.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_early_nfl"
ALIAS = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV"}
TRIALS = ["T1_clv", "T2_maker_first_bid", "T3_taker_first_ask"]
BAR = 0.05 / len(TRIALS)
GAP, MAXSPR, MAKER, TAKER = 0.04, 0.10, 0.0175, 0.07
DEV_WEEKS = 12


def num(x):
    return pd.to_numeric(x, errors="coerce")


def dev_schedule() -> pd.DataFrame:
    g = pd.read_parquet(DATA / "nfl_games.parquet")
    s = pd.read_csv(DATA / "nfl_schedule.csv", usecols=["game_id", "week", "game_type"])
    g = g.merge(s.rename(columns={"game_type": "sched_type"}), on="game_id")
    g = g[(g.season == 2025) & (g.sched_type == "REG") & (g.week <= DEV_WEEKS)].copy()
    assert g.week.max() <= DEV_WEEKS and (g.season == 2025).all()
    g["date_et"] = pd.to_datetime(g.start_utc, utc=True).dt.tz_convert("America/New_York").dt.strftime("%y%b%d").str.upper()
    return g


def markets() -> pd.DataFrame:
    rows, cursor = [], None
    while True:
        p = {"series_ticker": "KXNFLGAME", "limit": 1000} | ({"cursor": cursor} if cursor else {})
        d = fetch(f"{API}/historical/markets", p, cache=True)
        rows += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            break
    m = pd.DataFrame(rows).drop_duplicates("ticker")
    m["code"] = m.ticker.str.split("-").str[-1].map(lambda c: ALIAS.get(c, c))
    m["date"] = m.event_ticker.str.split("-").str[1].str[:7]
    return m


def cmd_build() -> int:
    g = dev_schedule()
    mk = markets()
    rows, frames, unmatched = [], [], []
    for _, r in g.iterrows():
        m = mk[(mk.date == r.date_et) & mk.code.isin([r.home, r.away])]
        if m.event_ticker.nunique() != 1 or len(m) != 2:
            unmatched.append(r.game_id)
            continue
        ko = int(pd.Timestamp(r.start_utc).timestamp())
        for _, x in m.iterrows():
            ot = int(pd.Timestamp(x.open_time).timestamp())
            side = "home" if x.code == r.home else "away"
            rows.append(dict(game_id=r.game_id, week=int(r.week), side=side, ticker=x.ticker, open_ts=ot, ko_ts=ko,
                             result=x.result))
            cs = []
            s = ot - 3600
            while s < ko:  # chunk to stay under the API's per-request candle cap
                e = min(ko, s + 3600 * 2000)
                j = fetch(f"{API}/historical/markets/{x.ticker}/candlesticks",
                          {"start_ts": s, "end_ts": e, "period_interval": 60}, cache=True)
                cs += j.get("candlesticks", [])
                s = e
            for c in cs:
                frames.append(dict(game_id=r.game_id, side=side, end=int(c["end_period_ts"]),
                                   bid=num((c.get("yes_bid") or {}).get("close")),
                                   ask=num((c.get("yes_ask") or {}).get("close")),
                                   tlow=num((c.get("price") or {}).get("low")),
                                   thigh=num((c.get("price") or {}).get("high")),
                                   vol=num(c.get("volume"))))
    pd.DataFrame(rows).to_parquet(META, index=False)
    cd = pd.DataFrame(frames).drop_duplicates(["game_id", "side", "end"])
    cd.to_parquet(OUT, index=False)
    print(f"{len(g)} dev games; {len(g) - len(unmatched)} matched; unmatched {unmatched}; {len(cd)} candles -> {OUT}")
    return 0


# ------------------------------------------------------------------------------------------- model
def build_features(drop_after_week: int = DEV_WEEKS):
    """nfl_simple_model.build_all, with 2025 weeks > 12 (sealed) dropped from pbp and schedule first."""
    seasons = list(range(2006, 2026))
    p = nsm.load_pbp(seasons)
    p = p[~((p.season == 2025) & (p.week > drop_after_week))]
    sched = nsm.load_schedule()
    sched = sched[sched.season.isin(seasons) & sched.home_score.notna() & sched.game_id.isin(set(p.game_id))]
    tf = nsm.team_features(nsm.team_games(p))
    qf = nsm.qb_features(nsm.qb_games(p), sched)
    pts = sched.groupby("season").apply(lambda d: (d.home_score + d.away_score).mean())
    lg_pts = pd.Series({s: pts.get(s - 1, np.nan) for s in seasons})
    return sched, tf, qf, lg_pts


def model_probs(meta: pd.DataFrame) -> pd.DataFrame:
    sched, tf, qf, lg_pts = build_features()
    g_std = nsm.assemble(sched, tf, qf, lg_pts)
    g_std = g_std[g_std.season >= nsm.FIRST_FIT].dropna(subset=nsm.BASE_S + nsm.BASE_T)
    train = g_std[g_std.season <= 2019]
    cols = nsm.MODELS["B"][0]
    _, m, _ = nsm.fit_predict(train, train, cols, "margin", True)
    sigma = float(np.sqrt(np.mean(np.asarray(m.resid) ** 2)))
    # as-of-listing remap: (target game, team) -> (team's first game with kickoff >= listing time)
    ko = pd.read_parquet(DATA / "nfl_games.parquet", columns=["game_id", "start_utc"])
    ko["ko"] = pd.to_datetime(ko.start_utc, utc=True).map(lambda t: int(t.timestamp()))
    s25 = sched[sched.season == 2025].merge(ko[["game_id", "ko"]], on="game_id")
    listing = meta.groupby("game_id").open_ts.min()
    remap = []
    for gid, L in listing.items():
        r = s25[s25.game_id == gid]
        if r.empty:
            continue
        r = r.iloc[0]
        for t in (r.home_team, r.away_team):
            tg = s25[((s25.home_team == t) | (s25.away_team == t)) & (s25.ko >= L)].sort_values("ko")
            remap.append(dict(game_id=gid, team=t, src=tg.iloc[0].game_id if len(tg) else gid))
    remap = pd.DataFrame(remap)
    tf_a = remap.merge(tf.drop(columns=["game_id"]).assign(src=tf.game_id), on=["src", "team"], how="left")
    qf_a = remap.merge(qf.drop(columns=["game_id"]).assign(src=qf.game_id), on=["src", "team"], how="left")
    tf_a = tf_a.drop(columns="src").assign(season=2025)
    qf_a = qf_a.drop(columns="src")
    g_list = nsm.assemble(sched[sched.game_id.isin(remap.game_id)], tf_a.drop(columns=["season", "week"], errors="ignore")
                          .merge(tf[["game_id", "team", "season", "week"]], on=["game_id", "team"]), qf_a, lg_pts)
    out = {}
    for name, D in (("list", g_list), ("std", g_std[g_std.game_id.isin(remap.game_id)])):
        pr, _, _ = nsm.fit_predict(train, D, cols, "margin", True)
        out[name] = pd.DataFrame({"game_id": D.game_id.to_numpy(), f"margin_{name}": np.asarray(pr),
                                  f"p_{name}": norm.cdf(np.asarray(pr) / sigma)})
    res = out["list"].merge(out["std"], on="game_id", how="outer")
    nch = (remap.src != remap.game_id).groupby(remap.game_id).any()
    res["asof_differs"] = res.game_id.map(nch)
    res.attrs["sigma"] = sigma
    return res


# ------------------------------------------------------------------------------------------- prices
def quotes(cd: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (gid, side), c in cd.groupby(["game_id", "side"]):
        mm = meta[(meta.game_id == gid) & (meta.side == side)].iloc[0]
        c = c[c.end <= mm.ko_ts].sort_values("end")
        two = c[(c.bid > 0) & (c.ask < 1) & c.bid.notna() & c.ask.notna()]
        ok = two[(two.ask - two.bid) <= MAXSPR]
        if ok.empty:
            continue
        f, l = ok.iloc[0], ok.iloc[-1]
        later = c[(c.end > f.end)]
        def filled(px):
            return bool((later.tlow <= px - 0.01 + 1e-9).any())
        rows.append(dict(game_id=gid, side=side, won=mm.result == "yes", open_ts=mm.open_ts, ko_ts=mm.ko_ts,
                         first_two_end=int(two.iloc[0].end), first_two_spread=float(two.iloc[0].ask - two.iloc[0].bid),
                         first_end=int(f.end), fbid=f.bid, fask=f.ask, fmid=(f.bid + f.ask) / 2,
                         lmid=(l.bid + l.ask) / 2, fill_bid=filled(f.bid), fill_ask1=filled(f.ask - 0.01),
                         vol_pre=float(c.vol.sum()), vol_24h=float(c[c.end <= f.end + 86400].vol.sum())))
    return pd.DataFrame(rows)


def boot(v: np.ndarray, stake: np.ndarray | None = None, n=5000):
    rng = np.random.default_rng(0)
    if len(v) == 0:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, p=1.0)
    idx = rng.integers(0, len(v), (n, len(v)))
    if stake is None:
        b = v[idx].mean(1); est = v.mean()
    else:
        b = v[idx].sum(1) / stake[idx].sum(1); est = v.sum() / stake.sum()
    return dict(n=int(len(v)), mean=float(est), lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)),
                p=float(np.mean(b <= 0)))


def bets(q: pd.DataFrame, mp: pd.DataFrame, pcol: str) -> pd.DataFrame:
    h = q[q.side == "home"].set_index("game_id")
    a = q[q.side == "away"].set_index("game_id")
    ids = sorted(set(h.index) | set(a.index))
    rows = []
    for gid in ids:
        hm = h.fmid.get(gid, np.nan); am = a.fmid.get(gid, np.nan)
        f_home = hm if pd.notna(hm) else (1 - am if pd.notna(am) else np.nan)
        hl = h.lmid.get(gid, np.nan); al = a.lmid.get(gid, np.nan)
        l_home = hl if pd.notna(hl) else (1 - al if pd.notna(al) else np.nan)
        r = mp[mp.game_id == gid]
        if r.empty or pd.isna(f_home) or pd.isna(r[pcol].iloc[0]):
            continue
        pm = float(r[pcol].iloc[0])
        rows.append(dict(game_id=gid, p_model=pm, f_home=f_home, l_home=l_home, gap=pm - f_home))
    b = pd.DataFrame(rows)
    b["side"] = np.where(b.gap > 0, "home", "away")
    b["sel"] = b.gap.abs() >= GAP
    b["clv"] = np.where(b.side == "home", b.l_home - b.f_home, b.f_home - b.l_home)
    qq = q.set_index(["game_id", "side"])
    for c in ("fbid", "fask", "won", "fill_bid", "fill_ask1"):
        b[c] = [qq[c].get((g, s), np.nan) for g, s in zip(b.game_id, b.side)]
    return b


def score(b: pd.DataFrame, weeks: pd.Series) -> dict:
    s = b[b.sel].copy()
    s["week"] = s.game_id.map(weeks)
    won = s.won.astype(float)
    out = {}
    def add(name, d, kind):
        halves = {}
        for lab, part in (("w1-6", d[d.week <= 6]), ("w7-12", d[d.week > 6])):
            if kind == "clv":
                halves[lab] = [round(float(part.clv.mean()), 4) if len(part) else None, int(len(part))]
            else:
                halves[lab] = [round(float(part.pnl.sum() / part.cost.sum()), 4) if len(part) else None, int(len(part))]
        r = boot(d.clv.dropna().to_numpy()) if kind == "clv" else boot(d.pnl.to_numpy(), d.cost.to_numpy())
        out[name] = r | {"halves": halves}
    add("T1_clv", s.dropna(subset=["clv"]), "clv")
    m = s[s.fill_bid == True].dropna(subset=["fbid"]).copy()
    m["cost"] = m.fbid + MAKER * m.fbid * (1 - m.fbid); m["pnl"] = np.where(m.won == 1, 1, 0) - m.cost
    add("T2_maker_first_bid", m, "roi")
    t = s.dropna(subset=["fask"]).copy()
    t["cost"] = t.fask + TAKER * t.fask * (1 - t.fask); t["pnl"] = np.where(t.won == 1, 1, 0) - t.cost
    add("T3_taker_first_ask", t, "roi")
    m1 = s[s.fill_ask1 == True].dropna(subset=["fask"]).copy()
    m1["px"] = m1.fask - 0.01
    m1["cost"] = m1.px + MAKER * m1.px * (1 - m1.px); m1["pnl"] = np.where(m1.won == 1, 1, 0) - m1.cost
    add("U_maker_first_ask_minus_1c", m1, "roi")
    out["_bet_set"] = dict(games=int(len(b)), selected=int(len(s)), maker_fill_rate_bid=float((s.fill_bid == True).mean()),
                           maker_fill_rate_ask1=float((s.fill_ask1 == True).mean()), home_share=float((s.side == "home").mean()))
    return out


def cmd_dev() -> int:
    cd, meta = pd.read_parquet(OUT), pd.read_parquet(META)
    assert meta.week.max() <= DEV_WEEKS
    weeks = meta.groupby("game_id").week.first()
    mp = model_probs(meta)
    q = quotes(cd, meta)
    res = {"sigma": mp.attrs["sigma"], "n_games_meta": int(meta.game_id.nunique()), "n_games_quoted": int(q.game_id.nunique())}
    # descriptive
    ql = q.assign(lead_d=(q.ko_ts - q.open_ts) / 86400, first_lag_h=(q.first_end - q.open_ts) / 3600,
                  first_two_lag_h=(q.first_two_end - q.open_ts) / 3600, spread=q.fask - q.fbid)
    res["describe"] = {
        "listing_lead_days": ql.groupby("game_id").lead_d.first().describe().round(2).to_dict(),
        "listing_lead_days_by_week": ql.assign(week=ql.game_id.map(weeks)).groupby("week").lead_d.median().round(1).to_dict(),
        "hours_listing_to_first_two_sided": ql.first_two_lag_h.describe().round(1).to_dict(),
        "first_two_sided_spread": ql.first_two_spread.describe().round(3).to_dict(),
        "hours_listing_to_first_qualifying": ql.first_lag_h.describe().round(1).to_dict(),
        "spread_at_first_qualifying": ql.spread.describe().round(3).to_dict(),
        "volume_first_24h_contracts": ql.vol_24h.describe().round(0).to_dict(),
        "volume_pregame_contracts": ql.vol_pre.describe().round(0).to_dict(),
    }
    # vs closing moneyline (de-vigged multiplicative), uncounted
    g = pd.read_parquet(DATA / "nfl_games.parquet", columns=["game_id", "home_ml", "away_ml"])
    imp = lambda ml: np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    g["p_close"] = imp(g.home_ml) / (imp(g.home_ml) + imp(g.away_ml))
    res["primary"] = {}
    for name, pcol in (("as_of_listing", "p_list"), ("std_week_features", "p_std")):
        b = bets(q, mp, pcol).merge(g[["game_id", "p_close"]], on="game_id", how="left")
        r = score(b, weeks)
        r["_vs_close"] = dict(
            mae_first_vs_close=float((b.f_home - b.p_close).abs().mean()),
            mae_last_vs_close=float((b.l_home - b.p_close).abs().mean()),
            mae_model_vs_close=float((b.p_model - b.p_close).abs().mean()),
            corr_gap_vs_move=float(np.corrcoef(b.gap, b.l_home - b.f_home)[0, 1]),
            brier_first=float(((b.f_home - b.game_id.map(home_won(q)))**2).mean()),
            brier_last=float(((b.l_home - b.game_id.map(home_won(q)))**2).mean()),
            brier_model=float(((b.p_model - b.game_id.map(home_won(q)))**2).mean()),
            brier_close=float(((b.p_close - b.game_id.map(home_won(q)))**2).mean()),
            n=int(b.p_close.notna().sum()))
        res["primary"][name] = r
        b.to_parquet(DATA / f"lab/kalshi_early_nfl_bets_{name}.parquet", index=False)
    res["asof_differs_share"] = float(mp.asof_differs.mean())
    print(json.dumps(res, indent=1, default=float))
    (DATA / "lab" / "kalshi_early_nfl_results.json").write_text(json.dumps(res, indent=1, default=float))
    if "--ledger" in sys.argv:
        recs = []
        for i, name in enumerate(TRIALS, 1):
            r = res["primary"]["as_of_listing"][name]
            h = r["halves"]
            both = all(v[0] is not None and v[0] > 0 for v in h.values())
            recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_early_nfl_{name}",
                         "sha": hashlib.sha256(f"kalshi_early_nfl:{name}".encode()).hexdigest()[:12], "dataset": DATASET,
                         "metric": "clv_prob" if name == "T1_clv" else "roi", "trial": name,
                         "hypothesis": f"pre-registered in scripts/kalshi_early_nfl.py ({name})",
                         "source": "lead brief 2026-10-25: Kalshi first-listed NFL prices stale vs legacy model B?",
                         "status": "ok", "trial_number": i, "study_trials": len(TRIALS), "bonferroni_bar": BAR,
                         "bets": r["n"], "roi" if name != "T1_clv" else "mean": r["mean"], "ci_lo": r["lo"], "ci_hi": r["hi"],
                         "p": r["p"], "per_fold": h, "passed": bool(r["p"] < BAR and r["mean"] > 0 and both)})
        with LEDGER.open("a") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
        print(f"ledger += {len(recs)} rows")
    return 0


def home_won(q: pd.DataFrame) -> pd.Series:
    h = q[q.side == "home"].set_index("game_id").won.astype(float)
    a = 1 - q[q.side == "away"].set_index("game_id").won.astype(float)
    return h.combine_first(a)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    sys.exit({"build": cmd_build, "dev": cmd_dev}[cmd]())

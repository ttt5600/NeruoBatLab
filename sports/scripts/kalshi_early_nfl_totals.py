"""KALSHI EARLY NFL TOTALS: are Kalshi's early NFL total-points ladder prices stale vs a simple model, and does a
TAKER order at the FIRST TIGHT quote profit? DEV only.

    python scripts/kalshi_early_nfl_totals.py level   # pre-registered level check (model vs CLOSING total, wk 1-4)
    python scripts/kalshi_early_nfl_totals.py build   # hourly Kalshi candles for every ladder strike, DEV games only
    python scripts/kalshi_early_nfl_totals.py dev [--ledger]

Follow-up to L35 (scripts/kalshi_early_nfl.py, game-winner market: first prices stale, CLV +3.3c, maker adversely
selected; first two-sided quote is a ~59c placeholder, first TIGHT quote ~2 h after listing). Here: the totals ladder.

PRE-REGISTERED 2026-10-25, written BEFORE any Kalshi price (candle) for a DEV totals market was pulled. Seen
beforehand: series KXNFLTOTAL metadata only (6,243 markets 2025-08..2026-08; ladder of "Over X.5" YES markets,
strike_type greater, 9-81 strikes per event; open_time per market) and the L35 game-winner results.

  split  : DEV = 2025 REG weeks 1-12 (as scripts/nfl_ingame.py / kalshi_early_nfl.py). SEALED = 2025 weeks 13+ / POST /
           all 2026: never fetched. 2025 pbp + schedule rows from weeks >= 13 are dropped before features are built.
  market : KXNFLTOTAL. One event per game, YES = total points > floor_strike (strikes at X.5, no pushes).
  model  : LEGACY model B TOTALS (nfl_simple_model MODELS["B"][1], target "points", no neutral term), OLS fit on
           2007-2019 -> out of sample for 2025. Chosen over legacy C because C carries a known level bias (-1.7 pts
           on 2020-25, -3.2 on 2026 wk 1-7; nfl_simple_model.md) while legacy B's was -0.16 / +0.44.
           Features AS OF LISTING (counted), exactly as L35: for each team, the pre-game feature row of its first
           game with kickoff >= the event's listing time (min open_time over the event's markets).
           Usual "as of that week" features = secondary, uncounted diagnostic.
  LEVEL RULE (decided here, before prices): compute bias = mean(model total - CLOSING total) over 2025 weeks 1-4
           (closing total_line from data/nfl_games.parquet -- a sportsbook close, not a Kalshi price), as-of-listing
           features. If |bias| <= 1.5 pts: no correction, score weeks 1-12 (halves 1-6 / 7-12).
           If |bias| > 1.5: model total -= bias, score weeks 5-12 only (halves 5-8 / 9-12).
           (`level` command; its output is recorded in notes/kalshi_early_nfl_totals.md before `build` is run.)
  prices : hourly candles (period_interval=60) per strike market, from that market's open_time - 1h to kickoff.
           Each strike's quote at hour h = close yes bid/ask of its latest candle ending <= h (carried forward).
           A strike QUALIFIES at h if bid > 0, ask < 1 and ask - bid <= 0.10; mid = P(over strike).
           Implied total at h: sort qualifying strikes; enforce monotone non-increasing mids (pool-adjacent-violators);
           linearly interpolate the strike where mid = 0.5 between the adjacent bracketing pair (mid >= .5 at the lower,
           <= .5 at the upper strike). Defined only if such a bracketing pair exists.
           FIRST TIGHT ladder = first hourly candle end (>= event listing) at which the implied total is defined.
           LAST pre-kick = last candle end <= kickoff at which it is defined.
  bet set: games with |model - first implied| >= 2.5 pts. Side = OVER if model > first implied else UNDER.
  T1 CLV  : (last implied - first implied) * (+1 over / -1 under), in points. Mean over games.
  T2 taker, model strike: at the first tight hour, among QUALIFYING strikes take the one nearest the MODEL total
           (tie -> the one nearer the implied total). OVER = buy YES at its ask; UNDER = buy NO at 1 - its bid.
           Exact Kalshi taker fee for a 1-contract order: ceil(100 * 0.07 * p * (1-p)) / 100 (rounded up to the cent).
           Held to settlement (market result). ROI = sum pnl / sum cost.
  T3 taker, implied strike: same, strike nearest the first implied total (tie -> nearer the model).
  stats  : one bet per game; one-sided bootstrap over games (5000 draws, seed 0), p = P(boot stat <= 0).
           Bar p < 0.05/3 = 0.0167. PASS = p < bar AND estimate > 0 AND same sign in both halves.
  uncounted: standard-week features; unrounded fee; first/last implied vs the closing total; lead time, spread,
           number of strikes at the first tight ladder; volume; maker-at-first-bid shown descriptively only.
  power (stated before the join): ~150-176 dev games (or ~115 if the level rule triggers); with an 8-12 pt-wide
           early-vs-model disagreement band, maybe 50-90 in the bet set. SD of a totals line move ~1.5 pts ->
           detectable mean CLV at 80% power ~ (2.13 + 0.84) * 1.5 / sqrt(70) ~ 0.55 pts. ROI per $ of a ~50c contract
           has SD ~1 -> only ROI > ~35% detectable. T2/T3 are underpowered; a fail there is weak evidence of no edge.
  Caveats: no historical order book (depth at the first quote unknown); hourly bars hide intra-hour quotes; the taker
           price is the candle-close quote, assumed still available; QB term from the schedule's starter of the
           as-of game (projected-starter proxy), as L35.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch
from kalshi_longshot import API
import nfl_simple_model as nsm
from kalshi_early_nfl import dev_schedule, build_features, boot, DEV_WEEKS

DATA = ROOT / "data"
OUT = DATA / "kalshi_early_nfl_totals_candles.parquet"
META = DATA / "kalshi_early_nfl_totals_markets.parquet"
LEVEL = DATA / "lab" / "kalshi_early_nfl_totals_level.json"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_early_nfl_totals"
SERIES = "KXNFLTOTAL"
ALIAS = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV"}
REV = {}
for k, v in ALIAS.items():
    REV.setdefault(v, []).append(k)
TRIALS = ["T1_clv_pts", "T2_taker_model_strike", "T3_taker_implied_strike"]
BAR = 0.05 / len(TRIALS)
GAP, MAXSPR, TAKER, LEVEL_TOL = 2.5, 0.10, 0.07, 1.5


def num(x):
    return pd.to_numeric(x, errors="coerce")


def fee(p, rounded=True):
    f = TAKER * p * (1 - p)
    return math.ceil(round(100 * f, 9)) / 100 if rounded else f


def markets() -> pd.DataFrame:
    rows, cursor = [], None
    while True:
        p = {"series_ticker": SERIES, "limit": 1000} | ({"cursor": cursor} if cursor else {})
        d = fetch(f"{API}/historical/markets", p, cache=True)
        rows += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            break
    m = pd.DataFrame(rows).drop_duplicates("ticker")
    m["date"] = m.event_ticker.str.split("-").str[1].str[:7]
    m["teams"] = m.event_ticker.str.split("-").str[1].str[7:]
    m["strike"] = num(m.floor_strike)
    return m


def codes(t: str) -> list[str]:
    return [t] + REV.get(t, [])


def match_events(g: pd.DataFrame, mk: pd.DataFrame) -> dict:
    ev = mk.groupby(["date", "teams"]).event_ticker.first()
    out = {}
    for _, r in g.iterrows():
        hits = {ev[(r.date_et, a + h)] for a in codes(r.away) for h in codes(r.home)
                if (r.date_et, a + h) in ev.index}
        if len(hits) == 1:
            out[r.game_id] = hits.pop()
    return out


# ------------------------------------------------------------------------------------------- model
def model_totals(listing: pd.Series) -> pd.DataFrame:
    """Legacy model B totals, as-of-listing ('list') and usual week ('std') features. listing: game_id -> open ts."""
    sched, tf, qf, lg_pts = build_features()
    g_std = nsm.assemble(sched, tf, qf, lg_pts)
    g_std = g_std[g_std.season >= nsm.FIRST_FIT].dropna(subset=nsm.BASE_S + nsm.BASE_T)
    train = g_std[g_std.season <= 2019]
    cols = nsm.MODELS["B"][1]
    ko = pd.read_parquet(DATA / "nfl_games.parquet", columns=["game_id", "start_utc"])
    ko["ko"] = pd.to_datetime(ko.start_utc, utc=True).map(lambda t: int(t.timestamp()))
    s25 = sched[sched.season == 2025].merge(ko[["game_id", "ko"]], on="game_id")
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
    tf_a = tf_a.drop(columns="src")
    qf_a = qf_a.drop(columns="src")
    g_list = nsm.assemble(sched[sched.game_id.isin(remap.game_id)],
                          tf_a.drop(columns=["season", "week"], errors="ignore")
                          .merge(tf[["game_id", "team", "season", "week"]], on=["game_id", "team"]), qf_a, lg_pts)
    out = {}
    for name, D in (("list", g_list), ("std", g_std[g_std.game_id.isin(remap.game_id)])):
        pr, _, _ = nsm.fit_predict(train, D, cols, "points", False)
        out[name] = pd.DataFrame({"game_id": D.game_id.to_numpy(), f"total_{name}": np.asarray(pr)})
    res = out["list"].merge(out["std"], on="game_id", how="outer")
    res["asof_differs"] = res.game_id.map((remap.src != remap.game_id).groupby(remap.game_id).any())
    return res


def listing_times(g: pd.DataFrame, mk: pd.DataFrame, evmap: dict) -> pd.Series:
    mk = mk.assign(ot=pd.to_datetime(mk.open_time, utc=True).map(lambda t: int(t.timestamp())))
    first = mk.groupby("event_ticker").ot.min()
    return pd.Series({gid: int(first[ev]) for gid, ev in evmap.items()})


def cmd_level() -> int:
    g = dev_schedule()
    mk = markets()
    evmap = match_events(g, mk)
    mt = model_totals(listing_times(g, mk, evmap))
    d = g[["game_id", "week", "total_line"]].merge(mt, on="game_id")
    w = d[d.week <= 4].dropna(subset=["total_list", "total_line"])
    bias = float((w.total_list - w.total_line).mean())
    res = dict(n_wk1_4=int(len(w)), bias_list_wk1_4=bias, bias_std_wk1_4=float((w.total_std - w.total_line).mean()),
               mae_list_wk1_4=float((w.total_list - w.total_line).abs().mean()),
               rule="no correction, score wk 1-12" if abs(bias) <= LEVEL_TOL else f"subtract {bias:+.2f}, score wk 5-12",
               correct=bool(abs(bias) > LEVEL_TOL), n_events_matched=len(evmap), n_dev_games=int(len(g)))
    LEVEL.parent.mkdir(parents=True, exist_ok=True)
    LEVEL.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


# ------------------------------------------------------------------------------------------- prices
def cmd_build() -> int:
    assert LEVEL.exists(), "run `level` first (pre-registered order)"
    g = dev_schedule()
    mk = markets()
    evmap = match_events(g, mk)
    rows, frames = [], []
    for _, r in g.iterrows():
        ev = evmap.get(r.game_id)
        if ev is None:
            continue
        ko = int(pd.Timestamp(r.start_utc).timestamp())
        for _, x in mk[mk.event_ticker == ev].iterrows():
            ot = int(pd.Timestamp(x.open_time).timestamp())
            rows.append(dict(game_id=r.game_id, week=int(r.week), ticker=x.ticker, strike=float(x.strike),
                             open_ts=ot, ko_ts=ko, result=x.result))
            cs, s = [], ot - 3600
            while s < ko:
                e = min(ko, s + 3600 * 2000)
                j = fetch(f"{API}/historical/markets/{x.ticker}/candlesticks",
                          {"start_ts": s, "end_ts": e, "period_interval": 60}, cache=True)
                cs += j.get("candlesticks", [])
                s = e
            for c in cs:
                frames.append(dict(game_id=r.game_id, ticker=x.ticker, end=int(c["end_period_ts"]),
                                   bid=num((c.get("yes_bid") or {}).get("close")),
                                   ask=num((c.get("yes_ask") or {}).get("close")),
                                   vol=num(c.get("volume"))))
    pd.DataFrame(rows).to_parquet(META, index=False)
    cd = pd.DataFrame(frames).drop_duplicates(["ticker", "end"])
    cd.to_parquet(OUT, index=False)
    print(f"{len(g)} dev games; {len(evmap)} matched events; {len(rows)} strike markets; {len(cd)} candles -> {OUT}")
    return 0


def pava_dec(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pool-adjacent-violators for a NON-INCREASING fit."""
    y = -np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; cnt[-2] += cnt[-1]; vals[-2] = v
            vals.pop(); wts.pop(); cnt.pop()
    return -np.repeat(vals, cnt)


def implied(strikes: np.ndarray, mids: np.ndarray) -> float:
    o = np.argsort(strikes)
    k, m = strikes[o], pava_dec(mids[o])
    for i in range(len(k) - 1):
        if m[i] >= 0.5 >= m[i + 1] and k[i + 1] > k[i]:
            if m[i] == m[i + 1]:
                return float((k[i] + k[i + 1]) / 2)
            return float(k[i] + (m[i] - 0.5) / (m[i] - m[i + 1]) * (k[i + 1] - k[i]))
    return np.nan


def ladders(cd: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per game: first-tight and last pre-kick implied total, plus the first-tight ladder quotes."""
    out = []
    for gid, c in cd.groupby("game_id"):
        mm = meta[meta.game_id == gid]
        ko, L = int(mm.ko_ts.iloc[0]), int(mm.open_ts.min())
        c = c[c.end <= ko]
        if c.empty:
            continue
        piv_b = c.pivot(index="end", columns="ticker", values="bid").sort_index().ffill()
        piv_a = c.pivot(index="end", columns="ticker", values="ask").sort_index().ffill()
        sk = mm.set_index("ticker").strike
        res = mm.set_index("ticker").result
        series, first = [], None
        for t in piv_b.index:
            if t < L:
                continue
            b, a = piv_b.loc[t], piv_a.loc[t]
            ok = (b > 0) & (a < 1) & ((a - b) <= MAXSPR + 1e-9) & b.notna() & a.notna()
            if ok.sum() < 2:
                continue
            tk = ok[ok].index
            imp = implied(sk[tk].to_numpy(float), ((b[tk] + a[tk]) / 2).to_numpy(float))
            if np.isnan(imp):
                continue
            series.append((t, imp))
            if first is None:
                first = dict(t=int(t), imp=imp, quotes=pd.DataFrame(dict(ticker=tk, strike=sk[tk].to_numpy(float),
                             bid=b[tk].to_numpy(float), ask=a[tk].to_numpy(float),
                             result=res[tk].to_numpy())))
        if first is None:
            continue
        # first two-sided quote anywhere on the ladder (descriptive)
        two = c[(c.bid > 0) & (c.ask < 1)]
        q = first["quotes"]
        nearest = q.iloc[(q.strike - first["imp"]).abs().argsort()].iloc[0]
        out.append(dict(game_id=gid, open_ts=L, ko_ts=ko, first_t=first["t"], f_imp=first["imp"],
                        l_t=int(series[-1][0]), l_imp=float(series[-1][1]), n_strikes_first=len(q),
                        first_two_t=int(two.end.min()) if len(two) else np.nan,
                        spread_nearest_first=float(nearest.ask - nearest.bid),
                        vol_pre=float(c.vol.sum()), vol_24h=float(c[c.end <= L + 86400].vol.sum()),
                        quotes=q.to_dict("records")))
    return pd.DataFrame(out)


def pick(quotes: list, target: float, tiebreak: float) -> dict:
    q = pd.DataFrame(quotes)
    q["d"] = (q.strike - target).abs()
    q["d2"] = (q.strike - tiebreak).abs()
    return q.sort_values(["d", "d2"]).iloc[0].to_dict()


def taker(row, strike_q: dict, rounded=True) -> dict:
    over = row.side == "over"
    p = float(strike_q["ask"]) if over else float(1 - strike_q["bid"])
    won = (strike_q["result"] == "yes") if over else (strike_q["result"] == "no")
    cost = p + fee(p, rounded)
    return dict(px=p, cost=cost, pnl=float(won) - cost, won=bool(won), strike=float(strike_q["strike"]))


def bets(lad: pd.DataFrame, mt: pd.DataFrame, col: str, shift: float, weeks: pd.Series, rounded=True) -> pd.DataFrame:
    b = lad.merge(mt[["game_id", col]], on="game_id", how="inner").dropna(subset=[col])
    b["model"] = b[col] - shift
    b["week"] = b.game_id.map(weeks)
    b["gap"] = b.model - b.f_imp
    b["sel"] = b.gap.abs() >= GAP
    b["side"] = np.where(b.gap > 0, "over", "under")
    b["clv"] = (b.l_imp - b.f_imp) * np.where(b.side == "over", 1, -1)
    for tag, tgt, tie in (("t2", "model", "f_imp"), ("t3", "f_imp", "model")):
        recs = [taker(r, pick(r.quotes, r[tgt], r[tie]), rounded) for _, r in b.iterrows()]
        for k in ("px", "cost", "pnl", "won", "strike"):
            b[f"{tag}_{k}"] = [x[k] for x in recs]
    return b


def score(b: pd.DataFrame, halves: tuple) -> dict:
    s = b[b.sel]
    out = {}
    for name, kind in zip(TRIALS, ("clv", "t2", "t3")):
        hv = {}
        for lab, (lo, hi) in halves:
            part = s[(s.week >= lo) & (s.week <= hi)]
            if kind == "clv":
                hv[lab] = [round(float(part.clv.mean()), 4) if len(part) else None, int(len(part))]
            else:
                hv[lab] = [round(float(part[f"{kind}_pnl"].sum() / part[f"{kind}_cost"].sum()), 4) if len(part) else None,
                           int(len(part))]
        r = boot(s.clv.to_numpy()) if kind == "clv" else boot(s[f"{kind}_pnl"].to_numpy(), s[f"{kind}_cost"].to_numpy())
        both = all(v[0] is not None and v[0] > 0 for v in hv.values())
        out[name] = r | {"halves": hv, "passed": bool(r["p"] < BAR and r["mean"] > 0 and both)}
        if kind != "clv":
            out[name]["win_rate"] = float(s[f"{kind}_won"].mean()) if len(s) else None
            out[name]["mean_px"] = float(s[f"{kind}_px"].mean()) if len(s) else None
    out["_bet_set"] = dict(games=int(len(b)), selected=int(len(s)), over_share=float((s.side == "over").mean()))
    return out


def cmd_dev() -> int:
    lv = json.loads(LEVEL.read_text())
    cd, meta = pd.read_parquet(OUT), pd.read_parquet(META)
    assert meta.week.max() <= DEV_WEEKS
    weeks = meta.groupby("game_id").week.first()
    listing = meta.groupby("game_id").open_ts.min()
    mt = model_totals(listing)
    lad = ladders(cd, meta)
    shift = lv["bias_list_wk1_4"] if lv["correct"] else 0.0
    wk_lo = 5 if lv["correct"] else 1
    halves = ((("w5-8", (5, 8)), ("w9-12", (9, 12))) if lv["correct"] else (("w1-6", (1, 6)), ("w7-12", (7, 12))))
    lad_s = lad[lad.game_id.map(weeks) >= wk_lo]
    g = pd.read_parquet(DATA / "nfl_games.parquet", columns=["game_id", "total_line", "home_score", "away_score"])
    res = {"level": lv, "n_games_meta": int(meta.game_id.nunique()), "n_games_laddered": int(len(lad)),
           "n_games_scored_window": int(len(lad_s))}
    L = lad.assign(lead_d=(lad.ko_ts - lad.open_ts) / 86400, lag_h=(lad.first_t - lad.open_ts) / 3600,
                   lag_two_h=(lad.first_two_t - lad.open_ts) / 3600, before_ko_d=(lad.ko_ts - lad.first_t) / 86400)
    res["describe"] = {k: L[k].describe().round(2).to_dict() for k in
                       ("lead_d", "lag_two_h", "lag_h", "before_ko_d", "n_strikes_first", "spread_nearest_first",
                        "vol_24h", "vol_pre")}
    res["describe"]["lead_days_by_week"] = L.assign(w=L.game_id.map(weeks)).groupby("w").lead_d.median().round(1).to_dict()
    res["primary"] = {}
    for name, col in (("as_of_listing", "total_list"), ("std_week_features", "total_std")):
        b = bets(lad_s, mt, col, shift, weeks).merge(g, on="game_id", how="left")
        r = score(b, halves)
        b["pts"] = b.home_score + b.away_score
        r["_vs_close"] = dict(n=int(b.total_line.notna().sum()),
                              mae_first_vs_close=float((b.f_imp - b.total_line).abs().mean()),
                              mae_last_vs_close=float((b.l_imp - b.total_line).abs().mean()),
                              mae_model_vs_close=float((b.model - b.total_line).abs().mean()),
                              bias_model_vs_close=float((b.model - b.total_line).mean()),
                              bias_first_vs_close=float((b.f_imp - b.total_line).mean()),
                              corr_gap_vs_move=float(np.corrcoef(b.gap, b.l_imp - b.f_imp)[0, 1]),
                              rmse_first_vs_pts=float(np.sqrt(((b.f_imp - b.pts) ** 2).mean())),
                              rmse_last_vs_pts=float(np.sqrt(((b.l_imp - b.pts) ** 2).mean())),
                              rmse_close_vs_pts=float(np.sqrt(((b.total_line - b.pts) ** 2).mean())),
                              rmse_model_vs_pts=float(np.sqrt(((b.model - b.pts) ** 2).mean())))
        s = b[b.sel]
        r["_uncounted"] = dict(
            unrounded_fee_t2_roi=float(bets(lad_s, mt, col, shift, weeks, False).query("sel").pipe(
                lambda d: d.t2_pnl.sum() / d.t2_cost.sum())),
            clv_by_gap={lab: [round(float(x.clv.mean()), 3), int(len(x))] for lab, x in
                        (("2.5-4", s[s.gap.abs() < 4]), ("4-6", s[(s.gap.abs() >= 4) & (s.gap.abs() < 6)]),
                         (">=6", s[s.gap.abs() >= 6]))},
            clv_over=[float(s[s.side == "over"].clv.mean()), int((s.side == "over").sum())],
            clv_under=[float(s[s.side == "under"].clv.mean()), int((s.side == "under").sum())],
            blind_drift_all=float((b.l_imp - b.f_imp).mean()),
            clv_excl_wk1_2=float(s[s.week > 2].clv.mean()) if (s.week > 2).any() else None)
        res["primary"][name] = r
        b.drop(columns=["quotes"]).to_parquet(DATA / f"lab/kalshi_early_nfl_totals_bets_{name}.parquet", index=False)
    print(json.dumps(res, indent=1, default=float))
    (DATA / "lab" / "kalshi_early_nfl_totals_results.json").write_text(json.dumps(res, indent=1, default=float))
    if "--ledger" in sys.argv:
        recs = []
        for i, name in enumerate(TRIALS, 1):
            r = res["primary"]["as_of_listing"][name]
            recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "strategy": f"kalshi_early_nfl_totals_{name}",
                         "sha": hashlib.sha256(f"kalshi_early_nfl_totals:{name}".encode()).hexdigest()[:12],
                         "dataset": DATASET, "metric": "clv_pts" if name.startswith("T1") else "roi", "trial": name,
                         "hypothesis": f"pre-registered in scripts/kalshi_early_nfl_totals.py ({name})",
                         "source": "lead brief 2026-10-25: Kalshi early NFL totals ladder stale vs legacy model B totals?",
                         "status": "ok", "trial_number": i, "study_trials": len(TRIALS), "bonferroni_bar": BAR,
                         "bets": r["n"], ("mean" if name.startswith("T1") else "roi"): r["mean"],
                         "ci_lo": r["lo"], "ci_hi": r["hi"], "p": r["p"], "per_fold": r["halves"],
                         "passed": r["passed"]})
        with LEDGER.open("a") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
        print(f"ledger += {len(recs)} rows")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    sys.exit({"level": cmd_level, "build": cmd_build, "dev": cmd_dev}[cmd]())

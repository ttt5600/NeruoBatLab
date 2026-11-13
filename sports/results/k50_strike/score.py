"""k50_strike dev scoring (pre-registered in notes/k50_strike.md). Rules K50 (model vs Kalshi's own ~50c strike, k fixed)
and LO (ladder-optimal EV under Normal(m, sigma)). Leagues: NBA (L32b), CFB (L36), NFL (L33).
Usage: score.py -> rows_<lg>.csv, summary.json."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = Path(__file__).resolve().parent
XL = OUT.parent / "kalshi_openers_xleague"
NB = OUT.parent / "nba_l32b_exec"
sys.path.insert(0, str(XL))
from kx import side_x  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
SIG = json.load(open(OUT / "sigma.json"))
K_PRIMARY = {("nba", "total"): 3.0, ("cfb", "total"): 3.5, ("nfl", "total"): 2.5, ("nfl", "spread"): 2.0}
K_GRID = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
EV_MIN = 0.03


def fee(p, n=100):
    return math.ceil(round(n * 0.07 * p * (1 - p) * 100, 6)) / 100 / n


def secs(T):
    return int((pd.Timestamp(T) - EPOCH) // pd.Timedelta(seconds=1))


def et_date(T):
    return pd.Timestamp(T).tz_convert("America/New_York").strftime("%Y-%m-%d")


def ladder(c, T, max_age, ev=None, mk="total"):
    """c: candles (ticker, strike, end, bid, ask[, event]) -> list of rungs on the over axis."""
    t = secs(T)
    c = c[(c.end <= t) & (c.end >= t - max_age) & c.bid.notna() & c.ask.notna()]
    c = c.sort_values("end").groupby("ticker").tail(1)
    c = c[(c.bid > 0) & (c.ask < 1)]
    out = []
    for r in c.itertuples():
        x, sg = side_x(ev or getattr(r, "event", ""), r.ticker, r.strike, mk)
        mid = (r.bid + r.ask) / 2
        # cost of "axis > x" (up) and "axis < x" (down)
        up_cost = r.ask if sg > 0 else 1 - r.bid
        dn_cost = 1 - r.bid if sg > 0 else r.ask
        out.append(dict(ticker=r.ticker, x=x, p_over=mid if sg > 0 else 1 - mid, spread=r.ask - r.bid,
                        up=up_cost, dn=dn_cost, age_min=(t - r.end) / 60))
    return out


def anchor(L, m):
    return min(L, key=lambda z: (round(abs(z["p_over"] - .5), 6), round(z["spread"], 6), abs(z["x"] - m)))


def bet_row(z, up, final, close, how):
    ask = z["up"] if up else z["dn"]
    cost = ask + fee(ask)
    r = dict(rule=how, S=z["x"], pick="up" if up else "down", ask=ask, cost=cost, spread_c=round(z["spread"] * 100),
             age_min=z["age_min"], ticker=z["ticker"])
    if final is not None and np.isfinite(final):
        won = final > z["x"] if up else final < z["x"]
        r.update(final=final, won=bool(won), roi=(1 - cost) / cost if won else -1.0)
    if close is not None and np.isfinite(close):
        r["clv"] = (close - z["x"]) if up else (z["x"] - close)
    return r


def decide(L, m, sigma, k, final, close):
    """-> list of bet dicts for K50 at k (plus grid) and LO."""
    res = []
    if not L:
        return res
    a = anchor(L, m)
    g = m - a["x"]
    for kk in sorted(set(K_GRID + [k])):
        if abs(g) >= kk:
            res.append({**bet_row(a, g > 0, final, close, f"k50_k{kk}"), "gap": g, "primary": kk == k})
    best = None
    for z in L:
        p_up = 1 - norm.cdf((z["x"] - m) / sigma)
        for up, p in ((True, p_up), (False, 1 - p_up)):
            ask = z["up"] if up else z["dn"]
            ev = p - (ask + fee(ask))
            if best is None or ev > best[0]:
                best = (ev, z, up)
    if best[0] >= EV_MIN:
        res.append({**bet_row(best[1], best[2], final, close, "lo"), "gap": m - best[1]["x"], "ev": best[0],
                    "primary": True})
    return res


# ---------------- NBA ----------------
def nba():
    S = json.load(open(Path.home() / ".claude/quant-sports/data/l32b_state.json"))
    Lst = json.load(open(Path.home() / ".claude/quant-kalshi/data/open_totals_state.json"))
    C = pd.concat([pd.read_parquet(NB / "q2_candles.parquet"), pd.read_parquet(OUT / "nba_candles.parquet")])
    M = pd.concat([pd.read_parquet(NB / "q2_markets.parquet"), pd.read_parquet(OUT / "nba_markets.parquet")])
    res_by_t = M.drop_duplicates("ticker").set_index("ticker").result.to_dict()
    rows = []
    for r in S.values():
        if r["kind"] != "live" or pd.Timestamp(r["t"]) >= pd.Timestamp("2026-11-11", tz="UTC"):
            continue
        k = r["key"]
        g = Lst[k]
        m = r["model"] - r["c"]
        T = pd.Timestamp(r["t"])
        base = dict(lg="nba", mk="total", key=k, date=k.split("|")[0], T=T, m=m, open=r["open_line"],
                    close=(g.get("pin_last") or {}).get("line"))
        L = ladder(C[C.key == k], T, 3600, mk="total")
        if not L:
            rows.append({**base, "status": "no quote"})
            continue
        bets = decide(L, m, SIG["nba"]["sd_total"], K_PRIMARY[("nba", "total")], np.nan, base["close"])
        a = anchor(L, m)
        rows.append({**base, "status": "ladder", "S50": a["x"], "S50_mid": a["p_over"], "S50_spread_c": round(a["spread"] * 100),
                     "n_rungs": len(L)})
        for b in bets:  # grade by the Kalshi settlement of the bought market (YES = total > strike)
            rz = res_by_t.get(b["ticker"])
            if rz in ("yes", "no"):
                yes = rz == "yes"
                won = yes if b["pick"] == "up" else not yes
                b.update(final_result=rz, won=won, roi=(1 - b["cost"]) / b["cost"] if won else -1.0)
                fin = r.get("final_total")
                if fin is not None and np.isfinite(fin):
                    assert (fin > b["S"]) == yes, (k, fin, b)
            rows.append({**base, "status": "bet", **b})
    return pd.DataFrame(rows)


# ---------------- CFB ----------------
def cfb():
    R = pd.read_csv(XL / "q1_cfb_rows.csv")
    R = R[(R.mk == "total") & R.ft_t.notna()]
    C = pd.read_parquet(XL / "cfb_candles.parquet")
    G = pd.read_parquet(XL / "cfb_games.parquet")
    K = pd.read_parquet(XL / "cfb_kmarkets.parquet")
    P = pd.read_parquet(XL / "cfb_preds_2026.parquet")
    P["event_id"] = P.event_id.astype(str)
    G["event_id"] = G.event_id.astype(str)
    G = G.merge(P[["event_id", "pred_total", "fbs_game"]], on="event_id", how="left")
    ev2key = K.drop_duplicates("event").set_index("event").key
    gk = G.drop_duplicates("key").set_index("key")
    rows = []
    for r in R.itertuples():
        g = gk.loc[ev2key[r.event]]
        base = dict(lg="cfb", mk="total", key=r.event)
        if pd.isna(g.pred_total) or not bool(g.fbs_game):
            rows.append({**base, "status": "no pred/non-FBS"})
            continue
        T = pd.Timestamp(r.ft_t)
        kick = pd.Timestamp(r.occ)
        if kick - T < pd.Timedelta(minutes=30):
            rows.append({**base, "status": "ft < 30 min"})
            continue
        m = float(g.pred_total)
        fin = float(g.home_score + g.away_score) if pd.notna(g.home_score) else np.nan
        close = float(g.total_close) if pd.notna(g.total_close) else np.nan
        base.update(date=et_date(kick), T=T, m=m, open=g.total_open, close=close)
        L = ladder(C[C.event == r.event], T, 6 * 3600, ev=r.event, mk="total")
        if not L:
            rows.append({**base, "status": "no quote"})
            continue
        a = anchor(L, m)
        rows.append({**base, "status": "ladder", "S50": a["x"], "S50_mid": a["p_over"], "S50_spread_c": round(a["spread"] * 100),
                     "n_rungs": len(L)})
        for b in decide(L, m, SIG["cfb"]["sd_total"], K_PRIMARY[("cfb", "total")], fin, close):
            rows.append({**base, "status": "bet", **b})
    return pd.DataFrame(rows)


# ---------------- NFL ----------------
def nfl():
    s = json.load(open(Path.home() / ".claude/quant-kalshi/data/open_lines_nfl_state.json"))["_model_cache"]
    mc = pd.DataFrame([dict(cache_t=v["t"], **r) for w, v in s.items() for r in v["rows"]])
    G = pd.read_parquet(XL / "nfl_games.parquet")
    C = pd.read_parquet(XL / "nfl_candles.parquet")
    R = pd.read_csv(XL / "q1_nfl_rows.csv")
    K = pd.read_parquet(XL / "nfl_kmarkets.parquet")
    code = {"WAS": "WSH", "LA": "LAR"}
    G["start"] = pd.to_datetime(G.start_utc, utc=True)
    rows = []
    for c in mc.itertuples():
        kick = pd.Timestamp(c.kickoff_utc)
        h, a = code.get(c.home, c.home), code.get(c.away, c.away)
        gg = G[(G.home == h) & (G.away == a) & ((G.start - kick).abs() < pd.Timedelta(hours=2))]
        if len(gg) != 1:
            rows.append(dict(lg="nfl", key=c.game_id, status=f"no espn game ({len(gg)})"))
            continue
        g = gg.iloc[0]
        if g.status != "STATUS_FINAL":
            rows.append(dict(lg="nfl", key=c.game_id, status="not final"))
            continue
        for mk in ("total", "spread"):
            evs = K[(K.key == g.key) & (K.mk == mk)].event.unique()
            base = dict(lg="nfl", mk=mk, key=c.game_id, date=et_date(kick))
            if len(evs) != 1:
                rows.append({**base, "status": "no kalshi event"})
                continue
            ev = evs[0]
            rr = R[R.event == ev]
            ft = pd.Timestamp(rr.ft_t.iloc[0]) if len(rr) and pd.notna(rr.ft_t.iloc[0]) else None
            if ft is None:
                rows.append({**base, "status": "never tight"})
                continue
            T = max(pd.Timestamp(c.cache_t), ft)
            T = pd.Timestamp(math.ceil(secs(T) / 3600) * 3600, unit="s", tz="UTC")  # first hourly candle end >= T
            if kick - T < pd.Timedelta(minutes=30):
                rows.append({**base, "status": "T < 30 min"})
                continue
            if mk == "total":
                m, fin, close, op = c.model_total, g.home_score + g.away_score, g.total_close, g.total_open
            else:
                m, fin, close, op = c.model_spread_home, g.home_score - g.away_score, -g.home_spread_close, -g.home_spread_open
            base.update(T=T, m=m, open=op, close=close)
            L = ladder(C[C.event == ev], T, 6 * 3600, ev=ev, mk=mk)
            if not L:
                rows.append({**base, "status": "no quote"})
                continue
            an = anchor(L, m)
            rows.append({**base, "status": "ladder", "S50": an["x"], "S50_mid": an["p_over"],
                         "S50_spread_c": round(an["spread"] * 100), "n_rungs": len(L)})
            sd = SIG["nfl"]["sd_total" if mk == "total" else "sd_spread"]
            for b in decide(L, m, sd, K_PRIMARY[("nfl", mk)], float(fin), float(close)):
                rows.append({**base, "status": "bet", **b})
    return pd.DataFrame(rows)


def cboot(x, col, seed=0, B=4000):
    x = x[x[col].notna()]
    if len(x) < 2:
        return [float(x[col].mean()) if len(x) else None, None, None, None]
    dates = x.date.unique()
    grp = {d: x.loc[x.date == d, col].to_numpy(float) for d in dates}
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for i in range(B):
        pick = rng.choice(dates, len(dates))
        v = np.concatenate([grp[d] for d in pick])
        bs[i] = v.mean()
    return [round(float(x[col].mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4), round(float((bs <= 0).mean()), 4)]


def summarize(d, lg):
    out = {"status": d.drop_duplicates(["key", "mk", "status"]).status.value_counts().to_dict() if len(d) else {}}
    lad = d[d.status == "ladder"]
    if len(lad):
        out["ladder"] = dict(n=len(lad), S50_mid_mean=round(lad.S50_mid.mean(), 3), S50_spread_c_med=float(lad.S50_spread_c.median()),
                             S50_minus_open_abs=round((lad.S50 - lad.open).abs().mean(), 2),
                             m_minus_S50_sd=round((lad.m - lad.S50).std(), 2), m_minus_S50_mean=round((lad.m - lad.S50).mean(), 2))
    b = d[(d.status == "bet")].copy()
    if len(b):
        pr = b[b.rule.str.startswith("k50") & (b.primary == True)].copy()
        pr["rule"] = "K50_PRIMARY"
        b = pd.concat([b, pr])
    for rule, x in b.groupby("rule"):
        x = x[x.roi.notna()]
        if not len(x):
            continue
        s = dict(n=len(x), W=int((x.won == True).sum()), dates=int(x.date.nunique()), roi=cboot(x, "roi"),
                 clv=cboot(x, "clv"), avg_cost=round(x.cost.mean(), 4), avg_ask=round(x.ask.mean(), 4),
                 ups=int((x.pick == "up").sum()), primary=bool(x.primary.iloc[0]))
        if lg == "nfl":
            s["by_mk"] = {mk: dict(n=len(y), W=int((y.won == True).sum()), roi=round(y.roi.mean(), 4),
                                   clv=round(y.clv.mean(), 3), cost=round(y.cost.mean(), 3)) for mk, y in x.groupby("mk")}
        out[rule] = s
    return out


if __name__ == "__main__":
    allS = {}
    for lg, fn in (("nba", nba), ("cfb", cfb), ("nfl", nfl)):
        d = fn()
        d.to_csv(OUT / f"rows_{lg}.csv", index=False)
        allS[lg] = summarize(d, lg)
        print(lg, json.dumps(allS[lg], default=str))
    json.dump(allS, open(OUT / "summary.json", "w"), indent=1, default=str)

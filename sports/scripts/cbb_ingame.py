"""Men's COLLEGE BASKETBALL in-game: simple win-probability model vs Kalshi KXNCAAMBGAME in-game prices.

    python scripts/cbb_ingame.py espn SEASON        # SEASON = 2024 (2023-24), 2025, 2026 (2025-26), 2027 (2026-27 so far)
    python scripts/cbb_ingame.py fit                # fit on 2023-24 + 2024-25 (before any Kalshi CBB market existed)
    python scripts/cbb_ingame.py kalshi             # 1-min candles for every KXNCAAMBGAME game (home-team market)
    python scripts/cbb_ingame.py build              # snapshot table -> data/cbb_ingame/snapshots.parquet (edgelab loader)
    python scripts/cbb_ingame.py brier              # descriptive: Kalshi mid vs model vs ESPN WP by minutes-left / volume

DATA. ESPN summary API (free, no key): plays[] with homeScore/awayScore/period/clock/team/type/wallclock and
winprobability[] (ESPN's own model; first value = ESPN pregame home p0). Only compact per-play rows are stored
(data/cbb_ingame/plays_<season>.parquet), never the raw summaries (~200 KB each). Kalshi: KXNCAAMBGAME, first game
2025-11-03; one market per team; we pull the HOME team's market: buying home = YES at yes_ask, buying away = NO at
1 - yes_bid (same payoff as the away YES contract).

MODEL (Brownian-motion probit, from the home side), fit on games dated < 2025-11-01 only:
    P(home) = Phi(b0 + b1*m/sqrt(tau) + b2*z0*sqrt(tau) + b3*poss/sqrt(tau) + b4*m)
    m = home margin; tau = (game seconds left + 4)/2400 (OT: seconds left in the OT period); z0 = Phi^-1(prior);
    poss = +1/-1/0 from a play-type heuristic (rebound/steal -> that team; turnover, made FG, made FT -> other team).
    Fit prior = ESPN pregame p0. Test prior = Kalshi pregame mid (see edgelab/rules/cbb_ingame.py), else ESPN p0.

PRE-REGISTRATION lives in edgelab/specs/cbb_ig_*.yaml + edgelab/rules/cbb_ingame.py (registered before any Kalshi
candle was pulled in bulk; 2 candle files were probed for the schema on 2026-11-05).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))

import numpy as np
import pandas as pd
from scipy.stats import norm

from sports.http import fetch

D = ROOT / "data" / "cbb_ingame"
COEF = D / "coef.json"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
KAPI = "https://api.elections.kalshi.com/trade-api/v2"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2025-11-01")
SEASONS = {2024: ("2023-11-06", "2024-04-08"), 2025: ("2024-11-04", "2025-04-07"),
           2026: ("2025-11-03", "2026-04-06"), 2027: ("2026-11-02", "2026-11-04")}
SKIP_TYPES = {"Substitution", "OfficialTVTimeOut", "ShortTimeOut", "RegularTimeOut", "FullTimeOut"}


def secs(c):
    try:
        m, s = c.split(":") if ":" in c else ("0", c)
        return int(m) * 60 + float(s)
    except Exception:
        return np.nan


def scoreboard(season: int) -> pd.DataFrame:
    a, b = SEASONS[season]
    rows = []
    for d in pd.date_range(a, b):
        sb = fetch(f"{ESPN}/scoreboard", {"dates": d.strftime("%Y%m%d"), "groups": 50, "limit": 400},
                   cache=d < pd.Timestamp.now() - pd.Timedelta(days=2))
        for e in sb.get("events", []):
            c = e["competitions"][0]
            if c.get("status", {}).get("type", {}).get("completed") is not True:
                continue
            tm = {x["homeAway"]: x for x in c["competitors"]}
            rows.append({"eid": e["id"], "date": d, "start": e.get("date"), "neutral": bool(c.get("neutralSite")),
                         "home": tm["home"]["team"].get("abbreviation"), "away": tm["away"]["team"].get("abbreviation"),
                         "home_name": tm["home"]["team"].get("location"), "away_name": tm["away"]["team"].get("location"),
                         "home_id": tm["home"]["team"]["id"], "away_id": tm["away"]["team"]["id"],
                         "stype": e.get("season", {}).get("type")})
    return pd.DataFrame(rows).drop_duplicates("eid")


def parse(eid: str, j: dict):
    comp = j["header"]["competitions"][0]
    side = {c["id"]: c["homeAway"] for c in comp["competitors"]}
    wp = j.get("winprobability") or []
    plays = j.get("plays") or []
    if not plays:
        return None, []
    last = plays[-1]
    hs, as_ = last.get("homeScore"), last.get("awayScore")
    pc = (j.get("pickcenter") or [{}])[0]
    meta = {"eid": eid, "p0": wp[0]["homeWinPercentage"] if wp else np.nan, "home_won": int(hs > as_),
            "final_margin": hs - as_, "n_periods": max(p.get("period", {}).get("number", 0) for p in plays),
            "spread": pc.get("spread"), "home_ml": (pc.get("homeTeamOdds") or {}).get("moneyLine"),
            "away_ml": (pc.get("awayTeamOdds") or {}).get("moneyLine")}
    wpd = {w["playId"]: w["homeWinPercentage"] for w in wp}
    rows, poss = [], 0
    for p in plays:
        typ = p.get("type", {}).get("text", "")
        if typ in SKIP_TYPES:
            continue
        pn = p.get("period", {}).get("number", 0)
        tid = (p.get("team") or {}).get("id")
        s = side.get(tid)
        sgn = 1 if s == "home" else -1 if s == "away" else 0
        made = bool(p.get("scoringPlay"))
        if sgn:
            if "Rebound" in typ or typ == "Steal":
                poss = sgn
            elif "Turnover" in typ:
                poss = -sgn
            elif made:
                poss = -sgn
        cs = secs(p.get("clock", {}).get("displayValue", ""))
        left = cs + (1200 if pn == 1 else 0) if pn <= 2 else cs
        rows.append({"eid": eid, "seq": int(p.get("sequenceNumber", 0)), "period": pn, "left": left,
                     "hs": p.get("homeScore"), "as": p.get("awayScore"), "poss": poss, "wall": p.get("wallclock"),
                     "espn_wp": wpd.get(p["id"], np.nan)})
    return meta, rows


def _one(eid):
    try:
        return parse(eid, fetch(f"{ESPN}/summary", {"event": eid}, cache=False))
    except Exception as ex:
        print("skip", eid, ex, flush=True)
        return None, []


def cmd_espn() -> int:
    season = int(sys.argv[2])
    D.mkdir(parents=True, exist_ok=True)
    g = scoreboard(season)
    print(f"season {season}: {len(g)} completed D1 games", flush=True)
    metas, rows = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (m, rr) in enumerate(ex.map(_one, g.eid.tolist())):
            if m:
                metas.append(m)
                rows += rr
            if i % 500 == 0:
                print(season, i, len(rows), flush=True)
    meta = g.merge(pd.DataFrame(metas), on="eid")
    meta["season"] = season
    pl = pd.DataFrame(rows)
    pl["wall"] = pd.to_datetime(pl.wall, utc=True, format="ISO8601", errors="coerce")
    meta.to_parquet(D / f"games_{season}.parquet", index=False)
    pl.to_parquet(D / f"plays_{season}.parquet", index=False)
    print(f"-> {len(meta)} games, {len(pl)} plays; wallclock filled {pl.wall.notna().mean():.3f}")
    return 0


# ------------------------------------------------------------------ model
def feats(m, left, z0, poss):
    tau = (np.asarray(left, float) + 4) / 2400
    st = np.sqrt(tau)
    m = np.asarray(m, float)
    return np.column_stack([np.ones_like(tau), m / st, np.asarray(z0, float) * st, np.asarray(poss, float) / st, m])


def predict(b, m, left, z0, poss):
    return norm.cdf(feats(m, left, z0, poss) @ np.asarray(b))


def load(seasons):
    g = pd.concat([pd.read_parquet(D / f"games_{s}.parquet") for s in seasons], ignore_index=True)
    p = pd.concat([pd.read_parquet(D / f"plays_{s}.parquet") for s in seasons], ignore_index=True)
    s = p.merge(g[["eid", "date", "season", "p0", "home_won"]], on="eid", how="inner", validate="many_to_one")
    s = s.dropna(subset=["left", "hs", "as", "p0"])
    s["m"] = s.hs - s["as"]
    s["z0"] = norm.ppf(np.clip(s.p0, 0.01, 0.99))
    return s


def fit_probit(s):
    import statsmodels.api as sm
    X = feats(s.m, s.left, s.z0, s.poss)
    return sm.GLM(s.home_won.to_numpy(), X, family=sm.families.Binomial(sm.families.links.Probit())).fit().params


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cmd_fit() -> int:
    s = load([2024, 2025])
    assert s.date.max() < TRAIN_END
    # one row per (game, 15-s bucket of time left) keeps the fit from over-weighting foul-heavy endings
    s = s.assign(bk=(s.left // 15)).drop_duplicates(["eid", "period", "bk"], keep="last")
    tr, va = s[s.season == 2024], s[s.season == 2025]
    b = fit_probit(tr)
    print("validation coef (fit 2023-24):", np.round(b, 4))
    v = va.dropna(subset=["espn_wp"]).copy()
    v["pm"] = predict(b, v.m, v.left, v.z0, v.poss)
    y = v.home_won.to_numpy()
    print(f"OOS 2024-25 n={len(v)} games={v.eid.nunique()}  model LL {ll(y, v.pm):.4f} Brier {np.mean((v.pm - y) ** 2):.4f}"
          f" | ESPN LL {ll(y, v.espn_wp):.4f} Brier {np.mean((v.espn_wp - y) ** 2):.4f}")
    v["bin"] = pd.cut(v.pm, [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1])
    print(v.groupby("bin", observed=True).agg(n=("pm", "size"), games=("eid", "nunique"), model=("pm", "mean"),
                                               real=("home_won", "mean")).round(3).to_string())
    bf = fit_probit(s)
    print("final coef (2023-24 + 2024-25):", np.round(bf, 4))
    COEF.write_text(json.dumps({"coef": list(map(float, bf)), "val_coef": list(map(float, b)),
                                "train_end": str(TRAIN_END.date()), "games": int(s.eid.nunique())}))
    return 0


# ------------------------------------------------------------------ Kalshi
def kalshi_markets() -> pd.DataFrame:
    rows = []
    for path, extra, cache in ((f"{KAPI}/historical/markets", {}, True), (f"{KAPI}/markets", {"status": "settled"}, False)):
        cursor = None
        while True:
            p = {"series_ticker": "KXNCAAMBGAME", "limit": 1000, **extra}
            if cursor:
                p["cursor"] = cursor
            d = fetch(path, p, cache=cache)
            rows += d.get("markets", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    m = pd.DataFrame(rows).drop_duplicates("ticker")
    m = m[m.result.isin(["yes", "no"])].copy()
    m["code"] = m.ticker.str.split("-").str[-1]
    m["kdate"] = pd.to_datetime(m.event_ticker.str.split("-").str[1].str[:7], format="%y%b%d", errors="coerce")
    m["volume"] = pd.to_numeric(m.get("volume_fp"), errors="coerce")
    return m[["ticker", "event_ticker", "code", "kdate", "result", "volume", "close_time", "open_time", "yes_sub_title"]]


def match(g: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    """ESPN game -> the Kalshi event on the same ET date whose two market codes are the ESPN abbreviations (after
    alias), else whose yes_sub_titles contain both school names. Unique matches only."""
    out = []
    by_date = {d: x for d, x in mk.groupby("kdate")}
    for _, r in g.iterrows():
        et = pd.Timestamp(r.start).tz_convert("America/New_York").normalize().tz_localize(None)
        x = by_date.get(et)
        if x is None:
            continue
        cand = []
        for ev, e in x.groupby("event_ticker"):
            if len(e) != 2:
                continue
            codes = set(e.code)
            subs = " | ".join(e.yes_sub_title.astype(str)).lower()
            if {r.home, r.away} == codes or (str(r.home_name).lower() in subs and str(r.away_name).lower() in subs):
                cand.append(e)
        if len(cand) != 1:
            continue
        e = cand[0]
        hm = e[(e.code == r.home)] if r.home in set(e.code) else e[e.yes_sub_title.str.lower().str.startswith(str(r.home_name).lower())]
        if len(hm) != 1:
            continue
        h = hm.iloc[0]
        out.append({"eid": r.eid, "ticker": h.ticker, "event_ticker": h.event_ticker, "home_result": h.result,
                    "volume": float(e.volume.sum())})
    return pd.DataFrame(out)


def cmd_kalshi() -> int:
    g = pd.concat([pd.read_parquet(D / f"games_{s}.parquet") for s in (2026, 2027) if (D / f"games_{s}.parquet").exists()])
    mk = kalshi_markets()
    mt = match(g, mk)
    mt = mt.drop_duplicates("ticker", keep=False)
    print(f"ESPN games {len(g)}; Kalshi events {mk.event_ticker.nunique()}; matched {len(mt)}", flush=True)
    pl = pd.concat([pd.read_parquet(D / f"plays_{s}.parquet") for s in (2026, 2027) if (D / f"plays_{s}.parquet").exists()])
    span = pl.dropna(subset=["wall"]).groupby("eid").wall.agg(["min", "max"])
    mt = mt[mt.eid.isin(span.index)]

    def one(r):
        st = span.loc[r.eid]
        s0 = int((st["min"] - EPOCH) // pd.Timedelta(seconds=1)) - 6 * 3600
        e0 = int((st["max"] - EPOCH) // pd.Timedelta(seconds=1)) + 1800
        for path in (f"{KAPI}/historical/markets/{r.ticker}/candlesticks",
                     f"{KAPI}/series/KXNCAAMBGAME/markets/{r.ticker}/candlesticks"):
            try:
                j = fetch(path, {"start_ts": s0, "end_ts": e0, "period_interval": 1}, cache=False)
            except Exception:
                continue
            if j.get("candlesticks"):
                return r.eid, j["candlesticks"]
        return r.eid, []

    def px(x, k, f="close"):
        d = x.get(k) or {}
        v = d.get(f, d.get(f + "_dollars"))   # historical endpoint: close; live endpoint: close_dollars
        return np.nan if v is None else float(v)

    frames = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (eid, c) in enumerate(ex.map(one, [r for _, r in mt.iterrows()])):
            if c:
                frames.append(pd.DataFrame([{"eid": eid, "end": x["end_period_ts"], "bid": px(x, "yes_bid"),
                                             "ask": px(x, "yes_ask"), "vol": float(x.get("volume") or x.get("volume_fp") or 0),
                                             "tlo": px(x, "price", "low"), "thi": px(x, "price", "high")} for x in c]))
            if i % 500 == 0:
                print("candles", i, flush=True)
    cd = pd.concat(frames, ignore_index=True)
    cd.to_parquet(D / "candles.parquet", index=False)
    mt.to_parquet(D / "kalshi_match.parquet", index=False)
    print(f"-> candles for {cd.eid.nunique()} games, {len(cd)} rows")
    return 0


# ------------------------------------------------------------------ snapshot table (edgelab dataset cbb_ingame)
def ux(ts) -> np.ndarray:
    return ((pd.to_datetime(ts, utc=True) - EPOCH) // pd.Timedelta(seconds=1)).to_numpy(float)


def cmd_build() -> int:
    from traps import check_epoch_seconds
    mt = pd.read_parquet(D / "kalshi_match.parquet")
    cd = pd.read_parquet(D / "candles.parquet")
    g = pd.concat([pd.read_parquet(D / f"games_{s}.parquet") for s in (2026, 2027) if (D / f"games_{s}.parquet").exists()])
    pl = pd.concat([pd.read_parquet(D / f"plays_{s}.parquet") for s in (2026, 2027) if (D / f"plays_{s}.parquet").exists()])
    pl = pl.dropna(subset=["wall", "left", "hs", "as"])
    pl["wu"] = ux(pl.wall)
    check_epoch_seconds(pl.wu.to_numpy(), "play wall")
    check_epoch_seconds(cd.end.to_numpy(float), "candle end")
    g = g.set_index("eid")
    cds = {e: x.sort_values("end").reset_index(drop=True) for e, x in cd.groupby("eid")}
    out = []
    for _, r in mt.iterrows():
        if r.eid not in cds or r.eid not in g.index:
            continue
        gp = pl[pl.eid == r.eid].sort_values(["wu", "seq"])
        if len(gp) < 50:
            continue
        c = cds[r.eid]
        first, last = gp.wu.iloc[0], gp.wu.iloc[-1]
        pre = c[c.end <= first]
        two = pre[(pre.bid >= 0.01) & (pre.ask <= 0.99) & ((pre.ask - pre.bid) <= 0.10)]
        if len(two):
            prior, src, pas = (two.bid.iloc[-1] + two.ask.iloc[-1]) / 2, "kalshi", float(two.end.iloc[-1])
        else:
            prior, src, pas = g.loc[r.eid, "p0"], "espn", first - 60
        t = np.arange(np.ceil((first + 60) / 60) * 60, np.floor(last / 60) * 60 + 1, 60)
        si = np.searchsorted(gp.wu.to_numpy(), t - 60, side="right") - 1
        qi = np.searchsorted(c.end.to_numpy(), t, side="right") - 1
        ok = (si >= 0) & (qi >= 0)
        t, si, qi = t[ok], si[ok], qi[ok]
        s = gp.iloc[si]
        q = c.iloc[qi]
        out.append(pd.DataFrame({"eid": r.eid, "ticker": r.ticker, "date": g.loc[r.eid, "date"], "t": t,
                                 "m": (s.hs - s["as"]).to_numpy(), "left": s.left.to_numpy(), "period": s.period.to_numpy(),
                                 "poss": s.poss.to_numpy(), "state_wall": s.wu.to_numpy(), "espn_wp": s.espn_wp.to_numpy(),
                                 "yes_bid": q.bid.to_numpy(), "yes_ask": q.ask.to_numpy(), "q_end": q.end.to_numpy(float),
                                 "prior": prior, "prior_src": src, "prior_as_of": pas, "p0": g.loc[r.eid, "p0"],
                                 "pre_vol": float(pre.vol.sum()), "volume": r.volume,
                                 "game_end": pd.Timestamp(last, unit="s", tz="UTC"),
                                 "result": r.home_result, "home_won": int(g.loc[r.eid, "home_won"])}))
    snap = pd.concat(out, ignore_index=True)
    # sanity: the ticker's result must agree with ESPN's final score (row misalignment check)
    agree = (snap.drop_duplicates("eid").pipe(lambda x: ((x.result == "yes") == (x.home_won == 1)).mean()))
    print(f"result/ESPN agreement {agree:.4f}")
    snap.to_parquet(D / "snapshots.parquet", index=False)
    print(f"-> {D / 'snapshots.parquet'}: {snap.eid.nunique()} games, {len(snap)} snapshots, "
          f"{snap.date.min().date()}..{snap.date.max().date()}; prior from Kalshi {(snap.drop_duplicates('eid').prior_src == 'kalshi').mean():.3f}")
    return 0


def cmd_brier() -> int:
    """DEV window only (2025-11-03 .. 2026-01-31): Kalshi mid vs model vs ESPN WP, home side, per snapshot."""
    import statsmodels.api as sm
    sys.path.insert(0, str(ROOT))
    import edgelab.rules.cbb_ingame as R
    x = pd.read_parquet(D / "snapshots.parquet")
    x = x[(x.date >= "2025-11-03") & (x.date <= "2026-01-31")]
    x = x[(x.yes_bid >= 0.01) & (x.yes_ask <= 0.99) & (x.left > 0)].copy()
    x["mid"] = (x.yes_bid + x.yes_ask) / 2
    x["spr"] = x.yes_ask - x.yes_bid
    x["model"] = R.model_home(x)
    x["y"] = (x.result == "yes").astype(float)
    print(f"DEV snapshots {len(x)}, games {x.eid.nunique()}; median |mid - model| {(x.mid - x.model).abs().median():.3f}, "
          f"median |mid - ESPN| {(x.mid - x.espn_wp).abs().median():.3f} (alignment sanity)")
    x["mins"] = pd.cut(x.left / 60, [-1, 2, 5, 10, 20, 30, 41], labels=["0-2", "2-5", "5-10", "10-20", "20-30", "30-40"])
    x.loc[x.period > 2, "mins"] = np.nan
    gv = x.drop_duplicates("eid").set_index("eid").volume
    cuts = gv.quantile([1 / 3, 2 / 3]).to_numpy()
    x["vol"] = pd.cut(x.eid.map(gv), [-1, cuts[0], cuts[1], np.inf], labels=[f"low <{cuts[0]:,.0f}", "mid", f"high >{cuts[1]:,.0f}"])
    rng = np.random.default_rng(0)

    def tab(by):
        rows = []
        for k, z in x.groupby(by, observed=True):
            d = ((z.mid - z.y) ** 2 - (z.model - z.y) ** 2).groupby(z.eid).sum()
            n = z.groupby("eid").size()
            bs = [d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))).pipe(lambda s: s.sum() / n[s.index].sum())
                  for _ in range(300)]
            rows.append({by: k, "snaps": len(z), "games": z.eid.nunique(), "Kalshi": ((z.mid - z.y) ** 2).mean(),
                         "model": ((z.model - z.y) ** 2).mean(), "ESPN": ((z.espn_wp - z.y) ** 2).mean(),
                         "K-M": d.sum() / n.sum(), "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975),
                         "med_spread": z.spr.median()})
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    tab("mins")
    tab("vol")
    y = x.y.to_numpy()
    lg = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
    X = sm.add_constant(np.column_stack([lg(x.mid), lg(x.model)]))
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.eid)[0]})
    print("outcome ~ logit(Kalshi mid) + logit(model): coef", np.round(r.params, 3), "CI", np.round(r.conf_int(), 3).tolist())
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"espn": cmd_espn, "fit": cmd_fit, "kalshi": cmd_kalshi, "build": cmd_build, "brier": cmd_brier}.get(cmd, lambda: print(__doc__) or 1)())

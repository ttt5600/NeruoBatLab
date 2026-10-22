"""NBA LATE-GAME CALIBRATION: P(win | margin, time left, possession, pregame strength) vs Kalshi KXNBAGAME. DEV only.

User idea (2026-10-21): "Pacers down ~3 priced at 23% -- I bet it and they won; Pistons came back from 20 down. The
market may not have the context in close games." This is NOT the lag hypothesis (H1-H3 in nba_ingame.py, closed);
it asks whether Kalshi's LEVEL is miscalibrated in late, close states.

    python scripts/nba_late_wp.py states        # parse cached ESPN summaries -> data/nba_late_states.parquet
    python scripts/nba_late_wp.py fit           # fit + out-of-sample validation of the WP model, descriptive tables
    python scripts/nba_late_wp.py kalshi        # pull Kalshi candles for 2024-25 playoff games (2025-26 reused)
    python scripts/nba_late_wp.py dev [--ledger]

DATA. ESPN summary API per game (cached): plays[] with homeScore/awayScore/period/clock/team/type/wallclock and
winprobability[] (ESPN's model; its FIRST value = ESPN's pregame home win prob p0, used as the strength input).
Seasons 2021-22 .. 2025-26 (cached). Kalshi: KXNBAGAME 1-min candles (data/nba_ingame.parquet for 2025-26, plus a
new pull for the 2025-04..06 playoffs, the first period Kalshi lists NBA games).

MODEL (Brownian-motion form, probit). From the home side:
    P(home wins) = Phi(b0 + b1*m/sqrt(tau) + b2*z0*sqrt(tau) + b3*poss/sqrt(tau) + b4*m)
    m = home score - away score; tau = (seconds left in the current period + 4)/2880 for Q4/OT;
    z0 = Phi^-1(ESPN pregame p0); poss = +1/-1/0 home/away/unknown possession from a play-type heuristic
    (rebound -> rebounder's team; turnover, made field goal, made last free throw -> the other team).
Fit on period >= 4 plays of games dated < 2025-04-15 (2021-22 .. 2024-25 regular season), so every Kalshi game is
out of sample. Validation: fit 2021-22 + 2022-23, score 2023-24 + 2024-25 regular season (log loss / Brier vs ESPN's
own WP on the same plays, calibration by bins).

PRE-REGISTERED 2026-10-21, before any Kalshi price was joined to a late-game state:
  snapshot : every 1-min Kalshi candle (per side) whose end c falls in period >= 4; state = the last ESPN play with
             wallclock <= c - 60 s (60 s data latency; clock = that play's clock, i.e. slightly stale).
             Two-sided candle, ask - bid <= 0.05. Price = candle close ask; fee 0.07*p*(1-p) per contract.
  T1 trail_buy : Q4 with <= 360 s left OR any OT, trailing by 1..8 points; buy the TRAILING team's YES when
                 model P(trailer wins) - ask - fee(ask) >= 0.04. First trigger per game, hold to settlement.
  T2 any_side  : same window and |margin| <= 8 (ties included); buy EITHER side's YES when model - ask - fee >= 0.04.
                 First trigger per game, hold to settlement.
  split    : DEV = Kalshi games dated 2025-04-15 .. 2026-01-14 (2024-25 playoffs + 2025-26 regular season before
             01-15, same boundary as nba_ingame.py). SEALED = 2026-01-15 onward incl. 2026-27 (user's yes; not touched).
  test     : one-sided bootstrap over GAMES (3000 resamples) of summed P&L; 2 trials -> bar p < 0.025.
  calibration (descriptive, not a trial): trailing team's Kalshi mid in late states vs realised win rate, bins
             0-5-10-20-30-40-50c, CIs by game-cluster bootstrap; plus logistic regression of the outcome on
             logit(Kalshi mid) and logit(model) -- if the model's coefficient is ~0 the market already has the context.
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

STATES = ROOT / "data" / "nba_late_states.parquet"
CANDLES_OLD = ROOT / "data" / "nba_ingame.parquet"
CANDLES_NEW = ROOT / "data" / "nba_late_candles_2025po.parquet"
COEF = ROOT / "data" / "nba_late_wp_coef.json"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_late_wp"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ALIAS = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2025-04-15")
DEV_START, SPLIT = pd.Timestamp("2025-04-15"), pd.Timestamp("2026-01-15")
TRIALS = ["T1_trail_buy", "T2_any_side"]
FEE = 0.07
EDGE = 0.04


def fee(p):
    return FEE * p * (1 - p)


def secs(c):
    try:
        m, s = c.split(":") if ":" in c else ("0", c)
        return int(m) * 60 + float(s)
    except Exception:
        return np.nan


def parse_game(eid: str) -> tuple[dict, list[dict]]:
    j = fetch(f"{ESPN}/summary?event={eid}")
    comp = j["header"]["competitions"][0]
    side = {c["id"]: c["homeAway"] for c in comp["competitors"]}
    abbr = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
    wp = j.get("winprobability") or []
    plays = j.get("plays") or []
    if not plays or not wp:
        return {}, []
    last = plays[-1]
    hs, as_ = last.get("homeScore"), last.get("awayScore")
    meta = {"eid": eid, "home": ALIAS.get(abbr["home"], abbr["home"]), "away": ALIAS.get(abbr["away"], abbr["away"]),
            "p0": wp[0]["homeWinPercentage"], "home_won": int(hs > as_), "final_margin": hs - as_}
    wpd = {w["playId"]: w["homeWinPercentage"] for w in wp}
    rows, poss, per = [], 0, 0
    for p in plays:
        pn = p.get("period", {}).get("number", 0)
        if pn != per:
            per, poss = pn, 0
        typ = p.get("type", {}).get("text", "")
        tid = (p.get("team") or {}).get("id")
        s = side.get(tid)
        sgn = 1 if s == "home" else -1 if s == "away" else 0
        made = bool(p.get("scoringPlay"))
        if sgn:
            if "Rebound" in typ:
                poss = sgn
            elif "Turnover" in typ:
                poss = -sgn
            elif "Free Throw" in typ:
                parts = typ.replace("Free Throw - ", "").split(" of ")
                if made and len(parts) == 2 and parts[0] == parts[1]:
                    poss = -sgn
            elif made and p.get("shootingPlay"):
                poss = -sgn
        if pn >= 4:
            rows.append({"eid": eid, "seq": int(p.get("sequenceNumber", 0)), "period": pn, "clock_s": secs(p.get("clock", {}).get("displayValue", "")),
                         "hs": p.get("homeScore"), "as": p.get("awayScore"), "poss": poss, "wall": p.get("wallclock"),
                         "espn_wp": wpd.get(p["id"], np.nan)})
    return meta, rows


def cmd_states() -> int:
    g = pd.read_parquet(ROOT / "data" / "nba_games.parquet")
    g = g[g.season >= 2021]
    metas, rows = [], []
    for i, (_, r) in enumerate(g.iterrows()):
        try:
            m, rr = parse_game(r.game_id)
        except Exception as ex:
            print("skip", r.game_id, ex)
            continue
        if not m:
            continue
        metas.append({**m, "date": r.date, "season": r.season, "gtype": r.game_type})
        rows += rr
        if i % 1000 == 0:
            print(i, len(rows), flush=True)
    meta = pd.DataFrame(metas)
    s = pd.DataFrame(rows).merge(meta, on="eid")
    s["wall"] = pd.to_datetime(s.wall, utc=True, format="ISO8601", errors="coerce")
    s.to_parquet(STATES, index=False)
    print(f"-> {STATES}: {meta.eid.nunique()} games, {len(s)} period>=4 plays")
    return 0


def feats(m, clock_s, z0, poss):
    tau = (np.asarray(clock_s, float) + 4) / 2880
    st = np.sqrt(tau)
    return np.column_stack([np.ones_like(tau), m / st, z0 * st, poss / st, m])


def prep(s):
    s = s.dropna(subset=["clock_s", "hs", "as"]).copy()
    s["m"] = s.hs - s["as"]
    s["z0"] = norm.ppf(np.clip(s.p0, 0.01, 0.99))
    return s


def fit_probit(s):
    import statsmodels.api as sm
    X = feats(s.m.to_numpy(), s.clock_s.to_numpy(), s.z0.to_numpy(), s.poss.to_numpy())
    return sm.GLM(s.home_won.to_numpy(), X, family=sm.families.Binomial(sm.families.links.Probit())).fit().params


def predict(b, m, clock_s, z0, poss):
    return norm.cdf(feats(np.asarray(m, float), clock_s, np.asarray(z0, float), np.asarray(poss, float)) @ b)


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cmd_fit() -> int:
    s = prep(pd.read_parquet(STATES))
    s = s[s.date < TRAIN_END]
    tr, va = s[s.season <= 2022], s[s.season >= 2023]
    b = fit_probit(tr)
    print("validation fit coef (2021-22+2022-23):", np.round(b, 4))
    v = va.dropna(subset=["espn_wp"]).copy()
    v["pm"] = predict(b, v.m, v.clock_s, v.z0, v.poss)
    late = v[(v.period > 4) | (v.clock_s <= 360)]
    for nm, x in (("all Q4/OT", v), ("Q4<=6min/OT", late), ("...and |m|<=8", late[late.m.abs() <= 8])):
        y = x.home_won.to_numpy()
        print(f"OOS 2023-24+2024-25 {nm:15s} n={len(x):7d} games={x.eid.nunique()}  model LL {ll(y, x.pm):.4f} Brier {np.mean((x.pm - y) ** 2):.4f}"
              f" | ESPN LL {ll(y, x.espn_wp):.4f} Brier {np.mean((x.espn_wp - y) ** 2):.4f}")
    x = late[late.m.abs() <= 8]
    x = x.assign(bin=pd.cut(x.pm, [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1]))
    print("OOS calibration (late, |m|<=8): bin, n plays, games, mean model, realised")
    print(x.groupby("bin", observed=True).agg(n=("pm", "size"), games=("eid", "nunique"), model=("pm", "mean"), real=("home_won", "mean")).round(3).to_string())
    bfull = fit_probit(s)
    print("final coef (all games < 2025-04-15):", np.round(bfull, 4))
    COEF.write_text(json.dumps({"coef": list(map(float, bfull)), "val_coef": list(map(float, b)), "train_end": str(TRAIN_END.date())}))
    # descriptive table for the user: trailing team, Q4, by deficit and minutes left, from the trailer's view
    q = s[(s.period == 4) & (s.m != 0)].copy()
    q["trail_won"] = np.where(q.m < 0, q.home_won, 1 - q.home_won)
    q["deficit"] = q.m.abs()
    q["trail_p0"] = np.where(q.m < 0, q.p0, 1 - q.p0)
    q["min_left"] = pd.cut(q.clock_s, [0, 30, 60, 120, 180, 300, 420, 600, 720], include_lowest=True)
    q = q.sort_values(["eid", "seq"]).drop_duplicates(["eid", "deficit", "min_left"])  # one obs per game per cell
    for lo, hi, lab in ((0, 1.01, "all"), (0, 0.4, "trailer was pregame underdog (p0<.40)"), (0.6, 1.01, "trailer was pregame favourite (p0>.60)")):
        qq = q[(q.trail_p0 >= lo) & (q.trail_p0 < hi)]
        t = qq[qq.deficit.isin([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])].pivot_table(index="deficit", columns="min_left", values="trail_won", aggfunc="mean", observed=True)
        n = qq[qq.deficit.isin([3, 10, 20])].pivot_table(index="deficit", columns="min_left", values="trail_won", aggfunc="size", observed=True)
        print(f"\nEMPIRICAL P(trailing team wins), Q4, {lab}; one obs per game per cell; games < 2025-04-15")
        print(t.round(3).to_string())
        print("n games (deficit 3/10/20):\n" + n.to_string())
    return 0


def kalshi_markets():
    from kalshi_longshot import settled
    mk = settled("KXNBAGAME")
    mk["code"] = mk.ticker.str.split("-").str[-1]
    mk["kdate"] = mk.event_ticker.str.split("-").str[1].str[:7]
    return mk


def pull_candles(games: pd.DataFrame, s: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    from kalshi_longshot import API
    span = s.groupby("eid").wall.agg(["min", "max"])
    frames, unmatched = [], 0
    for _, r in games.iterrows():
        if r.eid not in span.index or pd.isna(span.loc[r.eid, "min"]):
            unmatched += 1
            continue
        st = span.loc[r.eid]
        # the Q4 states start late in the game; tip-off is ~2 h earlier -> start 3.5 h before the first Q4 play
        kd_candidates = {(st["min"] - pd.Timedelta(hours=h)).tz_convert("America/New_York").strftime("%y%b%d").upper() for h in (2, 3)}
        m = mk[mk.kdate.isin(kd_candidates) & mk.code.isin([r.home, r.away])]
        if m.event_ticker.nunique() != 1 or len(m) != 2:
            unmatched += 1
            continue
        s0 = int((st["min"] - EPOCH) // pd.Timedelta(seconds=1)) - 4 * 3600
        e0 = int((st["max"] - EPOCH) // pd.Timedelta(seconds=1)) + 900
        for _, k in m.iterrows():
            c = None
            for path in (f"{API}/historical/markets/{k.ticker}/candlesticks", f"{API}/series/KXNBAGAME/markets/{k.ticker}/candlesticks"):
                try:
                    j = fetch(path, {"start_ts": s0, "end_ts": e0, "period_interval": 1}, cache=True)
                except Exception:
                    continue
                if j.get("candlesticks"):
                    c = j["candlesticks"]
                    break
            if not c:
                continue
            f = lambda x, kk: pd.to_numeric((x.get(kk) or {}).get("close_dollars", (x.get(kk) or {}).get("close")), errors="coerce")
            frames.append(pd.DataFrame([{"end": x["end_period_ts"], "bid": f(x, "yes_bid"), "ask": f(x, "yes_ask")} for x in c])
                          .assign(eid=r.eid, side="home" if k.code == r.home else "away", result=k.result, date=r.date))
    print(f"matched {len({f.eid.iloc[0] for f in frames})} games, unmatched {unmatched}")
    return pd.concat(frames, ignore_index=True)


def cmd_kalshi() -> int:
    s = pd.read_parquet(STATES)
    meta = s.drop_duplicates("eid")[["eid", "home", "away", "date"]]
    po = meta[(meta.date >= DEV_START) & (meta.date < pd.Timestamp("2025-07-01"))]
    cd = pull_candles(po, s, kalshi_markets())
    cd.to_parquet(CANDLES_NEW, index=False)
    print(f"-> {CANDLES_NEW}: {cd.eid.nunique()} games")
    return 0


def snapshots(s: pd.DataFrame, cd: pd.DataFrame, b, lag: int = 60) -> pd.DataFrame:
    out = []
    s = s.dropna(subset=["wall"])
    for eid, gp in s.groupby("eid"):
        c = cd[cd.eid == eid]
        if c.side.nunique() != 2:
            continue
        gp = gp.sort_values(["wall", "seq"])
        wt = ((gp.wall - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()
        h = c[c.side == "home"].set_index("end")[["bid", "ask", "result"]]
        a = c[c.side == "away"].set_index("end")[["bid", "ask"]]
        j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner").reset_index()
        j = j[(j.end >= wt.min() + lag) & (j.end <= wt.max() + 120)]
        idx = np.searchsorted(wt, j.end.to_numpy() - lag, side="right") - 1
        ok = idx >= 0
        j, idx = j[ok], idx[ok]
        st = gp.iloc[idx]
        out.append(pd.DataFrame({"eid": eid, "end": j.end.to_numpy(), "period": st.period.to_numpy(), "clock_s": st.clock_s.to_numpy(),
                                 "m": st.m.to_numpy(), "poss": st.poss.to_numpy(), "z0": st.z0.to_numpy(), "p0": st.p0.to_numpy(),
                                 "espn_wp": st.espn_wp.to_numpy(), "home_won": (j.result == "yes").astype(int).to_numpy(),
                                 "bid_h": j.bid_h.to_numpy(), "ask_h": j.ask_h.to_numpy(), "bid_a": j.bid_a.to_numpy(), "ask_a": j.ask_a.to_numpy()}))
    x = pd.concat(out, ignore_index=True)
    x["pm"] = predict(b, x.m, x.clock_s, x.z0, x.poss)
    return x


def side_view(x: pd.DataFrame, which: str) -> pd.DataFrame:
    """Rows from one side's view: 'home' / 'away'. fair = model P(that side wins)."""
    if which == "home":
        return x.assign(side="home", fair=x.pm, bid=x.bid_h, ask=x.ask_h, won=x.home_won, marg=x.m, espn=x.espn_wp)
    return x.assign(side="away", fair=1 - x.pm, bid=x.bid_a, ask=x.ask_a, won=1 - x.home_won, marg=-x.m, espn=1 - x.espn_wp)


def late_rows(x):
    v = pd.concat([side_view(x, "home"), side_view(x, "away")], ignore_index=True)
    v = v[(v.period > 4) | ((v.period == 4) & (v.clock_s <= 360))]
    v = v[(v.bid > 0) & (v.ask < 1) & (v.ask - v.bid <= 0.05) & (v.marg.abs() <= 8)]
    return v


def trial(v, name):
    cost = v.ask + fee(v.ask)
    hit = (v.fair - cost >= EDGE)
    if name == "T1_trail_buy":
        hit &= (v.marg <= -1)
    t = v[hit].sort_values(["eid", "end"]).drop_duplicates("eid")
    cost = t.ask + fee(t.ask)
    return t.assign(stake=cost, pnl=np.where(t.won == 1, 1 - cost, -cost))


def boot(ev, n=3000):
    rng = np.random.default_rng(0)
    b = np.array([rng.choice(ev, len(ev)).sum() for _ in range(n)]) if len(ev) else np.zeros(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975)), float(np.mean(b <= 0))


def cmd_dev() -> int:
    b = np.array(json.loads(COEF.read_text())["coef"])
    s = prep(pd.read_parquet(STATES))
    old = pd.read_parquet(CANDLES_OLD)
    cd = pd.concat([old[["end", "bid", "ask", "eid", "side", "result", "date"]], pd.read_parquet(CANDLES_NEW)], ignore_index=True)
    dev_ids = cd[(cd.date >= DEV_START) & (cd.date < SPLIT)].eid.unique()
    assert s[s.eid.isin(dev_ids)].date.min() >= TRAIN_END, "train/test overlap"
    x = snapshots(s[s.eid.isin(dev_ids)], cd[cd.eid.isin(dev_ids)], b)
    v = late_rows(x)
    v["mid"] = (v.bid + v.ask) / 2
    print(f"DEV late states: {v.eid.nunique()} games, {len(v)} side-snapshots (Q4<=6min/OT, |m|<=8, spread<=5c)")
    print(f"sanity |Kalshi mid - ESPN WP| median {np.nanmedian((v.mid - v.espn).abs()):.3f}; |mid - model| median {np.median((v.mid - v.fair).abs()):.3f}")
    # calibration of Kalshi: trailing team's mid vs outcome, one snapshot per game per bin (first)
    tv = v[v.marg <= -1].copy()
    tv["bin"] = pd.cut(tv.mid, [0, .05, .1, .2, .3, .4, .5, .6], include_lowest=True)
    one = tv.sort_values(["eid", "end"]).drop_duplicates(["eid", "bin"])
    rng = np.random.default_rng(1)
    print("\nKALSHI CALIBRATION, trailing team (first snapshot per game per bin): bin, games, mean mid, mean model, realised [95% CI]")
    calib = []
    for bn, gg in one.groupby("bin", observed=True):
        w = gg.won.to_numpy()
        bs = [rng.choice(w, len(w)).mean() for _ in range(2000)]
        r = {"bin": str(bn), "games": len(gg), "mid": round(gg.mid.mean(), 3), "model": round(gg.fair.mean(), 3), "espn": round(gg.espn.mean(), 3),
             "realised": round(w.mean(), 3), "lo": round(float(np.quantile(bs, .025)), 3), "hi": round(float(np.quantile(bs, .975)), 3)}
        calib.append(r)
        print(r)
    # does the model add information beyond the price? logistic on logit(mid), logit(model); cluster bootstrap by game
    import statsmodels.api as sm
    lg = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))
    tv1 = v.sort_values(["eid", "end"]).groupby(["eid", "side"]).nth(slice(None, None, 3))  # thin to every 3rd minute
    X = np.column_stack([np.ones(len(tv1)), lg(tv1.mid), lg(tv1.fair)])
    coef = sm.GLM(tv1.won.to_numpy(), X, family=sm.families.Binomial()).fit().params
    ids = tv1.eid.unique()
    grp = {e: i for i, e in enumerate(ids)}
    gi = tv1.eid.map(grp).to_numpy()
    rows_by = [np.where(gi == k)[0] for k in range(len(ids))]
    bc = []
    for _ in range(300):
        pick = np.concatenate([rows_by[k] for k in rng.integers(0, len(ids), len(ids))])
        try:
            bc.append(sm.GLM(tv1.won.to_numpy()[pick], X[pick], family=sm.families.Binomial()).fit().params)
        except Exception:
            pass
    bc = np.array(bc)
    print(f"\noutcome ~ logit(mid) + logit(model): coef mid {coef[1]:.3f} [{np.quantile(bc[:, 1], .025):.3f}, {np.quantile(bc[:, 1], .975):.3f}], "
          f"model {coef[2]:.3f} [{np.quantile(bc[:, 2], .025):.3f}, {np.quantile(bc[:, 2], .975):.3f}]  (n={len(tv1)}, games={len(ids)})")
    y = v.won.to_numpy()
    print(f"Brier on all late side-snapshots: Kalshi mid {np.mean((v.mid - y) ** 2):.4f}  model {np.mean((v.fair - y) ** 2):.4f}  ESPN {np.nanmean((v.espn - y) ** 2):.4f}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    for name in TRIALS:
        t = trial(v, name)
        ev = t.groupby("eid").pnl.sum().to_numpy()
        lo, hi, p = boot(ev)
        r = {"bets": int(len(t)), "games": int(len(ev)), "staked": float(t.stake.sum()), "units": float(t.pnl.sum()),
             "roi": float(t.pnl.sum() / t.stake.sum()) if len(t) else 0.0, "ci_lo": lo, "ci_hi": hi, "p": p,
             "win_rate": float(t.won.mean()) if len(t) else 0.0, "mean_fair": float(t.fair.mean()) if len(t) else 0.0,
             "mean_ask": float(t.ask.mean()) if len(t) else 0.0}
        sha = hashlib.sha256(("nba_late_wp:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:13s} bets {r['bets']:4d} units {r['units']:+7.2f} ROI {r['roi']:+6.2%} 95%CI [{lo:+.1f}, {hi:+.1f}] p {p:.4f} "
              f"| win {r['win_rate']:.3f} vs model {r['mean_fair']:.3f} vs ask {r['mean_ask']:.3f}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nba_late_wp_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/nba_late_wp.py ({name})",
                     "source": "user idea 10-21: Kalshi misprices late close NBA states (Pacers -3 at 23c)", "status": "ok",
                     "trial_number": len(shas), **r})
    print(f"{DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    (ROOT / "data" / "nba_late_calib.json").write_text(json.dumps(calib))
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"states": cmd_states, "fit": cmd_fit, "kalshi": cmd_kalshi, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())

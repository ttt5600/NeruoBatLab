"""KFG1 forward PAPER logger (prereg sports/notes/kalshi_fg_markets.md section 5). READ-ONLY: public Kalshi GETs, no keys,
no orders.

  kfg1_paper.py tick   [--out DIR] [--lo 50 --hi 70]   NFL games with kickoff in [now+lo, now+hi] min: price KXNFLFG with
                                                       the frozen M1 (scripts/kfg1_m1.json); one row per team-game
                                                       (bet or skip) to DIR/picks.jsonl; once per game.
  kfg1_paper.py settle [--out DIR]                     fill Kalshi results + CLV (side mid at last 1-min candle <= kickoff)
                                                       -> DIR/settled.jsonl (append-only)
  kfg1_paper.py report [--out DIR]                     n, ROI at ask, game-clustered CI, CLV; judge at 100, kill rule

Exit codes: 0 ok, 1 fetch/parse error (fail loudly), 2 stale schedule.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kx_throttle import kget  # noqa: E402

HERE = Path(__file__).resolve().parent
COEF = HERE / "kfg1_m1.json"
SCHED = Path.home() / ".claude/quant-kalshi/data/nfl_schedule.csv"
DEFAULT_OUT = Path.home() / ".claude/quant-sports/data/kfg1"
SERIES = "KXNFLFG"
THR = 0.04
FIX = {"JAC": "JAX", "LAR": "LA"}          # Kalshi team code -> nflverse
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def now_utc():
    return pd.Timestamp.now(tz="UTC")


def epoch_s(ts):
    return int((ts - EPOCH) // pd.Timedelta(seconds=1))


def fee(p):
    return math.ceil(round(100 * 0.07 * 10 * p * (1 - p), 9)) / 1000


def get(path, params=None):
    j = kget(path, params, raise_4xx=True)
    if j is None:
        raise RuntimeError(f"Kalshi GET failed: {path} {params}")
    return j


def model_ladder(implied, indoor):
    c = json.loads(COEF.read_text())["coef"]
    x = np.array([implied, implied ** 2 / 10, float(indoor)])
    p = [1 / (1 + math.exp(-(c[str(k)]["intercept"] + float(np.dot(c[str(k)]["b"], x))))) for k in range(1, 5)]
    return list(np.minimum.accumulate(p))


def schedule():
    age_h = (time.time() - SCHED.stat().st_mtime) / 3600
    if age_h > 36:
        print(f"STALE schedule {SCHED} age {age_h:.1f} h", file=sys.stderr)
        sys.exit(2)
    s = pd.read_csv(SCHED)
    s["kick"] = pd.to_datetime(s.gameday + " " + s.gametime.fillna("00:00")).dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    hist = s[s.season.between(2023, 2026) & s.roof.notna()]
    mode = hist.groupby("stadium_id").roof.agg(lambda r: r.value_counts().index[0]).to_dict()
    s["roof_used"] = s.roof.where(s.roof.notna(), s.stadium_id.map(mode))
    s["roof_src"] = np.where(s.roof.notna(), "schedule", "stadium_mode")
    return s


def load_jsonl(p):
    return [json.loads(l) for l in p.open()] if p.exists() else []


def tick(out):
    lo, hi = float(arg("--lo", 50)), float(arg("--hi", 70))
    s = schedule()
    t = now_utc()
    g = s[(s.kick >= t + pd.Timedelta(minutes=lo)) & (s.kick <= t + pd.Timedelta(minutes=hi))]
    picks = out / "picks.jsonl"
    done = {r["game_id"] for r in load_jsonl(picks)}
    g = g[~g.game_id.isin(done)]
    if g.empty:
        return 0          # idle: silent (qs_job logs only non-empty output)
    evs = get("/events", {"series_ticker": SERIES, "status": "open", "limit": 200}).get("events") or []
    rows = []
    for r in g.itertuples():
        kdate = r.kick.tz_convert("America/New_York").strftime("%y%b%d").upper()
        cand = []
        for e in evs:
            et = e["event_ticker"]
            tail = et.split("-")[1]
            teams = tail[7:]
            if tail[:7] != kdate:
                continue
            for a in range(2, 5):
                ka, kh = teams[:a], teams[a:]
                if FIX.get(ka, ka) == r.away_team and FIX.get(kh, kh) == r.home_team:
                    cand.append(et)
        if len(cand) != 1:
            raise RuntimeError(f"{r.game_id}: expected 1 KXNFLFG event for {kdate}, found {cand}")
        et = cand[0]
        mk = get("/markets", {"event_ticker": et, "limit": 100}).get("markets") or []
        if len(mk) != 8:
            raise RuntimeError(f"{et}: expected 8 markets, got {len(mk)}")
        if pd.isna(r.total_line) or pd.isna(r.spread_line):
            raise RuntimeError(f"{r.game_id}: no total/spread in schedule")
        indoor = int(r.roof_used in ("dome", "closed"))
        for team, implied in ((r.home_team, (r.total_line + r.spread_line) / 2), (r.away_team, (r.total_line - r.spread_line) / 2)):
            P = model_ladder(implied, indoor)
            opts, quotes = [], {}
            for m in mk:
                code = m["ticker"].split("-")[2]
                kt, k = code[:-1], int(code[-1])
                if FIX.get(kt, kt) != team:
                    continue
                ask, bid = float(m["yes_ask_dollars"]), float(m["yes_bid_dollars"])
                quotes[k] = {"ticker": m["ticker"], "yes_ask": ask, "yes_bid": bid,
                             "ask_size": m.get("yes_ask_size_fp"), "bid_size": m.get("yes_bid_size_fp"), "status": m.get("status")}
                p = P[k - 1]
                for side, cost, ps in (("yes", ask, p), ("no", round(1 - bid, 4), 1 - p)):
                    if bid > 0 and 0.03 <= cost <= 0.97:
                        opts.append(dict(k=k, side=side, ticker=m["ticker"], cost=cost, fee=fee(cost), p_side=ps,
                                         edge=round(ps - cost - fee(cost), 4)))
            if len(quotes) != 4:
                raise RuntimeError(f"{et} {team}: expected 4 strikes, got {sorted(quotes)}")
            best = max(opts, key=lambda o: o["edge"]) if opts else None
            bet = best is not None and best["edge"] >= THR
            rows.append(dict(ts=t.isoformat(), game_id=r.game_id, event_ticker=et, team=team, kick=r.kick.isoformat(),
                             mins_to_kick=round((r.kick - t) / pd.Timedelta(minutes=1), 1), total_line=r.total_line,
                             spread_line=r.spread_line, implied=implied, roof=r.roof_used, roof_src=r.roof_src, indoor=indoor,
                             p=[round(float(x), 4) for x in P], quotes=quotes, best=best,
                             decision="bet" if bet else "skip",
                             reason="" if bet else ("no valid quote" if best is None else f"best edge {best['edge']:.3f} < {THR}")))
    out.mkdir(parents=True, exist_ok=True)
    with picks.open("a") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
            b = x["best"] or {}
            print(f"{x['game_id']} {x['team']} {x['decision']} {b.get('ticker','')} {b.get('side','')} cost {b.get('cost','')} "
                  f"edge {b.get('edge','')} p {x['p']} | {x['reason']}")
    return 0


def settle(out):
    picks = load_jsonl(out / "picks.jsonl")
    sp = out / "settled.jsonl"
    if not picks:
        return 0
    done = {(r["game_id"], r["team"]) for r in load_jsonl(sp)}
    n = 0
    with sp.open("a") as f:
        for r in picks:
            if r["decision"] != "bet" or (r["game_id"], r["team"]) in done:
                continue
            if pd.Timestamp(r["kick"]) > now_utc() - pd.Timedelta(hours=3):
                continue      # game not over yet
            b = r["best"]
            m = get(f"/markets/{b['ticker']}")["market"]
            if m.get("result") not in ("yes", "no"):
                continue
            kick = pd.Timestamp(r["kick"])
            ks = epoch_s(kick)
            c = get(f"/series/{SERIES}/markets/{b['ticker']}/candlesticks",
                    {"start_ts": ks - 1800, "end_ts": ks, "period_interval": 1}).get("candlesticks") or []
            c = [x for x in c if int(x["end_period_ts"]) <= ks]
            clv = None
            if c:
                last = max(c, key=lambda x: int(x["end_period_ts"]))
                a = float(last["yes_ask"].get("close_dollars") or "nan"); bd = float(last["yes_bid"].get("close_dollars") or "nan")
                mid = (a + bd) / 2
                clv = round((mid if b["side"] == "yes" else 1 - mid) - b["cost"], 4) if not math.isnan(mid) else None
            win = int((m["result"] == "yes") == (b["side"] == "yes"))
            tc = b["cost"] + b["fee"]
            f.write(json.dumps(dict(ts=now_utc().isoformat(), game_id=r["game_id"], team=r["team"], ticker=b["ticker"], side=b["side"],
                                    result=m["result"], win=win, cost=b["cost"], fee=b["fee"], pnl=round(win - tc, 4), clv=clv)) + "\n")
            n += 1
    if n:
        print(f"settled {n} new")
    return 0


def report(out):
    picks = load_jsonl(out / "picks.jsonl")
    s = pd.DataFrame(load_jsonl(out / "settled.jsonl"))
    nb = sum(r["decision"] == "bet" for r in picks)
    print(f"team-games logged {len(picks)}, bets {nb}, settled {len(s)}")
    if s.empty:
        return 0
    s["tc"] = s.cost + s.fee
    roi = s.pnl.sum() / s.tc.sum()
    g = s.groupby("game_id").agg(p=("pnl", "sum"), c=("tc", "sum"))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), size=(10000, len(g)))
    r = g.p.to_numpy()[idx].sum(1) / g.c.to_numpy()[idx].sum(1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    print(f"ROI {roi:+.4f} game-CI [{lo:+.4f}, {hi:+.4f}] wins {int(s.win.sum())}/{len(s)} CLV mean {s.clv.dropna().mean():+.4f} (n {s.clv.notna().sum()})")
    if len(s) >= 60 and hi < 0:
        print("KILL: ROI CI upper < 0 after >= 60 bets")
    if len(s) >= 100:
        print("JUDGE (100 bets): " + ("PASS (CI lower > 0)" if lo > 0 else "NOT PASS"))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    out = Path(arg("--out", str(DEFAULT_OUT)))
    fn = {"tick": tick, "settle": settle, "report": report}.get(cmd)
    if fn is None:
        print(__doc__); sys.exit(1)
    try:
        rc = fn(out)
    except Exception as e:  # fail loudly
        print(f"ERROR {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)

"""NBA IN-GAME: Kalshi live game-winner prices vs ESPN's per-play win probability (QUEUE M4). DEV only.

    python scripts/nba_ingame.py build
    python scripts/nba_ingame.py dev [--ledger]

Data (probed 2026-10-20): ESPN summary API per game has plays[] with wallclock (UTC, 100% filled on 2025-11-10
LAL@CHA) and winprobability[] (homeWinPercentage per playId) -- ESPN's own model. Kalshi KXNBAGAME 1-min candles.
Lesson carried from nfl_ingame.py: Kalshi repriced NFL big plays within ~1 min and beat the public model; epoch
conversion must be unit-safe ((t - epoch) // 1 s).

Pre-registered 2026-10-20, before any NBA in-game price was pulled:
  model    : after play i, fair home prob = ESPN homeWinPercentage at play i (ESPN's value is the state AFTER the
             play); away = 1 - home.
  latency  : act at play wallclock + 60 s: first 1-min candle ending >= t+60 s, two-sided, spread <= 5c; fee
             0.07*p*(1-p) each side.
  H1 gap_hold : first time in a game a side's model - (ask + fee) >= 0.05 -> buy at ask, hold. One per game-side.
  H2 lag_flip : plays with model change >= 0.05 in that side's favour, and model - ask >= 0.03 at t+60 s -> buy at ask,
                sell at the bid of the candle ending >= t + 5 min.
  H3 late_fav : 4th quarter (or OT), <= 120 s left, model side >= 0.95, ask <= 0.92 -> buy, hold. One per game.
  split    : DEV = 2025-26 regular-season games dated < 2026-01-15 (ET); SEALED = later 2025-26 games (incl. playoffs)
             and 2026-27 (user's yes).
  3 trials -> bar p < 0.0167, one-sided bootstrap over GAMES.
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

from sports.http import fetch, fetch_many
from kalshi_longshot import API, settled

OUT = ROOT / "data" / "nba_ingame.parquet"
PLAYS = ROOT / "data" / "nba_ingame_plays.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_ingame"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ALIAS = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHX": "PHX", "BKN": "BKN"}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
SPLIT = pd.Timestamp("2026-01-15")
TRIALS = ["H1_gap_hold", "H2_lag_flip", "H3_late_fav"]
FEE = 0.07


def fee(p):
    return FEE * p * (1 - p)


def espn_games() -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2025-10-21", "2026-06-25"):
        sb = fetch(f"{ESPN}/scoreboard", {"dates": d.strftime("%Y%m%d")}, cache=True)
        for e in sb.get("events", []):
            c = e["competitions"][0]
            if c.get("status", {}).get("type", {}).get("completed") is not True:
                continue
            tm = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
            rows.append({"eid": e["id"], "date": d, "home": ALIAS.get(tm["home"], tm["home"]), "away": ALIAS.get(tm["away"], tm["away"]),
                         "stype": e.get("season", {}).get("type")})
    return pd.DataFrame(rows)


def cmd_build() -> int:
    g = espn_games()
    print(f"ESPN completed games {len(g)}", flush=True)
    sums = fetch_many([f"{ESPN}/summary?event={e}" for e in g.eid])
    prow = []
    for (_, r), s in zip(g.iterrows(), sums):
        wp = {w["playId"]: w["homeWinPercentage"] for w in s.get("winprobability", [])}
        for p in s.get("plays", []):
            if p.get("wallclock") and p["id"] in wp:
                prow.append({"eid": r.eid, "pid": p["id"], "t": p["wallclock"], "period": p.get("period", {}).get("number"),
                             "clock": p.get("clock", {}).get("displayValue"), "home_wp": wp[p["id"]]})
    plays = pd.DataFrame(prow)
    plays["t"] = pd.to_datetime(plays.t, utc=True, format="ISO8601")
    def secs(c):
        try:
            m, s = c.split(":") if ":" in c else ("0", c)
            return int(m) * 60 + float(s)
        except Exception:
            return np.nan
    plays["clock_s"] = plays.clock.map(secs)
    plays.to_parquet(PLAYS, index=False)
    mk = settled("KXNBAGAME")
    mk["code"] = mk.ticker.str.split("-").str[-1]
    mk["kdate"] = mk.event_ticker.str.split("-").str[1].str[:7]
    frames, unmatched = [], 0
    span = plays.groupby("eid").t.agg(["min", "max"])
    for _, r in g.iterrows():
        if r.eid not in span.index:
            continue
        st = span.loc[r.eid]
        kd = st["min"].tz_convert("America/New_York").strftime("%y%b%d").upper()
        m = mk[(mk.kdate == kd) & mk.code.isin([r.home, r.away])]
        if m.event_ticker.nunique() != 1 or len(m) != 2:
            unmatched += 1
            continue
        s0, e0 = int((st["min"] - EPOCH) // pd.Timedelta(seconds=1)) - 900, int((st["max"] - EPOCH) // pd.Timedelta(seconds=1)) + 900
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
                          .assign(eid=r.eid, side="home" if k.code == r.home else "away", result=k.result, date=r.date, stype=r.stype))
    cd = pd.concat(frames, ignore_index=True)
    cd.to_parquet(OUT, index=False)
    print(f"-> {OUT}: {cd.eid.nunique()} games with candles; {unmatched} ESPN games unmatched to Kalshi")
    return 0


def snapshots(plays, cd) -> pd.DataFrame:
    out = []
    for eid, gp in plays.groupby("eid"):
        c = cd[cd.eid == eid]
        if c.side.nunique() != 2:
            continue
        gp = gp.sort_values(["t", "pid"])
        ts = ((gp.t - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()
        home = gp.home_wp.to_numpy()
        prev = np.r_[home[0], home[:-1]]
        for side in ("home", "away"):
            cs = c[c.side == side].sort_values("end")
            ends = cs.end.to_numpy()
            i1, i5 = np.searchsorted(ends, ts + 60), np.searchsorted(ends, ts + 300)
            ok = (i1 < len(cs)) & (i5 < len(cs))
            fair = home if side == "home" else 1 - home
            pv = prev if side == "home" else 1 - prev
            b, a = cs.bid.to_numpy(), cs.ask.to_numpy()
            out.append(pd.DataFrame({"eid": eid, "side": side, "pid": gp.pid.to_numpy()[ok], "period": gp.period.to_numpy()[ok],
                                     "clock_s": gp.clock_s.to_numpy()[ok], "fair": fair[ok], "dfair": (fair - pv)[ok],
                                     "bid1": b[i1[ok]], "ask1": a[i1[ok]], "bid5": b[i5[ok]],
                                     "won": cs.result.iloc[0] == "yes"}))
    s = pd.concat(out, ignore_index=True)
    return s[(s.bid1 > 0) & (s.ask1 < 1) & ((s.ask1 - s.bid1) <= 0.05)]


def trial(s, name):
    if name == "H1_gap_hold":
        x = s[s.fair - (s.ask1 + fee(s.ask1)) >= 0.05].sort_values(["eid", "side", "pid"]).drop_duplicates(["eid", "side"])
        cost = x.ask1 + fee(x.ask1)
        return x.assign(pnl=np.where(x.won, 1 - cost, -cost), stake=cost)
    if name == "H2_lag_flip":
        x = s[(s.dfair >= 0.05) & (s.fair - s.ask1 >= 0.03) & (s.bid5 > 0)]
        cost = x.ask1 + fee(x.ask1)
        return x.assign(pnl=x.bid5 - fee(x.bid5) - cost, stake=cost)
    x = s[(s.period >= 4) & (s.clock_s <= 120) & (s.fair >= 0.95) & (s.ask1 <= 0.92)].sort_values(["eid", "pid"]).drop_duplicates("eid")
    cost = x.ask1 + fee(x.ask1)
    return x.assign(pnl=np.where(x.won, 1 - cost, -cost), stake=cost)


def cmd_dev() -> int:
    cd = pd.read_parquet(OUT)
    plays = pd.read_parquet(PLAYS)
    dev = cd[(cd.date < SPLIT) & (cd.stype == 2)].eid.unique()
    s = snapshots(plays[plays.eid.isin(dev)], cd[cd.eid.isin(dev)])
    gap = ((s.bid1 + s.ask1) / 2 - s.fair).abs()
    print(f"DEV {s.eid.nunique()} games, {len(s)} snapshots; median |Kalshi mid - ESPN model| {gap.median():.3f} "
          f"(sanity: NFL fixed run was 0.028; a large value means misalignment)")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    for name in TRIALS:
        t = trial(s, name)
        ev = t.groupby("eid").pnl.sum().to_numpy()
        rng = np.random.default_rng(0)
        boots = np.array([rng.choice(ev, len(ev)).sum() for _ in range(3000)]) if len(ev) else np.zeros(1)
        r = {"bets": int(len(t)), "games": int(len(ev)), "staked": float(t.stake.sum()), "units": float(t.pnl.sum()),
             "roi": float(t.pnl.sum() / t.stake.sum()) if len(t) else 0.0, "ci_lo": float(np.quantile(boots, .025)),
             "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0))}
        sha = hashlib.sha256(("nba_ingame:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:12s} bets {r['bets']:5d} games {r['games']:4d} units {r['units']:+7.2f} ROI {r['roi']:+6.2%} "
              f"95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] p {r['p']:.4f}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nba_ingame_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/nba_ingame.py ({name})",
                     "source": "user idea: in-game odds lag (NBA)", "status": "ok", "trial_number": len(shas), **r})
    print(f"{DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())

"""NFL in-game OVERSHOOT FADE as a maker -- FORWARD paper test on games after the pre-registration. No orders placed.

    python scripts/nfl_ingame_fade.py interim     # descriptive look at forward games so far (not ledgered)
    python scripts/nfl_ingame_fade.py score       # ONE-TIME score once games through 2026-11-30 are in (refuses earlier)

Origin (post-hoc, so dev cannot test it): nfl_ingame.py event study on 2025 wk1-12 -- after big plays Kalshi moves
~87% of the model jump within a minute, then gives back ~1.8 pts over the next 7 min. Taker round trip (~1c spread +
2 x 0.07*p*(1-p) fees ~4.5c) eats that, so the only version worth testing is a resting (maker) entry.
User approved a paper test 2026-10-20.

Pre-registered 2026-10-20 (before any game in the test window was played):
  window   : NFL games with kickoff >= 2026-10-22 00:00 ET and <= 2026-11-30 23:59 ET (REG season).
  big play : |vegas_home_wp(next play) - vegas_home_wp(this play)| >= 0.08, t = time_of_day (nflverse pbp).
  side     : the team whose model prob FELL (the overshoot reverts toward it).
  entry    : at t+60 s, rest a YES bid on that side at the bid of the first 1-min candle ending >= t+60 s (spread <= 5c,
             0.03 <= bid <= 0.97). FILL iff some candle ending in (t+60 s, t+240 s] has yes_ask LOW <= bid - 0.01
             (trade-through: someone sold below our price). Maker fee 0.0175*p*(1-p).
  exit     : sell at the bid of the first candle ending >= t+480 s; taker fee 0.07*p*(1-p). Unfilled orders = no bet.
  score    : P&L per filled contract; one-sided bootstrap over GAMES; PASS iff p < 0.05 and units > 0. Scored once.
Kalshi maker fee coefficient (0.0175) is from memory of the fee schedule -- verify before any real use.
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
import requests

from sports.http import fetch
from kalshi_longshot import API, settled
from nfl_ingame import ALIAS, PBP

START = pd.Timestamp("2026-10-22", tz="America/New_York")
END = pd.Timestamp("2026-11-30 23:59", tz="America/New_York")
LOCK = ROOT / "lab" / "lockbox.jsonl"
DATASET = "nfl_ingame_fade"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def fresh_pbp() -> pd.DataFrame:
    p = PBP / "play_by_play_2026.parquet"
    r = requests.get("https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet", timeout=300)
    r.raise_for_status()
    p.write_bytes(r.content)  # nflverse rebuilds nightly; always re-download for this test
    d = pd.read_parquet(p, columns=["game_id", "play_id", "time_of_day", "home_team", "away_team", "vegas_home_wp", "season_type"])
    d = d[d.time_of_day.notna() & d.vegas_home_wp.notna() & (d.season_type == "REG")].copy()
    d["t"] = pd.to_datetime(d.time_of_day, utc=True, format="ISO8601")
    return d.sort_values(["game_id", "t", "play_id"])


def candles(tk: str, s: int, e: int) -> pd.DataFrame:
    for path in (f"{API}/historical/markets/{tk}/candlesticks", f"{API}/series/KXNFLGAME/markets/{tk}/candlesticks"):
        try:
            j = fetch(path, {"start_ts": s, "end_ts": e, "period_interval": 1}, cache=False)
        except Exception:
            continue
        c = j.get("candlesticks", [])
        if c:
            f = lambda x, k, f2: pd.to_numeric((x.get(k) or {}).get(f2 + "_dollars", (x.get(k) or {}).get(f2)), errors="coerce")
            return pd.DataFrame([{"end": x["end_period_ts"], "bid": f(x, "yes_bid", "close"), "ask": f(x, "yes_ask", "close"),
                                  "ask_low": f(x, "yes_ask", "low")} for x in c])
    return pd.DataFrame()


def bets() -> pd.DataFrame:
    plays = fresh_pbp()
    g = plays.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"), start=("t", "min"), end=("t", "max"))
    g = g[(g.start >= START) & (g.start <= END)]
    g["date_et"] = g.start.dt.tz_convert("America/New_York").dt.strftime("%y%b%d").str.upper()
    mk = settled("KXNFLGAME")
    mk["code"] = mk.ticker.str.split("-").str[-1].map(lambda c: ALIAS.get(c, c))
    mk["date"] = mk.event_ticker.str.split("-").str[1].str[:7]
    out = []
    for gid, gr in g.iterrows():
        m = mk[(mk.date == gr.date_et) & mk.code.isin([gr.home, gr.away])]
        if m.event_ticker.nunique() != 1 or len(m) != 2:
            continue
        s, e = int(gr.start.timestamp()) - 600, int(gr.end.timestamp()) + 900
        cs = {("home" if r.code == gr.home else "away"): candles(r.ticker, s, e) for _, r in m.iterrows()}
        if any(c.empty for c in cs.values()):
            continue
        gp = plays[plays.game_id == gid].copy()
        gp["d"] = gp.vegas_home_wp.shift(-1) - gp.vegas_home_wp
        for _, p in gp[gp.d.abs() >= 0.08].iterrows():
            side = "away" if p.d > 0 else "home"  # the side whose prob fell
            c = cs[side].sort_values("end")
            t = int((p.t - EPOCH) // pd.Timedelta(seconds=1))
            i = np.searchsorted(c.end.to_numpy(), t + 60)
            j = np.searchsorted(c.end.to_numpy(), t + 480)
            if i >= len(c) or j >= len(c):
                continue
            q = c.iloc[i]
            if not (q.ask - q.bid <= 0.05 and 0.03 <= q.bid <= 0.97):
                continue
            win = c[(c.end > q.end) & (c.end <= t + 240)]
            filled = bool(len(win)) and bool((win.ask_low <= q.bid - 0.01 + 1e-9).any())
            if not filled:
                continue
            b, x = q.bid, c.iloc[j].bid
            pnl = (x - 0.07 * x * (1 - x)) - (b + 0.0175 * b * (1 - b))
            out.append({"game_id": gid, "play_id": p.play_id, "side": side, "entry": b, "exit": x, "pnl": pnl,
                        "stake": b + 0.0175 * b * (1 - b)})
    return pd.DataFrame(out)


def summary(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"bets": 0, "games": 0, "units": 0.0, "roi": 0.0, "p": 1.0}
    ev = t.groupby("game_id").pnl.sum().to_numpy()
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(ev, len(ev)).sum() for _ in range(3000)])
    return {"bets": int(len(t)), "games": int(len(ev)), "staked": float(t.stake.sum()), "units": float(t.pnl.sum()),
            "roi": float(t.pnl.sum() / t.stake.sum()), "ci_lo": float(np.quantile(boots, .025)),
            "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0))}


def cmd_interim() -> int:
    t = bets()
    r = summary(t)
    print(f"INTERIM (descriptive, not a score): forward games so far -> {r}")
    return 0


def cmd_score() -> int:
    if pd.Timestamp.now(tz="America/New_York") < END + pd.Timedelta(days=2):
        print("window not complete (games through 2026-11-30 + 2 days for pbp); use `interim`")
        return 1
    prior = [json.loads(l) for l in LOCK.read_text().splitlines() if l.strip()]
    sha = hashlib.sha256(b"nfl_ingame_fade:F1").hexdigest()[:12]
    if any(x["sha"] == sha for x in prior):
        print("already scored; refusing")
        return 1
    r = summary(bets())
    verdict = "PASS" if (r["p"] < 0.05 and r["units"] > 0) else "FAIL"
    with LOCK.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "nfl_ingame_fade_F1",
                            "sha": sha, "dataset": DATASET, "promotion_number": 1, **r, "verdict": verdict}) + "\n")
    print(f"FORWARD F1: {r} -> {verdict}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"interim": cmd_interim, "score": cmd_score}.get(cmd, lambda: print(__doc__) or 1)())

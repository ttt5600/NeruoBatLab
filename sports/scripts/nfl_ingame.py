"""NFL IN-GAME: Kalshi live game-winner prices vs a play-by-play win-probability model. DEV only.

    python scripts/nfl_ingame.py build          # 1-min Kalshi candles per game + nflverse pbp (time_of_day per play)
    python scripts/nfl_ingame.py dev [--ledger]

User 2026-10-20: test "game scenarios when odds can't update fast enough" during football games -- combine a model
built on past information with live prices, backtest on real in-game quotes.
Data: Kalshi KXNFLGAME markets trade all game (DAL@PHI 2025-09-04: ~$13M per side); 1-minute candles with yes bid/ask
from /historical/markets/{t}/candlesticks?period_interval=1. nflverse play_by_play_<season>.parquet has time_of_day
(UTC wall clock of each play, 97% filled) and nflfastR's vegas_home_wp (win prob incl. the pregame spread; model fitted
on older seasons -> out of sample for 2025-26).

Pre-registered 2026-10-20, before any in-game price was pulled:
  model    : after play i, fair home win prob = vegas_home_wp of the NEXT play in the game (state after play i);
             after the last play, the final result. Away fair = 1 - home fair.
  latency  : we act 60 s after the play's time_of_day: the first 1-min candle whose end >= t + 60 s; its close
             bid/ask is the quote we trade (two-sided, spread <= 5c). Fee 0.07*p*(1-p) per contract, each side.
  H1 gap_hold  : first time in a game a side's model prob - (ask + fee) >= 0.05 -> buy YES on that side at the ask, hold
                 to settlement. At most one bet per game per side.
  H2 lag_flip  : plays with |model change| >= 0.08 (scores, turnovers, big gains): if, at t+60 s, the side the play
                 favoured has model - ask >= 0.03 -> buy at the ask, SELL at the bid of the candle ending >= t + 6 min
                 (pure catch-up trade; P&L = exit bid - entry ask - both fees). Every such play is a bet.
  H3 late_fav  : 4th quarter, <= 300 s left, model side prob >= 0.95, ask <= 0.92 -> buy, hold. One per game.
  split    : DEV = 2025 REG weeks 1-12; SEALED = 2025 weeks 13+ and POST, and all 2026 games (user's yes)
  3 trials -> bar p < 0.0167, one-sided bootstrap over GAMES.
Caveats written down now: (1) 60 s is slow vs pro feeds -- a lag edge at this latency is what a home setup could
capture; (2) Kalshi team codes vs nflverse abbreviations differ (LA/LAR, JAC/JAX, WAS/WSH) -> alias table, unmatched
games dropped and counted; (3) nflfastR WP ignores injuries/weather -> H1 may just measure model error.
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

PBP = ROOT / "data" / "pbp"
OUT = ROOT / "data" / "nfl_ingame.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nfl_ingame"
FEE = 0.07
ALIAS = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LV": "LV", "LVR": "LV", "ARZ": "ARI", "HST": "HOU", "BLT": "BAL", "CLV": "CLE"}
TRIALS = ["H1_gap_hold", "H2_lag_flip", "H3_late_fav"]


def pbp(season: int) -> pd.DataFrame:
    p = PBP / f"play_by_play_{season}.parquet"
    if not p.exists():
        import requests
        r = requests.get(f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet", timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    d = pd.read_parquet(p, columns=["game_id", "play_id", "time_of_day", "qtr", "game_seconds_remaining", "home_team",
                                    "away_team", "vegas_home_wp", "season_type", "week", "home_score", "away_score",
                                    "result"])
    d["season"] = season
    return d


def candles(tk: str, s: int, e: int) -> pd.DataFrame:
    rows = []
    for path in (f"{API}/historical/markets/{tk}/candlesticks", f"{API}/series/KXNFLGAME/markets/{tk}/candlesticks"):
        j = fetch(path, {"start_ts": s, "end_ts": e, "period_interval": 1}, cache=True)
        c = j.get("candlesticks", [])
        if c:
            for x in c:
                g = lambda k: pd.to_numeric((x.get(k) or {}).get("close_dollars", (x.get(k) or {}).get("close")), errors="coerce")
                rows.append({"end": x["end_period_ts"], "bid": g("yes_bid"), "ask": g("yes_ask")})
            break
    return pd.DataFrame(rows)


def cmd_build() -> int:
    plays = pd.concat([pbp(2025), pbp(2026)], ignore_index=True)
    plays = plays[plays.time_of_day.notna() & plays.vegas_home_wp.notna()].copy()
    plays["t"] = pd.to_datetime(plays.time_of_day, utc=True, format="ISO8601")
    plays = plays.sort_values(["game_id", "t", "play_id"])
    games = plays.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"), start=("t", "min"),
                                         end=("t", "max"), season=("season", "first"), week=("week", "first"),
                                         stype=("season_type", "first"), result=("result", "last"))
    games["date_et"] = games.start.dt.tz_convert("America/New_York").dt.strftime("%y%b%d").str.upper()
    mk = settled("KXNFLGAME")
    mk["code"] = mk.ticker.str.split("-").str[-1].map(lambda c: ALIAS.get(c, c))
    mk["date"] = mk.event_ticker.str.split("-").str[1].str[:7]
    frames, unmatched = [], 0
    for gid, g in games.iterrows():
        m = mk[(mk.date == g.date_et) & mk.code.isin([g.home, g.away])]
        if m.event_ticker.nunique() != 1 or len(m) != 2:
            unmatched += 1
            continue
        s, e = int(g.start.timestamp()) - 1800, int(g.end.timestamp()) + 1800
        for _, r in m.iterrows():
            c = candles(r.ticker, s, e)
            if c.empty:
                continue
            frames.append(c.assign(game_id=gid, side="home" if r.code == g.home else "away", ticker=r.ticker,
                                   result=r.result))
        print(f"{gid}: ok", flush=True) if len(frames) % 40 == 0 else None
    cd = pd.concat(frames, ignore_index=True)
    cd.to_parquet(OUT, index=False)
    plays.to_parquet(ROOT / "data" / "nfl_ingame_plays.parquet", index=False)
    games.to_parquet(ROOT / "data" / "nfl_ingame_games.parquet")
    print(f"-> {OUT}: {cd.game_id.nunique()} games with candles, {unmatched} of {len(games)} games unmatched to Kalshi")
    return 0


def snapshots(plays: pd.DataFrame, cd: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per (play, side): model prob after the play and the Kalshi quote at t+60 s and t+6 min."""
    out = []
    for gid, gp in plays.groupby("game_id"):
        c = cd[cd.game_id == gid]
        if c.side.nunique() != 2:
            continue
        gp = gp.sort_values(["t", "play_id"]).copy()
        res = games.loc[gid, "result"]
        final_home = 1.0 if res > 0 else (0.0 if res < 0 else 0.5)
        gp["fair_home"] = gp.vegas_home_wp.shift(-1).fillna(final_home)
        gp["prev_home"] = gp.vegas_home_wp
        for side in ("home", "away"):
            cs = c[c.side == side].sort_values("end")
            ends = cs.end.to_numpy()
            ts = (gp.t.astype("int64") // 10 ** 9).to_numpy()
            i1 = np.searchsorted(ends, ts + 60)
            i6 = np.searchsorted(ends, ts + 360)
            ok = (i1 < len(cs)) & (i6 < len(cs))
            fair = gp.fair_home.to_numpy() if side == "home" else 1 - gp.fair_home.to_numpy()
            prev = gp.prev_home.to_numpy() if side == "home" else 1 - gp.prev_home.to_numpy()
            b, a = cs.bid.to_numpy(), cs.ask.to_numpy()
            out.append(pd.DataFrame({"game_id": gid, "side": side, "play_id": gp.play_id.to_numpy()[ok],
                                     "qtr": gp.qtr.to_numpy()[ok], "secs": gp.game_seconds_remaining.to_numpy()[ok],
                                     "fair": fair[ok], "dfair": (fair - prev)[ok],
                                     "bid1": b[i1[ok]], "ask1": a[i1[ok]], "bid6": b[i6[ok]], "ask6": a[i6[ok]],
                                     "won": (cs.result.iloc[0] == "yes"), "week": games.loc[gid, "week"],
                                     "season": games.loc[gid, "season"], "stype": games.loc[gid, "stype"]}))
    s = pd.concat(out, ignore_index=True)
    return s[(s.bid1 > 0) & (s.ask1 < 1) & ((s.ask1 - s.bid1) <= 0.05)]


def fee(p):
    return FEE * p * (1 - p)


def trial(s: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "H1_gap_hold":
        x = s[s.fair - (s.ask1 + fee(s.ask1)) >= 0.05].sort_values(["game_id", "side", "play_id"]).drop_duplicates(["game_id", "side"])
        cost = x.ask1 + fee(x.ask1)
        return x.assign(pnl=np.where(x.won, 1 - cost, -cost), stake=cost)
    if name == "H2_lag_flip":
        x = s[(s.dfair >= 0.08) & (s.fair - s.ask1 >= 0.03) & s.bid6.notna() & (s.bid6 > 0)]
        cost = x.ask1 + fee(x.ask1)
        return x.assign(pnl=x.bid6 - fee(x.bid6) - cost, stake=cost)
    x = s[(s.qtr == 4) & (s.secs <= 300) & (s.fair >= 0.95) & (s.ask1 <= 0.92)].sort_values(["game_id", "play_id"]).drop_duplicates("game_id")
    cost = x.ask1 + fee(x.ask1)
    return x.assign(pnl=np.where(x.won, 1 - cost, -cost), stake=cost)


def cmd_dev() -> int:
    cd = pd.read_parquet(OUT)
    plays = pd.read_parquet(ROOT / "data" / "nfl_ingame_plays.parquet")
    games = pd.read_parquet(ROOT / "data" / "nfl_ingame_games.parquet")
    dev_games = games[(games.season == 2025) & (games.stype == "REG") & (games.week <= 12)].index
    s = snapshots(plays[plays.game_id.isin(dev_games)], cd[cd.game_id.isin(dev_games)], games)
    print(f"DEV {s.game_id.nunique()} games, {len(s)} (play, side) snapshots; median |Kalshi mid - model| "
          f"{((s.bid1 + s.ask1) / 2 - s.fair).abs().median():.3f}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    for name in TRIALS:
        t = trial(s, name)
        ev = t.groupby("game_id").pnl.sum().to_numpy()
        rng = np.random.default_rng(0)
        boots = np.array([rng.choice(ev, len(ev)).sum() for _ in range(3000)]) if len(ev) else np.zeros(1)
        r = {"bets": int(len(t)), "games": int(len(ev)), "staked": float(t.stake.sum()), "units": float(t.pnl.sum()),
             "roi": float(t.pnl.sum() / t.stake.sum()) if len(t) else 0.0, "ci_lo": float(np.quantile(boots, .025)),
             "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0))}
        sha = hashlib.sha256(("nfl_ingame:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:12s} bets {r['bets']:5d} games {r['games']:4d} staked {r['staked']:8.1f} units {r['units']:+7.2f} "
              f"ROI {r['roi']:+6.2%} 95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] p {r['p']:.4f}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nfl_ingame_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/nfl_ingame.py ({name})",
                     "source": "user idea: in-game odds lag", "status": "ok", "trial_number": len(shas), **r})
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

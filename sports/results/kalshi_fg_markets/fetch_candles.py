"""1-min candles for KXNFLFG markets over [T-180min, T], T = kickoff - 60 min. DEV weeks 1-7 ONLY (prereg). Read-only; restartable."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from kx_throttle import kget
OUT = Path(__file__).resolve().parent
CACHE = OUT / "candles_dev.jsonl"
FIX = {"JAC": "JAX", "LAR": "LA"}
def games():
    d = json.load(open(OUT / "census_raw.json"))["KXNFLFG"]
    hist = {m["ticker"] for m in d["hist_markets"]}
    df = pd.DataFrame(d["markets"] + d["hist_markets"]).drop_duplicates("ticker")
    df["src"] = df.ticker.isin(hist)
    df["kteam"] = df.ticker.str.split("-").str[2].str[:-1]
    df["team"] = df.kteam.replace(FIX)
    df["k"] = df.ticker.str[-1].astype(int)
    df["edate"] = pd.to_datetime(df.event_ticker.str.split("-").str[1].str[:7], format="%y%b%d")
    s = pd.read_csv(Path.home() / ".claude/quant-kalshi/data/nfl_schedule.csv")
    s = s[s.season == 2026].copy()
    s["kick"] = pd.to_datetime(s.gameday + " " + s.gametime).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    s["gd"] = pd.to_datetime(s.gameday)
    long = pd.concat([s.assign(team=s.home_team), s.assign(team=s.away_team)])[["game_id", "week", "gd", "kick", "team"]]
    m = df.merge(long, on="team")
    m = m[(m.gd - m.edate).abs() <= pd.Timedelta(days=1)]
    assert not m.duplicated("ticker").any()
    print("markets", len(df), "matched", len(m), "events", m.event_ticker.nunique(), "games", m.game_id.nunique())
    # one game per event
    assert (m.groupby("event_ticker").game_id.nunique() == 1).all()
    return m
if __name__ == "__main__":
    m = games()
    m.drop(columns=[c for c in m.columns if m[c].map(lambda v: isinstance(v, (list, dict))).any()]).to_parquet(OUT / "markets_matched.parquet")
    d = m[m.week <= 7]
    print("dev markets", len(d), "games", d.game_id.nunique())
    done = {json.loads(l)["ticker"] for l in CACHE.open()} if CACHE.exists() else set()
    with CACHE.open("a") as f:
        for i, r in enumerate(d.itertuples()):
            if r.ticker in done: continue
            T = int((r.kick - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)) - 3600
            p = {"start_ts": T - 3 * 3600, "end_ts": T, "period_interval": 1}
            paths = ([f"/historical/markets/{r.ticker}/candlesticks"] if r.src else []) + [f"/series/KXNFLFG/markets/{r.ticker}/candlesticks"]
            c = []
            for path in paths:
                c = (kget(path, p) or {}).get("candlesticks") or []
                if c: break
            f.write(json.dumps({"ticker": r.ticker, "T": T, "c": c}) + "\n")
            if i % 100 == 0: f.flush(); print(i, r.kick, len(c), flush=True)
    print("done")

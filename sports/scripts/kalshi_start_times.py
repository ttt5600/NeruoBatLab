"""Real start times for Kalshi sports events (ESPN scoreboards) -> leak-free sealed decision prices.

    python scripts/kalshi_start_times.py match     # ESPN schedules, match to Kalshi events, validate on 2025
    python scripts/kalshi_start_times.py reprice   # re-price SEALED (2026+) markets at real start - 1 h

Why: in 2026 Kalshi set some NFL expected_expiration_time = start + 6 h (was + 3 h), so the
dev rule (expiration - 4 h) can land after kickoff. This builds prices only; it never scores
outcomes (scoring is `kalshi_longshot.py sealed --confirm`, the user's decision).
Validation: on 2025 events the ESPN start must equal expiration - 3 h (the convention checked
2026-10-17); the match rate and agreement are printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports import espn
import kalshi_longshot as K

STARTS = ROOT / "data" / "kalshi_event_starts.parquet"
SEALED_PX = ROOT / "data" / "kalshi_sports_prices_sealed.parquet"
LEAGUE = {"NFL": ("nfl", [2025, 2026]), "NCAAF": ("cfb", [2025, 2026]), "NBA": ("nba", [2026, 2027]),
          "WNBA": ("wnba", [2025, 2026]), "NHL": ("nhl", [2026, 2027]), "MLB": ("mlb", [2025, 2026])}


def league_of(series: str) -> str:
    s = series.removeprefix("KX")
    for k in sorted(LEAGUE, key=len, reverse=True):
        if s.startswith(k):
            return k
    raise KeyError(series)


def cmd_match() -> int:
    px = pd.read_parquet(K.OUT)
    ev = px.drop_duplicates("event_ticker")[["series", "event_ticker", "event_date", "decision_ts"]].copy()
    ev["lg"] = ev.series.map(league_of)
    ev["teams"] = ev.event_ticker.str.split("-").str[1].str[7:]
    out = []
    for lg, (key, seasons) in LEAGUE.items():
        g = espn.games(key, seasons)
        g = g.dropna(subset=["start_utc", "home", "away"])
        g["d"] = pd.to_datetime(g.start_utc, utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        by_day = {d: grp for d, grp in g.groupby("d")}
        e = ev[ev.lg == lg]
        for r in e.itertuples():
            day = by_day.get(r.event_date)
            st = pd.NaT
            # 2026 MLB-style tickers embed the ET start: 26AUG171340STLCING1 -> 13:40 ET (+ doubleheader tag)
            if r.teams[:4].isdigit():
                st = (pd.Timestamp(f"{r.event_date:%Y-%m-%d} {r.teams[:2]}:{r.teams[2:4]}")
                      .tz_localize("America/New_York").tz_convert("UTC"))
            elif day is not None:
                exact = day[(day.away.str.upper() + day.home.str.upper()) == r.teams]
                if len(exact) == 1:
                    st = exact.start_utc.iloc[0]
                else:  # codes differ between Kalshi and ESPN for some teams: unique prefix+suffix match
                    c = day[day.apply(lambda x: r.teams.startswith(str(x.away).upper()[:2]) and
                                      r.teams.endswith(str(x.home).upper()[-2:]), axis=1)]
                    if len(c) == 1:
                        st = c.start_utc.iloc[0]
            out.append({"event_ticker": r.event_ticker, "series": r.series, "lg": lg, "event_date": r.event_date,
                        "start_utc": st, "exp_minus4_ts": r.decision_ts})
        print(f"{lg}: espn games {len(g)}, kalshi events {len(e)}", flush=True)
    s = pd.DataFrame(out)
    s["start_utc"] = pd.to_datetime(s.start_utc, utc=True)
    s.to_parquet(STARTS, index=False)
    s["exp_minus3"] = pd.to_datetime(s.exp_minus4_ts + 3600, unit="s", utc=True)
    s["off_h"] = (s.exp_minus3 - s.start_utc).dt.total_seconds() / 3600
    for (lg, yr), grp in s.groupby(["lg", s.event_date.dt.year]):
        m = grp.start_utc.notna()
        ok = (grp.off_h.abs() <= 0.25)[m]
        print(f"  {lg} {yr}: matched {m.mean():.0%} of {len(grp)}; Kalshi exp-3h within 15 min of ESPN start: "
              f"{ok.mean():.0%}; exp-4h after start: {(grp.off_h[m] >= 1).sum()}")
    return 0


def cmd_reprice() -> int:
    s = pd.read_parquet(STARTS)
    px = pd.read_parquet(K.OUT)
    sealed = px[px.event_date >= K.SPLIT].merge(s[["event_ticker", "start_utc"]], on="event_ticker", how="left")
    n0 = sealed.event_ticker.nunique()
    sealed = sealed.dropna(subset=["start_utc"]).copy()
    dec = pd.to_datetime(sealed.start_utc, utc=True) - pd.Timedelta(hours=1)
    sealed["decision_ts"] = ((dec - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(int)
    frames = []
    for series, mk in sealed.groupby("series"):
        p = K.prices(series, mk.drop(columns=["yes_bid", "yes_ask"]))
        frames.append(p)
        print(f"{series}: {mk.event_ticker.nunique()} events repriced, priced {p.yes_ask.notna().sum()}/{len(p)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(SEALED_PX, index=False)
    print(f"sealed events {n0} -> with real start {out.event_ticker.nunique()} -> {SEALED_PX} (prices only; NOT scored)")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"match": cmd_match, "reprice": cmd_reprice}.get(cmd, lambda: print(__doc__) or 1)())

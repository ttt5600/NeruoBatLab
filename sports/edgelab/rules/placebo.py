"""Placebo strategies (GOAL.md rule 8): no information by construction; run through the full pipeline.

nfl_* : lab/datasets.py nfl_total dev rows (one row per game x over/under, CLOSING odds). decided_at = kickoff - 10 min,
        price = 1/decimal odds (real, with vig) or, with params.fair, the proportional no-vig price of that side (a
        zero-edge market by construction: isolates the statistics from the vig). close = price (bet at the close).
kt_*  : lab/datasets.py kalshi_temp dev rows (every tradeable bracket, both sides, 22:00-local ask + taker fee).
        Timestamps approximated conservatively: decided 02:00Z on the date (<= 22:00 local for every US city),
        event_start 04:00Z. No close is observed for these rows -> CLV null -> rule 5e fails; the report also gives the
        rate with 5e ignored.
Kinds: random_side (every game, coin-flip side), coin_flip (each row bet with prob `rate`), shuffled_signal (a real
signal -- under when wind >= 15 mph for NFL; E1's consensus-reject flag for Kalshi -- permuted across rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _nfl_rows(f: pd.DataFrame, fair: bool) -> pd.DataFrame:
    ko = pd.to_datetime(f["start_utc"], utc=True)
    price = f["_fair"].to_numpy(float) if fair else 1 / f["dec_odds"].to_numpy(float)
    b = pd.DataFrame({"bet_id": (f["opp_id"].astype(str) + "|" + f["selection"].astype(str)).to_numpy(),
                      "cluster": pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
                      "event_start": ko.to_numpy(), "decided_at": (ko - pd.Timedelta(minutes=10)).to_numpy(),
                      "inputs_as_of": (ko - pd.Timedelta(days=1)).to_numpy(),
                      "price_time": (ko - pd.Timedelta(minutes=10)).to_numpy(), "price": np.round(price, 6),
                      "price_source": "book_quote", "side": f["selection"].to_numpy(), "stake": 1.0})
    b["close_price"] = b["price"]
    return b


def nfl(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(params["seed"]))
    kind, fair = params["kind"], bool(params.get("fair", False))
    f = frame[frame["dec_odds"].notna() & frame["start_utc"].notna()].copy()
    imp = 1 / f["dec_odds"].to_numpy(float)
    f["_fair"] = imp / pd.Series(imp, index=f.index).groupby(f["opp_id"].to_numpy()).transform("sum").to_numpy()
    f = f[f.groupby("opp_id")["selection"].transform("size") == 2]   # two-way markets only (de-vig needs both)
    games = np.sort(f["opp_id"].unique())
    if kind == "random_side":
        pick = dict(zip(games, rng.choice(["over", "under"], len(games))))
        f = f[f["selection"].to_numpy() == f["opp_id"].map(pick).to_numpy()]
    elif kind == "coin_flip":
        rate = float(params.get("rate", 0.25))
        g = set(games[rng.random(len(games)) < rate])
        side = dict(zip(games, rng.choice(["over", "under"], len(games))))
        f = f[f["opp_id"].isin(g) & (f["selection"].to_numpy() == f["opp_id"].map(side).to_numpy())]
    elif kind == "shuffled_signal":
        u = f[f["selection"] == "under"].drop_duplicates("opp_id")
        sig = (u["wind_mph"].fillna(0).to_numpy() >= 15)
        sig = rng.permutation(sig)
        chosen = set(u["opp_id"].to_numpy()[sig])
        f = f[f["opp_id"].isin(chosen) & (f["selection"] == "under")]
    else:
        raise ValueError(kind)
    return _nfl_rows(f, fair).reset_index(drop=True)


def kalshi(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(params["seed"]))
    kind = params["kind"]
    f = frame[frame["ask"].notna() & (frame["ask"] > 0) & (frame["ask"] < 1)].copy()
    if kind == "coin_flip":
        f = f[rng.random(len(f)) < float(params.get("rate", 0.03))]
        f = f.drop_duplicates("opp_id")
    elif kind == "shuffled_signal":
        from importlib import import_module
        import sys
        from ..paths import SPORTS
        if str(SPORTS) not in sys.path:
            sys.path.insert(0, str(SPORTS))
        rule = import_module("lab.strategies.kalshi_reject_safe_check")
        sig = rule.bet(None, f).to_numpy() > 0
        f = f[rng.permutation(sig)]
    else:
        raise ValueError(kind)
    d = pd.to_datetime(f["date"]).dt.tz_localize("UTC")
    b = pd.DataFrame({"bet_id": (f["opp_id"].astype(str) + "|" + f["selection"].astype(str)).to_numpy(),
                      "cluster": d.dt.strftime("%Y-%m-%d").to_numpy(),
                      "event_start": (d + pd.Timedelta(hours=4)).to_numpy(),
                      "decided_at": (d + pd.Timedelta(hours=2)).to_numpy(),
                      "inputs_as_of": (d - pd.Timedelta(hours=2)).to_numpy(),
                      "price_time": (d + pd.Timedelta(hours=2)).to_numpy(),
                      "price": np.round(f["ask"].to_numpy(float), 4), "price_source": "taker",
                      "side": f["selection"].to_numpy(), "stake": 1.0})
    return b.reset_index(drop=True)

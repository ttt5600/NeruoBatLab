"""NBA rules.

l2_sides: L2 team-rating side leans (results/l2_team_rating_sides/bets_*.csv, one trial at a time via params.trial).
  Price = the ESPN CLOSING moneyline of the picked side (a book quote; the close is by definition the last pre-tip
  price) -> price_time = decided_at = tip - 1 min. inputs_as_of = 00:00 ET of the game date (ratings use games with ET
  date < d). Tip times from data/espn/nba/games_2026.parquet. close_price = price (bet at the close -> CLV 0).
l32b_dev: L32b de-biased opening totals. ESPN opens carry NO timestamp, so price_time is unknown -> left NaT and the
  PIT check rejects every row (GOAL rule 4 fails closed). `assume_open_time_h` (sensitivity only, never a verdict
  spec) stamps the open at tip - h hours.
l32b_forward: the forward paper log (scripts/l32b_logger.py). It records a no-vig probability but no QUOTED price ->
  price NaN -> schema rejects (sportsbook verdicts need the quoted price); backfill decisions are stamped after tip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import DATA


def _am_to_prob(a):
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def _tips() -> dict:
    g = pd.read_parquet(DATA / "espn" / "nba" / "games_2026.parquet")
    col = "start_utc" if "start_utc" in g.columns else "date"
    return dict(zip(g["event_id"].astype(str), pd.to_datetime(g[col], utc=True)))


def l2_sides(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    trial = (params or {})["trial"]
    f = frame[frame["trial"] == trial]
    tips = _tips()
    tip = pd.to_datetime([tips.get(str(e), pd.NaT) for e in f["event_id"]], utc=True)
    et0 = pd.to_datetime(f["date"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    ml = np.where(f["side"] == "home", f["home_ml_close"], f["away_ml_close"])
    b = pd.DataFrame({"bet_id": f["trial"].astype(str) + "|" + f["event_id"].astype(str),
                      "cluster": f["date"].astype(str).str[:10].to_numpy(),
                      "event_start": tip, "decided_at": tip - pd.Timedelta(minutes=1),
                      "inputs_as_of": et0.to_numpy(), "price_time": tip - pd.Timedelta(minutes=1),
                      "price": np.round(_am_to_prob(ml), 6), "price_source": "book_quote",
                      "side": f["side"].to_numpy(), "stake": 1.0})
    b["close_price"] = b["price"]
    return b.reset_index(drop=True)


def l32b_dev(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    if "type" in frame.columns or frame.empty:
        return l32b_forward(frame, params)
    params = params or {}
    tip = pd.to_datetime(frame["start_utc"], utc=True)
    over = frame["side"].to_numpy() > 0
    odds = np.where(over, frame["over_odds_open"], frame["under_odds_open"])
    et0 = pd.to_datetime(frame["date"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    h = params.get("assume_open_time_h")
    ptime = (tip - pd.Timedelta(hours=float(h))) if h is not None else pd.Series(pd.NaT, index=frame.index)
    # CLV: no-vig close prob of our side at the close line, shifted to OUR open line at 0.022 prob/pt (NBA total sd ~18)
    pf_c = frame["pf_over_close"].to_numpy(float)
    dl = (frame["tot_close_same"] - frame["total_open"]).to_numpy(float)
    p_close_ours = np.where(over, pf_c + 0.022 * dl, (1 - pf_c) - 0.022 * dl)
    p_open_nv = np.where(over, frame["pf_over_open"], 1 - frame["pf_over_open"])
    b = pd.DataFrame({"bet_id": frame["event_id"].astype(str).to_numpy(),
                      "cluster": frame["date"].astype(str).str[:10].to_numpy(), "event_start": tip.to_numpy(),
                      "decided_at": ptime.to_numpy() if h is not None else (tip - pd.Timedelta(hours=1)).to_numpy(),
                      "inputs_as_of": et0.to_numpy(), "price_time": ptime.to_numpy(),
                      "price": np.round(_am_to_prob(odds), 6), "price_source": "book_quote",
                      "side": np.where(over, "over", "under"), "stake": 1.0})
    # same-units close: our quoted open price moved by the fair-prob change (keeps the vig out of the CLV sign)
    b["close_price"] = b["price"].to_numpy() + (p_close_ours - p_open_nv)
    return b


def l32b_forward(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    if frame.empty or "pick" not in frame.columns:
        return pd.DataFrame(columns=["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price",
                                     "price_source", "price_time", "side", "stake"])
    d = frame[(frame.get("type") == "decision") & frame["pick"].notna()]
    return pd.DataFrame({"bet_id": d["key"].to_numpy(), "cluster": d["key"].str[:10].to_numpy(),
                         "event_start": pd.to_datetime(d["start"], utc=True).to_numpy(),
                         "decided_at": pd.to_datetime(d["t"], utc=True).to_numpy(),
                         "inputs_as_of": pd.to_datetime(d["t"], utc=True).to_numpy(),
                         "price_time": pd.to_datetime(d["open_t"], utc=True).to_numpy(),
                         "price": np.nan, "price_source": "book_quote", "side": d["pick"].to_numpy(), "stake": 1.0})


def paint_sides(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """L30 P-PAINT fade (notes/nba_l30_paint.md): ATS side against the team conceding the higher season-to-date share of
    opponents' points in the paint, |z| >= 1, at the closing spread. params.trial: PP1 (-110, main line) | PP2 (ESPN core
    DK/ESPN BET close spread + odds). No tip times in the dev frame: event_start = game date 16:00 UTC (convention of
    edgelab.rules.game_release), decided_at = price_time = start - 1 min, inputs_as_of = 00:00 ET game date (features
    use games with date < d). Bet at the close -> close_price = price."""
    f = frame[frame["trial"] == (params or {})["trial"]]
    d = pd.to_datetime(f["date"]).dt.normalize()
    start = d.dt.tz_localize("UTC") + pd.Timedelta(hours=16)
    et0 = d.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    b = pd.DataFrame({"bet_id": (f["game_id"].astype(str) + "|" + f["trial"].astype(str)).to_numpy(),
                      "cluster": d.dt.strftime("%Y-%m-%d").to_numpy(),
                      "event_start": start.to_numpy(), "decided_at": (start - pd.Timedelta(minutes=1)).to_numpy(),
                      "inputs_as_of": et0.to_numpy(), "price_time": (start - pd.Timedelta(minutes=1)).to_numpy(),
                      "price": np.round(1.0 / f["dec"].to_numpy(float), 6), "price_source": "book_quote",
                      "side": f["sel"].to_numpy(), "stake": 1.0})
    b["close_price"] = b["price"]
    return b.reset_index(drop=True)

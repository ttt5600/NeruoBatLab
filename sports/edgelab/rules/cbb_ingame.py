"""Men's college basketball IN-GAME: simple WP model vs Kalshi KXNCAAMBGAME (scripts/cbb_ingame.py builds the data).

Frame (loader cbb_ingame): one row per game x decision time t (every 60 s of wallclock, on candle-end boundaries,
from first ESPN play + 60 s to the last play). Columns: eid, ticker (HOME team's market), date (ET), t (unix s),
state as of the last ESPN play with wallclock <= t - 60 s (m = home margin, left = game seconds left, period, poss,
state_wall), the quote = last 1-min candle with end <= t (yes_bid, yes_ask, q_end), prior (Kalshi pregame mid = last
two-sided candle with spread <= 0.10 ending <= first-play wallclock, else ESPN pregame p0; prior_src, prior_as_of),
pre_vol (Kalshi contracts traded on the home market before the first play), game_end (last ESPN play wallclock).
attrs['candles'] = {eid: DataFrame(end, bid, ask, tlo, thi)} for maker fills and the CLV mark. Outcome = hidden.

MODEL (frozen; fit by scripts/cbb_ingame.py fit on ESPN 2023-24 + 2024-25, games < 2025-11-01, i.e. before the first
Kalshi CBB market): P(home) = Phi(b . [1, m/sqrt(tau), z0*sqrt(tau), poss/sqrt(tau), m]), tau = (left + 4)/2400,
z0 = Phi^-1(clip(prior, .01, .99)).

Sides: home = buy YES on the home market at yes_ask; away = buy NO at 1 - yes_bid (pays iff the away team wins).
Quote filter: yes_bid >= 0.01, yes_ask <= 0.99, price in [0.03, 0.97].

T1 taker_all : any t with left > 0; a side with model_side - price - 0.07 p(1-p) >= 0.05 -> buy at the ask (taker).
               First trigger per game, hold to settlement.
T2 taker_lowvol : T1 restricted to games whose pre_vol is in the bottom third of the games in the evaluated frame
               (pre_vol uses only pre-tip trading; the tercile is computed on the frame the vault hands over).
T3 maker     : a side with model_side - q - 0.0175 q(1-q) >= 0.05 where q = that side's bid + 0.01 and spread >= 0.02;
               post q for 180 s. Filled iff a later 1-min candle (end in (t, t+180]) printed a trade through q
               (home: YES trade low <= q; away: YES trade high >= 1 - q). Fill time = that candle's end. Unfilled
               -> cancelled, the game stays eligible. First FILL per game, hold to settlement.
event_start = game_end (the outcome exists only at the final horn; in-game information is the point of the bet).
CLV mark (close_price) = same side's ASK on the last candle ending <= decision (or fill) + 600 s, capped at game_end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

COEF = [0.011268, 0.064196, 0.96394, 0.023993, -0.003372]  # data/cbb_ingame/coef.json (fit 2026-11-05, 12,286 games 2023-24 + 2024-25), frozen here
EDGE = 0.05
TAKER, MAKER = 0.07, 0.0175
POST_S = 180
CLV_S = 600


def model_home(f: pd.DataFrame) -> np.ndarray:
    b = np.asarray(COEF, float)
    tau = (f["left"].to_numpy(float) + 4) / 2400
    st = np.sqrt(tau)
    m = f["m"].to_numpy(float)
    z0 = norm.ppf(np.clip(f["prior"].to_numpy(float), 0.01, 0.99))
    X = np.column_stack([np.ones_like(tau), m / st, z0 * st, f["poss"].to_numpy(float) / st, m])
    return norm.cdf(X @ b)


def _ts(x):
    return pd.to_datetime(np.asarray(x, float), unit="s", utc=True)


def _candidates(frame: pd.DataFrame, maker: bool) -> pd.DataFrame:
    f = frame[(frame["left"] > 0) & (frame["yes_bid"] >= 0.01) & (frame["yes_ask"] <= 0.99)].copy()
    f.attrs = {}                                  # candles stay on `frame`; pandas cannot compare dict-of-frame attrs
    ph = model_home(f)
    out = []
    for side in ("yes", "no"):
        fair = ph if side == "yes" else 1 - ph
        if maker:
            bid = f["yes_bid"].to_numpy() if side == "yes" else 1 - f["yes_ask"].to_numpy()
            px = np.round(bid + 0.01, 2)
            ok = (f["yes_ask"] - f["yes_bid"]).to_numpy() >= 0.02 - 1e-9
            gap = fair - px - MAKER * px * (1 - px)
        else:
            px = f["yes_ask"].to_numpy() if side == "yes" else 1 - f["yes_bid"].to_numpy()
            px = np.round(px, 4)
            ok = np.ones(len(f), bool)
            gap = fair - px - TAKER * px * (1 - px)
        ok &= (px >= 0.03) & (px <= 0.97) & (gap >= EDGE)
        out.append(f[ok].assign(side=side, px=px[ok], fair=fair[ok]))
    c = pd.concat(out, ignore_index=True)
    return c.sort_values(["eid", "t", "side"]).reset_index(drop=True)


def _mark(cd: pd.DataFrame, side: str, t_unix: float, end_unix: float) -> float:
    c = cd[cd["end"] <= min(t_unix + CLV_S, end_unix)]
    if c.empty:
        return np.nan
    r = c.iloc[-1]
    return float(r["ask"]) if side == "yes" else float(1 - r["bid"])


def _rows(c: pd.DataFrame, frame: pd.DataFrame, price_time, src: str) -> pd.DataFrame:
    cds = frame.attrs["candles"]
    t_dec = _ts(c["t"])
    b = pd.DataFrame({"bet_id": c["ticker"].astype(str) + "|" + c["t"].astype(int).astype(str),
                      "cluster": pd.to_datetime(c["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
                      "event_start": pd.to_datetime(c["game_end"], utc=True).to_numpy(),
                      "decided_at": t_dec,
                      "inputs_as_of": pd.to_datetime(np.maximum(c["state_wall"].to_numpy(float), c["prior_as_of"].to_numpy(float)),
                                                     unit="s", utc=True),
                      "price": c["px"].to_numpy(), "price_source": src, "price_time": price_time,
                      "side": c["side"].to_numpy(), "stake": 1.0, "contracts": 1.0})
    ge = (pd.to_datetime(c["game_end"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    b["close_price"] = [_mark(cds[e], s, float(t), float(g)) for e, s, t, g in zip(c["eid"], c["side"], c["mark_t"], ge)]
    return b


def _taker(frame: pd.DataFrame, lowvol: bool) -> pd.DataFrame:
    if lowvol:
        g = frame.drop_duplicates("eid")[["eid", "pre_vol"]]
        cut = g["pre_vol"].quantile(1 / 3)
        frame = frame[frame["eid"].isin(g.loc[g["pre_vol"] <= cut, "eid"])]
        frame.attrs = dict(frame.attrs)
    c = _candidates(frame, maker=False).drop_duplicates("eid").reset_index(drop=True)
    c["mark_t"] = c["t"]
    return _rows(c, frame, _ts(c["q_end"]), "taker")


def t1_taker_all(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _taker(frame, lowvol=False)


def t2_taker_lowvol(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    lo = frame.copy()
    lo.attrs = dict(frame.attrs)
    return _taker(lo, lowvol=True)


def t3_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    cds = frame.attrs["candles"]
    c = _candidates(frame, maker=True)
    keep, fill_t = [], []
    done = set()
    busy_until = {}
    for i, r in enumerate(c.itertuples(index=False)):
        if r.eid in done or r.t < busy_until.get(r.eid, -1):
            continue
        busy_until[r.eid] = r.t + POST_S          # one resting order at a time per game
        cd = cds[r.eid]
        w = cd[(cd["end"] > r.t) & (cd["end"] <= r.t + POST_S)]
        hit = w[w["tlo"] <= r.px + 1e-9] if r.side == "yes" else w[w["thi"] >= 1 - r.px - 1e-9]
        if hit.empty:
            continue
        done.add(r.eid)
        keep.append(i)
        fill_t.append(float(hit["end"].iloc[0]))
    c = c.iloc[keep].reset_index(drop=True)
    c["mark_t"] = np.asarray(fill_t, float)
    b = _rows(c, frame, _ts(fill_t), "maker_tape")
    b["fill_deadline"] = _ts(c["t"].to_numpy(float) + POST_S)
    return b

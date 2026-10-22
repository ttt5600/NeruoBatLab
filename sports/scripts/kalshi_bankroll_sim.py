"""$50 bankroll simulation of the two sealed-PASSED Kalshi temperature streams under the LIVE executor caps.

    ../.venv_quant/bin/python scripts/kalshi_bankroll_sim.py [--fee-cent]   # prints everything; writes nothing

NOT a new trial and NOT a re-score: it replays the exact bet lists that were already scored once on the lockbox
(E1 = kalshi_reject_safe_check on kalshi_temp 2025Q3..2026Q4, 1,844 bets; E2 = X1 on wx_expand date >= 2026-03-01,
2,464 bets) and asks what the live caps do to them. The script first reproduces both lockbox totals and refuses to
continue if they do not match.

Live caps modelled (~/.claude/quant-kalshi/.env 2026-10-22): 1 contract per pick; per-order <= $1.05 (orig7) /
$1.00 (x1); <= $15 placed per night; <= $20 open; loss stops 24h <= -$10 -> skip the night, 7d <= -$20 -> KILL
(stream stops for good). Timing: picks on night n are for market date n+1 and settle the morning of n+2, so at night n
the bets of night n-1 are still OPEN and the 24 h stop sees night n-2's result.
Within a night the executor takes picks in log order; here a random order (cap-binding nights drop random picks).
Price = the 22:00 quote the lockbox used (1 - yes_bid + fee); real fills can be worse (executor skips >1c moves).

Bootstrap: whole calendar nights (zero-bet nights included) resampled with replacement from the window where BOTH
streams exist (2026-03-01 .. last sealed date), so same-night weather correlation is kept; seasonality is not.
A 7-night block bootstrap is printed as a sensitivity check.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))
sys.path.insert(0, str(ROOT / "scripts"))

from sports import wx  # noqa: E402

FEE = wx.FEE_RATE
CAP_ORDER = {"orig7": 1.05, "x1": 1.00}
CAP_NIGHT, CAP_OPEN, STOP_DAY, STOP_WEEK = 15.0, 20.0, -10.0, -20.0
BANK0 = 50.0


FEE_MODE = "cent" if "--fee-cent" in sys.argv else "exact"


def fee_ceil(p):
    """1-contract fee. exact (default): direct Kalshi members are charged the fee model at $0.0001 balance precision
    (docs.kalshi.com/getting_started/fee_rounding, read 2026-10-22) -> ~0.07 p (1-p). --fee-cent: rounded UP to the
    cent per order (what kalshi_exec.fee_total assumes for sizing, and what $0.01-precision FCM accounts, e.g. a
    broker-routed account, would pay) -> $0.01 per 1-contract order."""
    if FEE_MODE == "exact":
        return np.ceil(np.round(FEE * p * (1 - p) * 1e4, 6)) / 1e4
    return np.ceil(np.round(FEE * p * (1 - p) * 100, 6)) / 100


def e1_bets() -> pd.DataFrame:
    import datasets as D
    spec = importlib.util.spec_from_file_location("s", ROOT / "lab" / "strategies" / "kalshi_reject_safe_check.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    df = D.load("kalshi_temp", "lockbox")
    pick = S.bet(None, df) > 0
    b = df[pick].copy()
    units = np.where(b.won == 1, b.dec_odds - 1, -1.0).sum()
    assert len(b) == 1844 and abs(units - 26.8247) < 0.01, (len(b), units)
    no = b["ask"].astype(float)  # NO ask = 1 - yes_bid
    return pd.DataFrame({"date": pd.to_datetime(b["date"]), "city": b["city"], "cohort": "orig7", "no": no,
                         "win": b.won.astype(int), "lockbox_unit_pnl": np.where(b.won == 1, b.dec_odds - 1, -1.0)})


def e2_bets() -> pd.DataFrame:
    import wx_expand as X
    df = pd.read_parquet(X.OUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date >= X.SPLIT]
    s = X.picks(df[df.kind == "high"])  # identical selection to wx_expand.cmd_sealed
    units = s.pnl.sum()
    if not (len(s) == 2464 and abs(units - 19.2179) < 0.01):
        raise SystemExit(f"E2 reproduction mismatch: {len(s)} bets {units:.4f}u (lockbox 2464 / 19.2179)")
    return pd.DataFrame({"date": s.date, "city": s.series, "cohort": "x1", "no": 1 - s.yes_bid.astype(float),
                         "win": (s.result == "no").astype(int), "lockbox_unit_pnl": s.pnl})


def per_contract(b: pd.DataFrame) -> pd.DataFrame:
    """$ P&L of ONE contract with the executor's rounded-up fee (what the live account actually books)."""
    cost = b.no + fee_ceil(b.no)
    return b.assign(cost=cost, pnl=np.where(b.win == 1, 1 - cost, -cost), ok=cost <= b.cohort.map(CAP_ORDER) + 1e-9)


def simulate(nights: list[np.ndarray], costs: list[np.ndarray], rng) -> dict:
    """nights[i] = per-contract P&L of night i's picks, costs[i] their cost. Returns path stats."""
    n = len(nights)
    pnl = np.zeros(n)
    placed = np.zeros(n)
    nbets = np.zeros(n, int)
    killed_at, day_stops = None, 0
    for i in range(n):
        if i >= 2 and pnl[i - 2] <= STOP_DAY:
            day_stops += 1
            continue
        if i >= 2 and pnl[max(0, i - 8):i - 1].sum() <= STOP_WEEK:
            killed_at = i
            break
        open_prev = placed[i - 1] if i >= 1 else 0.0
        room = min(CAP_NIGHT, CAP_OPEN - open_prev)
        p, c = nights[i], costs[i]
        if len(p) == 0 or room <= 0:
            continue
        order = rng.permutation(len(p))
        cum = np.cumsum(c[order])
        take = order[cum <= room + 1e-9]
        pnl[i], placed[i], nbets[i] = p[take].sum(), c[take].sum(), len(take)
    eq = BANK0 + np.cumsum(pnl)
    dd = (np.maximum.accumulate(np.concatenate([[BANK0], eq])) - np.concatenate([[BANK0], eq])).max()
    hit100 = np.argmax(eq >= 100) if (eq >= 100).any() else None
    return {"pnl": pnl, "eq": eq, "dd": dd, "killed": killed_at, "day_stops": day_stops, "nbets": nbets,
            "placed": placed, "t100": hit100}


def main() -> int:
    rng = np.random.default_rng(20261022)
    b1, b2 = per_contract(e1_bets()), per_contract(e2_bets())
    print(f"reproduced lockbox: E1 {len(b1)} bets {b1.lockbox_unit_pnl.sum():+.2f}u (ROI {b1.lockbox_unit_pnl.mean():+.2%}); "
          f"E2 {len(b2)} bets {b2.lockbox_unit_pnl.sum():+.2f}u (ROI {b2.lockbox_unit_pnl.mean():+.2%})")
    allb = pd.concat([b1, b2], ignore_index=True)
    print(f"fee mode {FEE_MODE}: 1-contract $: E1 ${b1.pnl.sum():+.2f} on ${b1.cost.sum():.2f} ({b1.pnl.sum() / b1.cost.sum():+.2%}), "
          f"E2 ${b2.pnl.sum():+.2f} on ${b2.cost.sum():.2f} ({b2.pnl.sum() / b2.cost.sum():+.2%}); "
          f"mean NO cost ${allb.cost.mean():.3f}; {(~allb.ok).sum()} picks above the per-order cap (dropped)")
    allb = allb[allb.ok].copy()
    # --edge=conservative: shift each stream's per-contract P&L so its ROI equals the sealed CI LOWER bound
    # (lockbox.jsonl ci_lo / staked: E1 10.86/1844 = +0.59%, E2 4.74/2464 = +0.19%); --edge=zero: ROI 0 (pure noise).
    # Night-level correlation is kept; only the mean moves. Default (observed) treats the sample edge as truth.
    edge = next((a.split("=")[1] for a in sys.argv if a.startswith("--edge=")), "observed")
    if edge != "observed":
        tgt = {"orig7": 10.855721 / 1844, "x1": 4.744270 / 2464} if edge == "conservative" else {"orig7": 0.0, "x1": 0.0}
        for coh, t in tgt.items():
            m = allb.cohort == coh
            roi = allb.loc[m, "pnl"].sum() / allb.loc[m, "cost"].sum()
            allb.loc[m, "pnl"] -= (roi - t) * allb.loc[m, "cost"]
        print(f"edge scenario {edge}: per-contract ROI forced to " + ", ".join(f"{k} {v:+.2%}" for k, v in tgt.items()))
    lo, hi = pd.Timestamp("2026-03-01"), allb.date.max()
    w = allb[(allb.date >= lo) & (allb.date <= hi)]
    cal = pd.date_range(lo, hi, freq="D")
    g = {d: x for d, x in w.groupby("date")}
    nights = [g[d].pnl.to_numpy() if d in g else np.zeros(0) for d in cal]
    costs = [g[d].cost.to_numpy() if d in g else np.zeros(0) for d in cal]
    per = np.array([len(x) for x in nights])
    print(f"\nwindow {lo.date()}..{hi.date()} ({len(cal)} nights): picks/night mean {per.mean():.1f}, median "
          f"{np.median(per):.0f}, max {per.max()}, zero-pick nights {(per == 0).mean():.0%}; uncapped $/night "
          f"{np.mean([x.sum() for x in nights]):+.3f}")

    hist = simulate(nights, costs, rng)
    print(f"HISTORICAL path under caps (actual order of nights): P&L ${hist['pnl'].sum():+.2f} over {len(cal)} nights, "
          f"{hist['nbets'].sum()} bets (of {per.sum()}), max DD ${hist['dd']:.2f}, day-stops {hist['day_stops']}, "
          f"KILL {'night ' + str(hist['killed']) if hist['killed'] is not None else 'never'}")
    worst = sorted(((x.sum(), str(d.date()), len(x)) for d, x in zip(cal, nights)))[:5]
    print("worst uncapped nights ($, date, picks):", [(round(a, 2), d, k) for a, d, k in worst])

    for block in (1, 7):
        H = 365 * 3
        R = 4000
        res = []
        for _ in range(R):
            if block == 1:
                idx = rng.integers(0, len(cal), H)
            else:
                starts = rng.integers(0, len(cal) - block, H // block + 1)
                idx = (starts[:, None] + np.arange(block)).ravel()[:H]
            res.append(simulate([nights[i] for i in idx], [costs[i] for i in idx], rng))
        print(f"\n=== bootstrap, {block}-night blocks, {R} paths x {H} nights ===")
        for m, d in (("month 1", 30), ("month 3", 91), ("month 12", 365)):
            v = np.array([r["pnl"][:d].sum() for r in res])
            print(f"  {m:8s} P&L $: mean {v.mean():+.2f}  median {np.median(v):+.2f}  5% {np.percentile(v, 5):+.2f}  "
                  f"95% {np.percentile(v, 95):+.2f}  P(<0) {np.mean(v < 0):.0%}")
        for d in (30, 91, 365):
            dd = []
            for r in res:
                eq = np.concatenate([[BANK0], r["eq"][:d]])
                dd.append((np.maximum.accumulate(eq) - eq).max())
            dd = np.array(dd)
            print(f"  max drawdown within {d:3d} nights: median ${np.median(dd):.2f}  95% ${np.percentile(dd, 95):.2f}  "
                  f"max ${dd.max():.2f}")
        for d in (30, 91, 365):
            ds = np.mean([(r["day_stops"] > 0) if d >= H else any(r["pnl"][j] <= STOP_DAY for j in range(min(d, H) - 2))
                          for r in res])
            ks = np.mean([r["killed"] is not None and r["killed"] < d for r in res])
            print(f"  within {d:3d} nights: P(24h -$10 stop fires) {ds:.1%}   P(7d -$20 KILL) {ks:.1%}")
        t = np.array([r["t100"] if r["t100"] is not None else np.inf for r in res])
        print(f"  nights to reach $100: median {np.median(t):.0f}  25% {np.percentile(t, 25):.0f}  75% "
              f"{np.percentile(t, 75):.0f}; P(by 1y) {np.mean(t < 365):.0%}  P(by 2y) {np.mean(t < 730):.0%}  "
              f"P(by 3y) {np.mean(t < H):.0%}")
        nb = np.array([r["nbets"][:365].sum() for r in res])
        pl = np.array([r["placed"][:365].sum() for r in res])
        print(f"  year-1 bets placed: median {np.median(nb):.0f} (${np.median(pl):.0f} turnover)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

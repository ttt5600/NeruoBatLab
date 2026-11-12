"""F2 on FX (pre-registered, notes/e5_fix.md): obs trigger times from IEM ASOS for every settled FX pick.
FX fills have no timestamps, so report BOUNDS: optimistic = a fill survives only if the pick never triggered before
the cutoff; pessimistic = every fill survives (B0). Also how often a LOSER's trigger fires before local 09/12/15h."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/agent-harness/lib"))
from traps import check_epoch_seconds  # noqa: E402

OUTD = Path(__file__).parent
FEE = 0.01
TZ = {"LGA": "America/New_York", "LAX": "America/Los_Angeles", "SEA": "America/Los_Angeles", "BNA": "America/Chicago",
      "DCA": "America/New_York", "ATL": "America/New_York", "PHL": "America/New_York", "SFO": "America/Los_Angeles",
      "MSP": "America/Chicago", "OKC": "America/Chicago", "BOS": "America/New_York", "CLT": "America/New_York",
      "DFW": "America/Chicago", "PHX": "America/Phoenix", "MDW": "America/Chicago", "SAT": "America/Chicago",
      "MIA": "America/New_York", "MSY": "America/Chicago", "LAS": "America/Los_Angeles", "DTW": "America/Detroit",
      "AUS": "America/Chicago", "HOU": "America/Chicago", "BKF": "America/Denver", "JAX": "America/New_York"}
OBS_LAG = 15 * 60


def triggers(s, o):
    g = {k: v[["ts", "tmpf"]].to_numpy() for k, v in o.groupby("stn")}
    res = {x: [] for x in (2, 4, 6)}
    clear, mids, maxd = [], [], []
    for r in s.itertuples():
        z = TZ[r.code]
        mid = (pd.Timestamp(r.event_date, tz=z) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
        mids.append(mid)
        a = g[r.code]
        w = a[(a[:, 0] >= mid) & (a[:, 0] < mid + 86400)]
        rm = np.maximum.accumulate(w[:, 1])
        av = w[:, 0] + OBS_LAG
        maxd.append(rm[-1] if len(rm) else np.nan)
        if r.side == "NO":  # above-side: lose iff max > X  -> L = X + 1
            for x in res:
                h = np.nonzero(rm >= r.X + 1 - x)[0]
                res[x].append(av[h[0]] if len(h) else np.inf)
            clear.append(np.inf)
        else:  # below-side YES wins iff max > X: cleared once running max >= X + 1
            for x in res:
                res[x].append(np.inf)
            h = np.nonzero(rm >= r.X + 1)[0]
            clear.append(av[h[0]] if len(h) else np.inf)
    for x in res:
        s[f"trig{x}"] = res[x]
    s["clear_t"], s["mid_D"], s["obs_max"] = clear, mids, maxd
    check_epoch_seconds(s.mid_D, "mid_D")
    return s


def main():
    s = pd.read_csv(OUTD / "fx_picks_d1.csv")
    o = pd.read_parquet(OUTD / "asos_obs.parquet")
    o["stn"] = o.stn.astype(str).str[-3:]
    o["ts"] = (o.valid - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    s = triggers(s, o)
    # sanity: METAR max vs settlement (WU daily = METAR max): NO loses iff max > X
    s["obs_says_yes"] = s.obs_max > s.X
    agree = float((s.obs_says_yes == (s.yes_won == 1)).mean())
    s["lo"] = s[["low_d", "low_d1"]].min(axis=1)
    filled = (s.lo < s.side_bid).to_numpy()
    won = s.won.to_numpy().astype(bool)
    q = s.side_bid.to_numpy()
    pnl = np.where(won, 1 / (q + FEE) - 1, -1.0)
    out = {"metar_max_vs_settlement_agree": round(agree, 4), "n": len(s)}
    for x in (2, 4, 6):
        noon = s.mid_D + 12 * 3600
        trig_any = np.where(s.side == "NO", s[f"trig{x}"] < np.inf, ~(s.clear_t < noon))
        keep_opt = filled & ~trig_any
        n = int(keep_opt.sum())
        out[f"X{x}"] = {"picks_triggered": int(trig_any.sum()), "fills_on_untriggered": n,
                        "losses_on_untriggered": int((keep_opt & ~won).sum()),
                        "roi_optimistic": round(float(pnl[keep_opt].mean()), 4) if n else None,
                        "units_optimistic": round(float(pnl[keep_opt].sum()), 2),
                        "fills_on_triggered": int((filled & trig_any).sum()),
                        "winner_fills_on_triggered": int((filled & trig_any & won).sum()),
                        "loser_trigger_local_hour": sorted(round(float(h), 1) for h in
                                                           ((s[f"trig{x}"] - s.mid_D) / 3600)[~won & (s.side == "NO")])}
    out["losers"] = s[~won][["ticker", "side", "X", "side_bid", "low_d", "obs_max", "trig4"]].assign(
        trig4_local_h=lambda d: (d.trig4 - s.mid_D[~won]) / 3600).drop(columns="trig4").to_dict("records")
    out["B0"] = {"fills": int(filled.sum()), "roi": round(float(pnl[filled].mean()), 4),
                 "units": round(float(pnl[filled].sum()), 2)}
    json.dump(out, open(OUTD / "fx_f2.json", "w"), indent=1, default=float)
    s.to_csv(OUTD / "fx_f2_picks.csv", index=False)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

"""Ex-ante activity proxy for Polymarket (lifetime volume is EX-POST): number of hourly display-price CHANGES in the
72 h before the decision time t, from the cached CLOB history. Adds column `act72` to data/smallmarket/obs.parquet rows
of Polymarket and re-runs the longshot-gap / rule tables by activity bin -> data/smallmarket/exante.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from smallmarket_calibration import CLOB, HORIZONS, OUT, get, pmap
from smallmarket_analyze import summarize


def act(row):
    end = int(row.end.timestamp())
    d = get(f"{CLOB}/prices-history", {"market": row.token0, "startTs": end - 8 * 86400, "endTs": end + 3600, "fidelity": 60})
    out = {"market_id": row.market_id}
    if not d or not d.get("history"):
        return out
    ts = np.array([x["t"] for x in d["history"]]); ps = np.array([x["p"] for x in d["history"]], float)
    gs = pd.to_datetime(row.game_start, utc=True, errors="coerce")
    S = min(end, int(gs.timestamp())) if pd.notna(gs) else end
    for k, hh in HORIZONS.items():
        t = S - hh * 3600
        m = (ts > t - 72 * 3600) & (ts <= t)
        out[f"act_{k}"] = int((np.abs(np.diff(ps[m])) > 1e-9).sum()) if m.sum() > 1 else 0
    return out


def main():
    p = pd.read_parquet(OUT / "pm_priced2.parquet")
    a = pd.DataFrame(pmap(act, list(p.itertuples()), workers=8))
    obs = pd.read_parquet(OUT / "obs.parquet")
    o = obs[obs.venue == "polymarket"].copy()
    o["market_id"] = o.mkt
    am = a.set_index("market_id")
    o["act72"] = [am.at[m, f"act_{h}"] if m in am.index and f"act_{h}" in am.columns else np.nan for m, h in zip(o.market_id, o.h)]
    o.to_parquet(OUT / "obs_pm_act.parquet", index=False)
    res = {}
    for h, g in o.groupby("h"):
        g = g.assign(ab=pd.cut(g.act72, [-1, 0, 3, 10, 30, 1e9], labels=["0", "1-3", "4-10", "11-30", ">30"]))
        res[h] = {str(b): {**summarize(gb), "vol_life_median": float(gb.vol_life.median())} for b, gb in g.groupby("ab", observed=True)}
        res[h]["corr_act_logvol"] = float(np.corrcoef(g.act72.fillna(0), np.log1p(g.vol_life))[0, 1])
    (OUT / "exante.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({h: {b: {k: v for k, v in s.items() if k in ("n", "events", "ls_gap", "ece", "vol_life_median")}
                          | ({"FAV": s["FAV"]["roi_net"], "FAV_mid": s["FAV"]["roi_mid"]} if "FAV" in s else {})
                          | ({"NO_LONG": s["NO_LONG"]["roi_net"]} if "NO_LONG" in s else {})
                          for b, s in r.items() if isinstance(s, dict)} | {"corr": r["corr_act_logvol"]} for h, r in res.items()}, indent=0))


if __name__ == "__main__":
    main()

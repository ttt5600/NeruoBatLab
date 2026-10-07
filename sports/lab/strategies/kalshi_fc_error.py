import numpy as np
import pandas as pd
from scipy import stats

DATASET = "kalshi_temp"
HYPOTHESIS = ("Bracket prices at 10pm the night before are not calibrated to the GFS day-ahead "
              "forecast's own error distribution; price each bracket from a per-city normal error "
              "model (bias + sd learned on earlier quarters) and buy the side with EV >= 5% after fees.")
SOURCE = "agent"
EV_MIN = 0.05


def fit(train):
    days = train.drop_duplicates("event_ticker").dropna(subset=["fc_max_lead1", "actual"])
    err = days["actual"] - days["fc_max_lead1"]
    by_city = err.groupby(days["city"]).agg(["mean", "std", "size"])
    pooled = (float(err.mean()), float(err.std()))
    return {"city": by_city, "pooled": pooled}


def _p_yes(model, slate):
    mu_b = slate["city"].map(model["city"]["mean"]).fillna(model["pooled"][0])
    sd = slate["city"].map(model["city"]["std"]).fillna(model["pooled"][1])
    m = (slate["fc_max_lead1"] + mu_b).to_numpy(float)
    s = sd.to_numpy(float)
    lo = slate["floor_strike"].to_numpy(float)
    hi = slate["cap_strike"].to_numpy(float)
    st = slate["strike_type"].to_numpy()
    cdf = lambda x: stats.norm.cdf(x, m, s)
    # Official highs are whole degrees: "greater than F" means >= F+1, "less than C" means <= C-1.
    p = np.where(st == "greater", 1 - cdf(lo + 0.5),
        np.where(st == "less", cdf(hi - 0.5),
                 cdf(hi + 0.5) - cdf(lo - 0.5)))
    return np.clip(p, 1e-4, 1 - 1e-4)


def bet(model, slate):
    p_yes = _p_yes(model, slate)
    p_side = np.where(slate["selection"].to_numpy() == "yes", p_yes, 1 - p_yes)
    ev = p_side * slate["dec_odds"].to_numpy(float) - 1
    ev = np.where(np.isnan(slate["fc_max_lead1"].to_numpy(float)), -1, ev)
    s = pd.Series(np.where(ev >= EV_MIN, 1.0, 0.0), index=slate.index)
    best = pd.Series(ev, index=slate.index).groupby(slate["opp_id"]).transform("max")
    s[pd.Series(ev, index=slate.index) < best] = 0.0
    return s

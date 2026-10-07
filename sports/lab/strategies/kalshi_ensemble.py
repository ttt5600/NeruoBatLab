import numpy as np
import pandas as pd
from scipy import stats

DATASET = "kalshi_temp"
HYPOTHESIS = ("A multi-model day-ahead ensemble (GFS, ECMWF, ICON, GEM, NBM where archived) is ~23% sharper "
              "than GFS alone; price brackets from its per-city bias/sd (learned on earlier quarters, "
              ">= 3 members required) and buy the side with EV >= 5% after fees.")
SOURCE = "agent"
EV_MIN = 0.05
MEMBERS = ["fc_max_lead1", "fc_ecmwf_ifs025", "fc_icon_seamless", "fc_gem_seamless", "fc_ncep_nbm_conus"]


def _ens(df):
    cols = [c for c in MEMBERS if c in df.columns]
    m = df[cols]
    return m.mean(axis=1).where(m.notna().sum(axis=1) >= 3)


def fit(train):
    days = train.drop_duplicates("event_ticker").copy()
    days["ens"] = _ens(days)
    days = days.dropna(subset=["ens", "actual"])
    if len(days) < 150:
        return None
    err = days["actual"] - days["ens"]
    return {"city": err.groupby(days["city"]).agg(["mean", "std"]), "pooled": (err.mean(), err.std())}


def bet(model, slate):
    s = pd.Series(0.0, index=slate.index)
    if model is None:
        return s
    ens = _ens(slate)
    mu = (ens + slate["city"].map(model["city"]["mean"]).fillna(model["pooled"][0])).to_numpy(float)
    sd = slate["city"].map(model["city"]["std"]).fillna(model["pooled"][1]).to_numpy(float)
    lo, hi = slate["floor_strike"].to_numpy(float), slate["cap_strike"].to_numpy(float)
    st = slate["strike_type"].to_numpy()
    cdf = lambda x: stats.norm.cdf(x, mu, sd)
    p = np.where(st == "greater", 1 - cdf(lo + 0.5), np.where(st == "less", cdf(hi - 0.5),
                 cdf(hi + 0.5) - cdf(lo - 0.5)))
    p_side = np.where(slate["selection"].to_numpy() == "yes", p, 1 - p)
    ev = np.where(np.isnan(mu), -1.0, p_side * slate["dec_odds"].to_numpy(float) - 1)
    ev_s = pd.Series(ev, index=slate.index)
    s[(ev_s >= EV_MIN) & (ev_s >= ev_s.groupby(slate["opp_id"]).transform("max"))] = 1.0
    return s

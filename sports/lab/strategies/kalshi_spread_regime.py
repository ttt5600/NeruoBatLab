import numpy as np
import pandas as pd
from scipy import stats

DATASET = "kalshi_temp"
HYPOTHESIS = ("Cross-model forecast disagreement (ensemble spread) is a same-day uncertainty signal that a "
              "flat per-city width ignores. This does NOT re-fit the forecast mean/bias (that already lost "
              "twice): it keeps mu = ensemble mean and only lets the pricing sd depend on which pooled "
              "disagreement tercile today's spread falls in (fit on earlier quarters). Bet EV>=5% sides only "
              "in the top or bottom disagreement tercile, where a regime-conditional width should diverge "
              "most from the market's presumably flatter one; skip the middle tercile, where a flat model "
              "is probably adequate and we have no edge.")
SOURCE = "agent"
EV_MIN = 0.05
MEMBERS = ["fc_max_lead1", "fc_ecmwf_ifs025", "fc_icon_seamless", "fc_gem_seamless", "fc_ncep_nbm_conus"]


def _ens_spread(df):
    cols = [c for c in MEMBERS if c in df.columns]
    m = df[cols]
    n_avail = m.notna().sum(axis=1)
    ens = m.mean(axis=1).where(n_avail >= 3)
    spread = m.std(axis=1).where(n_avail >= 3)
    return ens, spread


def fit(train):
    days = train.drop_duplicates("event_ticker").dropna(subset=["actual"]).copy()
    ens, spread = _ens_spread(days)
    days["ens"], days["spread"] = ens, spread
    days = days.dropna(subset=["ens", "spread"])
    if len(days) < 150:
        return None
    cuts = days["spread"].quantile([0.33, 0.66]).to_numpy()
    regime = np.where(days["spread"] <= cuts[0], "low", np.where(days["spread"] <= cuts[1], "mid", "high"))
    err = days["actual"] - days["ens"]
    sd_by_regime = err.groupby(regime).std()
    return {"cuts": cuts, "sd": sd_by_regime}


def bet(model, slate):
    s = pd.Series(0.0, index=slate.index)
    if model is None:
        return s
    ens, spread = _ens_spread(slate)
    cuts, sd_map = model["cuts"], model["sd"]
    regime = np.where(spread <= cuts[0], "low", np.where(spread <= cuts[1], "mid", "high"))
    sd = pd.Series(regime, index=slate.index).map(sd_map).to_numpy(float)
    mu = ens.to_numpy(float)
    lo, hi = slate["floor_strike"].to_numpy(float), slate["cap_strike"].to_numpy(float)
    st = slate["strike_type"].to_numpy()
    cdf = lambda x: stats.norm.cdf(x, mu, sd)
    p = np.where(st == "greater", 1 - cdf(lo + 0.5),
        np.where(st == "less", cdf(hi - 0.5), cdf(hi + 0.5) - cdf(lo - 0.5)))
    p_side = np.where(slate["selection"].to_numpy() == "yes", p, 1 - p)
    ev = p_side * slate["dec_odds"].to_numpy(float) - 1
    active = ens.notna().to_numpy() & spread.notna().to_numpy() & (regime != "mid") & ~np.isnan(sd)
    ev = np.where(active, ev, -1.0)
    ev_s = pd.Series(ev, index=slate.index)
    s[(ev_s >= EV_MIN) & (ev_s >= ev_s.groupby(slate["opp_id"]).transform("max"))] = 1.0
    return s

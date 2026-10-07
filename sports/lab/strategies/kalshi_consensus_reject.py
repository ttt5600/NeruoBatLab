import numpy as np
import pandas as pd

DATASET = "kalshi_temp"
HYPOTHESIS = ("When every available forecast model's point estimate sits outside a bracket's range by more "
              "than a 3F safety margin (even the most extreme of 5 models doesn't reach it), the bracket is "
              "a near-unanimous reject regardless of any single model's bias or the ensemble's average "
              "error. If Kalshi's YES ask is still priced above a 1c floor on such brackets, buy NO. This "
              "uses raw cross-model agreement as the signal -- no fitted bias or variance model, no "
              "training -- so it cannot be re-fitting the error model that already lost twice.")
SOURCE = "agent"
MEMBERS = ["fc_max_lead1", "fc_ecmwf_ifs025", "fc_icon_seamless", "fc_gem_seamless", "fc_ncep_nbm_conus"]
MARGIN = 3.0


def fit(train):
    return None


def bet(model, slate):
    cols = [c for c in MEMBERS if c in slate.columns]
    m = slate[cols]
    n_avail = m.notna().sum(axis=1).to_numpy()
    hi = m.max(axis=1).to_numpy(float)
    lo = m.min(axis=1).to_numpy(float)
    floor_s = slate["floor_strike"].to_numpy(float)
    cap_s = slate["cap_strike"].to_numpy(float)
    st = slate["strike_type"].to_numpy()
    rejected = np.where(st == "greater", hi + MARGIN < floor_s + 1,
               np.where(st == "less", lo - MARGIN > cap_s - 1,
                        (hi + MARGIN < floor_s) | (lo - MARGIN > cap_s)))
    rejected = rejected & (n_avail >= 3)
    pick = rejected & (slate["selection"].to_numpy() == "no") & (slate["yes_ask"].to_numpy(float) > 0.01)
    return pd.Series(pick.astype(float), index=slate.index)

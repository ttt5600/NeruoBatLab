import numpy as np
import pandas as pd

DATASET = "kalshi_temp"
HYPOTHESIS = (
    "Mirror of the ledger's kalshi_consensus_reject (p=0.0006): when every available forecast "
    "model's point estimate sits comfortably INSIDE a bracket's range (all models clear the "
    "bracket's boundary by the same 3F margin already validated on the reject side), the "
    "bracket is a near-unanimous accept regardless of any single model's bias or the ensemble's "
    "average error. Kalshi brackets rarely trade all the way to a 99c ceiling even when "
    "consensus is this strong -- the same longshot-style asymmetry that makes 1c-floor NOs "
    "worth fading, mirrored at the top -- so if the YES ask is still below a 99c ceiling, buy "
    "YES. Same raw cross-model agreement mechanism and the same pre-registered margin as the "
    "winning strategy -- no fitted bias, no variance model, no training -- applied to the "
    "opposite, previously untested tail of the same signal."
)
SOURCE = "agent"
MEMBERS = ["fc_max_lead1", "fc_ecmwf_ifs025", "fc_icon_seamless", "fc_gem_seamless", "fc_ncep_nbm_conus"]
MARGIN = 3.0
PRICE_CEIL = 0.99


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
    accepted = np.where(st == "greater", lo - MARGIN > floor_s + 1,
               np.where(st == "less", hi + MARGIN < cap_s - 1,
                        (lo - MARGIN > floor_s) & (hi + MARGIN < cap_s)))
    accepted = accepted & (n_avail >= 3)
    pick = accepted & (slate["selection"].to_numpy() == "yes") & (slate["yes_ask"].to_numpy(float) < PRICE_CEIL)
    return pd.Series(pick.astype(float), index=slate.index)

import pandas as pd

DATASET = "nfl_spread"
HYPOTHESIS = "Home underdogs cover late in the season (week >= 15) as attention shifts to playoff races (Borghesi 2007)."
SOURCE = "registry:S003"


def fit(train):
    return None


def bet(model, slate):
    pick = (slate["selection"] == "home") & (slate["spread_line"] < 0) & (slate["week"] >= 15)
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

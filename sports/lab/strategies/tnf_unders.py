import pandas as pd

DATASET = "nfl_total"
HYPOTHESIS = "Thursday games on short rest score less than totals imply; bet the under on regular-season Thursday games after week 1."
SOURCE = "registry:S032"


def fit(train):
    return None


def bet(model, slate):
    thu = pd.to_datetime(slate["date"]).dt.dayofweek == 3
    pick = thu & (slate["week"] > 1) & (slate["selection"] == "under")
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

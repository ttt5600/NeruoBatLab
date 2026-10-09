import pandas as pd

DATASET = "nfl_total"
HYPOTHESIS = ("Divisional unders only from week 10 on (second meetings, late-season weather). "
              "POST-HOC split found after seeing div_unders by week -- counts as its own trial.")
SOURCE = "scripts/nfl_scan.py follow-up"


def fit(train):
    return None


def bet(model, slate):
    pick = slate["div_game"].astype(bool) & (slate["week"] >= 10) & (slate["selection"] == "under")
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

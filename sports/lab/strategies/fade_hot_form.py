import pandas as pd

DATASET = "nfl_spread"
HYPOTHESIS = ("Overreaction: when one team's 10-game average margin exceeds the opponent's by >= 14 points, "
              "the spread overshoots; bet the cold team.")
SOURCE = "registry:S005"


def fit(train):
    return None


def bet(model, slate):
    gap = slate["home_form_margin"] - slate["away_form_margin"]
    pick = ((gap >= 14) & (slate["selection"] == "away")) | ((gap <= -14) & (slate["selection"] == "home"))
    return pd.Series(pick.fillna(False).astype(float).to_numpy(), index=slate.index)

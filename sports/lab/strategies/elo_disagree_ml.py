import pandas as pd

DATASET = "nfl_moneyline"
HYPOTHESIS = ("A rating model decorrelated from the book can profit where it disagrees strongly: when pre-game "
              "Elo's home win probability differs from the de-vigged market's by >= 0.08, bet the side Elo favours "
              "relative to the market, on the moneyline.")
SOURCE = "Hubacek et al. 2019 (decorrelation from the book); Elo ratings in the dataset"


def fit(train):
    return None


def bet(model, slate):
    d = slate["p_home_elo"] - slate["p_home_ml_mkt"]
    pick = ((d >= 0.08) & (slate["selection"] == "home")) | ((d <= -0.08) & (slate["selection"] == "away"))
    return pd.Series(pick.fillna(False).astype(float).to_numpy(), index=slate.index)

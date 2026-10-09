import pandas as pd

DATASET = "nfl_spread"
HYPOTHESIS = ("Post-2011 CBA the market overprices bye-week rest (implied +0.97 pts vs real +0.31). "
              "When exactly one team is off a bye (rest >= 13 days), bet the OTHER team ATS.")
SOURCE = ("Lopez & Bliss 2024, Frontiers in Behavioral Economics: "
          "https://www.frontiersin.org/journals/behavioral-economics/articles/10.3389/frbhe.2024.1479832/full")


def fit(train):
    return None


def bet(model, slate):
    hb, ab = slate["home_rest_days"] >= 13, slate["away_rest_days"] >= 13
    pick = ((ab & ~hb) & (slate["selection"] == "home")) | ((hb & ~ab) & (slate["selection"] == "away"))
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

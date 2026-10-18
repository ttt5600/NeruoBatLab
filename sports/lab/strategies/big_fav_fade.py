import pandas as pd

DATASET = "nfl_spread"
HYPOTHESIS = ("Public money overbets big favourites, so double-digit spreads are shaded too far. When the closing "
              "spread is >= 10 points, bet the underdog ATS.")
SOURCE = "widely repeated betting lore (big-favourite fade); Levitt 2004 books shade against public bias"


def fit(train):
    return None


def bet(model, slate):
    sp = slate["spread_line"]  # + = home favoured
    pick = ((sp >= 10) & (slate["selection"] == "away")) | ((sp <= -10) & (slate["selection"] == "home"))
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

import pandas as pd

DATASET = "nfl_moneyline"
HYPOTHESIS = ("Favourite-longshot bias: heavy moneyline underdogs are overpriced, so heavy favourites "
              "(decimal odds <= 1.333, i.e. -300 or shorter) win more often than their price implies. Bet them.")
SOURCE = "favourite-longshot bias literature (Griffith 1949; Kalshi: Burgi, Deng & Whelan 2025)"


def fit(train):
    return None


def bet(model, slate):
    pick = slate["dec_odds"] <= 1.3334
    return pd.Series(pick.fillna(False).astype(float).to_numpy(), index=slate.index)

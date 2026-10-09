import pandas as pd

DATASET = "nfl_total"
HYPOTHESIS = ("Division rivals know each other's schemes; games score below the closing total. "
              "Bet the under in every divisional game.")
SOURCE = "scripts/nfl_scan.py (div_game total, q_bh 0.02 over 72 dev tests); widely published angle"


def fit(train):
    return None


def bet(model, slate):
    pick = slate["div_game"].astype(bool) & (slate["selection"] == "under")
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

import numpy as np
import pandas as pd

DATASET = "kalshi_temp"
HYPOTHESIS = ("Favourite-longshot bias: brackets quoted at a YES ask <= 5c resolve YES less often than "
              "the price implies; buy NO (at 1 - YES bid, fee included) on every such bracket.")
SOURCE = "registry:S001 (favourite-longshot) applied to Kalshi weather"


def fit(train):
    return None


def bet(model, slate):
    pick = (slate["selection"] == "no") & (slate["yes_ask"] <= 0.05)
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

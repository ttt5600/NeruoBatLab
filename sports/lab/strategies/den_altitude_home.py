import pandas as pd

DATASET = "nfl_spread"
HYPOTHESIS = ("Altitude: visiting teams at Denver (5,280 ft) tire late; if the spread only prices a "
              "generic home edge, DEN covers at home. Bet DEN ATS in non-neutral home games. "
              "(User idea 2026-10-15: location features.)")
SOURCE = ("Common betting-folklore claim that Denver's altitude adds home edge; no specific paper "
          "verified -- treat as an unsourced hypothesis.")


def fit(train):
    return None


def bet(model, slate):
    spot = (slate["home"] == "DEN") & ~slate["neutral"].astype(bool)
    pick = spot & (slate["selection"] == "home")
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

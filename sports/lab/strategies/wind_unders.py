"""NFL unders in high wind at outdoor stadiums."""
import pandas as pd

DATASET = "nfl_total"
HYPOTHESIS = ("Totals under-adjust for wind >= 15 mph at outdoor games; bet the under. "
              "Caveat: wind is the observed kickoff reading, a forecast proxy.")
SOURCE = "sports/README.md wind finding; widely published angle"


def fit(train):
    return None


def bet(model, slate):
    windy = (slate["wind_mph"] >= 15) & (~slate["is_dome"].astype(bool))
    return pd.Series(((slate["selection"] == "under") & windy).astype(float).to_numpy(), index=slate.index)

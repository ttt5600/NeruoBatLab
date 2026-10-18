import pandas as pd

DATASET = "nfl_total"
HYPOTHESIS = ("Prime-time games (kickoff >= 8pm ET) draw the most recreational money, which leans to overs; "
              "bet the under.")
SOURCE = "widely repeated online claim (prime-time unders)"


def fit(train):
    return None


def bet(model, slate):
    et = pd.to_datetime(slate["start_utc"], utc=True).dt.tz_convert("America/New_York")
    pick = (et.dt.hour >= 20) & (slate["selection"] == "under")
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

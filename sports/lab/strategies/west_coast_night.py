import pandas as pd

DATASET = "nfl_spread"
HYPOTHESIS = ("Circadian edge: Pacific-time teams playing East/Central teams at night (kickoff >= 8pm ET) "
              "cover. Bet the Pacific team ATS.")
SOURCE = "AASM/SLEEP 2013 (66% ATS 1970-2011): https://aasm.org/nfl-teams-on-west-coast-may-have-circadian-edge-in-night-games/"

PACIFIC = {"SF", "SEA", "OAK", "LV", "SD", "LAC", "LA"}


def fit(train):
    return None


def bet(model, slate):
    et = pd.to_datetime(slate["start_utc"], utc=True).dt.tz_convert("America/New_York")
    night = et.dt.hour >= 20
    hw, aw = slate["home"].isin(PACIFIC), slate["away"].isin(PACIFIC)
    pick = night & (((hw & ~aw) & (slate["selection"] == "home")) | ((aw & ~hw) & (slate["selection"] == "away")))
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

import pandas as pd

DATASET = "nfl_spread"
HYPOTHESIS = ("Circadian penalty, mirror of west_coast_night: a Pacific-time team playing AWAY at an "
              "Eastern-time venue with kickoff before 2pm ET plays at <= 10-11am body-clock time. "
              "Bet the Eastern HOME team ATS. (User idea 2026-10-15: location features.)")
SOURCE = ("Smith, Guilleminault & Efron 1997, Sleep 20(9): circadian rhythms and NFL performance; "
          "AASM 2013 follow-up: https://aasm.org/nfl-teams-on-west-coast-may-have-circadian-edge-in-night-games/")

PACIFIC = {"SF", "SEA", "OAK", "LV", "SD", "LAC", "LA"}
EASTERN = {"ATL", "BAL", "BUF", "CAR", "CIN", "CLE", "DET", "IND", "JAX", "MIA", "NE", "NYG", "NYJ",
           "PHI", "PIT", "TB", "WAS"}


def fit(train):
    return None


def bet(model, slate):
    et = pd.to_datetime(slate["start_utc"], utc=True).dt.tz_convert("America/New_York")
    early = et.dt.hour < 14
    spot = early & slate["away"].isin(PACIFIC) & slate["home"].isin(EASTERN) & ~slate["neutral"].astype(bool)
    pick = spot & (slate["selection"] == "home")
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

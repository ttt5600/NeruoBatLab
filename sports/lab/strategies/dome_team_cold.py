import pandas as pd

DATASET = "nfl_spread"
HYPOTHESIS = ("Teams that play home games in domes are unaccustomed to cold; when a dome team plays an outdoor "
              "road game below 35F, the line under-prices it. Bet the (non-dome) home team ATS.")
SOURCE = "betting lore (dome teams in the cold); dome status learned from training seasons only"


def fit(train):
    home = train.drop_duplicates("opp_id")
    share = home.groupby("home")["is_dome"].mean()
    return set(share[share >= 0.75].index)


def bet(model, slate):
    cold = (~slate["is_dome"]) & (slate["temp_f"] < 35)
    pick = cold & slate["away"].isin(model) & ~slate["home"].isin(model) & (slate["selection"] == "home")
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

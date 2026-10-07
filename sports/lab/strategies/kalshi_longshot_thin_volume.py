import pandas as pd

DATASET = "kalshi_temp"
HYPOTHESIS = ("Favourite-longshot bias (cheap YES brackets resolve YES less than their price implies, "
              "ledger: kalshi_fade_longshots, p=0.687, closest to breakeven of anything tried) should be "
              "strongest where it doesn't pay anyone to arbitrage it away: thin-volume brackets. This "
              "restricts the known YES-ask<=5c fade rule to the bottom tercile of same-city trading volume "
              "(volume_to_decision, tercile fit on earlier quarters) instead of firing on every cheap YES "
              "regardless of liquidity -- a liquidity-friction mechanism, not a re-run of the same rule.")
SOURCE = "registry:S001 (favourite-longshot) + agent (liquidity-friction conditioning)"
MIN_CITY_N = 100


def fit(train):
    counts = train.groupby("city")["volume_to_decision"].size()
    city_cut = train.groupby("city")["volume_to_decision"].quantile(0.33)
    city_cut = city_cut.where(counts >= MIN_CITY_N)
    pooled_cut = float(train["volume_to_decision"].quantile(0.33))
    return {"city_cut": city_cut, "pooled_cut": pooled_cut}


def bet(model, slate):
    cut = slate["city"].map(model["city_cut"]).fillna(model["pooled_cut"])
    pick = (slate["selection"] == "no") & (slate["yes_ask"] <= 0.05) & (slate["volume_to_decision"] <= cut)
    return pd.Series(pick.astype(float).to_numpy(), index=slate.index)

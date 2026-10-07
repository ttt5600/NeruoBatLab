import pandas as pd

DATASET = "mlb_k_props"
HYPOTHESIS = "Strikeout overs are overpriced for starters on short rest (<= 4 days); bet the under on each start's main line."
SOURCE = "registry:S014"


def fit(train):
    return None


def bet(model, slate):
    gap = (slate["p_over_mkt"] - 0.5).abs()
    main = gap == gap.groupby([slate["game_pk"], slate["pitcher_id"]]).transform("min")
    cand = slate[main & (slate["rest_days"] <= 4) & (slate["selection"] == "under")]
    keep = cand.drop_duplicates(["game_pk", "pitcher_id"]).index  # one bet per start
    s = pd.Series(0.0, index=slate.index)
    s[keep] = 1.0
    return s

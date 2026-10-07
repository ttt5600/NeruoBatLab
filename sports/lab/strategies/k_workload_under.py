import pandas as pd

DATASET = "mlb_k_props"
HYPOTHESIS = ("Lines anchor on trailing batters faced while workloads shrink; bet the under on the main line "
              "when the starter averaged < 85 pitches over his last 3 starts.")
SOURCE = "registry:S023"


def fit(train):
    return None


def bet(model, slate):
    gap = (slate["p_over_mkt"] - 0.5).abs()
    main = gap == gap.groupby([slate["game_pk"], slate["pitcher_id"]]).transform("min")
    cand = slate[main & (slate["pitches_last3"] < 85) & (slate["selection"] == "under")]
    keep = cand.drop_duplicates(["game_pk", "pitcher_id"]).index  # one bet per start
    s = pd.Series(0.0, index=slate.index)
    s[keep] = 1.0
    return s

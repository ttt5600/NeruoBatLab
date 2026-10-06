"""Control: bet every home moneyline. Should lose roughly the vig."""
import pandas as pd

DATASET = "nfl_moneyline"
HYPOTHESIS = "Control: no edge expected; measures the cost of the vig."
SOURCE = "control"


def fit(train):
    return None


def bet(model, slate):
    return pd.Series((slate["selection"] == "home").astype(float).to_numpy(), index=slate.index)

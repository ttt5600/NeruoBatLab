"""Adversarial test: tries to read the cached lockbox table. Must be REJECTED."""
import pandas as pd

DATASET = "nfl_total"
HYPOTHESIS = "cheat"
SOURCE = "test"


def fit(train):
    return pd.read_parquet("../data/lab/nfl_total.parquet")


def bet(model, slate):
    return pd.Series(0.0, index=slate.index)

import numpy as np
import pandas as pd

DATASET = "kalshi_temp"
HYPOTHESIS = (
    "Kalshi's bracket quotes are set at 10pm the night before, pinned close to that evening's "
    "lead-1-day forecast (fc_max_lead1). fc_max_lead2 is the prior day's forecast for the same "
    "target date, so fc_max_lead1 - fc_max_lead2 measures how much new information arrived "
    "overnight. If quotes are even slightly sticky and don't fully react to a late, large "
    "revision, the side the revision moved toward should be underpriced in proportion to the "
    "revision's size. This fits only a per-city revision-size threshold (top tercile of "
    "|fc_max_lead1 - fc_max_lead2|, from earlier quarters) and requires the bracket's own price "
    "to still be <= 50c -- it is a same-day information-lag momentum signal using a single "
    "model's own day-over-day change, not a re-fit of the forecast's absolute bias or variance, "
    "which already lost twice on this dataset."
)
SOURCE = "agent"
MARGIN = 2.0
PRICE_CEIL = 0.50
REVISION_PCTL = 0.67
MIN_CITY_N = 100


def fit(train):
    days = train.drop_duplicates("event_ticker").dropna(subset=["fc_max_lead1", "fc_max_lead2"]).copy()
    if len(days) < 150:
        return None
    abs_rev = (days["fc_max_lead1"] - days["fc_max_lead2"]).abs()
    counts = days.groupby("city").size()
    city_cut = abs_rev.groupby(days["city"]).quantile(REVISION_PCTL)
    city_cut = city_cut.where(counts >= MIN_CITY_N)
    pooled_cut = float(abs_rev.quantile(REVISION_PCTL))
    return {"city_cut": city_cut, "pooled_cut": pooled_cut}


def bet(model, slate):
    s = pd.Series(0.0, index=slate.index)
    if model is None:
        return s
    st = slate["strike_type"].to_numpy()
    tail = (st == "greater") | (st == "less")
    lead1 = slate["fc_max_lead1"].to_numpy(float)
    lead2 = slate["fc_max_lead2"].to_numpy(float)
    revision = lead1 - lead2
    cut = slate["city"].map(model["city_cut"]).fillna(model["pooled_cut"]).to_numpy(float)
    big = np.abs(revision) >= cut
    floor_s = slate["floor_strike"].to_numpy(float)
    cap_s = slate["cap_strike"].to_numpy(float)
    sel = slate["selection"].to_numpy()
    is_yes = sel == "yes"

    # Does the newest forecast (lead1) clear this row's selection, with a safety margin?
    favors_yes = np.where(st == "greater", lead1 > floor_s + 1 + MARGIN, lead1 < cap_s - 1 - MARGIN)
    favors_no = np.where(st == "greater", lead1 < floor_s + 1 - MARGIN, lead1 > cap_s - 1 + MARGIN)
    favors_this = np.where(is_yes, favors_yes, favors_no)

    # Did the overnight revision move in the direction that would confirm this call?
    need_pos = np.where(st == "greater", is_yes, ~is_yes)
    dir_ok = np.where(need_pos, revision > 0, revision < 0)

    price = slate["ask"].to_numpy(float)
    pick = tail & big & favors_this & dir_ok & (price <= PRICE_CEIL) & np.isfinite(revision)
    s[pick] = 1.0
    return s

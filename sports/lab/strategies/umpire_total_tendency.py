import numpy as np
import pandas as pd
from scipy import optimize, stats

DATASET = "mlb_total"
HYPOTHESIS = ("Home-plate umpires differ in strike-zone size, which moves walk rate and pitch counts and "
              "therefore scoring; crews are known well before the close but the posted total does not key "
              "off the umpire. Empirical-Bayes shrinkage (prior strength k=40 games, chosen in advance as "
              "roughly a season of starts) of each umpire's historical mean(actual minus close) from "
              "training folds only is applied as a mean shift in a NegBin total-runs model; bets only "
              "umpires with >=30 training starts where the shrunk effect implies EV >= 3 percent at the "
              "close.")
SOURCE = "agent (focus: umpire total tendencies)"

K_PRIOR = 40.0
MIN_TRAIN_GAMES = 30
EV_MIN = 0.03


def fit(train):
    g = train[train["selection"] == "over"].copy()
    g["resid"] = g["actual"].to_numpy(float) - g["ou_close"].to_numpy(float)
    global_drift = float(g["resid"].mean())
    by_ump = g.groupby("officials")["resid"].agg(["mean", "count"])
    shrunk = (by_ump["count"] / (by_ump["count"] + K_PRIOR)) * (by_ump["mean"] - global_drift)
    effect = shrunk.to_dict()
    n_train = by_ump["count"].to_dict()
    mu = g["ou_close"].to_numpy(float) + global_drift + \
        g["officials"].map(effect).fillna(0.0).to_numpy(float)
    mu = np.clip(mu, 0.5, None)
    y = g["actual"].to_numpy(float)

    def nb(log_r):
        r = np.exp(log_r)
        return -stats.nbinom.logpmf(y, r, r / (r + mu)).sum()

    r = float(np.exp(optimize.minimize_scalar(nb, bounds=(0.0, 6.0), method="bounded").x))
    return {"effect": effect, "n_train": n_train, "global_drift": global_drift, "r": r}


def bet(model, slate):
    effect = slate["officials"].map(model["effect"]).fillna(0.0).to_numpy(float)
    n_train = slate["officials"].map(model["n_train"]).fillna(0.0).to_numpy(float)
    mu = np.clip(slate["ou_close"].to_numpy(float) + model["global_drift"] + effect, 0.5, None)
    r = model["r"]
    line = slate["ou_close"].to_numpy(float)
    p_over = stats.nbinom.sf(np.floor(line), r, r / (r + mu))
    p_push = np.where(line == np.floor(line), stats.nbinom.pmf(line, r, r / (r + mu)), 0.0)
    p_under = 1.0 - p_over - p_push
    p_side = np.where(slate["selection"].to_numpy() == "over", p_over, p_under)
    ev = p_side * slate["dec_odds"].to_numpy() + p_push - 1.0
    known = n_train >= MIN_TRAIN_GAMES
    s = pd.Series(np.where(known & (ev >= EV_MIN), 1.0, 0.0), index=slate.index)
    best = pd.Series(ev, index=slate.index).groupby(slate["opp_id"]).transform("max")
    s[pd.Series(ev, index=slate.index) < best] = 0.0
    return s

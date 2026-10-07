import numpy as np
import pandas as pd
from scipy import optimize, stats

DATASET = "mlb_total"
HYPOTHESIS = ("Wrigley Field's small, exposed outfield is famous for amplifying wind far beyond the "
              "league-wide effect the already-tested weather model averages across every park; a NegBin "
              "GLM adds one Wrigley-only wind interaction term on top of the same global temperature and "
              "wind controls, and bets only Wrigley games where that extra term implies EV >= 3 percent "
              "at the close.")
SOURCE = "agent (focus: park x wind, e.g. Wrigley)"

OUT = {"Out To CF": 1.0, "Out To LF": 1.0, "Out To RF": 1.0,
       "In From CF": -1.0, "In From LF": -1.0, "In From RF": -1.0}
EV_MIN = 0.03


def _is_wrigley(df):
    return df["venue"].fillna("").str.contains("Wrigley", case=False)


def _X(df):
    sealed = df["condition"].isin(["Dome", "Roof Closed"]).to_numpy()
    temp = (df["temp_f"].fillna(72).to_numpy() - 72.0) / 10.0
    wind = df["wind_mph"].fillna(0).to_numpy() * df["wind_dir"].map(OUT).fillna(0.0).to_numpy() / 10.0
    temp[sealed] = 0.0
    wind[sealed] = 0.0
    wrig_wind = wind * _is_wrigley(df).to_numpy()
    return np.column_stack([np.ones(len(df)), temp, wind, wrig_wind])


def fit(train):
    g = train[train["selection"] == "over"]
    X, y = _X(g), g["actual"].to_numpy(float)
    off = np.log(g["ou_close"].to_numpy(float))

    def nll(b):
        eta = off + X @ b
        return -(y * eta - np.exp(eta)).sum()

    def grad(b):
        return -X.T @ (y - np.exp(off + X @ b))

    b = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x
    mu = np.exp(off + X @ b)

    def nb(log_r):
        r = np.exp(log_r)
        return -stats.nbinom.logpmf(y, r, r / (r + mu)).sum()

    r = float(np.exp(optimize.minimize_scalar(nb, bounds=(0.0, 6.0), method="bounded").x))
    return {"b": b, "r": r}


def bet(model, slate):
    mu = np.exp(np.log(slate["ou_close"].to_numpy(float)) + _X(slate) @ model["b"])
    r = model["r"]
    line = slate["ou_close"].to_numpy(float)
    p_over = stats.nbinom.sf(np.floor(line), r, r / (r + mu))
    p_push = np.where(line == np.floor(line), stats.nbinom.pmf(line, r, r / (r + mu)), 0.0)
    p_under = 1.0 - p_over - p_push
    p_side = np.where(slate["selection"].to_numpy() == "over", p_over, p_under)
    ev = p_side * slate["dec_odds"].to_numpy() + p_push - 1.0
    wrig = _is_wrigley(slate).to_numpy()
    s = pd.Series(np.where(wrig & (ev >= EV_MIN), 1.0, 0.0), index=slate.index)
    best = pd.Series(ev, index=slate.index).groupby(slate["opp_id"]).transform("max")
    s[pd.Series(ev, index=slate.index) < best] = 0.0
    return s

import numpy as np
import pandas as pd
from scipy import optimize, stats

DATASET = "mlb_total"
HYPOTHESIS = ("The already-tested weather model controls for temperature and wind but not sky/precipitation "
              "condition; overcast or rainy air is denser and more humid, which should suppress fly-ball "
              "carry below what temperature and wind alone predict, and is a secondary factor the market "
              "may price less precisely than the two headline variables. A NegBin GLM adds cloud and "
              "precipitation dummies (matched by keyword on the condition string) on top of the same "
              "temperature/wind controls, and bets only open-air games where one of those two dummies is "
              "active and the extra term implies EV >= 3 percent at the close.")
SOURCE = "agent (focus: condition/precipitation beyond temp+wind)"

OUT = {"Out To CF": 1.0, "Out To LF": 1.0, "Out To RF": 1.0,
       "In From CF": -1.0, "In From LF": -1.0, "In From RF": -1.0}
EV_MIN = 0.03


def _flags(df):
    cond = df["condition"].fillna("")
    sealed = cond.isin(["Dome", "Roof Closed"]).to_numpy()
    precip = cond.str.contains("Rain|Drizzle|Shower|Snow", case=False, na=False).to_numpy()
    cloud = cond.str.contains("Cloud|Overcast", case=False, na=False).to_numpy() & ~precip
    return sealed, precip & ~sealed, cloud & ~sealed


def _X(df):
    sealed, precip, cloud = _flags(df)
    temp = (df["temp_f"].fillna(72).to_numpy(float) - 72.0) / 10.0
    wind = df["wind_mph"].fillna(0).to_numpy(float) * df["wind_dir"].map(OUT).fillna(0.0).to_numpy() / 10.0
    temp[sealed] = 0.0
    wind[sealed] = 0.0
    return np.column_stack([np.ones(len(df)), temp, wind, precip.astype(float), cloud.astype(float)])


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
    _, precip, cloud = _flags(slate)
    mu = np.exp(np.log(slate["ou_close"].to_numpy(float)) + _X(slate) @ model["b"])
    r = model["r"]
    line = slate["ou_close"].to_numpy(float)
    p_over = stats.nbinom.sf(np.floor(line), r, r / (r + mu))
    p_push = np.where(line == np.floor(line), stats.nbinom.pmf(line, r, r / (r + mu)), 0.0)
    p_under = 1.0 - p_over - p_push
    p_side = np.where(slate["selection"].to_numpy() == "over", p_over, p_under)
    ev = p_side * slate["dec_odds"].to_numpy() + p_push - 1.0
    target = precip | cloud
    s = pd.Series(np.where(target & (ev >= EV_MIN), 1.0, 0.0), index=slate.index)
    best = pd.Series(ev, index=slate.index).groupby(slate["opp_id"]).transform("max")
    s[pd.Series(ev, index=slate.index) < best] = 0.0
    return s

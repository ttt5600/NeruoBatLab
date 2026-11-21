"""Variant check on VALIDATION only: MR + 10-8 style dominance term (rounds with |s| > tau count extra)."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import judges as J
def rs_fn(theta, inv, X, nf):
    w, k, g, tau = theta[:-3], theta[-3], theta[-2], theta[-1]
    s = X @ w
    pts = 2 * expit(s) - 1 + g * (expit(4 * (s - tau)) - expit(4 * (-s - tau)))
    return k * np.bincount(inv, weights=pts, minlength=nf)
def nll(theta, inv, X, y, nf):
    p = np.clip(expit(rs_fn(theta, inv, X, nf)), 1e-9, 1 - 1e-9)
    w = theta[:-3]
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() + 1e-3 * w @ w
u, inv, X, y = J.pack(J.fit)
th0 = np.r_[J.fit_round(J.fit), 0.5, 2.5]
res = minimize(nll, th0, args=(inv, X, y, len(u)), method="L-BFGS-B", bounds=[(None, None)] * (len(th0) - 2) + [(0, 3), (0.5, 6)])
uv, iv, Xv, yv = J.pack(J.val)
p = expit(rs_fn(res.x, iv, Xv, len(uv)))
print("VAL MR+dominance", J.score(p, yv), "g,tau", np.round(res.x[-2:], 3))

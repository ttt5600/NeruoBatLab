"""Blue-wave-adjusted NYT needle: rebuild P(Dem wins) as Normal(dem_margin_median + shift, sd), sd from the
margin_quantile IQR (/1.349). shift=0 must reproduce NYT win_probability (checked in snapshot)."""
import math

PARTY = {"DEM": "D", "GOP": "R", "REP": "R"}


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def race_model(r):
    t = r.get("top_reporting_unit") or {}
    est = t.get("nyt_model_estimates") or {}
    mq = est.get("margin_quantile")
    sh, wp = {}, {}
    for c in t.get("candidates") or []:
        e = c.get("nyt_model_estimates") or {}
        pa = PARTY.get(e.get("party_caucus"))
        if pa and e.get("voteshare_quantile"):
            sh[pa] = e["voteshare_quantile"][2]
            wp[pa] = wp.get(pa, 0) + float(e.get("win_probability") or 0)
    if not mq or len(mq) < 5 or "D" not in sh or "R" not in sh:
        return None
    sd = max((mq[3] - mq[1]) / 1.349, 1e-4)
    ve = est.get("votes_estimated") or 0
    rem = max(0.0, min(1.0, (est.get("votes_remaining") or 0) / ve)) if ve else 1.0
    return {"dm": sh["D"] - sh["R"], "sd": sd, "wp": wp, "rem": rem}


def p_dem(mod, shift):
    """shift = Dem margin swing on the UNCOUNTED vote, so it moves the final margin by shift * share remaining
    (a flat shift flipped near-final FL-13 from D 0.008 to 0.898)."""
    return phi((mod["dm"] + shift * mod["rem"]) / mod["sd"])

"""Edge math: the fund's one calculator for every edge claim (standard: notes/EDGE_MATH.md).

Library + CLI. Pure python (stdlib only; numpy optional for the drawdown simulation). No orders, no network.

    edge_math.py bet --p 0.92 --price 0.90 --venue forecastex              # full table for one bet
    edge_math.py bet --wins 209 --n 219 --price 0.8877 --venue forecastex  # p and its Wilson LCB from a record
    edge_math.py bet --p 0.55 --odds -110 --venue book                     # sportsbook (american or decimal odds)
    edge_math.py devig --odds -120 +100                                    # proportional / power / Shin
    edge_math.py clv --bet-prob 0.50 0.52 --close-prob 0.53 0.51           # CLV in prob points + significance
    edge_math.py power --mu 0.02 --sigma 1.0                               # bets needed to detect an edge
    edge_math.py ruin --p 0.95 --price 0.90 --frac 0.5 --level 0.5         # P(ever losing half), max drawdown sim

Conventions
- A binary contract pays $1. `price` is what you pay per contract before fees (0-1). Sportsbook odds are converted to
  the same form: cost c = 1/decimal_odds per $1 returned. All-in cost c = price + fee per contract.
- p = your probability that the bought side wins. Net odds b = (1 - c)/c (profit per $1 staked if it wins).
- EV per $ staked = p/c - 1. EV per contract = p - c. Break-even probability = c.
- Kelly fraction of bankroll f* = (b p - q)/b = (p - c)/(1 - c) (Kelly 1956; Thorp 2006). Fund rule: plug in the
  LOWER confidence bound of p (Wilson), then take a fraction (1/4 default, 1/2 max), then apply the per-bet,
  per-night (correlation) and liquidity caps.
- Fees: Kalshi taker 0.07*C*p*(1-p), maker 0.0175*C*p*(1-p) only on series with fee_type quadratic_with_maker_fees
  (weather/mentions are 'quadratic' = maker 0; notes/kalshi_temp_maker.md); Kalshi rounds the per-ORDER fee up to the
  next cent (`round_fee=True`). Polymarket US taker 0.0695*C*p*(1-p) (notes/venues.md, eff. 2026-10-19). ForecastEx
  $0.01 per contract (notes/forecastex_wx.md). Sportsbooks: the vig is inside the odds, fee 0.
"""
from __future__ import annotations

import argparse
import math
import random
from statistics import NormalDist

N01 = NormalDist()

# ------------------------------------------------------------------------------------------------ prices / odds

def american_to_decimal(a: float) -> float:
    if -100 < a < 100:
        raise ValueError(f"american odds must be <= -100 or >= +100, got {a}")
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def decimal_to_american(d: float) -> float:
    if d <= 1:
        raise ValueError("decimal odds must be > 1")
    return (d - 1) * 100 if d >= 2 else -100 / (d - 1)


def prob_to_american(p: float) -> float:
    return decimal_to_american(1 / p)


def implied_prob(x: float) -> float:
    """Raw implied probability (vig included) of one quote: contract price (0-1), decimal (1-50] or American."""
    if 0 < x < 1:
        return float(x)
    if 1 < x <= 50:
        return 1 / x
    return 1 / american_to_decimal(x)


# ------------------------------------------------------------------------------------------------ de-vig

def overround(probs) -> float:
    """Book margin: sum of implied probabilities - 1 (e.g. -110/-110 -> 0.0476)."""
    return sum(probs) - 1


def devig_proportional(probs) -> list[float]:
    """Divide each implied prob by the sum. Puts the margin on every outcome in proportion to its price, which
    UNDER-states favourites and OVER-states longshots (the favourite-longshot bias lives in the longshot), so it
    fakes longshot CLV (FUND_BOOK lesson 6)."""
    s = sum(probs)
    return [x / s for x in probs]


def _bisect(f, lo, hi, tol=1e-12, it=200):
    flo = f(lo)
    for _ in range(it):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def devig_power(probs) -> list[float]:
    """Find k with sum(q_i^k) = 1 (Clarke et al. 2017). k > 1 removes more margin from longshots."""
    if abs(sum(probs) - 1) < 1e-12:
        return list(probs)
    k = _bisect(lambda k: sum(q ** k for q in probs) - 1, 0.2, 10.0)
    return [q ** k for q in probs]


def devig_shin(probs) -> list[float]:
    """Shin (1993) insider-trading model, closed form per outcome given z (Jullien & Salanie 1994 / Strumbelj 2014):
    p_i = (sqrt(z^2 + 4(1-z) q_i^2 / S) - z) / (2(1-z)), S = sum q_i; z solved so sum p_i = 1."""
    s = sum(probs)
    if abs(s - 1) < 1e-12:
        return list(probs)

    def ps(z):
        return [(math.sqrt(z * z + 4 * (1 - z) * q * q / s) - z) / (2 * (1 - z)) for q in probs]

    z = _bisect(lambda z: sum(ps(z)) - 1, 0.0, 0.99)
    return ps(z)


DEVIG = {"proportional": devig_proportional, "power": devig_power, "shin": devig_shin}


def devig(quotes, method: str = "power") -> list[float]:
    """quotes: every outcome's price/odds from ONE book (american, decimal or contract prices)."""
    return DEVIG[method]([implied_prob(x) for x in quotes])


# ------------------------------------------------------------------------------------------------ fees / cost

VENUES = ("kalshi", "kalshi_maker", "kalshi_maker_fee", "polymarket_us", "forecastex", "book", "none")


def fee_per_contract(price: float, venue: str, contracts: int = 1, round_fee: bool = False) -> float:
    """Fee per contract in $. round_fee: Kalshi charges ceil(fee for the whole order to the cent)."""
    p = price
    if venue == "kalshi":
        tot = 0.07 * contracts * p * (1 - p)
    elif venue == "kalshi_maker_fee":
        tot = 0.0175 * contracts * p * (1 - p)
    elif venue == "polymarket_us":
        tot = 0.0695 * contracts * p * (1 - p)
    elif venue == "forecastex":
        tot = 0.01 * contracts
    elif venue in ("kalshi_maker", "book", "none"):
        tot = 0.0
    else:
        raise ValueError(f"unknown venue {venue}; one of {VENUES}")
    if round_fee and venue.startswith("kalshi"):
        tot = math.ceil(round(tot * 100, 9)) / 100
    return tot / contracts


def all_in_cost(price: float | None = None, venue: str = "book", odds: float | None = None,
                contracts: int = 1, round_fee: bool = False) -> float:
    """Cost per $1 of payout including fees. Give a contract `price` or sportsbook `odds`."""
    if odds is not None:
        return implied_prob(odds)
    return price + fee_per_contract(price, venue, contracts, round_fee)


def break_even_prob(cost: float) -> float:
    return cost


def net_odds(cost: float) -> float:
    """b: profit per $1 staked on a win."""
    return (1 - cost) / cost


def ev_per_dollar(p: float, cost: float) -> float:
    return p / cost - 1


def ev_per_contract(p: float, cost: float) -> float:
    return p - cost


def sigma_per_dollar(p: float, cost: float) -> float:
    """SD of the per-$ return of one binary bet."""
    return math.sqrt(p * (1 - p)) / cost


# ------------------------------------------------------------------------------------------------ confidence

def wilson(k: float, n: float, conf: float = 0.95, one_sided: bool = False) -> tuple[float, float]:
    """Wilson score interval for k successes in n. one_sided=True returns the (lower, upper) one-sided bounds at conf."""
    if n <= 0:
        return 0.0, 1.0
    z = N01.inv_cdf(conf if one_sided else 1 - (1 - conf) / 2)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def z_score(mean: float, se: float) -> float:
    return mean / se if se > 0 else float("inf")


def p_one_sided(z: float) -> float:
    return 1 - N01.cdf(z)


# ------------------------------------------------------------------------------------------------ Kelly / growth

def kelly(p: float, cost: float) -> float:
    """Full Kelly fraction f* = (b p - q)/b = (p - c)/(1 - c); negative = don't bet."""
    b = net_odds(cost)
    return (b * p - (1 - p)) / b


def log_growth(f: float, p: float, cost: float) -> float:
    """Expected log growth per bet g = p ln(1 + f b) + q ln(1 - f)."""
    if f <= 0:
        return 0.0
    if f >= 1:
        return float("-inf")
    b = net_odds(cost)
    return p * math.log1p(f * b) + (1 - p) * math.log1p(-f)


def log_growth_var(f: float, p: float, cost: float) -> float:
    if f <= 0:
        return 0.0
    b = net_odds(cost)
    a, l = math.log1p(f * b), math.log1p(-f)
    m = p * a + (1 - p) * l
    return p * (a - m) ** 2 + (1 - p) * (l - m) ** 2


def stake_fraction(p_lcb: float, cost: float, frac: float = 0.25, cap: float = 0.05) -> float:
    """Fund sizing: frac x Kelly on the LOWER bound of p, clipped to [0, cap]."""
    return max(0.0, min(cap, frac * kelly(p_lcb, cost)))


def risk_of_ruin(p: float, cost: float, f: float, level: float = 0.5) -> float:
    """P(bankroll EVER falls to `level` x start) under the diffusion approximation:
    P = level^(2 g / v), g = mean log growth per bet, v = its variance (Thorp 2006 s.7; MacLean-Thorp-Ziemba 2011).
    At full Kelly this is ~level (P(ever halve) ~ 1/2); at fraction k of Kelly ~ level^(2/k - 1)."""
    g, v = log_growth(f, p, cost), log_growth_var(f, p, cost)
    if f <= 0 or v == 0:
        return 0.0
    if g <= 0:
        return 1.0
    return level ** (2 * g / v)


def simulate_drawdown(p: float, cost: float, f: float, n_bets: int = 1000, n_paths: int = 2000,
                      seed: int = 0) -> dict:
    """Monte Carlo of n_bets independent bets at fixed fraction f: expected / median / 95th pct max drawdown,
    median final multiple, P(final < start)."""
    rng = random.Random(seed)
    b = net_odds(cost)
    up, dn = math.log1p(f * b), math.log1p(-f)
    mdds, finals = [], []
    for _ in range(n_paths):
        x = peak = 0.0
        mdd = 0.0
        for _ in range(n_bets):
            x += up if rng.random() < p else dn
            if x > peak:
                peak = x
            elif peak - x > mdd:
                mdd = peak - x
        mdds.append(1 - math.exp(-mdd))
        finals.append(math.exp(x))
    mdds.sort()
    finals.sort()
    return {"exp_max_dd": sum(mdds) / n_paths, "median_max_dd": mdds[n_paths // 2],
            "p95_max_dd": mdds[int(0.95 * n_paths)], "median_final": finals[n_paths // 2],
            "p_final_below_1": sum(v < 1 for v in finals) / n_paths}


# ------------------------------------------------------------------------------------------------ power / CLV

def n_required(mu: float, sigma: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Bets to detect a mean per-bet return mu > 0 with SD sigma (one-sided): n = ((z_a + z_b) sigma / mu)^2."""
    if mu <= 0:
        return float("inf")
    return ((N01.inv_cdf(1 - alpha) + N01.inv_cdf(power)) * sigma / mu) ** 2


def n_required_roi(roi: float, cost: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Bets to detect a target ROI per $ on binary bets at all-in cost c (p implied by the ROI: p = c(1+roi))."""
    p = min(cost * (1 + roi), 1 - 1e-9)
    return n_required(roi, sigma_per_dollar(p, cost), alpha, power)


def clv_prob_points(bet_prob: float, close_prob: float) -> float:
    """CLV in probability points: de-vigged CLOSING prob of your side minus the de-vigged prob you paid
    (positive = you beat the close)."""
    return close_prob - bet_prob


def clv_test(clvs) -> dict:
    """Mean CLV, SE, z and one-sided p (H0: mean CLV <= 0). Cluster duplicates before calling (protocol rule 4)."""
    xs = list(clvs)
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("nan")
    z = z_score(m, se) if n > 1 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "se": se, "z": z, "p": p_one_sided(z) if n > 1 else float("nan")}


# ------------------------------------------------------------------------------------------------ the full table

def bet_table(p: float | None = None, price: float | None = None, venue: str = "book", odds: float | None = None,
              wins: int | None = None, n: int | None = None, p_lcb: float | None = None, conf: float = 0.95,
              frac: float = 0.5, cap: float = 0.05, bankroll: float | None = None, night_cap: float = 0.10,
              picks_per_night: int = 1, liquidity: float | None = None, contracts: int = 1,
              round_fee: bool = False, ruin_level: float = 0.5, n_sim: int = 0) -> dict:
    """Every step of the EDGE_MATH standard for one bet. Returns an ordered dict of named rows."""
    if wins is None and n and p is not None:
        wins = p * n                      # --p with --n: Wilson around a rate observed on n bets
    if wins is not None and n:
        p = wins / n if p is None else p
        if p_lcb is None:
            p_lcb = wilson(wins, n, conf)[0]
    if p is None:
        raise ValueError("need --p or --wins/--n")
    if price is None and odds is None:
        raise ValueError("need --price or --odds")
    raw = implied_prob(odds) if odds is not None else price
    fee = 0.0 if odds is not None else fee_per_contract(price, venue, contracts, round_fee)
    c = raw + fee
    out = {"raw_implied_prob": raw, "fee_per_contract": fee, "all_in_cost": c, "break_even_prob": c,
           "net_odds_b": net_odds(c), "fair_american_at_p": prob_to_american(p) if 0 < p < 1 else float("nan"),
           "p": p, "p_lcb": p_lcb, "edge_prob_points": p - c, "ev_per_dollar": ev_per_dollar(p, c),
           "ev_per_contract": ev_per_contract(p, c), "sigma_per_dollar": sigma_per_dollar(p, c),
           "kelly_full": kelly(p, c), f"kelly_{frac:g}": frac * kelly(p, c)}
    if p_lcb is not None:
        out["ev_per_dollar_at_lcb"] = ev_per_dollar(p_lcb, c)
        out["kelly_full_at_lcb"] = kelly(p_lcb, c)
        out[f"kelly_{frac:g}_at_lcb"] = frac * kelly(p_lcb, c)
    p_size = p_lcb if p_lcb is not None else p
    f = stake_fraction(p_size, c, frac, cap)
    out["stake_frac_after_pick_cap"] = f
    if picks_per_night > 1 and f * picks_per_night > night_cap:
        f = night_cap / picks_per_night
    out["stake_frac_after_night_cap"] = f
    if bankroll:
        dollars = f * bankroll
        out["stake_usd"] = dollars
        if liquidity is not None:
            dollars = min(dollars, liquidity * c)
            out["stake_usd_after_liquidity"] = dollars
        out["contracts"] = math.floor(dollars / c) if odds is None else float("nan")
    out["log_growth_per_bet"] = log_growth(f, p, c)
    out[f"risk_of_ruin_to_{ruin_level:g}"] = risk_of_ruin(p, c, f, ruin_level)
    out["n_bets_to_detect_ev"] = n_required(ev_per_dollar(p, c), sigma_per_dollar(p, c))
    if n_sim:
        out.update({f"sim{n_sim}_{k}": v for k, v in simulate_drawdown(p, c, f, n_sim, 1000).items()})
    return out


def fmt_table(d: dict) -> str:
    w = max(len(k) for k in d)
    rows = []
    for k, v in d.items():
        if v is None:
            s = "-"
        elif isinstance(v, int):
            s = str(v)
        elif abs(v) >= 1000 or v != v or v in (float("inf"), float("-inf")):
            s = f"{v:,.0f}" if v == v and abs(v) != float("inf") else str(v)
        else:
            s = f"{v:+.4f}" if k.startswith(("ev", "edge", "kelly", "clv")) else f"{v:.4f}"
        rows.append(f"{k:<{w}}  {s}")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bet")
    b.add_argument("--p", type=float)
    b.add_argument("--wins", type=int)
    b.add_argument("--n", type=int)
    b.add_argument("--p-lcb", type=float)
    b.add_argument("--conf", type=float, default=0.95)
    b.add_argument("--price", type=float, help="contract price 0-1 (before fee)")
    b.add_argument("--odds", type=float, help="sportsbook american or decimal odds")
    b.add_argument("--venue", default="book", choices=VENUES)
    b.add_argument("--frac", type=float, default=0.5, help="Kelly fraction (fund: 0.25 default, 0.5 max)")
    b.add_argument("--cap", type=float, default=0.05, help="per-pick cap, fraction of bankroll")
    b.add_argument("--night-cap", type=float, default=0.10)
    b.add_argument("--picks-per-night", type=int, default=1)
    b.add_argument("--bankroll", type=float)
    b.add_argument("--liquidity", type=float, help="contracts available at the price")
    b.add_argument("--contracts", type=int, default=1)
    b.add_argument("--round-fee", action="store_true")
    b.add_argument("--sim", type=int, default=0, help="also simulate max drawdown over this many bets")
    d = sub.add_parser("devig")
    d.add_argument("--odds", type=float, nargs="+", required=True)
    c = sub.add_parser("clv")
    c.add_argument("--bet-prob", type=float, nargs="+", required=True)
    c.add_argument("--close-prob", type=float, nargs="+", required=True)
    pw = sub.add_parser("power")
    pw.add_argument("--mu", type=float)
    pw.add_argument("--sigma", type=float)
    pw.add_argument("--roi", type=float)
    pw.add_argument("--cost", type=float)
    r = sub.add_parser("ruin")
    r.add_argument("--p", type=float, required=True)
    r.add_argument("--price", type=float, required=True)
    r.add_argument("--venue", default="none", choices=VENUES)
    r.add_argument("--frac", type=float, default=0.25, help="fraction of full Kelly")
    r.add_argument("--level", type=float, default=0.5)
    r.add_argument("--bets", type=int, default=1000)
    a = ap.parse_args(argv)

    if a.cmd == "bet":
        t = bet_table(a.p, a.price, a.venue, a.odds, a.wins, a.n, a.p_lcb, a.conf, a.frac, a.cap, a.bankroll,
                      a.night_cap, a.picks_per_night, a.liquidity, a.contracts, a.round_fee, n_sim=a.sim)
        print(fmt_table(t))
        verdict = "NO BET (EV <= 0 at the point estimate)" if t["ev_per_dollar"] <= 0 else (
            "NO BET (Kelly on the lower bound <= 0: edge not established)" if t["stake_frac_after_pick_cap"] <= 0
            else "BET at the sizing rows above (still needs: prereg/sealed status, liquidity, night total)")
        if t["p_lcb"] is None:
            verdict += " | WARNING: sized on the POINT estimate; give --wins/--n or --p-lcb (standard step 6)"
        print(f"VERDICT: {verdict}")
    elif a.cmd == "devig":
        raw = [implied_prob(x) for x in a.odds]
        print(f"raw implied {[round(x, 4) for x in raw]}  overround {overround(raw):+.4f}")
        for m in DEVIG:
            print(f"{m:<13}" + "  ".join(f"{x:.4f}" for x in devig(a.odds, m)))
    elif a.cmd == "clv":
        cl = [clv_prob_points(x, y) for x, y in zip(a.bet_prob, a.close_prob)]
        print(fmt_table({f"clv_{k}" if k in ("mean",) else k: v for k, v in clv_test(cl).items()}))
    elif a.cmd == "power":
        n = n_required_roi(a.roi, a.cost) if a.roi is not None else n_required(a.mu, a.sigma)
        print(f"bets needed (one-sided alpha .05, power .8): {n:,.0f}")
    elif a.cmd == "ruin":
        cst = all_in_cost(a.price, a.venue)
        f = a.frac * kelly(a.p, cst)
        print(f"cost {cst:.4f}  full Kelly {kelly(a.p, cst):+.4f}  f used {f:.4f}  g/bet {log_growth(f, a.p, cst):.6f}")
        print(f"P(ever at {a.level:g}x start) {risk_of_ruin(a.p, cst, f, a.level):.4f}")
        print(fmt_table(simulate_drawdown(a.p, cst, max(f, 0), a.bets)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

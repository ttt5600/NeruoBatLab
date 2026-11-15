"""Exp B: calibration of the last pre-start Kalshi mid vs outcomes, and vs the de-vigged book close (read-only, cached data).
Run from sports/: ../.venv_quant/bin/python results/clv_calibration/calib.py
Outputs results/clv_calibration/{markets.parquet, calib.csv, book.csv, report.txt}."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R = OUT.parent / "kalshi_openers_xleague"
sys.path.insert(0, str(R))
from kx import side_x  # noqa: E402

RNG = np.random.default_rng(20261115)
B = 2000
SPREAD_MAX, PLO, PHI = 0.04, 0.03, 0.97


def dev(o):  # american odds -> raw implied prob
    o = float(o)
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


rows = []
for lg in ["nfl", "nhl", "cfb", "cbb", "nba"]:
    src = OUT if lg == "nba" else R  # nba fetched here by nba_fetch.py
    m = pd.read_parquet(src / f"{lg}_kmarkets.parquet")
    g = pd.read_parquet(src / f"{lg}_games.parquet")
    c = pd.read_parquet(src / f"{lg}_candles.parquet")
    g["start_s"] = (g.start_utc - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)  # traps: no //1e9
    j = c.merge(m[["ticker", "mk", "result", "key"]], on="ticker").merge(
        g[["key", "start_s", "date_et", "total_close", "over_close", "under_close", "home_spread_close"]], on="key")
    pre = j[j.end <= j.start_s].sort_values("end").groupby("ticker").tail(1).copy()
    pre["age_min"] = (pre.start_s - pre.end) / 60
    pre["lg"] = lg
    rows.append(pre)
d = pd.concat(rows, ignore_index=True)
d = d[d.result.isin(["yes", "no"]) & d.bid.notna() & d.ask.notna()]
d["y"] = (d.result == "yes").astype(float)
d["mid"] = (d.bid + d.ask) / 2
d["spr"] = d.ask - d.bid
assert (d.age_min >= 0).all() and d.age_min.max() < 24 * 60, d.age_min.describe()  # lookahead / staleness guard
d.to_parquet(OUT / "markets.parquet", index=False)
f = d[(d.spr <= SPREAD_MAX + 1e-9) & d.mid.between(PLO, PHI) & (d.age_min <= 90)].copy()
f["resid"] = f.y - f.mid
print("markets with pre-start candle", len(d), "after filter", len(f), "games", f.key.nunique(), "dates", f.date_et.nunique())


def cboot(df, col="resid", cl="date_et", w=None):
    """mean of col and date-cluster bootstrap percentile 95% CI."""
    gs = df.groupby(cl)[col].agg(["sum", "count"])
    s, n = gs["sum"].to_numpy(), gs["count"].to_numpy()
    k = len(s)
    idx = RNG.integers(0, k, size=(B, k))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return s.sum() / n.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int(n.sum()), k, df.key.nunique()


def verdict(lo, hi):
    if lo > -0.0025 and hi < 0.0025:
        return "ACCEPT"
    if lo > 0 or hi < 0:
        return "REJECT"
    return "UNRESOLVED"


out = []
f["bin"] = pd.cut(f.mid, [0.03, 0.15, 0.35, 0.65, 0.85, 0.97], include_lowest=True).astype(str)
specs = [("all", f)] + [(f"{lg}", s) for lg, s in f.groupby("lg")] + [(f"{lg}|{mk}", s) for (lg, mk), s in f.groupby(["lg", "mk"])]
specs += [(f"bin {b}", s) for b, s in f.groupby("bin")]
specs += [(f"{lg}|{mk}|bin {b}", s) for (lg, mk, b), s in f.groupby(["lg", "mk", "bin"])]
# one market per game (strike nearest 50c) -> removes within-game ladder duplication
near = f.assign(dd=(f.mid - 0.5).abs()).sort_values("dd").groupby(["key", "mk"]).head(1)
specs += [(f"near50 {lg}|{mk}", s) for (lg, mk), s in near.groupby(["lg", "mk"])]
for name, s in specs:
    mu, lo, hi, n, k, ng = cboot(s)
    out.append(dict(cut=name, n_mkts=n, n_games=ng, n_dates=k, bias_c=100 * mu, lo_c=100 * lo, hi_c=100 * hi,
                    halfw_c=50 * (hi - lo), verdict=verdict(lo, hi)))
cal = pd.DataFrame(out)
cal.to_csv(OUT / "calib.csv", index=False)

# --- fallback: Kalshi close mid vs de-vigged book close at the same line (totals: strike == total_close) ---
t = d[(d.mk == "total") & d.over_close.notna() & d.under_close.notna() & (d.spr <= SPREAD_MAX + 1e-9)
      & (d.age_min <= 90)].copy()
t = t[np.isclose(t.strike, t.total_close)]
po, pu = t.over_close.map(dev), t.under_close.map(dev)
t["book_p"] = po / (po + pu)  # multiplicative de-vig
t["kb"] = t.mid - t.book_p  # Kalshi minus book (YES = over)
t["resid_book"] = t.y - t.book_p
bo = []
for name, s in [("all", t)] + list(t.groupby("lg")):
    mu, lo, hi, n, k, ng = cboot(s, "kb")
    mu2, lo2, hi2, *_ = cboot(s, "resid_book")
    sd = s.kb.std()
    bo.append(dict(cut=name, n=n, n_dates=k, kal_minus_book_c=100 * mu, lo_c=100 * lo, hi_c=100 * hi, sd_c=100 * sd,
                   mad_c=100 * s.kb.abs().mean(), verdict_vs_book=verdict(lo, hi),
                   outcome_minus_book_c=100 * mu2, ob_lo=100 * lo2, ob_hi=100 * hi2))
bk = pd.DataFrame(bo)
bk.to_csv(OUT / "book.csv", index=False)
pd.set_option("display.width", 250)
with open(OUT / "report.txt", "w") as fh:
    fh.write(cal.round(2).to_string() + "\n\n" + bk.round(2).to_string() + "\n")
print(cal.round(2).to_string())
print(bk.round(2).to_string())

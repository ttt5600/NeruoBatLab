"""Write the 6 edgelab specs for k50_strike (K50 + LO per league). Run from sports/."""
import yaml

base = dict(rule="edgelab.rules.forward:unwired", price_model="taker", min_n=100,
            kill=dict(kind="clv_negative_significant", after_n=100, p=0.05), fee=dict(rounding="exact"))
pre = "k50_strike (notes/k50_strike.md, pre-registered 2026-11-13 before any outcome). "
DT = {"nba": "L32b live decision tick (>=30 min pre-tip); last 1-min candle <= T",
      "cfb": "Kalshi first-tight hourly candle (>=30 min pre-kick)",
      "nfl": "later of L33 model-cache time and Kalshi first-tight hour (>=30 min pre-kick); hourly candles"}
FAM = {"nba": ("nba_totals", "nba_totals_open"), "cfb": ("cfb_lines", "kalshi_openers_cfb"),
       "nfl": ("nfl_lines", "kalshi_openers_nfl")}
model = {"nba": "L32b de-biased model (model - c)", "cfb": "frozen L36 pred_total",
         "nfl": "L33 logger models (total C, spread B home margin)"}
kk = {"nba": "3.0", "cfb": "3.5", "nfl": "2.5 total / 2.0 spread"}
sig = {"nba": "18.61", "cfb": "15.39", "nfl": "13.09 total / 12.51 margin"}
for lg in ("nba", "cfb", "nfl"):
    for r in ("k50", "lo"):
        fam, ds = FAM[lg]
        if r == "k50":
            h = pre + (f"K50-{lg.upper()}: {model[lg]} vs Kalshi's own anchor strike S50 (two-sided strike with mid nearest "
                       f"50c at T). Bet over/home if m - S50 >= k, under/away if <= -k, k = {kk[lg]}; buy S50 at the ask + "
                       f"exact taker fee; graded on final vs S50; CLV = close - S50 signed.")
        else:
            h = pre + (f"LO-{lg.upper()}: {model[lg]} as Normal(m, sigma={sig[lg]}, prior-season sd(final - open)); over "
                       f"every two-sided strike and side EV = P - (ask + fee); bet the max if EV >= 0.03/contract; graded "
                       f"on final; CLV = close - strike signed.")
        sp = dict(id=f"k50_{lg}_{r}_kalshi", family=fam, dataset=ds, hypothesis=h, decision_time=DT[lg],
                  split=f"vault:{ds}",
                  legacy=("new 2026-11-13 (agent k50_strike). Dev = same games as K1/K-CFB/K-NFL dev (disclosed). Pass "
                          "dev only if ROI p < family bar AND CLV > 0; GO forward if ROI>0, CLV>0, n>=15; kill ROI CI "
                          "upper < 0 or CLV <= 0 with n >= 20."), **base)
        yaml.safe_dump(sp, open(f"edgelab/specs/{sp['id']}.yaml", "w"), sort_keys=False, width=1000)
        print(sp["id"])

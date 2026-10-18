"""European soccer 1X2 (football-data.co.uk): price shopping vs the market / vs Pinnacle. DEV only.

    python scripts/soccer_fd.py build           # download 22 divisions x 2012-13..2025-26 (cached) -> data/soccer_fd.parquet
    python scripts/soccer_fd.py dev [--ledger]

Data: per match, pre-close odds collected by football-data.co.uk (Fri afternoon for weekend games,
Tue for midweek): Pinnacle PS*, Bet365 B365*, best price across books BbMx*/Max*, average BbAv*/Avg*;
and Pinnacle CLOSING PSC*. Outcome FTR (H/D/A).
Pre-registered 2026-10-18, before any download. DEV = seasons 2012-13 .. 2018-19; SEALED = 2019-20+.
All bets at PRE-CLOSE prices (what was available when the snapshot was taken).
  S1 kaunitz05   : p_cons = 1/Avg (with margin); bet the outcome if Max > 1/(p_cons - 0.05) (Kaunitz et al. 2017)
  S2 max_vs_pin3 : p_pin = Pinnacle pre-close de-vigged (proportional); bet at Max if Max*p_pin - 1 >= 0.03
  S3 b365_vs_pin3: same, at Bet365's price only (one soft book a US-style bettor could actually use)
  S4 fav_close13 : back the outcome priced <= 1.30 at Pinnacle CLOSE, at the close price (favourite-longshot)
Dataset soccer_fd, 4 trials -> bar p < 0.0125. Bootstrap over matches. Also printed (not trials): CLV of S1-S3 bets
= mean(price taken * Pinnacle-close fair prob) - 1.
Caveat: Max includes stale / error quotes that books void ("palps"); S3 is the honest version.

Added 2026-10-18 after S1-S4 were scored:
  S5 b365_clv (POST-HOC: CLV was printed alongside S3): H0 mean CLV <= 0 for the S3 bets; bootstrap over matches.
  S6 wh_vs_pin3 (pre-registered before the WH column was built): William Hill pre-close price, bet if
     WH*p_pin - 1 >= 0.03. PRIMARY statistic = CLV (bootstrap p, H0 mean CLV <= 0); ROI reported too.
  Why CLV: at mean odds ~6.5 ROI has sd ~4.6% after 3,000 bets; CLV vs the calibrated Pinnacle close is far
  less noisy and is how +EV bettors measure edge (Buchdahl). Assumes Pinnacle close is the truth.

2026-10-18 WITHDRAWN: S6's CLV pass (+3.97%, p<.001) is a de-vig artifact. Proportional de-vig of Pinnacle CLOSE
overstates longshots (hit/fair 0.80 at odds 10-20, 0.67 at 20+; power de-vig ~1.00 below 10). With power de-vig
S6 CLV = -0.32% (p .79), S3/S5 B365 = +0.50% (p .022). The selection rule had the same bias -> it picked longshots.
  S7 b365_pow3 / S8 wh_pow3 (flagged: method chosen after seeing dev calibration): select with POWER de-vig of
     Pinnacle pre-close, book price * p_pow - 1 >= 0.03 and price < 10; PRIMARY = CLV vs POWER de-vig of Pinnacle close.
  S9 bw_pow3 (pre-registered 2026-10-18 before the BW column was built): same rule at bwin (Entain, co-owner of BetMGM).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports.http import fetch

OUT = ROOT / "data" / "soccer_fd.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "soccer_fd"
DIVS = ["E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "D1", "D2", "I1", "I2", "SP1", "SP2",
        "F1", "F2", "N1", "B1", "P1", "T1", "G1"]
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2012, 2026)]
DEV = SEASONS[:7]  # 1213..1819
SIDES = "HDA"


def cmd_build() -> int:
    fs = []
    for s in SEASONS:
        for div in DIVS:
            try:
                b = fetch(f"https://www.football-data.co.uk/mmz4281/{s}/{div}.csv", binary=True)
            except Exception as e:
                print(s, div, "missing", str(e)[:60])
                continue
            d = pd.read_csv(io.BytesIO(b), encoding="latin1", on_bad_lines="skip")
            d = d.dropna(subset=["FTR"]) if "FTR" in d else d.iloc[0:0]
            row = {"season": s, "div": div, "date": d.get("Date"), "home": d.get("HomeTeam"), "away": d.get("AwayTeam"),
                   "ftr": d["FTR"]}
            for x in SIDES:
                for key, cands in (("pin", [f"PS{x}"]), ("b365", [f"B365{x}"]), ("wh", [f"WH{x}"]), ("bw", [f"BW{x}"]), ("max", [f"Max{x}", f"BbMx{x}"]),
                                   ("avg", [f"Avg{x}", f"BbAv{x}"]), ("pinc", [f"PSC{x}"])):
                    col = next((c for c in cands if c in d), None)
                    row[f"{key}_{x}"] = pd.to_numeric(d[col], errors="coerce") if col else np.nan
            fs.append(pd.DataFrame(row))
        print(s, "done", flush=True)
    out = pd.concat(fs, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out)} matches)")
    return 0


def _fair(d, key):
    inv = np.column_stack([1 / d[f"{key}_{x}"] for x in SIDES])
    return inv / inv.sum(1, keepdims=True)


def _power(d, key):
    """p_i = (1/o_i)^k, k solved so the three sum to 1 (puts more of the margin on longshots)."""
    from scipy.optimize import brentq
    inv = np.column_stack([1 / d[f"{key}_{x}"].to_numpy() for x in SIDES])
    out = np.empty_like(inv)
    for j, r in enumerate(inv):
        k = brentq(lambda k: (r ** k).sum() - 1, 0.3, 4.0)
        out[j] = r ** k
    return out


def bets(d: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    p_pin, p_pinc = _fair(d, "pin"), _fair(d, "pinc")
    p_pow = _power(d, "pin") if name.endswith("pow3") else None
    if p_pow is not None:
        p_pinc = _power(d, "pinc")  # CLV measured against the calibrated (power) close
    for i, x in enumerate(SIDES):
        won = (d.ftr == x).to_numpy()
        mx, avg, b365, pinc = (d[f"{k}_{x}"].to_numpy() for k in ("max", "avg", "b365", "pinc"))
        if name == "kaunitz05":
            price, sel = mx, mx > 1 / (1 / avg - 0.05)
        elif name == "max_vs_pin3":
            price, sel = mx, mx * p_pin[:, i] - 1 >= 0.03
        elif name in ("b365_vs_pin3", "b365_clv"):
            price, sel = b365, b365 * p_pin[:, i] - 1 >= 0.03
        elif name in ("b365_pow3", "wh_pow3", "bw_pow3"):
            bk = d[f"{name.split('_')[0]}_{x}"].to_numpy()
            price, sel = bk, (bk * p_pow[:, i] - 1 >= 0.03) & (bk < 10)
        elif name == "wh_vs_pin3":
            wh = d[f"wh_{x}"].to_numpy()
            price, sel = wh, wh * p_pin[:, i] - 1 >= 0.03
        else:
            price, sel = pinc, pinc <= 1.30
        sel = sel & np.isfinite(price) & (price > 1)
        rows.append(pd.DataFrame({"match": d.index[sel], "season": d.season[sel].to_numpy(), "price": price[sel],
                                  "won": won[sel], "clv": price[sel] * p_pinc[sel, i] - 1}))
    b = pd.concat(rows)
    b["pnl"] = np.where(b.won, b.price - 1, -1.0)
    return b


def score(b, rng) -> dict:
    pm = b.groupby("match")["pnl"].sum().to_numpy()
    boots = np.array([rng.choice(pm, len(pm)).sum() for _ in range(3000)])
    per = b.groupby("season")["pnl"].sum().round(2)
    return {"bets": int(len(b)), "staked": float(len(b)), "units": float(b.pnl.sum()), "roi": float(b.pnl.mean()),
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)),
            "p": float(np.mean(boots <= 0)), "folds_positive": int((per > 0).sum()), "per_fold": {str(k): v for k, v in per.items()}}


def clv_p(b, rng) -> float:
    pm = b.groupby("match")["clv"].sum().to_numpy()
    boots = np.array([rng.choice(pm, len(pm)).mean() for _ in range(3000)])
    return float(np.mean(boots <= 0))


def cmd_followup() -> int:
    d = pd.read_parquet(OUT)
    d = d[d.season.isin(DEV)].reset_index(drop=True)
    need = [f"{k}_{x}" for k in ("pin", "pinc") for x in SIDES]
    d = d.dropna(subset=need)
    d = d[(d[need] > 1).all(1)].reset_index(drop=True)
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    names = [a for a in sys.argv[2:] if not a.startswith("--")] or ["b365_clv", "wh_vs_pin3"]
    for name in names:
        b = bets(d, name)
        r = score(b, np.random.default_rng(0))
        pc = clv_p(b, np.random.default_rng(1))
        sha = hashlib.sha256(("soccer_fd:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:11s} bets {r['bets']:5d}  CLV {b.clv.mean():+.2%} (p {pc:.4f})  ROI {r['roi']:+6.2%} (p {r['p']:.3f})  "
              f"seasons ROI+ {r['folds_positive']}/7  CLV>0 seasons {int((b.groupby('season').clv.mean() > 0).sum())}/7  "
              f"avg price {b.price.mean():.2f}")
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"soccer_{name}", "sha": sha,
               "dataset": DATASET, "hypothesis": f"scripts/soccer_fd.py ({name}){' POST-HOC' if name == 'b365_clv' else ''}",
               "source": "CLV as edge test (Buchdahl)", "status": "ok", "trial_number": len(shas), **r,
               "clv": float(b.clv.mean()), "p_clv": pc}
        rec["p"] = pc  # primary statistic for these two is CLV
        recs.append(rec)
    print(f"{DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return 0


def cmd_sealed() -> int:
    """ONE-TIME sealed score (2019-20+), user's explicit yes required. Pre-registered 2026-10-18 before any sealed
    row was scored: trials named on the command line (S7 b365_pow3 and/or S8 wh_pow3), same rules as dev.
    PASS iff mean CLV (power de-vig of Pinnacle close) > 0 with bootstrap p < 0.05 / (#soccer_fd promotions incl.
    these) AND ROI > 0 is reported alongside (ROI is noisy; CLV is the primary statistic)."""
    if "--confirm" not in sys.argv:
        print("sealed scoring is the user's decision; pass --confirm")
        return 1
    names = [a for a in sys.argv[2:] if a in ("b365_pow3", "wh_pow3", "bw_pow3")]
    if not names:
        print("name the trial(s): b365_pow3 wh_pow3")
        return 1
    lock = ROOT / "lab" / "lockbox.jsonl"
    prior = [json.loads(l) for l in lock.read_text().splitlines() if l.strip()]
    d = pd.read_parquet(OUT)
    d = d[~d.season.isin(DEV)].reset_index(drop=True)
    need = [f"{k}_{x}" for k in ("pin", "pinc") for x in SIDES]
    d = d.dropna(subset=need)
    d = d[(d[need] > 1).all(1)].reset_index(drop=True)
    n_prom = sum(r["dataset"] == DATASET for r in prior) + len(names)
    bar = 0.05 / n_prom
    for name in names:
        sha = hashlib.sha256(("soccer_fd:" + name).encode()).hexdigest()[:12]
        if any(r["sha"] == sha for r in prior):
            print(f"{name} already scored on the lockbox; refusing")
            continue
        b = bets(d, name)
        r = score(b, np.random.default_rng(0))
        pc = clv_p(b, np.random.default_rng(1))
        verdict = "PASS" if (pc < bar and b.clv.mean() > 0) else "FAIL"
        with lock.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"soccer_{name}",
                                "sha": sha, "dataset": DATASET, "promotion_number": n_prom, **r,
                                "clv": float(b.clv.mean()), "p_clv": pc, "verdict": verdict}) + "\n")
        print(f"SEALED {name}: matches {len(d)} bets {r['bets']} CLV {b.clv.mean():+.2%} (p {pc:.4f}) ROI {r['roi']:+.2%} "
              f"(p {r['p']:.3f}) seasons ROI+ {r['folds_positive']}/{len(r['per_fold'])} -> {verdict} (bar {bar:.4f})")
    return 0


def cmd_dev() -> int:
    d = pd.read_parquet(OUT)
    d = d[d.season.isin(DEV)].reset_index(drop=True)
    need = [f"{k}_{x}" for k in ("pin", "max", "avg", "pinc") for x in SIDES]
    d = d.dropna(subset=need)
    d = d[(d[need] > 1).all(1)].reset_index(drop=True)
    over = lambda k: np.median(sum(1 / d[f"{k}_{x}"] for x in SIDES) - 1)
    print(f"DEV matches {len(d)} ({d.season.min()}..{d.season.max()}, {d['div'].nunique()} divs); overround median: "
          f"pin {over('pin'):+.3f} pinc {over('pinc'):+.3f} avg {over('avg'):+.3f} max {over('max'):+.3f} (max < 0 = arb-ish)")
    pc = _fair(d, "pinc")
    hit = np.column_stack([(d.ftr == x) for x in SIDES])
    print(f"sanity: Pinnacle-close Brier {np.mean((pc - hit) ** 2):.4f}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    for name in ("kaunitz05", "max_vs_pin3", "b365_vs_pin3", "fav_close13"):
        b = bets(d, name)
        r = score(b, np.random.default_rng(0))
        sha = hashlib.sha256(("soccer_fd:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:13s} bets {r['bets']:6d}  units {r['units']:+8.1f}  ROI {r['roi']:+6.2%}  "
              f"95%CI [{r['ci_lo']:+.0f}, {r['ci_hi']:+.0f}]  p {r['p']:.4f}  seasons+ {r['folds_positive']}/7  "
              f"CLV {b.clv.mean():+.2%}  avg price {b.price.mean():.2f}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"soccer_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/soccer_fd.py ({name})",
                     "source": "Kaunitz et al. 2017; +EV vs Pinnacle", "status": "ok", "trial_number": len(shas), **r})
    print(f"{DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev, "followup": cmd_followup, "sealed": cmd_sealed}.get(cmd, lambda: print(__doc__) or 1)())

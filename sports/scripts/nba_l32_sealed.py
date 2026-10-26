"""SEALED test L32: NBA simple-model TOTALS vs the OPENING total, scored ONCE on NBA 2026 (= 2025-26).

    ../.venv_quant/bin/python scripts/nba_l32_sealed.py devcheck --season 2025     # dev-only reproduction (safe)
    ../.venv_quant/bin/python scripts/nba_l32_sealed.py sealed --confirm-lockbox    # ONE shot (user said "yes L32" 2026-10-25)

Dev source: scripts/nba_simple_model.py trial T5_tot_strict_open_novig (notes/nba_simple_model.md): 1,190 bets, 55.5%,
no-vig ROI +8.7% [+3.1, +14.3], p .001; per season 2024 +8.5% (654), 2025 +9.0% (536). This file imports that
script's functions unchanged (build / fit_stack / predict / bets / boot) and copies only `markets` (parameterised by
season, plus a provider-name alias). nba_simple_model.py itself is not modified.

=====================================================================================================================
PRE-REGISTRATION (written 2026-10-25, BEFORE any 2025-26 outcome, model output or price comparison was looked at.
Only a schema/count check of lines_2026 / games_2026 was viewed: column names, provider x opener-field counts,
provider x month counts, status/season_type counts. No score, close line, or bet was read or computed.)
=====================================================================================================================
MODEL  T2_STRICT, frozen. Nothing is refit on 2025-26 data, nothing is tuned.
  - BPM-lite player weights: fit on all dev player-seasons 2021-22..2024-25 (results/nba_simple_model_frozen.json "bpm").
  - Totals stack T2_strict: OLS fit on all dev scored games 2022-23..2024-25 (frozen json fits.T2_strict, n 3,445):
    total ~ const + tot0 + Ofull + oAbs_strict + U1 pace + U2 own 3P% + U3 opp 3P% + U4 3PA-rate + U5 b2b, with the
    frozen training z-constants (zc) and standardisation (sc).
  - The script RE-DERIVES both from dev data only (seasons < 2026, same code as dev) and ABORTS unless they equal the
    frozen json to 1e-6 (guards against a silently different frozen file). The frozen values are then used.
  - Features for 2025-26 walk forward by ET date: every game on date D uses only games dated < D (team ridges,
    player decayed box rates with half-life 180 d so 2024-25 is the prior, team style half-life 60 d). Same code as dev.
  - STRICT availability: a team's players tonight = those who played its previous game (no lookahead).
GAMES  2025-26 regular season, not neutral site, both teams >= 5 earlier games that season (dev `scored` filter).
       Final games only (4 postponed rows drop out).
OPENER Per game, the DraftKings opening total if present, else the ESPN BET opening total -- the dev rule
       (nba_simple_model.markets: DraftKings first). Opener ROW selection is copied from dev unchanged: among that
       game's DraftKings / ESPN BET rows WITH a spread opener, DraftKings first; that row's total_open + over/under
       open prices are used (schema check: 1 row in 2025-26 has a total opener but no spread opener, 1 the reverse).
       Provider alias: ESPN spells it "Draft Kings" in 2025-26 and "DraftKings" in dev -- treated as the same book
       (naming fix only). Expected split from the schema count: ~978 DraftKings, ~251 ESPN BET openers.
       NOTE: notes/nba_simple_model.md s6 had floated "ESPN BET first (O1 order)"; the lead's L32 instruction and the
       dev code both say DraftKings first, so DraftKings first is the pre-registered rule. Per-book splits are reported.
RULE   bet OVER if T2 - opening total >= +3.0, UNDER if <= -3.0; flat 1 unit; one bet per game.
PRICE  proportional de-vig of that book's opening over/under prices (sports.market.devig_two_way); a win pays
       1/p_fair - 1, a push 0, a loss -1. Rows whose de-vigged price is missing are dropped (as in dev).
PRIMARY  mean no-vig P&L per bet (ROI). One-sided game bootstrap, 4,000 resamples (nba_simple_model.boot, seed 0):
         p = share of resampled means <= 0. PASS = p < 0.05 AND ROI > 0. One test, one shot, never re-scored.
SECONDARY (descriptive, no pass/fail): ROI at -110; CLV in points = bet-side move from the open to the SAME book's
         closing total; win rate graded against that same-book closing total; per-book and per-month splits; over/under split.
POWER  dev SE ~2.9% on 1,190 bets; 2025-26 should give ~600 bets, SE ~4%. At a true +4% (expected after shrinkage,
       since T5 failed the whole-ledger bar and the O1 sibling halved on sealed data) power is only ~25%. A miss is
       INCONCLUSIVE, not a refutation. A pass at ~+8% would be consistent with dev.
ONE SHOT  `sealed` needs --confirm-lockbox; it aborts if results/nba_l32_sealed.json exists or lab/lockbox.jsonl
       already has strategy nba_L32_tot_open. It writes results/nba_l32_sealed.json with status "started" BEFORE
       reading any 2025-26 data, so a crash also blocks a re-run (a human must then decide; do not delete it quietly).
DEV CHECK `devcheck --season 2025` runs the identical code path with history < 2025 only (stack fit on 2022-23 +
       2023-24, BPM on seasons < 2025, exactly the dev walk-forward fold) and must reproduce T5's 2025 figure
       (536 bets, +0.0895). It never reads 2026 (asserted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nba_simple_model as nsm  # noqa: E402

from sports.market import devig_two_way  # noqa: E402

ROOT = nsm.ROOT
OUT = ROOT / "results" / "nba_l32_sealed.json"
DEVCHECK = ROOT / "results" / "nba_l32_devcheck.json"
LOCKBOX = ROOT / "lab" / "lockbox.jsonl"
STRATEGY = "nba_L32_tot_open"
SEALED = 2026
FIRST = 2022                      # warm-up season (AMENDMENT in nba_simple_model.py)
STACK_FROM = 2023                 # stacks fit on scored games from 2022-23 on
MODEL, VAR, THR = "T2", "strict", nsm.TOT_X
assert THR == 3.0
BOOK_ALIAS = {"Draft Kings": "DraftKings"}
OPEN_ORDER = ["DraftKings", "ESPN BET"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def markets_season(d: pd.DataFrame, season: int) -> pd.DataFrame:
    """Copy of nsm.markets (totals part) for one season, with the provider alias. Opener DK first, else ESPN BET."""
    ln = pd.read_parquet(nsm.ESPN / f"lines_{season}.parquet")
    ln["provider"] = ln.provider.replace(BOOK_ALIAS)
    bk = ln[~ln.provider.isin(nsm.NOT_BOOKS)]
    c = bk.groupby("event_id").agg(tot_close=("total_close", "median")).reset_index()
    o = bk[bk.provider.isin(OPEN_ORDER)].dropna(subset=["home_spread_open"]).copy()
    o["pri"] = o.provider.map({p: i for i, p in enumerate(OPEN_ORDER)})
    o = o.sort_values("pri", kind="stable").drop_duplicates("event_id")
    o = o[["event_id", "provider", "total_open", "total_close", "over_odds_open", "under_odds_open"]].rename(
        columns={"provider": "open_book", "total_close": "tot_close_same"})
    m = c.merge(o, on="event_id", how="left")
    m["pf_over_open"] = devig_two_way(m.over_odds_open, m.under_odds_open)[0]
    return d.merge(m, on="event_id", how="left")


def same(a: dict, b: dict, path="") -> list:
    """Recursive numeric comparison; returns list of mismatching paths."""
    bad = []
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                bad.append(f"{path}.{k} missing")
            else:
                bad += same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} len"]
        for i, (x, y) in enumerate(zip(a, b)):
            bad += same(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)):
        if not np.isclose(float(a), float(b), rtol=1e-6, atol=1e-6):
            bad.append(f"{path} {a} vs {b}")
    return bad


def run(season: int) -> tuple[pd.DataFrame, dict]:
    """Build features for FIRST..season walk-forward; fit stack on scored STACK_FROM..season-1; bet season."""
    seasons = list(range(FIRST, season + 1))
    g, betas = nsm.build(seasons)
    base = g[g.scored & (g.season >= STACK_FROM)]
    tr = base[base.season < season].dropna(subset=["mu0", "tot0", "dPfull", "Ofull"])
    fit = nsm.fit_stack(tr, MODEL, VAR)
    info = {"stack_n": fit["n"], "stack_params": fit["params"], "sigma": fit["sigma"]}
    if season == SEALED:                                       # must equal the frozen dev coefficients, then use them
        fz = json.loads(nsm.FROZEN.read_text())
        bad = same(betas[SEALED], fz["bpm"], "bpm") + same(
            {k: fit[k] for k in ("params", "zc", "sc", "n")},
            {k: fz["fits"]["T2_strict"][k] for k in ("params", "zc", "sc", "n")}, "T2_strict")
        if bad:
            raise SystemExit(f"ABORT: re-derived dev coefficients differ from frozen json: {bad[:5]}")
        fit = fz["fits"]["T2_strict"]
        info["frozen_match"] = True
    te = base[base.season == season].copy()
    te["T2_strict"] = nsm.predict(fit, te)
    info["scored_games"] = int(len(te))
    d = markets_season(te, season)
    x = nsm.bets(d, "tot", "T2_strict", "total_open", "tot_close_same", "pf_over_open", THR)
    x = x[x.pnl_novig.notna()].copy()
    return x, info


def summarise(x: pd.DataFrame) -> dict:
    prim = nsm.boot(x.pnl_novig, x.season)                        # seed 0, 4000, one-sided
    r = {"bets": prim["bets"], "units": prim.get("units"), "roi_novig": prim["roi"], "ci_lo": prim.get("ci_lo"),
         "ci_hi": prim.get("ci_hi"), "p": prim["p"]}
    r["verdict"] = "PASS" if (r["p"] < 0.05 and r["roi_novig"] > 0) else "FAIL (inconclusive at ~25% power)"
    if len(x):
        r["win_rate_ex_push"] = float(x.win[~x.push].mean())
        r["roi_110"] = float(x.pnl_110.mean())
        c = x.dropna(subset=["clv"])
        r["clv_pts"] = float(c.clv.mean())
        r["clv_n"] = int(len(c))
        res_c = x.side * (x.total - x.tot_close_same)
        r["win_vs_same_close_ex_push"] = float((res_c[res_c != 0] > 0).mean())
        r["by_book"] = {b: {"n": int(len(s)), "roi_novig": float(s.pnl_novig.mean()), "clv": float(s.clv.mean())}
                        for b, s in x.groupby("open_book")}
        r["over_under"] = {k: {"n": int(len(s)), "roi_novig": float(s.pnl_novig.mean())}
                           for k, s in x.groupby(np.where(x.side > 0, "over", "under"))}
        mo = pd.to_datetime(x.start_utc).dt.tz_convert("US/Eastern").dt.strftime("%Y-%m")
        r["units_by_month"] = {k: round(float(v), 3) for k, v in x.pnl_novig.groupby(mo).sum().items()}
    return r


def show(r: dict, info: dict, tag: str):
    print(f"\n{tag}: scored games {info['scored_games']}; stack n {info['stack_n']} sigma {info['sigma']:.2f}")
    print(f"  bets {r['bets']}  no-vig ROI {r['roi_novig']:+.4f} CI [{r.get('ci_lo', float('nan')):+.4f},"
          f"{r.get('ci_hi', float('nan')):+.4f}] p {r['p']:.4f} -> {r['verdict']}")
    for k in ("units", "win_rate_ex_push", "roi_110", "clv_pts", "win_vs_same_close_ex_push", "by_book", "over_under",
              "units_by_month"):
        if k in r:
            print(f"  {k}: {r[k]}")


def cmd_devcheck(season: int) -> int:
    assert season in nsm.DEV and season >= 2024, "devcheck is for dev walk-forward folds 2024/2025 only"
    x, info = run(season)
    r = summarise(x)
    show(r, info, f"DEV CHECK season {season}")
    dev = json.loads((ROOT / "results" / "nba_simple_model.json").read_text())["trials"]["T5_tot_strict_open_novig"]
    want_roi, want_n = dev["per_fold"][str(season)]
    ok = (r["bets"] == want_n) and abs(round(r["roi_novig"], 4) - want_roi) < 1e-9
    wf = pd.read_pickle(nsm.OUTD / "wf.pkl")
    wf = wf[wf.season == season].set_index("event_id").T2_strict
    xs = x.set_index("event_id").T2_strict
    maxdiff = float((xs - wf.reindex(xs.index)).abs().max())
    print(f"\n  dev T5 {season}: n {want_n} roi {want_roi:+.4f} | this path: n {r['bets']} roi {r['roi_novig']:+.4f} "
          f"| max |pred - wf.pkl| on bet games {maxdiff:.2e} -> {'MATCH' if ok and maxdiff < 1e-6 else 'MISMATCH'}")
    DEVCHECK.write_text(json.dumps({"season": season, "match": bool(ok and maxdiff < 1e-6), "dev_per_fold": [want_roi, want_n],
                                    "this": r, "max_pred_diff_vs_wf": maxdiff, "info": info,
                                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1, default=float))
    return 0 if ok else 1


def cmd_sealed(confirm: bool) -> int:
    if not confirm:
        print("sealed: needs --confirm-lockbox and the user's explicit yes (given 2026-10-25: 'yes L32')")
        return 1
    if OUT.exists():
        print(f"ABORT: {OUT} exists -- L32 was already run (or crashed mid-run). Never re-score.")
        return 1
    if LOCKBOX.exists() and any(json.loads(l).get("strategy") == STRATEGY
                                for l in LOCKBOX.read_text().splitlines() if l.strip()):
        print("ABORT: L32 already in lab/lockbox.jsonl -- never re-score")
        return 1
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT.write_text(json.dumps({"status": "started", "ts": ts}))         # one-shot marker before any sealed read
    man = json.dumps(json.loads((ROOT / "data" / "frozen" / "2026-10-14" / "MANIFEST.json").read_text()))
    hashes = {}
    for f in ("games", "lines", "box"):
        h = sha(nsm.ESPN / f"{f}_{SEALED}.parquet")
        hashes[f] = {"sha256": h, "in_frozen_manifest": h in man}
        print(f"{f}_{SEALED} sha256 {h[:12]} in frozen MANIFEST: {h in man}")
    x, info = run(SEALED)
    r = summarise(x)
    show(r, info, f"SEALED L32 season {SEALED}")
    rec = {"ts": ts, "strategy": STRATEGY, "sha": hashlib.sha256(STRATEGY.encode()).hexdigest()[:12],
           "dataset": "nba_simple_model", "status": "done", "data_hashes": hashes, "info": info, **r}
    OUT.write_text(json.dumps(rec, indent=1, default=float))
    with LOCKBOX.open("a") as f:
        f.write(json.dumps({k: v for k, v in rec.items() if k not in ("info",)}, default=float) + "\n")
    x.drop(columns=["loc"], errors="ignore")[["event_id", "date", "home", "away", "open_book", "total_open", "T2_strict",
                                              "side", "pf_over_open", "total", "tot_close_same", "pnl_novig", "pnl_110",
                                              "clv"]].to_csv(ROOT / "results" / "nba_l32_sealed_bets.csv", index=False)
    print(f"wrote {OUT}, results/nba_l32_sealed_bets.csv; lockbox appended")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["devcheck", "sealed"])
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--confirm-lockbox", action="store_true")
    a = ap.parse_args()
    if a.cmd == "devcheck":
        return cmd_devcheck(a.season)
    return cmd_sealed(a.confirm_lockbox)


if __name__ == "__main__":
    raise SystemExit(main())

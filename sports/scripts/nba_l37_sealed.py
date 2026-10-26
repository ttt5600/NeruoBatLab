"""SEALED test L37: NBA mid-game comeback model vs Kalshi KXNBAGAME, scored ONCE on 2025-26 in-game candles.

    ../.venv_quant/bin/python scripts/nba_l37_sealed.py devcheck                    # dev code-path check (2025 playoffs; safe)
    ../.venv_quant/bin/python scripts/nba_l37_sealed.py sealed --confirm-lockbox     # ONE shot (user: "yes one shot tests" 2026-10-26)

Source of the rule and model: scripts/nba_midgame_scan.py (docstring = the frozen L37 forward-paper rule; committed in
2c29645) and results/nba_midgame_fit.json (frozen probit, fit on 2015-16..2024-25 pbp states only). This file IMPORTS
the frozen functions p_trailer / empirical / wilson / fee and the window constants from nba_midgame_scan.py and refuses
to run unless that file, the fit json and the dev neighbour table data/nba_midgame_states.parquet have the sha256 they
had at commit 2c29645 (constants below). The candle/pbp alignment is copied from scripts/nba_l28_sealed.py (same
rule as nba_late_wp.py); nothing is imported from L28.

=====================================================================================================================
PRE-REGISTRATION (written 2026-10-26 BEFORE any 2025-26 Kalshi price was joined to a mid-game state. What had been
viewed of the sealed files: schema, row counts, game counts by month / season type, date range, pbp + spread coverage
by eid. No bid, ask, result, score or spread VALUE of a 2025-26 game was read for this question.)
=====================================================================================================================
SEALED WINDOW  data/nba_ingame.parquet games with ET date >= 2026-01-15 (all season types: 626 regular, 6 play-in,
       81 playoffs = 713 games, 223,826 candle rows; 713/713 in nba_games.parquet for the settlement cross-check).
       Games dated 2025-10-21 .. 2026-01-14 (590) are EXCLUDED. Reason: they were DEV for
       scripts/nba_ingame.py (H1 gap_hold / H2 lag_flip scanned EVERY play of the game against ESPN WP, i.e. Q2-Q4
       mid-game states too) and for scripts/nba_late_wp.py (T1/T2 + calibration, Q4 <= 6:00 / OT, |margin| <= 8).
       One could argue nba_late_wp touched a disjoint region of state space (Q4 <= 6:00 close games vs L37's
       Q2 .. Q4 6:00 with deficits 6-25), but nba_ingame H1/H2 did not restrict the clock, and in any case the prices
       and outcomes of those games were analysed, so the conservative option is taken: every game date used as dev by
       any earlier trial is excluded. The 2025 playoffs (nba_late_candles_2025po.parquet) are dev too (and in-sample
       for the L37 fit, which includes 2024-25 playoffs) and are used only by `devcheck`.
KNOWN LEAK (score side, not price side, flagged in scripts/nba_l28_sealed.py): notes/nba_big_lead.md already reports
       that 2025-26 was a LOW-comeback season by pbp scores (20+ leads blown in 3.6% of games, 25+ in 0.8%), and
       nba_live_scan.py's history pools 2015-16..2025-26. That knowledge favours LEADER buys; L37 can buy either side.
       The trailer/leader split (secondary) must be read with this in mind. Prices were never joined.
DATA   Candles: data/nba_ingame.parquet (end = 1-min candle end, unix s; bid/ask = close YES bid/ask; side = home/away
       market; result = that market's Kalshi settlement). PBP: data/nba_big_lead_plays.parquet (ESPN; eid, seq,
       period, clock_s, hs, as, wall). Coverage (count check): 713/713 sealed games have pbp with wallclock.
SPREAD Pregame expected home margin hm = -(ESPN summary pickcenter[0].spread), from data/nba_big_lead_meta.parquet
       (cached post-game summary = the provider's last pregame line, i.e. the close). Count check: 713/713 sealed games
       have it. This is the same source and sign as the L37 dev fit for 2015-16..2022-23 (nba_comebacks build uses
       -pickcenter where present; verified identical to the states-table spread, corr 1.000 on 2022-23) and the live
       logger's primary source. A game without a spread can never be an entry (counted); no Kalshi-mid fallback is used.
       devcheck uses the dev fit's own spread (data/nba_comebacks/games.parquet, OddsData close for 2024-25), because
       ESPN pickcenter is empty for 2023-24/2024-25.
STATE  Per game, home and away candles are inner-joined on end time. For a candle ending at c: state = the last pbp
       play with wallclock <= c - 60 s (plays ordered by wallclock, then seq). Candles with c - 60 before the first
       play or c after the last play + 120 s have no state. el = (period-1)*720 + (720 - clock_s) for periods 1-4.
       Margin m = hs - as at that play; m == 0 (tied) -> no state.
       Trailer = the team behind; d = |m|; s = trailer's pregame expected margin (hm if home trails, else -hm).
WINDOW (frozen code, nba_midgame_scan.evaluate): period in {2,3,4} AND 720 <= el <= 2520 AND 6 <= d <= 25.
       Q2 any clock (incl. halftime = the Q2 0:00 play), Q3 any clock, Q4 clock >= 360 s. NOTE the frozen docstring
       says "Q4 clock > 360" while the frozen code (which the forward logger runs) includes clock == 360 exactly
       (el <= 2520); the CODE is used here so the sealed test matches the forward test. OT and Q1 never qualify.
MODEL  p_tr = nba_midgame_scan.p_trailer(fit, d, s, el) with the frozen coef; p_ld = 1 - p_tr. Nothing is refit.
EDGE   For each side X in {trailer, leader} with that side's candle CLOSE ask a: 0.01 <= a <= 0.99,
       edge_X = round(p_X - a - 0.07 a (1-a), 4) (exact taker fee, unrounded; rounded to 4 dp as the live code does).
QUOTE  (added for historical candles, per the lead's L37 sealed brief) the bought side's quote must be two-sided and
       tight: bid > 0, ask < 1, ask - bid <= 0.05.
EMPIRICAL AGREEMENT (frozen live condition): nba_midgame_scan.empirical(ST, d, s, el) on the FROZEN dev neighbour
       table data/nba_midgame_states.parquet (seasons 2015..2024 only; asserted): n_emp >= 30 AND
       p_emp_X - a - fee(a) > 0 (p_emp_ld = 1 - p_emp_tr).
ENTRY  the FIRST candle (by end time) in a game where some side X has edge_X >= 0.05 + quote OK + empirical agreement.
       If both sides qualify, the larger edge. ONE entry per game, 1 contract, taker at that close ask, hold to
       settlement, no exits/adds.
SETTLE Kalshi `result` of the bought side's market (yes = 1); cross-check vs data/nba_games.parquet final score
       (mismatches counted); ESPN final used only if the Kalshi result is missing (counted).
       P&L per contract = won - ask - fee(ask).
PRIMARY  mean P&L per contract over entries (= ROI per contract). One-sided bootstrap over GAMES (one entry per
       game): 10,000 resamples, numpy default_rng(0); p = share of resampled means <= 0.
       PASS = mean P&L per contract > 0 AND p < 0.05. One test, one shot, never re-scored.
SECONDARY (descriptive, no pass/fail): ROI per $ staked; win rate vs mean model vs mean ask; mean edge at entry;
       cent-rounded fee sensitivity (ceil(100 fee)/100 on a 1-lot); TRAILER vs LEADER split (n, W, P&L/contract, p);
       by period; settlement cross-check counts; entry lag (candle end - state play wallclock).
       Kalshi calibration in the window: trailer's Kalshi mid (two-sided, spread <= 5c) by bins
       0-5-10-15-20-30-40-50-100c, FIRST candle per game per bin: games, mean mid, mean model, realised, Wilson 95%;
       plus Brier of Kalshi mid vs model on that sample.
POWER  from `devcheck` (dev quantities only): the rule entered 54 of 85 dev window games (64%), so expect ~450
       entries in the 713 sealed games; dev P&L SD 0.52/contract -> SE ~0.025. Power (one-sided .05): ~65% at a
       true +0.05/contract, ~20% at +0.02. A FAIL is weak evidence against a small edge.
       Dev entries were 42/54 LEADER buys (model > Kalshi on leaders), so the KNOWN LEAK above (low-comeback
       sealed season) tilts toward a pass. Read a pass driven by the leader side with that caveat.
ONE SHOT  `sealed` needs --confirm-lockbox; it aborts if results/nba_l37_sealed.json exists or lab/lockbox.jsonl has
       strategy nba_L37_midgame. It writes results/nba_l37_sealed.json with status "started" BEFORE reading any sealed
       file, so a crash also blocks a re-run (a human must then decide; do not delete it quietly).
DEV CHECK `devcheck` runs the identical functions on data/nba_late_candles_2025po.parquet (2024-25 play-in + playoffs,
       the only non-sealed Kalshi NBA games). These games are IN-SAMPLE for the L37 fit, so its P&L is a code-path
       check only, not evidence. It never reads nba_ingame.parquet except eid/date columns for a count.
       Result (2026-10-26): 86 games, 85 with window states, 6 without a dev spread; 54 entries, 32 W,
       P&L/contract -0.014 [-0.153, +0.124], p .58 (FAIL in-sample); trailer 12 (+0.122), leader 42 (-0.053);
       median entry lag 69 s; Kalshi-vs-ESPN settlement mismatches 0; median |trailer mid - model| 0.025;
       Brier on the 1-per-game-per-bin sample: Kalshi mid 0.1811, model 0.1810.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SCAN = ROOT / "scripts" / "nba_midgame_scan.py"
FIT = ROOT / "results" / "nba_midgame_fit.json"
ST = ROOT / "data" / "nba_midgame_states.parquet"
FROZEN_SHA = {  # sha256 at commit 2c29645 (verified 2026-10-26, git status clean)
    SCAN: "ef242e3d5782546ff3037c339851acf5538719c3b940097acdbea883b6e1de50",
    FIT: "9b5fec9325be36a76072d130b3cd1fa017be4ec3843df1ff2dadb3ebe06ed7dd",
    ST: "64defe54510a8e22a27623fa440707044ad08dbb0f8f45dd9edaf566e9a80105",
}
SEALED_CANDLES = ROOT / "data" / "nba_ingame.parquet"
DEV_CANDLES = ROOT / "data" / "nba_late_candles_2025po.parquet"
PLAYS = ROOT / "data" / "nba_big_lead_plays.parquet"
META = ROOT / "data" / "nba_big_lead_meta.parquet"
DEV_GAMES = ROOT / "data" / "nba_comebacks" / "games.parquet"
GAMES = ROOT / "data" / "nba_games.parquet"
OUT = ROOT / "results" / "nba_l37_sealed.json"
BETS = ROOT / "results" / "nba_l37_sealed_bets.csv"
DEVCHECK = ROOT / "results" / "nba_l37_devcheck.json"
LOCKBOX = ROOT / "lab" / "lockbox.jsonl"
STRATEGY = "nba_L37_midgame"
SEALED_FROM = pd.Timestamp("2026-01-15")       # nba_ingame / nba_late_wp DEV ended 2026-01-14
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
LAG, TAIL = 60, 120
MAX_SPREAD = 0.05
N_BOOT, SEED = 10_000, 0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_frozen() -> None:
    bad = [str(p.relative_to(ROOT)) for p, h in FROZEN_SHA.items() if sha(p) != h]
    if bad:
        raise SystemExit(f"ABORT: frozen L37 artifacts changed since 2c29645: {bad}")


check_frozen()
import nba_midgame_scan as L37  # noqa: E402  (hash-verified above)

assert (L37.EDGE_MIN, L37.EMP_MIN_N, L37.D_LO, L37.D_HI, L37.EL_LO, L37.EL_HI) == (0.05, 30, 6, 25, 720, 2520)


def fee(p):
    return 0.07 * np.asarray(p, float) * (1 - np.asarray(p, float))


def fee_rounded(p):
    return np.ceil(np.round(100 * fee(p), 9)) / 100


# --------------------------------------------------------------------------------------------- alignment (from L28)
def load_pbp(eids) -> pd.DataFrame:
    p = pd.read_parquet(PLAYS)
    p = p[p.eid.isin(set(eids))].dropna(subset=["hs", "as", "clock_s", "wall", "period"]).copy()
    p["m"] = p.hs - p["as"]
    p["t"] = (p.wall - EPOCH) // pd.Timedelta(seconds=1)          # unit-safe epoch seconds
    return p.sort_values(["eid", "t", "seq"], kind="stable").reset_index(drop=True)


def candle_states(cd: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, candle end) with both sides' quotes and the state of the last play at or before end - 60 s."""
    out = []
    pg = dict(tuple(pbp.groupby("eid")))
    for eid, c in cd.groupby("eid"):
        gp = pg.get(eid)
        if gp is None or not len(gp) or c.side.nunique() != 2:
            continue
        h = c[c.side == "home"].set_index("end")[["bid", "ask", "result"]]
        a = c[c.side == "away"].set_index("end")[["bid", "ask", "result"]]
        j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner").reset_index()
        wt = gp.t.to_numpy()
        j = j[(j.end - LAG >= wt.min()) & (j.end <= wt.max() + TAIL)]
        if not len(j):
            continue
        idx = np.searchsorted(wt, j.end.to_numpy() - LAG, side="right") - 1
        st = gp.iloc[idx]
        out.append(pd.DataFrame({
            "eid": eid, "end": j.end.to_numpy(), "date": c.date.iloc[0],
            "stype": c.stype.iloc[0] if "stype" in c else -1,
            "bid_h": j.bid_h.to_numpy(float), "ask_h": j.ask_h.to_numpy(float), "res_h": j.result_h.to_numpy(),
            "bid_a": j.bid_a.to_numpy(float), "ask_a": j.ask_a.to_numpy(float), "res_a": j.result_a.to_numpy(),
            "period": st.period.to_numpy(), "clock_s": st.clock_s.to_numpy(float), "m": st.m.to_numpy(),
            "play_t": st.t.to_numpy()}))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------------------------- L37 rule
def window_rows(x: pd.DataFrame, hm: pd.Series, fitd: dict) -> pd.DataFrame:
    """In-window rows (frozen code window) with trailer/leader view, model prob and both sides' edges."""
    x = x[(x.m != 0) & x.period.isin([2, 3, 4])].copy()
    x["el"] = (x.period - 1) * 720 + (720 - x.clock_s)
    x["d"] = x.m.abs()
    x = x[(x.el >= L37.EL_LO) & (x.el <= L37.EL_HI) & (x.d >= L37.D_LO) & (x.d <= L37.D_HI)].copy()
    x["hm"] = x.eid.map(hm)
    x["home_trails"] = x.m < 0
    x["s_tr"] = np.where(x.home_trails, x.hm, -x.hm)
    for k in ("bid", "ask", "res"):
        x[f"{k}_tr"] = np.where(x.home_trails, x[f"{k}_h"], x[f"{k}_a"])
        x[f"{k}_ld"] = np.where(x.home_trails, x[f"{k}_a"], x[f"{k}_h"])
    x["trailer_side"] = np.where(x.home_trails, "home", "away")
    ok = x.s_tr.notna()
    x["p_tr"] = np.nan
    x.loc[ok, "p_tr"] = L37.p_trailer(fitd, x.d[ok].to_numpy(), x.s_tr[ok].to_numpy(), x.el[ok].to_numpy())
    x["p_ld"] = 1 - x.p_tr
    for s in ("tr", "ld"):
        a = x[f"ask_{s}"]
        valid = a.notna() & (a >= 0.01) & (a <= 0.99) & x.p_tr.notna()
        x[f"edge_{s}"] = np.where(valid, np.round(x[f"p_{s}"] - a - fee(a), 4), np.nan)
        b = x[f"bid_{s}"]
        x[f"quote_{s}"] = b.notna() & a.notna() & (b > 0) & (a < 1) & (a - b <= MAX_SPREAD + 1e-9)
    return x.sort_values(["eid", "end"], kind="stable").reset_index(drop=True)


def select(w: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """First qualifying candle per game. Empirical check computed only on model+quote candidates (same first entry)."""
    cand = w[((w.edge_tr >= L37.EDGE_MIN) & w.quote_tr) | ((w.edge_ld >= L37.EDGE_MIN) & w.quote_ld)]
    ents = []
    for eid, g in cand.groupby("eid", sort=False):
        for r in g.itertuples():
            pe, n, _ = L37.empirical(st, r.d, r.s_tr, r.el)
            ok = []
            for s in ("tr", "ld"):
                e, a = getattr(r, f"edge_{s}"), getattr(r, f"ask_{s}")
                if not (e == e and e >= L37.EDGE_MIN and getattr(r, f"quote_{s}")):
                    continue
                pe_s = pe if s == "tr" else (1 - pe if pe == pe else pe)
                if n >= L37.EMP_MIN_N and pe_s == pe_s and pe_s - a - float(fee(a)) > 0:
                    ok.append(s)
            if ok:
                s = max(ok, key=lambda z: getattr(r, f"edge_{z}"))
                ents.append({**r._asdict(), "side": s, "ask": getattr(r, f"ask_{s}"), "bid": getattr(r, f"bid_{s}"),
                             "p_model": getattr(r, f"p_{s}"), "edge": getattr(r, f"edge_{s}"), "emp_tr": pe,
                             "emp_n": n, "result": getattr(r, f"res_{s}"),
                             "team_side": r.trailer_side if s == "tr" else ("away" if r.trailer_side == "home" else "home")})
                break
    e = pd.DataFrame(ents)
    if len(e):
        e = e.drop(columns=["Index"], errors="ignore")
    return e


def settle(ent: pd.DataFrame) -> pd.DataFrame:
    if not len(ent):
        return ent
    g = pd.read_parquet(GAMES, columns=["game_id", "home_score", "away_score"]).drop_duplicates("game_id").set_index("game_id")
    hs, as_ = ent.eid.map(g.home_score), ent.eid.map(g.away_score)
    hw = (hs > as_).to_numpy()
    espn_won = np.where(ent.team_side == "home", hw, ~hw).astype(float)
    espn_won[hs.isna().to_numpy()] = np.nan
    kal = ent.result.map({"yes": 1.0, "no": 0.0}).to_numpy(float)
    won = np.where(np.isnan(kal), espn_won, kal)
    ent = ent.assign(won=won, kalshi_won=kal, espn_won=espn_won)
    ent["fee"] = fee(ent.ask)
    ent["fee_r"] = fee_rounded(ent.ask)
    ent["pnl"] = ent.won - ent.ask - ent.fee
    ent["pnl_r"] = ent.won - ent.ask - ent.fee_r
    ent["lag_s"] = ent.end - ent.play_t
    return ent


# --------------------------------------------------------------------------------------------- scoring
def boot_mean(v, n=N_BOOT, seed=SEED) -> dict:
    v = np.asarray(v, float)
    if not len(v):
        return {"p": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    s = rng.choice(v, (n, len(v))).mean(1)
    return {"p": float(np.mean(s <= 0)), "ci_lo": float(np.quantile(s, .025)), "ci_hi": float(np.quantile(s, .975))}


def summarise(ent: pd.DataFrame, w: pd.DataFrame, x: pd.DataFrame) -> dict:
    r = {"games_with_state": int(x.eid.nunique()), "games_in_window": int(w.eid.nunique()),
         "games_in_window_no_spread": int(w[w.hm.isna()].eid.nunique()),
         "window_candles": int(len(w))}
    # Kalshi calibration: y = Kalshi settlement of the CURRENT trailer's market at that candle
    res_by = w.assign(y=w.res_tr.map({"yes": 1.0, "no": 0.0}))
    t = res_by[res_by.quote_tr & res_by.p_tr.notna()].dropna(subset=["y"]).copy()
    t["mid"] = (t.bid_tr + t.ask_tr) / 2
    t["bin"] = pd.cut(t.mid, [0, .05, .10, .15, .20, .30, .40, .50, 1.0], include_lowest=True)
    one = t.drop_duplicates(["eid", "bin"])
    tab = []
    for b, g in one.groupby("bin", observed=True):
        lo, hi = L37.wilson(int(g.y.sum()), len(g))
        tab.append({"trailer_mid": str(b), "games": int(len(g)), "mean_mid": round(float(g.mid.mean()), 4),
                    "mean_model": round(float(g.p_tr.mean()), 4), "realised": round(float(g.y.mean()), 4),
                    "ci95": [round(lo, 4), round(hi, 4)]})
    r["calibration"] = {"table": tab, "rows": int(len(one)), "games": int(one.eid.nunique()),
                        "brier_kalshi_mid": float(np.mean((one.mid - one.y) ** 2)) if len(one) else None,
                        "brier_model": float(np.mean((one.p_tr - one.y) ** 2)) if len(one) else None,
                        "median_abs_mid_minus_model": float(np.median((t.mid - t.p_tr).abs())) if len(t) else None}
    ent = ent.dropna(subset=["won"]) if len(ent) else ent
    r["entries"] = int(len(ent))
    if not len(ent):
        r["passed"] = False
        return r
    b = boot_mean(ent.pnl)
    stake = float((ent.ask + ent.fee).sum())
    r.update({"wins": int(ent.won.sum()), "pnl_per_contract": float(ent.pnl.mean()), "boot_p": b["p"],
              "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "units": float(ent.pnl.sum()), "staked": stake,
              "roi_per_dollar": float(ent.pnl.sum() / stake), "win_rate": float(ent.won.mean()),
              "mean_model": float(ent.p_model.mean()), "mean_ask": float(ent.ask.mean()),
              "mean_edge": float(ent.edge.mean())})
    r["passed"] = bool(r["pnl_per_contract"] > 0 and r["boot_p"] < 0.05)
    br = boot_mean(ent.pnl_r)
    r["rounded_fee"] = {"pnl_per_contract": float(ent.pnl_r.mean()), "boot_p": br["p"]}
    r["by_side"] = {}
    for s, g in ent.groupby("side"):
        bb = boot_mean(g.pnl)
        r["by_side"]["trailer" if s == "tr" else "leader"] = {
            "n": int(len(g)), "wins": int(g.won.sum()), "pnl_per_contract": float(g.pnl.mean()), "boot_p": bb["p"],
            "mean_model": float(g.p_model.mean()), "mean_ask": float(g.ask.mean()), "win_rate": float(g.won.mean())}
    r["by_period"] = {int(k): {"n": int(len(g)), "wins": int(g.won.sum()), "pnl_per_contract": float(g.pnl.mean())}
                      for k, g in ent.groupby("period")}
    r["by_stype"] = {str(k): {"n": int(len(g)), "pnl_per_contract": float(g.pnl.mean())} for k, g in ent.groupby("stype")}
    r["lag_s_median"] = float(ent.lag_s.median())
    r["settle_kalshi_vs_espn_mismatch"] = int(((ent.kalshi_won != ent.espn_won) & ent.kalshi_won.notna()
                                               & ent.espn_won.notna()).sum())
    r["settle_kalshi_missing"] = int(ent.kalshi_won.isna().sum())
    return r


def show(r: dict, tag: str) -> None:
    print(f"\n== {tag} ==")
    for k, v in r.items():
        if k != "calibration":
            print(f"  {k}: {v}")
    c = r.get("calibration") or {}
    if c.get("table"):
        print(f"  Kalshi calibration, trailer mid (first candle per game per bin; {c['games']} games):")
        print(pd.DataFrame(c["table"]).to_string(index=False))
        print(f"  Brier mid {c['brier_kalshi_mid']:.4f} vs model {c['brier_model']:.4f}; median |mid - model| "
              f"{c['median_abs_mid_minus_model']:.3f}")
    if r.get("entries"):
        print(f"  VERDICT: {'PASS' if r['passed'] else 'FAIL'}  (n {r['entries']}, P&L/contract "
              f"{r['pnl_per_contract']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], one-sided p {r['boot_p']:.4f})")


def run(cd: pd.DataFrame, hm: pd.Series) -> tuple[dict, pd.DataFrame]:
    fitd = json.loads(FIT.read_text())
    st = pd.read_parquet(ST)
    assert st.season.max() <= 2024, "dev neighbour table must be seasons <= 2024-25"
    pbp = load_pbp(cd.eid.unique())
    missing = sorted(set(cd.eid) - set(pbp.eid))
    x = candle_states(cd, pbp)
    w = window_rows(x, hm, fitd)
    ent = settle(select(w, st))
    r = summarise(ent, w, x)
    r["games_candles"] = int(cd.eid.nunique())
    r["games_excluded_no_pbp"] = len(missing)
    return r, ent


def cmd_devcheck() -> int:
    cd = pd.read_parquet(DEV_CANDLES)
    assert cd.date.min() >= pd.Timestamp("2025-04-15") and cd.date.max() < pd.Timestamp("2025-07-01"), "DEV candles only"
    g = pd.read_parquet(DEV_GAMES, columns=["eid", "spread"]).dropna(subset=["eid"]).drop_duplicates("eid")
    hm = g.set_index("eid").spread                                    # dev fit's own spread (= home expected margin)
    r, ent = run(cd, hm)
    show(r, "DEV CHECK: L37 on 2025 play-in + playoffs (in-sample for the fit; code-path check only)")
    if len(ent):
        cols = ["eid", "side", "period", "clock_s", "d", "s_tr", "p_model", "emp_tr", "emp_n", "bid", "ask", "edge",
                "won", "pnl", "lag_s"]
        print(ent[cols].round(4).to_string(index=False))
    n_sealed = int(pd.read_parquet(SEALED_CANDLES, columns=["eid", "date"]).query("date >= @SEALED_FROM").eid.nunique())
    rate = r["entries"] / max(r["games_in_window"], 1)
    n_exp = n_sealed * rate
    sd = float(ent.pnl.std(ddof=1)) if len(ent) > 2 else 0.45
    pw = {"dev_entry_rate_per_window_game": rate, "sealed_games_count_only": n_sealed, "expected_sealed_entries": n_exp,
          "pnl_sd_assumed": sd}
    from scipy.stats import norm
    se = sd / math.sqrt(max(n_exp, 1))
    for mu in (0.02, 0.05):
        pw[f"power_true_pnl_{mu:+.2f}"] = float(1 - norm.cdf(1.645 - mu / se))
    print("\nPOWER (dev-only inputs):", json.dumps(pw, indent=1))
    DEVCHECK.write_text(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dev": r,
                                    "power": pw}, indent=1, default=str))
    print(f"wrote {DEVCHECK}")
    return 0


def cmd_sealed(confirm: bool) -> int:
    if not confirm:
        print("sealed: needs --confirm-lockbox and the user's explicit yes")
        return 1
    if OUT.exists():
        print(f"ABORT: {OUT} exists -- L37 was already run (or crashed mid-run). Never re-score.")
        return 1
    if LOCKBOX.exists() and any(json.loads(l).get("strategy") == STRATEGY for l in LOCKBOX.read_text().splitlines() if l.strip()):
        print("ABORT: L37 already in lab/lockbox.jsonl -- never re-score")
        return 1
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT.write_text(json.dumps({"status": "started", "ts": ts}))            # one-shot marker before any sealed read
    hashes = {f.name: sha(f) for f in (SEALED_CANDLES, PLAYS, META, GAMES, SCAN, FIT, ST, Path(__file__))}
    for k, v in hashes.items():
        print(f"sha256 {k}: {v}")
    cd = pd.read_parquet(SEALED_CANDLES)
    cd = cd[cd.date >= SEALED_FROM]
    assert cd.date.min() >= SEALED_FROM
    m = pd.read_parquet(META, columns=["eid", "spread"]).drop_duplicates("eid").set_index("eid")
    hm = -m.spread                                                    # ESPN pickcenter home spread -> home expected margin
    r, ent = run(cd, hm)
    show(r, f"SEALED L37 (nba_ingame games dated >= {SEALED_FROM.date()})")
    rec = {"ts": ts, "strategy": STRATEGY, "sha": hashlib.sha256(STRATEGY.encode()).hexdigest()[:12],
           "dataset": "nba_midgame", "status": "done", "window": f">= {SEALED_FROM.date()}", "data_hashes": hashes, **r}
    OUT.write_text(json.dumps(rec, indent=1, default=str))
    with LOCKBOX.open("a") as f:
        f.write(json.dumps({k: v for k, v in rec.items() if k != "calibration"}, default=str) + "\n")
    if len(ent):
        ent[["eid", "date", "stype", "side", "team_side", "period", "clock_s", "el", "d", "s_tr", "end", "bid", "ask",
             "p_model", "edge", "emp_tr", "emp_n", "fee", "won", "kalshi_won", "espn_won", "pnl", "pnl_r",
             "lag_s"]].to_csv(BETS, index=False)
    print(f"wrote {OUT}, {BETS}; lockbox appended")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["devcheck", "sealed"])
    ap.add_argument("--confirm-lockbox", action="store_true")
    a = ap.parse_args()
    return cmd_devcheck() if a.cmd == "devcheck" else cmd_sealed(a.confirm_lockbox)


if __name__ == "__main__":
    raise SystemExit(main())

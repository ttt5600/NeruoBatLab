"""SEALED test L28: NBA big-lead leader buy on Kalshi KXNBAGAME, scored ONCE on 2025-26 in-game candles.

    ../.venv_quant/bin/python scripts/nba_l28_sealed.py devcheck                    # dev reproduction (2025 playoffs; safe)
    ../.venv_quant/bin/python scripts/nba_l28_sealed.py sealed --confirm-lockbox     # ONE shot (user: "yes one shot tests" 2026-10-26)

Source of the rule: scripts/nba_bigleads_logger.py docstring (L28, pre-registered for forward paper 2026-10-25).
Dev history: notes/nba_big_lead.md (B1-B3 failed dev on the 2025 playoffs; B2 = Q4 > 6:00 lead >= 20 went 15/15 but
half its events had ask $1.00). This file re-implements L28 on historical 1-min candles; it imports nothing from the
dev scripts, so editing them cannot change this test.

=====================================================================================================================
PRE-REGISTRATION (written 2026-10-26 BEFORE any 2025-26 Kalshi price was joined to a game state. What had been viewed
of the sealed files: schema, row counts, game counts by month / season type, date range, pbp coverage by eid. No bid,
ask or result value of data/nba_ingame.parquet was read for this question.)
=====================================================================================================================
SEALED WINDOW  data/nba_ingame.parquet games with ET date >= 2026-01-15 (all season types: regular, play-in, playoffs).
       Games dated 2025-10-21 .. 2026-01-14 are EXCLUDED: their candles were DEV for scripts/nba_ingame.py (H1-H3,
       regular season < 01-15) and scripts/nba_late_wp.py (T1/T2, all games < 01-15). Different rules, but their prices
       and outcomes were already analysed, so they are not clean. Schema count: 713 games after the cut
       (626 regular, 6 play-in, 81 playoffs), 223,826 candle rows; 590 earlier games dropped.
       The frozen MANIFEST (data/frozen/2026-10-14) seals NBA season 2026 (= 2025-26) as a whole; nba_ingame.parquet is
       not itself listed there, so its sha256 is recorded at run time.
KNOWN LEAK (score side, not price side): notes/nba_big_lead.md already reports 2025-26 comeback rates from pbp scores
       (a 20+ lead was blown in 3.6% of games, 25+ in 0.8%, the lowest since 2016-17), and scripts/nba_live_scan.py's
       history table pools 2015-16..2025-26. So we already know the sealed season was a LOW-comeback season, which
       favours a pass. Prices were never joined. A pass should be read with this in mind.
DATA   Candles: data/nba_ingame.parquet (end = candle end, unix s; bid/ask = close YES bid/ask in dollars; side =
       home/away market; result = that market's Kalshi settlement yes/no). Play-by-play: data/nba_big_lead_plays.parquet
       (ESPN; columns eid, seq, period, clock_s, hs, as, wall). Coverage checked 2026-10-26: 1,303/1,303 candle games
       have pbp with wallclock, so no fetch is needed. A sealed game without usable pbp is EXCLUDED and counted.
STATE  For each 1-min candle of a market with end time c: state = the last pbp play with wallclock <= c - 60 s (plays
       ordered by wallclock, then seq; same rule as scripts/nba_late_wp.py). Lead = |home score - away score| at that
       play; period and game clock are that play's. Candles with c - 60 before the first play, or c after the last
       play + 120 s, have no state.
TRIGGER (L28) the market's team is the LEADER at the state and
       (a) period == 3 and lead >= 25 (any Q3 clock), or
       (b) period == 4 and lead >= 20 and clock > 360 s (strictly more than 6:00 left). OT never qualifies.
ENTRY  BUY the leader's YES at that candle's CLOSE ASK when the quote is two-sided on the leader's market
       (bid > 0 and ask < 1, both present), spread ask - bid <= 0.05, and ask <= 0.96. One entry per game: the FIRST
       candle (by end time) that meets trigger + quote + price. Flat 1 contract. Hold to settlement. No exits, no adds.
FEE    Primary: Kalshi taker fee 0.07 * p * (1-p) per contract, unrounded (prod fills were ~exact).
       Sensitivity: cent-rounded-up fee on a 1-lot, ceil(100 * 0.07 p (1-p)) / 100 (= $0.01 at every ask <= 0.96 here).
SETTLE Kalshi `result` of the leader's market (yes = 1). Cross-check against the official final score in
       data/nba_games.parquet; mismatches are counted and listed. If result is missing, the ESPN final is used (counted).
       P&L per contract = won - ask - fee.
PRIMARY  total P&L (units = sum over bets of P&L per 1 contract) and ROI = units / sum(ask + fee). One-sided bootstrap
       over GAMES (one bet per game): 10,000 resamples, numpy default_rng(0); p = share of resampled sums <= 0
       (H1: ROI > 0). PASS = ROI > 0 AND bootstrap p < 0.05 AND exact p < 0.05, where exact p = Poisson-binomial
       P(wins >= observed | each bet wins with probability = its break-even ask + fee). The exact co-condition is the
       project's rule since nba_big_lead B2 (amendment 2): with zero losses every bootstrap resample is profitable and
       the bootstrap p is 0 by construction, which is not evidence. One test, one shot, never re-scored.
SECONDARY (descriptive, no pass/fail): win rate vs mean break-even (ask + fee); rounded-fee units/ROI/p; per arm (Q3
       25+ vs Q4 20+) and per season type; list of every loss; entry lag (candle end - state play wallclock);
       TRIGGER GAMES = games with at least one trigger candle (any quote); of those without an entry, how many had a
       trigger candle priced ask > 0.96 (incl. $1.00 = no seller) -> "too high, no trade", vs no two-sided/tight quote;
       the leader's realised win rate in the no-trade games.
POWER  (dev-only inputs, printed by `devcheck`, results/nba_l28_devcheck.json.) A loss costs ~0.94-0.96 and a win
       earns ~0.04-0.06, so one loss erases ~15-25 wins. Expected n = 713 sealed games x 0.329 (share of 2024-25 games
       whose pbp reaches a trigger state) x 0.214 (6 of 28 dev trigger games had a buyable quote) ~ 50 bets. This is
       rough: the buyable share comes from 28 PLAYOFF games, and regular-season blowouts may be priced differently.
       With n ~ 50 and mean break-even 0.94 (dev), the exact co-condition passes ONLY with zero losses, and a zero-loss
       sample passes only if mean break-even^n < .05, i.e. n >= 49 / 59 / 74 / 99 at break-even .94 / .95 / .96 / .97.
       Simulated power at n 50: 5% / 9% / 13% / 22% at true win rate = break-even + 0 / 1 / 2 / 3 pts.
       (Bootstrap alone would "pass" any zero-loss sample; under H0 that happens with probability break-even^n =
       4.5% / 8% / 13% / 22% at .94 / .95 / .96 / .97 -- the reason for the exact co-condition.)
       So: a PASS needs a loss-free season with enough buyable games; a FAIL is weak evidence against a 1-3 pt edge.
ONE SHOT  `sealed` needs --confirm-lockbox; it aborts if results/nba_l28_sealed.json exists or lab/lockbox.jsonl has
       strategy nba_L28_biglead. It writes results/nba_l28_sealed.json with status "started" BEFORE reading any sealed
       file, so a crash also blocks a re-run (a human must then decide; do not delete it quietly).
DEV CHECK `devcheck` runs the identical functions on data/nba_late_candles_2025po.parquet (2024-25 play-in + playoffs,
       the only DEV Kalshi NBA games; already used by nba_big_lead.py B1-B3) and asserts it never touches nba_ingame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEALED_CANDLES = ROOT / "data" / "nba_ingame.parquet"
DEV_CANDLES = ROOT / "data" / "nba_late_candles_2025po.parquet"
PLAYS = ROOT / "data" / "nba_big_lead_plays.parquet"
META = ROOT / "data" / "nba_big_lead_meta.parquet"
GAMES = ROOT / "data" / "nba_games.parquet"
OUT = ROOT / "results" / "nba_l28_sealed.json"
BETS = ROOT / "results" / "nba_l28_sealed_bets.csv"
DEVCHECK = ROOT / "results" / "nba_l28_devcheck.json"
LOCKBOX = ROOT / "lab" / "lockbox.jsonl"
STRATEGY = "nba_L28_biglead"
SEALED_FROM = pd.Timestamp("2026-01-15")       # nba_ingame / nba_late_wp DEV ended 2026-01-14
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
LAG, TAIL = 60, 120
MAX_ASK, MAX_SPREAD = 0.96, 0.05
N_BOOT, SEED = 10_000, 0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fee(p):
    p = np.asarray(p, float)
    return 0.07 * p * (1 - p)


def fee_rounded(p):
    return np.ceil(np.round(100 * fee(p), 9)) / 100


def trigger(period, clock_s, lead):
    period, clock_s, lead = (np.asarray(x) for x in (period, clock_s, lead))
    return ((period == 3) & (lead >= 25)) | ((period == 4) & (lead >= 20) & (clock_s > 360))


def load_pbp(eids) -> pd.DataFrame:
    p = pd.read_parquet(PLAYS)
    p = p[p.eid.isin(set(eids))].dropna(subset=["hs", "as", "clock_s", "wall", "period"]).copy()
    p["m"] = p.hs - p["as"]
    p["t"] = (p.wall - EPOCH) // pd.Timedelta(seconds=1)          # unit-safe epoch seconds
    return p.sort_values(["eid", "t", "seq"], kind="stable").reset_index(drop=True)


def candle_states(cd: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side, candle) with the state of the last play at or before end - 60 s, from that side's view."""
    out = []
    pg = dict(tuple(pbp.groupby("eid")))
    for eid, c in cd.groupby("eid"):
        gp = pg.get(eid)
        if gp is None or not len(gp):
            continue
        wt = gp.t.to_numpy()
        c = c[(c.end - LAG >= wt.min()) & (c.end <= wt.max() + TAIL)]
        if not len(c):
            continue
        idx = np.searchsorted(wt, c.end.to_numpy() - LAG, side="right") - 1
        st = gp.iloc[idx]
        sgn = np.where(c.side.to_numpy() == "home", 1, -1)
        out.append(pd.DataFrame({"eid": eid, "side": c.side.to_numpy(), "end": c.end.to_numpy(),
                                 "bid": c.bid.to_numpy(float), "ask": c.ask.to_numpy(float),
                                 "result": c.result.to_numpy(), "date": c.date.to_numpy(),
                                 "stype": c.stype.to_numpy() if "stype" in c else -1,
                                 "period": st.period.to_numpy(), "clock_s": st.clock_s.to_numpy(),
                                 "marg": sgn * st.m.to_numpy(), "play_t": st.t.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def select(x: pd.DataFrame):
    """Apply L28. Returns (entries one per game, trigger-candle frame)."""
    x = x.assign(lead=x.marg)
    trig = x[(x.marg > 0) & trigger(x.period, x.clock_s, x.marg)].copy()
    q = trig.bid.notna() & trig.ask.notna() & (trig.bid > 0) & (trig.ask < 1) & (trig.ask - trig.bid <= MAX_SPREAD + 1e-9)
    trig["quote_ok"] = q
    trig["buy"] = q & (trig.ask <= MAX_ASK + 1e-9) & (trig.ask > 0)
    ent = trig[trig.buy].sort_values(["eid", "end"]).drop_duplicates("eid").copy()
    ent["arm"] = np.where(ent.period == 3, "Q3_lead25", "Q4_lead20_gt6min")
    return ent, trig


def settle(ent: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_parquet(GAMES, columns=["game_id", "home_score", "away_score"]).set_index("game_id")
    hw = (ent.eid.map(g.home_score) > ent.eid.map(g.away_score))
    espn_won = np.where(ent.side == "home", hw, ~hw).astype(float)
    espn_won[ent.eid.map(g.home_score).isna().to_numpy()] = np.nan
    kal = ent.result.map({"yes": 1.0, "no": 0.0}).to_numpy(float)
    won = np.where(np.isnan(kal), espn_won, kal)
    ent = ent.assign(won=won, kalshi_won=kal, espn_won=espn_won)
    ent["fee"] = fee(ent.ask)
    ent["fee_r"] = fee_rounded(ent.ask)
    ent["pnl"] = ent.won - ent.ask - ent.fee
    ent["pnl_r"] = ent.won - ent.ask - ent.fee_r
    ent["lag_s"] = ent.end - ent.play_t
    return ent


def pbinom_tail(ps, k):
    d = np.zeros(len(ps) + 1)
    d[0] = 1.0
    for q in ps:
        d[1:] = d[1:] * (1 - q) + d[:-1] * q
        d[0] *= (1 - q)
    return float(d[k:].sum())


def boot(v, n=N_BOOT, seed=SEED):
    v = np.asarray(v, float)
    if not len(v):
        return {"p": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    s = rng.choice(v, (n, len(v))).sum(1)
    return {"p": float(np.mean(s <= 0)), "ci_lo": float(np.quantile(s, .025)), "ci_hi": float(np.quantile(s, .975))}


def summarise(ent: pd.DataFrame, trig: pd.DataFrame, x: pd.DataFrame) -> dict:
    ent = ent.dropna(subset=["won"])
    r = {"games_with_candles_and_state": int(x.eid.nunique()), "trigger_games": int(trig.eid.nunique()),
         "bets": int(len(ent))}
    if len(ent):
        stake = float((ent.ask + ent.fee).sum())
        b = boot(ent.pnl)
        be = (ent.ask + ent.fee).to_numpy()
        r.update({"wins": int(ent.won.sum()), "losses": int((1 - ent.won).sum()), "win_rate": float(ent.won.mean()),
                  "mean_ask": float(ent.ask.mean()), "mean_fee": float(ent.fee.mean()), "mean_breakeven": float(be.mean()),
                  "units": float(ent.pnl.sum()), "pnl_per_contract": float(ent.pnl.mean()), "staked": stake,
                  "roi": float(ent.pnl.sum() / stake), "boot_p": b["p"], "units_ci_lo": b["ci_lo"], "units_ci_hi": b["ci_hi"],
                  "exact_p_vs_breakeven": pbinom_tail(be, int(ent.won.sum()))})
        br = boot(ent.pnl_r)
        r["rounded_fee"] = {"units": float(ent.pnl_r.sum()), "roi": float(ent.pnl_r.sum() / (ent.ask + ent.fee_r).sum()),
                            "boot_p": br["p"], "exact_p": pbinom_tail((ent.ask + ent.fee_r).to_numpy(), int(ent.won.sum()))}
        r["passed"] = bool(r["roi"] > 0 and r["boot_p"] < 0.05 and r["exact_p_vs_breakeven"] < 0.05)
        r["by_arm"] = {k: {"n": int(len(s)), "wins": int(s.won.sum()), "units": float(s.pnl.sum()), "mean_ask": float(s.ask.mean())}
                       for k, s in ent.groupby("arm")}
        r["by_stype"] = {str(k): {"n": int(len(s)), "wins": int(s.won.sum()), "units": float(s.pnl.sum())}
                         for k, s in ent.groupby("stype")}
        r["lag_s_median"] = float(ent.lag_s.median())
        r["settle_kalshi_vs_espn_mismatch"] = int(((ent.kalshi_won != ent.espn_won) & ent.kalshi_won.notna() & ent.espn_won.notna()).sum())
        r["settle_kalshi_missing"] = int(ent.kalshi_won.isna().sum())
        r["loss_list"] = ent[ent.won == 0][["eid", "date", "side", "period", "clock_s", "lead", "ask"]].astype(str).to_dict("records")
    else:
        r["passed"] = False
    # trigger games without an entry
    nt = trig[~trig.eid.isin(ent.eid)]
    hi = nt[nt.ask.notna() & (nt.ask > MAX_ASK + 1e-9)].eid.unique()
    r["no_trade_games"] = int(nt.eid.nunique())
    r["no_trade_ask_gt_096"] = int(len(hi))
    r["no_trade_no_tight_quote_only"] = int(nt.eid.nunique() - len(hi))
    if len(nt):
        first = nt.sort_values("end").drop_duplicates("eid")
        r["no_trade_first_trigger_ask_counts"] = {f"{k:.2f}": int(v) for k, v in first.ask.round(2).value_counts().sort_index().items()}
        r["no_trade_leader_win_rate"] = float(first.result.map({"yes": 1.0, "no": 0.0}).mean())
    return r


def show(r: dict, tag: str):
    print(f"\n== {tag} ==")
    for k, v in r.items():
        if k != "loss_list":
            print(f"  {k}: {v}")
    for z in r.get("loss_list", []):
        print("  LOSS", z)
    if r.get("bets"):
        print(f"  VERDICT: {'PASS' if r['passed'] else 'FAIL'}  (units {r['units']:+.3f}, ROI {r['roi']:+.2%}, boot p "
              f"{r['boot_p']:.4f}, exact p {r['exact_p_vs_breakeven']:.4f}; win {r['win_rate']:.4f} vs break-even "
              f"{r['mean_breakeven']:.4f})")


def run(cd: pd.DataFrame):
    pbp = load_pbp(cd.eid.unique())
    missing = sorted(set(cd.eid) - set(pbp.eid))
    x = candle_states(cd, pbp)
    ent, trig = select(x)
    ent = settle(ent)
    r = summarise(ent, trig, x)
    r["games_in_window"] = int(cd.eid.nunique())
    r["games_excluded_no_pbp"] = len(missing)
    return r, ent


def power_note(dev: dict, n_sealed_games: int) -> dict:
    """Expected sealed n and power, from DEV quantities only (2024-25 pbp scores and the 2025 playoff candles)."""
    m = pd.read_parquet(META, columns=["eid", "season"])
    ids = m[m.season == 2024].eid
    p = load_pbp(ids)
    p["lead"] = p.m.abs()
    trig_rate = float(p[trigger(p.period, p.clock_s, p.lead)].eid.nunique() / p.eid.nunique())
    buy_rate = dev["bets"] / dev["trigger_games"] if dev.get("trigger_games") else float("nan")
    n_exp = n_sealed_games * trig_rate * buy_rate
    be = dev.get("mean_breakeven", 0.97)
    out = {"dev_2024_25_share_games_reaching_trigger": trig_rate, "dev_share_trigger_games_buyable": buy_rate,
           "expected_sealed_bets": n_exp, "assumed_breakeven": be}
    n = max(int(round(n_exp)), 1)
    # largest loss count that still gives exact p < .05 when every bet has break-even `be`
    crit = max([L for L in range(n + 1) if pbinom_tail(np.full(n, be), n - L) < 0.05], default=None)
    out["max_losses_for_exact_pass"] = crit
    rng = np.random.default_rng(1)
    for d in (0.0, 0.01, 0.02, 0.03):
        q = min(be + d, 0.9999)
        hits = 0
        for _ in range(2000):
            losses = rng.binomial(n, 1 - q)
            units = (n - losses) * (1 - be) - losses * be
            hits += (crit is not None and losses <= crit and units > 0)
        out[f"power_true_winrate_breakeven_plus_{int(d * 100)}pt"] = hits / 2000
    # a zero-loss sample passes the exact co-condition only if be^n < .05; without it, bootstrap-only "passes" any
    # zero-loss sample, which under H0 (win prob = break-even) happens with probability be^n
    out["min_bets_for_zero_loss_exact_pass"] = {f"{b:.2f}": math.ceil(math.log(0.05) / math.log(b)) for b in (0.94, 0.95, 0.96, 0.97)}
    out["bootstrap_only_false_pass_prob_H0"] = {f"{b:.2f}": round(b ** n, 3) for b in (0.94, 0.95, 0.96, 0.97)}
    return out


def cmd_devcheck() -> int:
    cd = pd.read_parquet(DEV_CANDLES)
    assert cd.date.min() >= pd.Timestamp("2025-04-15") and cd.date.max() < pd.Timestamp("2025-07-01"), "DEV candles only"
    r, ent = run(cd)
    show(r, "DEV CHECK: L28 on 2025 play-in + playoffs (86 games; dev, already used by nba_big_lead B1-B3)")
    if len(ent):
        print(ent[["eid", "side", "arm", "period", "clock_s", "lead", "bid", "ask", "won", "pnl", "lag_s"]].round(4).to_string(index=False))
    n_sealed = int(pd.read_parquet(SEALED_CANDLES, columns=["eid", "date"]).query("date >= @SEALED_FROM").eid.nunique())  # count only
    pw = power_note(r, n_sealed)
    print("\nPOWER (dev-only inputs):", json.dumps(pw, indent=1))
    DEVCHECK.write_text(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dev": r, "power": pw,
                                    "sealed_games_in_window_count_only": n_sealed}, indent=1, default=str))
    print(f"wrote {DEVCHECK}")
    return 0


def cmd_sealed(confirm: bool) -> int:
    if not confirm:
        print("sealed: needs --confirm-lockbox and the user's explicit yes")
        return 1
    if OUT.exists():
        print(f"ABORT: {OUT} exists -- L28 was already run (or crashed mid-run). Never re-score.")
        return 1
    if LOCKBOX.exists() and any(json.loads(l).get("strategy") == STRATEGY for l in LOCKBOX.read_text().splitlines() if l.strip()):
        print("ABORT: L28 already in lab/lockbox.jsonl -- never re-score")
        return 1
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT.write_text(json.dumps({"status": "started", "ts": ts}))            # one-shot marker before any sealed read
    hashes = {f.name: sha(f) for f in (SEALED_CANDLES, PLAYS, GAMES, Path(__file__))}
    for k, v in hashes.items():
        print(f"sha256 {k}: {v}")
    cd = pd.read_parquet(SEALED_CANDLES)
    cd = cd[cd.date >= SEALED_FROM]
    assert cd.date.min() >= SEALED_FROM
    r, ent = run(cd)
    show(r, f"SEALED L28 (nba_ingame games dated >= {SEALED_FROM.date()})")
    rec = {"ts": ts, "strategy": STRATEGY, "sha": hashlib.sha256(STRATEGY.encode()).hexdigest()[:12],
           "dataset": "nba_big_lead", "status": "done", "window": f">= {SEALED_FROM.date()}", "data_hashes": hashes, **r}
    OUT.write_text(json.dumps(rec, indent=1, default=str))
    with LOCKBOX.open("a") as f:
        f.write(json.dumps({k: v for k, v in rec.items() if k != "loss_list"}, default=str) + "\n")
    ent[["eid", "date", "stype", "side", "arm", "period", "clock_s", "lead", "end", "bid", "ask", "fee", "won",
         "kalshi_won", "espn_won", "pnl", "pnl_r", "lag_s"]].to_csv(BETS, index=False)
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

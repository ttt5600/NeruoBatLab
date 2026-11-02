"""Migrate the fund's history into the registry, honestly (task step 7).

1. Every scored lab/ledger.jsonl trial (distinct dataset x strategy whose results were seen: status ok / done / rerun /
   invalidated / void_bug) -> one historical trial in its family, state dead (= dead or lead: not runnable here; a
   lead must be re-registered as a new edgelab trial, which counts).
2. FUND_BOOK 3d rows whose evidence is NOT a ledger trial ([R], [EA], [S] rows) -> historical trials with the number of
   tests the row states (e.g. "152 tests" counts 152).
3. The strategies that are live / forward today -> runnable edgelab trials (specs below), evaluated through
   run.evaluate() on their real rows: dev, then the sealed REPLAY (vault.yaml spent_unseals; no lockbox is opened for
   the first time here), then forward. Whatever edgelab says is their state; FUND_BOOK's claim is kept as a
   legacy_note for comparison and never moves the ladder.
Idempotent: re-running registers nothing twice (same spec -> same trial id) and skips evaluations already logged.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import paths
from .registry import Registry

SEEN_STATUSES = ("ok", "done", "bugfix_rerun", "invalidated", "void_bug")

# ledger dataset -> family (market + venue the selection searched over). Coarse on purpose: rule 2's intent is that a
# tweak of a related idea raises the bar of everything related. USER: confirm the grouping.
FAMILY = {
    **{d: "kalshi_weather_temp" for d in ["kalshi_temp", "kalshi_temp_maker", "kalshi_temp_expand", "kalshi_temp_dayof",
                                         "wx_timing", "wx_lockin", "pmus_temp", "forecastex_wx", "forecastex_wx2"]},
    **{d: "kalshi_weather_precip" for d in ["kalshi_rain", "kalshi_wx_winter"]},
    **{d: "kalshi_data_ladders" for d in ["e1x_tsa", "e1x_quake", "econ_reject", "kalshi_cpi", "kalshi_gas",
                                         "q017_ercot_load", "q015_approval_avg", "q016_commodities",
                                         "q012_crypto_hourly"]},
    **{d: "kalshi_retail_flb" for d in ["kalshi_mentions", "q003_retail_maker", "kalshi_fav_maker"]},
    **{d: "kalshi_sports_pregame" for d in ["kalshi_sports", "kalshi_soccer", "kalshi_early_nfl",
                                           "kalshi_early_nfl_totals"]},
    **{d: "kalshi_ingame" for d in ["ingame_longshot", "nba_big_lead", "nba_late_wp", "nba_ingame", "nfl_ingame",
                                   "nba_live_totals", "nba_midgame"]},
    **{d: "nba_totals" for d in ["nba_simple_model", "l32b_debiased_totals", "nba_referees", "nba_luck"]},
    **{d: "nba_sides" for d in ["nba_sbr", "nba_public_models", "l2_team_rating_sides", "nba_early_season_bias",
                               "nba_hype_vs_form", "nba_clock_calendar_arena", "nba_style_matchups",
                               "q011_nba_injury_lag"]},
    **{d: "nba_props" for d in ["nba_player_sim", "nba_props_model", "nba_props_dist", "nba_props_dist_sealed2026",
                               "nba_ref_ft_props", "nba_player_opp_venue", "nba_injury_teammate_props",
                               "nba_hot_streaks", "nba_matchup_graph"]},
    **{d: "nfl_lines" for d in ["nfl_spread", "nfl_total", "nfl_moneyline", "nfl_simple_model", "nfl_leads_open",
                               "nfl_recency", "nfl_kicker_wx_matchups"]},
    **{d: "nfl_props" for d in ["nfl_props_script", "nfl_kicker_system"]},
    **{d: "nhl_lines" for d in ["nhl_luck", "nhl_simple_model"]},
    **{d: "cfb_lines" for d in ["cfb_simple_model", "l36_cfb_sealed"]},
    **{d: "mlb_lines" for d in ["mlb_total", "mlb_ml_sbr"]},
    "mlb_k_props": "mlb_props",
    "soccer_fd": "soccer_books",
    **{d: "equities_options" for d in ["stocks_twins_yf", "company_graph_edgar_yf", "q009_spy_options",
                                      "l11_dolthub_single_name_options"]},
    "poly_earnings_beat": "polymarket_calibration",
    "rumor_gdelt_kalshi": "kalshi_news",
}

# FUND_BOOK 3d rows without a ledger trial: (id, family, n_trials, evidence). Counts are what the row states.
FUND_BOOK_3D = [
    ("cb1_fade_trailing_nfl_dogs", "kalshi_ingame", 1, "3d CB1 [EA]"),
    ("live_pinnacle_vs_kalshi_flags", "kalshi_sports_pregame", 1, "3d live scanner [R]"),
    ("nba_comebacks_matchups_152", "nba_sides", 152, "3d 'NBA comebacks / matchups / coaching / timeouts (152 tests)' [EA]"),
    ("nfl_comebacks_proe_70", "nfl_lines", 70, "3d 'NFL comebacks / PROE / pace / 4th-down / coach / head ref (70 tests)' [EA]"),
    ("nfl_weather_oracle", "nfl_lines", 1, "3d NFL weather oracle [R]"),
    ("mlb_bullpen_rest_21", "mlb_lines", 21, "3d 'MLB bullpen / starter rest / day-after-night 0/21 BH' [R]"),
    ("nhl_l34_bet_espnbet", "nhl_lines", 1, "3d L34-bet [EA]"),
    ("nba_multibook_consensus_2023", "nba_sides", 1, "3d multi-book consensus deviation [R]"),
    ("nfl_edge_report_shrinkage", "nfl_lines", 1, "3d edge report shrinkage [R]"),
    ("pattern_mine_2280", "pattern_mine", 2280, "3d PM10 backwards pattern mine 2,280 cells [R]"),
    ("nfl_prop_loop_252", "nfl_props", 252, "3d 'NFL 252-trial prop model loop' [R]"),
    ("nfl_fg_props_2", "nfl_props", 2, "3d NFL FG props (defense style + script; zone/blitz) [R]"),
    ("kalshi_vs_dk_props_scan", "nba_props", 1, "3d Kalshi vs DraftKings props scan [S]"),
    ("two_leg_parlays_2", "parlays", 2, "3d 2-leg parlays (props, NFL ML) [R]"),
    ("sleeper_pickem_vs_pinnacle", "pickem", 1, "3d Sleeper pick'em [R]"),
    ("e1_confidence_kelly_sizing", "kalshi_weather_temp", 1, "3d confidence-based Kelly sizing [EA]"),
    ("smallmarket_lifetime_volume", "polymarket_calibration", 1, "3d small markets mispriced [EA]"),
    ("poly_negrisk_overround", "polymarket_calibration", 1, "3d negRisk overround legs [EA]"),
    ("xvenue_kalshi_poly_midterms", "cross_venue", 1, "3d M1 [R]"),
    ("xvenue_kalshi_poly_sports", "cross_venue", 1, "3d M2 [R]"),
    ("xvenue_kalshi_poly_weather", "cross_venue", 1, "3d Kalshi vs Polymarket weather [EA]"),
    ("xvenue_pmus_vs_pinnacle", "cross_venue", 1, "3d PMUS vs Pinnacle [R]"),
    ("xvenue_kalshi_soccer_vs_pinnacle", "cross_venue", 1, "3d Kalshi soccer vs Pinnacle [R]"),
    ("q007_forecastex_families", "cross_venue", 1, "3d Q007 [Q]"),
    ("etf_rules_14", "equities_options", 14, "3d 14 ETF rules [lab/stocks_ledger.jsonl]"),
    ("classic_pairs_7", "equities_options", 7, "3d classic pairs KO/PEP + 6 [R]"),
    ("be_earnings_straddle", "equities_options", 1, "3d BE earnings straddle [S]"),
]

# Ledger strategy names that ARE the runnable trials below (not imported twice).
MIGRATED_LEDGER_NAMES = {"kalshi_reject_safe_check", "wx_expand_X1_new_highs", "kalshi_temp_maker_M3_N10",
                         "e1x_tsa_T1", "e1x_tsa_T2", "L32b_debiased_open_totals", "l2_T1_ml_gap05"}

WX_FEE_EXACT = {"rounding": "exact", "contracts_per_order": 1,
                "note": "direct-member fees at $0.0001 precision (docs fee_rounding; prod fills ~exact, "
                        "HANDOFF_2026-10-25); cent rounding is the sensitivity"}
CLV_KILL = {"kind": "clv_negative_significant", "after_n": 100, "p": 0.05}

SPECS = [
    dict(id="e1_taker", family="kalshi_weather_temp", dataset="kalshi_temp_e1", rule="edgelab.rules.kalshi_wx:taker",
         hypothesis="E1: buy NO on daily-high brackets all 5 forecast models miss by > 3F (fcsafe, <= 18:00 local), "
                    "taker at the 22:00-local NO ask + fee (lab/strategies/kalshi_reject_safe_check.py, frozen)",
         decision_time="22:00 station-local the night before", price_model="taker",
         split="vault:kalshi_temp_e1", min_n=200, kill=CLV_KILL, fee=WX_FEE_EXACT,
         legacy="FUND_BOOK 3a E1: sealed pass 10-08, LIVE taker since 10-22 (user yes)"),
    dict(id="e2_taker", family="kalshi_weather_temp", dataset="kalshi_temp_x1", rule="edgelab.rules.kalshi_wx:taker",
         hypothesis="E2/X1: the frozen E1 rule on 17 more daily-high cities (scripts/wx_expand.py picks), taker",
         decision_time="22:00 station-local the night before", price_model="taker", split="vault:kalshi_temp_x1",
         min_n=200, kill=CLV_KILL, fee=WX_FEE_EXACT, legacy="FUND_BOOK 3a E2: sealed pass 10-20, LIVE with E1"),
    dict(id="e4_maker_e1", family="kalshi_weather_temp", dataset="kalshi_temp_e1",
         rule="edgelab.rules.kalshi_wx:maker_m3", params={"n_min": 10},
         hypothesis="E4 (E1 cities): post-only NO bid at the mid (floored, <= ask-1c), filled iff >= 10 contracts print "
                    "at <= q in (T, close], no cancel, maker fee 0 (quadratic series)",
         decision_time="22:00 station-local", price_model="maker_tape", split="vault:kalshi_temp_e1", min_n=200,
         kill=CLV_KILL, fee={"rounding": "exact"}, legacy="FUND_BOOK 3a E4: sealed pass 10-23 (pooled E1+X1), LIVE maker 10-29"),
    dict(id="e4_maker_x1", family="kalshi_weather_temp", dataset="kalshi_temp_x1",
         rule="edgelab.rules.kalshi_wx:maker_m3", params={"n_min": 10},
         hypothesis="E4 (X1 cities): as e4_maker_e1", decision_time="22:00 station-local", price_model="maker_tape",
         split="vault:kalshi_temp_x1", min_n=200, kill=CLV_KILL, fee={"rounding": "exact"},
         legacy="FUND_BOOK 3a E4 (pooled with E1 cities in the original sealed test)"),
    dict(id="tsa_t1_taker", family="kalshi_data_ladders", dataset="kalshi_tsaw", rule="edgelab.rules.tsa:t1_taker",
         hypothesis="E1X-T T1: KXTSAW far-side rungs outside F*exp(r_min/max -+ .01) at open+2h, taker at ask + fee",
         decision_time="week open + 2 h (Monday)", price_model="taker", split="vault:kalshi_tsaw", min_n=60,
         kill={"kind": "roi_ci_below_zero", "after_n": 60}, fee={"rounding": "exact"},
         legacy="notes/e1x_tsa.md: T1 dev 0.0% (fail); sealed +0.96%"),
    dict(id="tsa_t2_maker", family="kalshi_data_ladders", dataset="kalshi_tsaw", rule="edgelab.rules.tsa:t2_maker",
         hypothesis="E1X-T T2: same rungs, maker 1c inside, first crossing tape print = fill, no cancel",
         decision_time="week open + 2 h (Monday)", price_model="maker_tape", split="vault:kalshi_tsaw", min_n=60,
         kill={"kind": "losses_in_first", "n": 30, "max_losses": 2},
         fee={"rounding": "exact"},
         legacy="FUND_BOOK/EDGES: SEALED weak pass 11-01 (24 fills, 0 losses), forward paper (judge >= 60 fills)"),
    dict(id="l32b_totals_open", family="nba_totals", dataset="nba_totals_open", rule="edgelab.rules.nba:l32b_dev",
         hypothesis="L32b: NBA total model minus trailing-100-game mean(model - open), bet |adj| >= 3 at the open",
         decision_time="at the opener (ESPN open, no timestamp)", price_model="book_quote",
         split="vault:nba_totals_open", min_n=200, kill=CLV_KILL, fee={},
         legacy="notes/l32b_debiased_totals.md: DEV PASS 11-01 at no-vig open (+8.1%, p .006); forward paper"),
    dict(id="l2_t1_ml_gap05", family="nba_sides", dataset="nba_sides_l2", rule="edgelab.rules.nba:l2_sides",
         params={"trial": "T1_ml_gap05"},
         hypothesis="L2 T1: bet the moneyline side where team_rating P(win) exceeds the power-de-vigged close by >= 5c",
         decision_time="at the close (tip)", price_model="book_quote", split="vault:nba_sides_l2", min_n=100,
         kill={"kind": "roi_ci_below_zero", "after_n": 100}, fee={},
         legacy="notes/l2_team_rating_sides.md: DEAD 10-29 (-13.9% on 789)"),
    dict(id="f1_nba_early_form", family="nba_sides", dataset="nba_early_form", rule="edgelab.rules.forward:unwired",
         hypothesis="F1 NBA early form (<= 8-game window): form-vs-market ML at the Kalshi ask, 2026-27 weeks 1-3",
         decision_time="pre-tip tick", price_model="taker", split="vault:nba_early_form", min_n=30,
         kill={"kind": "clv_negative_significant", "after_n": 30, "p": 0.05}, fee={"rounding": "exact"},
         legacy="LEAD -> forward paper (nba_early_form_logger.py); the dev window was a post-hoc sensitivity"),
    dict(id="kalshi_prop_ladders", family="nba_props", dataset="kalshi_prop_ladders",
         rule="edgelab.rules.forward:unwired",
         hypothesis="Kalshi NBA prop ladders: YES sim >= ask + .08 / NO sim <= bid - .08, vol >= 100, proj >= 20 min",
         decision_time="logger tick", price_model="taker", split="vault:kalshi_prop_ladders", min_n=150,
         kill={"kind": "clv_negative_significant", "after_n": 150, "p": 0.05}, fee={"rounding": "exact"},
         legacy="notes/kalshi_prop_ladders.md: forward paper since 11-01 (judge >= 150 picks & >= 20 games)"),
    dict(id="q017_b_far_bracket", family="kalshi_data_ladders", dataset="q017_ercot",
         rule="edgelab.rules.forward:unwired",
         hypothesis="Q017 B: ERCOT daily peak far-bracket NO (E1 pattern) at the 06/10 CT quote",
         decision_time="06:00 / 10:00 CT", price_model="taker", split="vault:q017_ercot", min_n=30,
         kill={"kind": "losses_in_first", "n": 30, "max_losses": 2}, fee={"rounding": "exact"},
         legacy="notes/q017_ercot_load.md: B lead (12/12, 7/7), forward logger built"),
]

# first time anyone looked at each migrated strategy's data (ledger / lockbox ts), for evidence R1
HISTORICAL_FIRST_LOOK = {}


def ledger_trials() -> list[dict]:
    rows = [json.loads(l) for l in (paths.LAB / "ledger.jsonl").read_text().splitlines() if l.strip()]
    seen: dict[tuple, dict] = {}
    for r in rows:
        st = str(r.get("status", ""))
        if not st.startswith(SEEN_STATUSES):
            continue
        k = (r.get("dataset"), r.get("strategy"))
        if k not in seen:
            seen[k] = r
    return list(seen.values())


def first_looks() -> dict:
    out = {}
    names = {"e1_taker": "kalshi_reject_safe_check", "e2_taker": "wx_expand_X1_new_highs",
             "e4_maker_e1": "kalshi_temp_maker_M3_N10", "e4_maker_x1": "kalshi_temp_maker_M3_N10",
             "tsa_t1_taker": "e1x_tsa_T1", "tsa_t2_maker": "e1x_tsa_T2", "l32b_totals_open": "L32b_debiased_open_totals",
             "l2_t1_ml_gap05": "l2_T1_ml_gap05"}
    rows = [json.loads(l) for l in (paths.LAB / "ledger.jsonl").read_text().splitlines() if l.strip()]
    for sid, nm in names.items():
        ts = sorted(r["ts"] for r in rows if r.get("strategy") == nm)
        out[sid] = ts[0] if ts else None
    return out


def dump_specs() -> None:
    d = paths.PKG / "specs"
    d.mkdir(exist_ok=True)
    for s in SPECS:
        (d / f"{s['id']}.yaml").write_text(yaml.safe_dump(s, sort_keys=False, width=120))


def migrate(reg: Registry | None = None, evaluate_live: bool = True, log=print) -> dict:
    from . import run
    reg = reg or Registry()
    have = {t["spec"]["id"] for t in reg.trials().values()}
    n_imp = 0
    for r in ledger_trials():
        ds, nm = r.get("dataset"), r.get("strategy")
        if nm in MIGRATED_LEDGER_NAMES:
            continue
        fam = FAMILY.get(ds, f"unmapped:{ds}")
        sid = f"ledger:{ds}:{nm}"
        if sid in have:
            continue
        reg.register({"id": sid, "family": fam, "hypothesis": (r.get("hypothesis") or "")[:300],
                      "dataset": ds, "ledger_ts": r["ts"], "ledger_sha": r.get("sha")},
                     origin="ledger_import", initial_state="dead", evidence=f"lab/ledger.jsonl ts={r['ts']}")
        n_imp += 1
    for sid, fam, n, ev in FUND_BOOK_3D:
        if f"fund_book_3d:{sid}" in have:
            continue
        reg.register({"id": f"fund_book_3d:{sid}", "family": fam, "hypothesis": ev}, origin="fund_book_3d",
                     initial_state="dead", n_trials=n, evidence=f"notes/FUND_BOOK.md {ev}")
        n_imp += 1
    log(f"historical trials imported this run: {n_imp}")
    dump_specs()
    ids = {}
    for s in SPECS:
        tid = reg.register(s, origin="edgelab")
        ids[s["id"]] = tid
        if not any(e["event"] == "legacy_note" and e["trial_id"] == tid for e in reg.events()):
            reg.append({"event": "legacy_note", "trial_id": tid, "note": s.get("legacy"),
                        "historical_first_look": first_looks().get(s["id"]),
                        "registered_after_data_seen": s["id"] in first_looks()})
    if evaluate_live:
        from . import vault
        for s in SPECS:
            tid = ids[s["id"]]
            splits = vault.config()[s["dataset"]]["splits"]
            for split in ("dev", "sealed", "forward"):
                if splits.get(split) is None or reg.evaluations(tid, split):
                    continue
                if split == "sealed" and not vault.sealed_gate(reg, tid, s["dataset"])["allowed"]:
                    log(f"  {s['id']}: sealed not loaded ({vault.sealed_gate(reg, tid, s['dataset'])['why']})")
                    continue
                try:
                    out = run.evaluate(tid, split, reg)
                except Exception as e:  # recorded by evaluate where possible; report and continue
                    log(f"  {s['id']} {split}: ERROR {type(e).__name__}: {e}")
                    continue
                log(f"  {s['id']} {split}: {out.get('verdict')} n={out.get('n')} roi={out.get('net_roi')} "
                    f"p={out.get('p')} bar={out.get('bar'):.2e} reasons={out.get('reasons')} -> {reg.state(tid)}")
    return ids

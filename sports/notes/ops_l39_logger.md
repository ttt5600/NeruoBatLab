# ops_l39_logger -- L39 logger fixes before the Nov 24-30 live window (agent `ops_l39_logger`, 2026-11-15)

Brief (lead): from notes/l39_expansion.md, (1) map the Dec snow series MKE/MSP/PIT/TTN, (2) refuse WPB / Napa / resort
snow by name with a logged reason, (3) `report` maker P&L on FILLED rows only, (4) coverage check of Nov rain + Dec
snow, (5) read-only check that lead_exec's L39 path reads the picks. Ops task: no trial, no outcome, no prereg needed.
Scratch: `sports/results/ops_l39_logger/`. No orders, no git, no deploy.

## Changes (worktree only) -- scripts/kalshi_wx_monthly_logger.py
- `SNOW = {**W.SNOW, **SNOW_EXTRA}`, SNOW_EXTRA = KXMKESNOWM MKE, KXMSPSNOWM MSP, KXPITSNOWM PIT, KXTTNSNOWM TTN.
  `kalshi_wx_winter.SNOW` (dev/sealed universe) deliberately NOT edited: these 4 series have 0 settled events
  (`results/ops_l39_logger/q_hist.py`), so they are forward-only additions.
- Station evidence: KXTTNSNOWM rules_secondary names "Trenton-Mercer Airport (KTTN; NWS climate report CLITTN)".
  MKE/MSP/PIT rules name only the city; contract terms SNOWOVERTIME (`results/ops_l39_logger/dl/SNOWOVERTIME.pdf`)
  say <area> = "primary official weather measurement station ... designated by the Source Agency" (TWC first, NWS
  second); Kalshi's KXSNOWXMASMKE/MSP/PIT rules name KMKE / KMSP / KPIT. ACIS MKE/MSP/PIT/TTN each return 31/31 Dec-2025
  days, 0 missing (Dec-2025 snow 8.8 / 16.0 / 18.4 / 8.1 in), NWS station meta resolves (q_hist.py output).
  Residual risk: MKE/MSP/PIT station is inferred, not stated in the monthly market's own rules.
- `EXCLUDE` (KXRAINWPBM: CLIDJT; KXRAINNAPAM: KAPC seasonal) + resort rule (rules mention Resort / Ski Area /
  OnTheSnow / Mountain). `open_ladders(month, skipped)` drops them, tick prints `EXCLUDED <reason>` and stores
  `state["excluded"]`; event-level skips (no station / no NWS meta) now print too.
- settle: `pnl_maker` = P&L only if tape-filled, else None; new `pnl_maker_if_filled` keeps the counterfactual.
- report: `maker_frame()` recomputes maker P&L from won + maker_q on `maker_filled == True` rows only (old rows carry
  pnl_maker on unfilled rows; also `astype(bool)` turned a missing maker_filled into True). Columns makerAll_* removed;
  maker_filled / maker_W / maker_L / maker_c/ct / maker_units are filled-only.
  On a copy of the deployed picks: 22 settled, 22-0, maker filled 15/22, +57.0 c/ct, 0.59 units -- identical to
  scorecards.py's filled-only figure (+57.0c/ct), so the old report's makerTape column was already right; the
  makerAll column (all 22) was the misleading one.
- New `coverage [YYMON]` command (read-only, exit 1 if any open ladder is UNMAPPED).

## Coverage (results/ops_l39_logger/coverage_26NOV.txt, coverage_26DEC.txt)
- Nov 2026 rain: 36 open KXRAIN* series (not 37: the l39_expansion census file open_2026-11-15.json also has 36
  rain series; "37" was a miscount). 34 included, each with a CLI station; excluded WPB (CLIDJT) and Napa (KAPC,
  events 01NOV26-30NOV26 / 01DEC26-31DEC26 / 01NOV26-31MAR27). UNMAPPED 0.
- Dec 2026 snow: 12 city monthly series included (BOS CHI DC DEN DET MKE MSP NYC PHIL PIT SEA TTN); resort monthlies
  Alta + Big Sky excluded, plus 6 Oct1-Dec24 resort ladders. UNMAPPED 0. Dec RAIN ladders were not listed yet on
  11-15: rerun `coverage 26DEC` around Dec 1-20.

## lead_exec L39 path (read-only)
- `lx.L39_PICKS` = ~/.claude/quant-kalshi/data/kalshi_wx_monthly_picks.jsonl (exists, 22 rows). Now: 0 signals
  (all settled). With result fields stripped, l39_signals returns all 22 with ticker / q = maker_q / pick_d = d
  matching row-for-row. It mirrors only unsettled, non-off_schedule picks with maker_q, so exclusions at the logger
  are sufficient. lead-exec launchd is loaded (StartInterval 900 s) vs max_age_h 3.0 -> fine. No lead_exec change.
- `pnl_maker: None` on new rows does not affect lead_exec (never read), scorecards.py (filters maker_filled first) or
  fund_stats.py (taker only).

## Tests
`tests/test_wx_monthly_logger.py` 8 tests (station map, exclusions, open_ladders, maker_frame, report, settle,
coverage incl. an unmapped series -> exit 1, lead_exec reader). Full suite: 495 passed, 2 failed in
tests/test_bet.py (bet.py price/count asserts; bet.py and its test are unchanged vs HEAD, unrelated to this work).

## Deploy proposal (lead / user runs; nothing copied)
    cp sports/scripts/kalshi_wx_monthly_logger.py ~/.claude/quant-kalshi/scripts/
    cd ~/.claude/quant-kalshi && <venv>/bin/python scripts/kalshi_wx_monthly_logger.py coverage 26NOV
(kalshi_wx_winter.py and lead_exec.py are unchanged; no launchd change.) Decision for the lead: the 4 new snow
series enter L39 LIVE in Dec 25-31 with no dev history at those stations (rule frozen, universe widened).

## Learnings
### Kalshi monthly snow rules do not always name the station
KXMKESNOWM / KXMSPSNOWM / KXPITSNOWM rules say only "snowfall in Milwaukee"; only TTN names KTTN/CLITTN. The
station has to be inferred from the contract terms ("primary official station") and the sibling KXSNOWXMAS<city>
series, which name KMKE/KMSP/KPIT (results/ops_l39_logger/q_rules.py, snow_open.json). Record such inferences as risk.
### A hard-coded station map fails silently when Kalshi lists new cities
4 of 12 Dec-2026 city snow ladders would have been skipped with no message. A `coverage` command that diffs the
exchange's open series against the map (exit 1 on UNMAPPED) turns that into a loud failure.
### astype(bool) on a column with NaN gives True
`pd.Series([np.nan]).astype(bool)` is True, so a missing maker_filled counts as a fill. Use `.eq(True)`.

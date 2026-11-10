# scorecards -- agent notes (agent `scorecards`, 2026-11-10)

(Named scorecards_agent.md, not scorecards.md: on this case-insensitive Mac that path IS notes/SCORECARDS.md, the
generated output -- the first run overwrote my notes.)

Standing user ask (STATE 11-10 "live scorecards for every strategy"). Deliverable: `scripts/scorecards.py` (read-only
over the deployed logs, runnable any time) -> `notes/SCORECARDS.md`.

## Rules (written before the script computed any card; I had eyeballed raw lead_exec rows while learning formats)
- Each card quotes the kill rule VERBATIM from its source line, read at run time (ACT_LIST.md B1-B6, FUND_BOOK.md 3b,
  edgelab registry preset_kill row, l32b notes, logger code), with file:line.
- Status, mechanical: KILLED-by-rule = written kill condition met; near-kill = one more loss trips a loss-count rule, or
  ROI CI upper < 0 / mean CLV < 0 before the judge n; too-early = < 10 settled and no kill; alive = otherwise.
- ROI = sum P&L / sum $ risked; 95% CI = bootstrap over bets (10,000, seed 2026). CI crossing 0 = not distinguishable.
- Script never orders/cancels. `--kalshi` = authenticated GETs only (/portfolio/fills, /portfolio/positions,
  /markets/{t}), cached at results/scorecards/kalshi_cache.json.

## Results (run 2026-11-10 04:25Z; numbers from notes/SCORECARDS.md)
- FIRED: L34 NHL vs opener -- 110 settled, mean CLV -0.0005 prob [-0.0044, +0.0033] <= 0 after 100
  (open_lines_logger.py:799 rule). ml 57: ROI -26.2%, CLV -0.0025; total 53: ROI -1.5%, CLV +0.0016 (a post-hoc split,
  no separate pre-set rule).
- near-kill: B3 F1 (paper CLV vs ask -0.0068 [-0.0120, -0.0016] n 29; vs mid -0.0018 [-0.0070, +0.0034]; user journal
  5-3 +$3.22 on $21), B6 NFLF clean record 4-7 -$0.23 (7 of the 8 losses allowed in the first 20), Q004 NHL/NFL/NBA
  (all below the pass bar; NHL ROI -27.6% [-49.4%, -5.0%] n 76).
- alive: B1 L39 paper 22-0 (maker 15/22 filled +4.0%), L32b 16-10 CLV +0.75 pts [-0.13, +1.63] n 26, L33 n 15.
- too-early: B2 DUL (starts 11-12), B4 TSA (8 resting live, 0 fills; paper 0 tape fills of 8 settled posts), B5 NHL
  live 2-3 -$2.04, paper 4-3 -25.5% n 7.
- Lock-in depth: 116 locks / 7 games / 1 night (11-09). 99c queue at the horn (KXNBAGAME +0 s) median 201,371
  (min 109,590), 40x the frozen N_QUEUE=5,000; over rungs median 24,866 (min 7,114). 0% of locks <= 5,000. Logger's
  share-of-volume reaching a joiner = 0.000 at every delay.

## FINDING: NFLF runaway exits also happened 11-08; the journal under-records (status: holds, not critic-reviewed)
- Exchange fills (GET 11-10): on 11-08 EVERY exit attempt filled, e.g. NYJ bought 9, 4 sell orders x 9 = 36 sold
  (journal exit_attempts 4); PHI bought 4, sold 75.58; WAS bought 5, sold 27. 9 tickers carry unintended NO legs.
- Clean record (entry + FIRST exit order only): 11 entries, 4-7, -$0.23 on $15.58. Journal view: 2-7, -$2.46 (it
  booked PHI at expiry -$1.82 although the first exit filled, +$0.11).
- Incident (every later sell): +$63.47, incl. BUFMIN unsettled, marked at last (MIN 0.01, BUF 0.99): MIN +$54.35,
  PHI +$35.56, BUF -$22.95. Luck, uncapped exposure; says nothing about the strategy.
- Independent check: my all-fills P&L per 11-08 ticker equals Kalshi /portfolio/settlements (revenue - yes_cost -
  no_cost - fee + $1 per YES/NO pair netted) on all 7 settled tickers, e.g. PHI 71.58 + 4 - 1.80 - 36.767 - 1.337 =
  +35.68; DAL -4.54; NYJ +3.86 (results/scorecards/check_settlements.py).
- Fills API convention: action=sell/side=no for both "sell YES" exits and "buy NO" entries -> -n YES. Verified vs
  position_fp (MIN 25-350=-325, BUF 3-63=-60) and NHL SJ journal P&L +0.122.
- Implication for the lead: the find_order miss was present from the first NFLF day, not only 11-09; STATE's "NFLF 1-7
  -$2.80" is the journal view. Re-run `scripts/scorecards.py --kalshi` after BUFMIN settles (close 11-12 01:15Z).

# ops_ot1_forward -- OT1 crypto-taker FORWARD paper logger (ops subagent, 2026-11-15)

Agent: ops_ot1_forward. Read-only GETs (Kalshi via scripts/kx_throttle.py, Deribit public API). NO orders, no git, no
.env / kill changes. launchd: user approved read-only paper loggers (STATE 11-13); lead brief 11-15.

## Question
Does the frozen OT1 rule (notes/options_implied_tails.md; edgelab trial ot1_crypto_taker-18d38e5c, rule
edgelab/rules/opt_tails.py:taker) make money forward on the LIVE book? DEV: +4.60% [+1.07, +7.62] n 387, p .0058 >
family bar 1.04e-3 (LEAD). Critic (inbox critic_ot1): plumbing clean, edge front-loaded (pre-2025-07 +7.3%, after
+1.7% [-4.6, +6.7]) -> forward paper only.

## Forward pre-registration (written 2026-11-15 ~14:40Z, BEFORE any forward decision exists; window from 2026-11-16)
- Universe: KXBTCD, KXETHD event `<S>-<yyMONdd>17` (17:00 ET daily close), every day incl. weekends.
- Decision: first launchd tick at/after 11:00 ET (America/New_York, DST-safe via zoneinfo). A decision counts as a BET
  only if the Kalshi snapshot starts in [11:00, 11:20) ET; later -> status `late` (logged, never scored); none by
  12:00 ET -> a `missed` day row.
- Inputs strictly older than the Kalshi snapshot: Deribit option trades of that currency in [t_d - 60 min, t_d), t_d =
  the Deribit fetch time, fetched BEFORE the Kalshi GET (t_k > t_d > every trade used; checked and logged per row).
  Smile/probabilities = the frozen DEV code (scripts/options_implied_tails.py smile, p_above_crypto, p_yes,
  tail_candidates) with T = t_k (tau_K = close - t_k).
- Kalshi quote: ONE GET /markets?event_ticker= per event (one moment): yes_bid/yes_ask + size at best. Rule input =
  that bid/ask (DEV used the hourly-candle close at T). Frame restricted to AMENDMENT 2 (3 closest-to-money tail
  candidates per event x side), exactly as DEV; picks outside that set are logged with in_frame=false (descriptive).
- Rule: edgelab/rules/opt_tails.py:taker (unchanged import): edge = pF - ask - 0.07 ask(1-ask) >= .03, ask in [.50,
  .99], spread <= 5c. 1 contract per pick (DEV: $1 stake at cost), BTC + ETH same day = one cluster.
- Depth: per pick, GET /markets/<t>/orderbook right after -> size at best ask on our side and cumulative within +1c/+2c.
- Fees logged: exact (the spec, `fee: rounding: exact`), cent-rounded per order for 1 contract and for a 100-lot.
- Settle: after 17:05 ET, one GET /markets?event_ticker= per event -> `result` of every market (picks + all rungs, for
  calibration). Unsettled -> retried every run.
- Judge (frozen spec): edgelab forward evaluation of ot1_crypto_taker-18d38e5c (adapter ot1_fwd.py), min_n 150 bets; KILL if the ROI CI (ask + exact fee) is entirely below 0 after 150 bets.
  Command: `python -m edgelab evaluate ot1_crypto_taker-18d38e5c --split forward` (spec `kill: roi_ci_below_zero, after_n 150`).
  Secondary (descriptive, reported, not a gate): ROI at cent-rounded 1-lot fee; one-bet-per-day (max edge among in-frame
  picks); live-depth-capped sizes. Expect ~1-2 bets/day -> 150 bets ~ 3-5 months.

## Files
- Logger `scripts/ot1_forward.py` (tick | report); adapter `edgelab/adapters/ot1_fwd.py` (vault.yaml kalshi_opt_tails
  forward_adapters.ot1_crypto_taker); scorecard `card_ot1` in scripts/scorecards.py; qs_job.sh case `ot1`;
  installer `scripts/install_ot1_agent.sh` (launchd com.jonathanwang.quant-sports-ot1, 120 s).
- Logs: ~/.claude/quant-sports/data/kx_forward/ot1.jsonl (dates >= 11-16), ot1_quotes.jsonl (every rung), dates
  before 11-16 -> ot1_shakedown*.jsonl; manual `tick --force-now` -> ot1_manual*.jsonl (never scored).
- Stop: `launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-ot1`.

## Log
- 14:30Z started; inbox `started` row.
- 14:33Z manual forced run (09:33 ET, off-time, ot1_manual.jsonl): Deribit www host 391 BTC / 117 ETH trades in the
  last 60 min, smile 18 / 7 points, input age 32 s / 98 s before the Kalshi GET (inputs_older_than_quote true); KXBTCD
  80 rungs, KXETHD 40; 1 rule pick on all rungs (KXBTCD-26NOV1517-T66999.99 NO ask .96, p_yes .0038, edge +3.4c,
  size at best 106, depth +1c 1,006) outside the amendment-2 frame -> no bet. Live spreads 1c on most tail rungs.
- Adapter sandbox test: the manual pick forced to status bet + a fake settle row passed vault.resolve_outcomes +
  schema.enforce (PIT clean: inputs 14:33:03 < quote 14:33:35 <= decided 14:33:37 < event_start 21:59:59).
- Found while inspecting (frozen DEV construction, NOT changed): P(S > K) from -dC/dK with a piecewise-linear smile is
  NOT monotone in K. Live: T67249.99 and T67499.99 p_yes = 0.0 (clipped) while T67749.99 = .0042. DEV: 432 of 1,020
  crypto events have >= 1 non-monotone step (834 steps), 427 of 13,615 rungs clipped to exactly 0 or 1 (data/opt_tails/
  implied.parquet). DEV OT1 bets with p_T == 0: 0 of 387 (one < .001), so the DEV result does not ride on the artefact,
  but a forward pick on a clipped rung is possible. A v2 (monotone / smoothed smile) would be a NEW trial.
- 14:3xZ launchd job installed from ~/.claude/quant-sports; the 11:00 ET run today writes ot1_shakedown.jsonl
  (11-15 < forward start), the first scored decision is 2026-11-16 11:00 ET.

## Caveats
- Mac asleep at 11:00-11:20 ET -> that day's picks are `off_time` (not scored); asleep until 12:00 -> `missed` row.
- Settlement is the exchange `result`; no CLV mark is logged (critic: CLV 1 h before settle is mostly outcome).
- Fill realism: the logged ask is the top of book at one moment; size_at_best / depth_{0,1,2}c say how many would fill.

## Verification (launchd, 2026-11-15)
- First launchd run 06:37 PT rc=0 idle (state/ot1_heartbeat). DECISION run 08:00:12 PT = 11:00:11 ET (logs/ot1.log):
  `2026-11-15 decided: 0 bets (0 rule picks all rungs); KXBTCD ok trades 251 smile 13 age 14.2s; KXETHD ok trades 94
  smile 6 age 233.5s`. ot1_shakedown.jsonl day row + 120 quote rows (80 BTC + 40 ETH rungs) in ot1_quotes_shakedown.jsonl;
  inputs_older_than_quote true for both; spot BTC 64,712 / ETH 1,809; tauD 16.0 h. 0 picks is plausible (gaps >= 3c
  were ~1/day in DEV and live spreads are 1c).
- ETH coverage is thin: 6 smile points -> implied p for only 6 of 40 ETH rungs (BTC 31 of 80); no extrapolation by rule.
- Settle source checked on a finished event (KXBTCD-26NOV1417, GET /markets?event_ticker): 80/80 finalized, results
  22 yes / 58 no, expiration_value 64,386.38. Today's shakedown + manual events settle via launchd after 17:05 ET.

## Learnings
### Kalshi /markets list carries size at best (one GET = one-moment book snapshot)
`yes_bid_size_fp` / `yes_ask_size_fp` come with yes_bid/ask_dollars in GET /markets?event_ticker=, so a whole 80-rung
ladder snapshot is one request at one moment; /orderbook (orderbook_fp, yes_dollars/no_dollars bids only) adds depth.
Source: scripts/ot1_forward.py, ot1_shakedown.jsonl.

### The frozen OT1 implied P(S>K) is not monotone in K
Central-difference -dC/dK on a piecewise-linear smile gives clipped 0s and reversals: 432 of 1,020 DEV crypto events
have >= 1 reversal, 427 of 13,615 rungs clipped to 0/1 (data/opt_tails/implied.parquet); live 11-15 T67249.99 = 0.0
while T67749.99 = .0042. DEV bets on clipped rungs: 0 of 387. Any v2 smoother is a new trial.

### ETH Deribit options are too sparse for a 60-min smile on most rungs
11-15 11:00 ET: 94 ETH trades -> 6 OTM smile points -> implied p on 6 of 40 KXETHD rungs (BTC 13 points, 31 of 80).
Expect OT1 forward bets to be mostly BTC.

# ot1_extension -- OT1's options-implied tail fade on OTHER Kalshi crypto ladders

Agent: ot1_extension (research subagent, 2026-11-16). Dev only (<= 2026-04-10), no orders, no git.
Script: `scripts/ot1_extension.py` (census -> deribit -> implied -> candles -> frame); scratch in
`results/ot1_extension/`. Reuses `scripts/options_implied_tails.py` (smile, p_above_crypto, p_yes, tail_candidates,
deribit_window) and the rule `edgelab/rules/opt_tails.py:taker` UNCHANGED (k = .10, margin .03, spread <= 5c,
ask in [.50, .99], exact quadratic fee). Family kalshi_data_ladders (52 trials before this).

## Why / what is new vs OT1
OT1 = KXBTCD/KXETHD 17:00 ET daily above/below, decision 11:00 ET on the close date: dev +4.60% [+1.07, +7.62] n387,
front-loaded (pre-2025-07 +7.3%, since +1.7% n.s.); maker dead. OT1 descriptive: rows where the event had been listed
> 19 h at T (= weekly-listed Friday events) +8.6% (74 bets) -- the hint that longer-lived ladders stay staler.
Not redone here: OT1 itself, the maker variant, index arm, Q012 hourly.

## Census (outcome-free; tickers / listing / settlement)
Kalshi crypto series (GET /series?category=Crypto, results/ot1_extension/series_crypto.txt, 302 series). Relevant:
- KXBTC / KXETH: daily RANGE ladders (`between` brackets + 2 tail rungs), same 17:00 ET event set as KXBTCD/KXETHD
  (hourly series also carries hourly events; only the 17:00 ET one is used). Settles on the 60-s simple average of
  CF Benchmarks BRTI / ETHUSD_RTI before 5 PM ET (rules_primary, live GET 11-16).
- Weekly: there is NO separate weekly series for "above X on date"; the FRIDAY 17:00 ET events of KXBTCD (and later
  KXETHD, KXBTC, KXETH, KXSOLD, KXSOLE) are listed ~7 days ahead (e.g. KXBTCD-25MAR1417 open 2025-03-07 21:00Z;
  KXBTCD-24NOV0817 open 2024-11-01). KXBTC Friday events were NOT weekly-listed in 2024 (KXBTC-24NOV0817 opened
  the day before). KXBTCMAXW (weekly max) has 2 events total (2024) -> nothing to test.
- SOL: KXSOLD (above/below) and KXSOLE (range), CF SOLUSD_RTI 60-s average; KXSOLD/KXSOLE absent on 2025-03 and
  2025-09 lookups, present 2026-03 (start date: census table below). Deribit SOL options = linear SOL_USDC-* under
  currency=USDC (history host); 23 SOL_USDC trades in 11:00 ET-1h on 2025-06-11, 36 on 2026-03-12, spread across
  4-6 expiries -> OT1's ">= 5 OTM points of ONE expiry in the hour" will usually fail. Feasibility counted below.
- Monthly / yearly: KXBTCMAXM (2025), KXBTCMAXMON / KXBTCMINMON / KXETHMAXMON / KXETHMINMON / KXSOLMAXMON (2026),
  KXBTCMAXY/MINY, KXBTCY/KXETHY (EOY range, 1 event each). These are ONE-TOUCH barrier contracts ("ever above $X
  through 11:59 PM ET on <date>") -- path-dependent, so OT1's terminal-density rule does not apply as is; and ~1
  event per month per series. Counts: results/ot1_extension/census_longdated.csv.

## PRE-REGISTRATION (written 2026-11-16 after market LISTS only were requested; no candle, quote or `result` read.
## The market-list response contains `result`; the builder writes it to a hidden column only the vault reads.)

### Common (identical to OT1 unless stated)
- Implied P(YES): OT1 smile() on Deribit option trades in [T - 60 min, T); target expiry = first Deribit expiry
  (08:00 UTC) at or after the Kalshi close among instruments traded in the window, >= 5 OTM points with both sides,
  sqrt(tau_D/tau_K) standardised-moneyness map, linear smile interpolation, no extrapolation, central-difference
  -dC/dK. When tau_D >> tau_K (weekly arm: next-day or next-Friday expiry) the same map is used (frozen, not re-fit).
- Kalshi quote = close of the last hourly candle ending <= T (yes_bid / yes_ask). Inputs strictly older than T
  (Deribit trades < T; candle end <= T). CLV mark = last candle ending <= close - 1 h.
- Candidates: OT1 amendment 2 (3 tail candidates closest to the money per (event, side of spot), per decision).
- RULE = edgelab/rules/opt_tails.py:taker, unchanged. Taker only. 1 contract each.
- Cluster = crypto|ET close date (all assets / arms of one close date share one cluster).
- Vault: dataset `kalshi_opt_tails_ext`, DEV <= 2026-04-10, SEALED 2026-04-11..2026-10-31 (never fetched), FWD >= 11-16.

### Trials (k = 3)
- X1 RANGE_DAILY: KXBTC / KXETH 17:00 ET events, T = 11:00 ET on the close date. (Same days as OT1 -> NOT
  independent of OT1; reported as such.)
- X2 WEEKLY_AHEAD: every 17:00 ET event listed > 48 h before its close (KXBTCD, KXETHD, KXBTC, KXETH; BTC/ETH ONLY -- SOL weekly
  events are excluded), decisions at 11:00 ET on each of the 4 days before the close date (only if the event was
  open at T - 1 h). One bet per market = the EARLIEST decision (ahead4 -> ahead1) at which the rule fires.
- X3 SOL_DAILY: KXSOLD / KXSOLE 17:00 ET events, T = 11:00 ET close date, Deribit SOL_USDC options (mark_price is
  USD for linear options -> divided by index_price before OT1's smile(), which multiplies it back).
- NOT tested (reason written now): monthly/yearly one-touch (path-dependent; needs a barrier model = a new rule, and
  DEV holds <= ~15 independent months per series); KXBTCMAXW (2 events).

### Bar, pass, kill
Family kalshi_data_ladders 52 -> 55 trials -> bar ~ 0.05/55 = 9.1e-4. PASS = edgelab dev verdict pass and n >= 150
-> sealed ask to the user (nothing more). n < 150 -> underpowered, no verdict. Too-good guard: ROI > +10% on > 150
bets = bug until rows inspected. Reported for every trial: pre-2025-07 vs since 2025-07 (OT1 decayed), BTC vs ETH,
and for X2 by arm (ahead4..ahead1). Kill: all three fail/underpowered -> "OT1 does not extend" into FUND_BOOK 3d.
Power (from OT1): only >= +4-5% ROI can pass at ~1e-3 with a few hundred clustered bets.

## Log
- 11-16 census started (scripts/ot1_extension.py census -> results/ot1_extension/markets.parquet, census.log).
- 11-16 Registered BEFORE any candle/quote/outcome: x1_range_daily_taker-6e34023c, x2_weekly_ahead_taker-1b764fd2,
  x3_sol_daily_taker-c829cc84 (family 55 -> bar 9.09e-4). Rule edgelab/rules/opt_tails_ext.py, vault
  kalshi_opt_tails_ext (dev 2024-10-24..2026-04-10), loader kalshi_opt_tails_ext, specs edgelab/specs/x*_taker.yaml.
- 11-16 Long-dated census (results/ot1_extension/census_longdated.csv; counts/volume only): DEV-closing events
  KXBTCMAXM 10 (monthly max, 2024-03..2025-12, 48 markets), KXBTCMAXMON 3, KXBTCMINMON 3, KXETHMAXMON 3, KXETHMINMON 2,
  KXSOLMAXMON 3, KXSOLMINMON 1, KXBTCMAXY/MINY 2/3, KXETHMAXY/MINY 2/2, KXBTCMAXW 2, KXBTCY/KXETHY 0 (first closes
  2027-01-01). ~13 distinct DEV months in total -> a clustered test cannot reach p ~ 1e-3 (needs dozens of independent
  clusters even at +10%). Not tested, as pre-registered. Volume is large since 2026 (KXBTCMAXMON 27.3M contracts over
  11 events), so a FORWARD barrier-model paper trial is the only honest route (proposal, below).
- 11-16 Daily/weekly census (results/ot1_extension/census.log; 17:00 ET events closing <= 2026-04-10; weekly = listed
  > 48 h, median 169 h): KXBTC 507 daily + 14 weekly (weekly from 2026-01-02); KXETH 507 + 14; KXBTCD weekly 75
  (Fridays since 2024-11-01); KXETHD weekly 14 (since 2026-01; in 2025 ETH Fridays were listed the day before);
  KXSOLE 109 daily from 2025-10-07, KXSOLD 108 from 2025-12-09 (+14 weekly each, excluded from X2 by prereg).
- 11-16 Implied (outcome-free; results/ot1_extension/implied.parquet): markets with a mapped P(YES): KXBTC same_day
  8,243 / 21,750, KXETH 4,417 / 20,990; X2 arms ahead4..1 KXBTCD 3,074 / 2,574 / 2,209 / 2,087 of 3,844 each; SOL only
  95 of 16,331 (a SOL smile -- >= 5 OTM points of one expiry in the hour -- exists on 71 of 465 days 2025-01..2026-04,
  and almost never for the first expiry after a 17:00 ET close) -> X3 will be underpowered (stated before candles).
- 11-16 Candles: 6,782 candidate rows / 6,137 markets (5,939 new fetches; rest from OT1's cache).
- 11-16 SOL Deribit units check (SOL_2025-05-29): linear SOL_USDC mark_price is USD; after /index and OT1's x index,
  the inverted mark IV matches Deribit's trade iv (e.g. 174-C .734 vs 70.4%, 150-P .736 vs 72.7%).
- 11-16 Frame (results/ot1_extension/frame.parquet): 6,782 candidate rows, 6,761 quoted; all T = 11:00 ET, inputs < T,
  quote_ts <= T, T < close (checked). Median quote age 0 h (max 42 h on a few ahead rows: no candle that hour; OT1 same
  convention). X3 SOL: 23 candidate rows -> 0 bets.

## Results (edgelab dev, ONE run each, lab/edgelab_registry.jsonl; bar 9.09e-4)
| trial | n | net ROI | 95% CI (cluster crypto day) | p | verdict |
|---|---|---|---|---|---|
| X1 range daily (KXBTC/KXETH) | 206 | +0.07% | [-4.51%, +4.22%] | 0.49 | FAIL (p, half <= 0, concentration) |
| X2 weekly ahead (BTC/ETH)   | 73  | +0.43% | [-6.47%, +6.45%] | 0.44 | UNDERPOWERED (n < 150), no verdict |
| X3 SOL daily                | 0   | --     | --                | --   | UNDERPOWERED (0 bets: no SOL smile) |
Descriptives (scripts/ot1_extension_desc.py -> results/ot1_extension/desc.json; own bootstrap, fee rounding differs
slightly from edgelab): X1 pre-2025-07 +1.8% [-4.2, +7.0] n108 vs since -1.5% [-8.9, +5.3] n98; BTC +1.1% n138, ETH
-1.6% n68; 205/206 bets are NO on a bracket. X2 pre +5.3% [-1.5, +12.3] n31 / 16 clusters vs since -1.7% [-13.1, +6.9]
n42 / 14; KXBTCD +6.6% [-0.0, +12.0] n54 (only 27 weeks) vs range/ETH negative; arms ahead4 +3.2% (26), ahead3 -9.2%
(18), ahead2 +13.9% (10), ahead1 +1.3% (19) -- no "earlier = staler" pattern. All small-n, not tests.
Calibration (results/ot1_extension/calibration.txt; quoted candidates, spread <= 5c, implied pT / Kalshi mid / realised):
KXBTC range (.05,.10] .0733 / .0866 / .0727 (867); KXETH .0719 / .0925 / .0859 (384); ahead arms .0732 / .0850 / .0714
(1,176). The options smile is calibrated on range brackets and on 1-4-day horizons too, and the Kalshi mid is 1-2c rich
on tails -- but that is spread-sized, so few rungs clear the 3c post-fee gap (206 of ~4,500 X1 candidates) and those
that do do not pay. Brier implied vs mid: KXBTC .04581 vs .04563, KXETH .03989 vs .03959, ahead .06046 vs .06034
(mid marginally better).

## Verdict
OT1 does NOT extend: X1 fails (+0.07%, CI +/-4%, n206), X2 and X3 underpowered (X2 n73 ~0%; X3 0 bets -- Deribit SOL
options too thin for a one-hour smile). Monthly/yearly one-touch: not testable on DEV (~13 independent months). Same
decay as OT1: positive pre-2025-07, ~0 or negative since. -> FUND_BOOK 3d candidate: "options-implied tail fade on
Kalshi crypto RANGE / weekly-ahead / SOL ladders" (dev dead or underpowered, 11-16).
PROPOSALS (lead/user decide; nothing done): (1) no forward logger for X1-X3. (2) The monthly one-touch ladders
(KXBTCMAXMON etc., 27.3M contracts over 11 events) would need a separate pre-registered BARRIER model (e.g. reflection
2 x P_RN(S_T > K) on the monthly Deribit smile) judged FORWARD only -- DEV cannot power it.

## Learnings
### OT1's options-vs-Kalshi tail fade does not extend to range brackets, weekly-ahead or SOL ladders
X1 KXBTC/KXETH daily range: +0.07% [-4.51, +4.22] n206 (p .49); X2 weekly-listed events decided 1-4 days ahead +0.43%
[-6.47, +6.45] n73 (underpowered); X3 SOL 0 bets. Source: notes/ot1_extension.md, lab/edgelab_registry.jsonl.
The OT1 lead is specific to the KXBTCD/KXETHD threshold ladder and its pre-2025-07 period.

### Deribit's smile is calibrated on Kalshi range brackets, but Kalshi's tail richness is only spread-sized
Bracket tails p in (.05,.10]: implied .0733 / Kalshi mid .0866 / realised .0727 (KXBTC, 867 quotes); weekly-ahead
.0732 / .0850 / .0714 (1,176). A 1-2c rich mid is eaten by the ask + 0.07p(1-p) fee. Source:
results/ot1_extension/calibration.txt.

### Deribit SOL options are too thin for a one-hour smile
SOL_USDC linear options (history host, currency=USDC; mark_price in USD): 23-36 trades/h across 4-6 expiries; >= 5
OTM points of one expiry in [T-1h, T) on 71 of 465 days, almost never for the expiry right after a 17:00 ET close.
A SOL model needs forward book/ticker snapshots or a longer window, not OT1's construction.

### "Weekly" Kalshi crypto ladders are the Friday 17:00 events listed 7 days ahead
No separate weekly above/below series; KXBTCD Friday events listed ~169 h ahead since 2024-11, the other BTC/ETH/SOL
series only since 2026-01 (KXBTCMAXW: 2 events, 2024). Source: results/ot1_extension/census.log.

### Kalshi monthly/yearly crypto max/min ladders are one-touch barriers with too few independent months to test
~13 DEV months across all series (results/ot1_extension/census_longdated.csv) -> no clustered test reaches ~1e-3;
a barrier model must be judged forward.

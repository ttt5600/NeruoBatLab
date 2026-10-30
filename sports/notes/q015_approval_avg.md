# Q015 -- Kalshi approval-average markets vs the already-published average (PRE-REGISTRATION)

Written 2026-10-29 BEFORE any Kalshi price was pulled or any P&L computed for these series.
Disclosure: while checking the input source I printed settlement values (expiration_value) for 12 recent KXAPRPOTUS
events and compared them with the RCP daily history (match rates below). No prices, no bets, no ROI were seen.

## 1. Contracts (read from the Kalshi API /series + market rules + contract-terms PDFs, cached in data/q015/)

| series | cadence | source | settles on | strikes | fee_type | settled events |
|---|---|---|---|---|---|---|
| KXAPRPOTUS (incl. legacy `APRPOTUS-` tickers) | weekly, Fri 11:00 ET (a few 10:00/12:00/16:00 closes) | RealClearPolitics "RCP Average" Approve value | value displayed at exactly 11:00 AM ET on the expiration date (terms APPROVE.pdf); post-expiration revisions ignored | 0.1 grid; "between" = floor..cap inclusive (e.g. 41.0 to 41.2), "greater" floor X = X+0.1 or above, "less" cap X = X-0.1 or below (from yes_sub_title) | quadratic x1 | 261 (2021-10-15 .. 2026-10-23) |
| KXTRUMPAPPROVE | daily, 13:00 ET (close 12:59 ET) | RCP, "Approve" column of "RCP Average" row of "President Trump Job Approval" | value at exactly 1:00 PM ET (POLITICALSTAT.pdf; source's 1-decimal precision; first publication only) | exact-value markets E38.4 (=38.4), A x (>x), U x (<x) | quadratic x1 | 58 (2026-09-02 .. 2026-10-29) |
| KXTRUMPVH | weekly, 10:00 ET | VoteHub approval polling average | value at 10:00 AM ET | one "above X" market per week | quadratic x1 | 20 single markets -> forward logger only |

Markets open: KXTRUMPAPPROVE at 10:01 ET the day before close. Settlement rule check (must pass before scoring): on every
market whose expiration_value is numeric, yes(expiration_value) computed from the strike fields == Kalshi result.
Fee: taker 0.07*p*(1-p) per contract (unrounded, as in other studies); sensitivity: Kalshi ceil-to-cent on a 1-contract order.

## 2. Why the "recompute from polls" version cannot be backtested point-in-time
- RCP poll tables are behind Cloudflare (HTTP 403 for curl/browser UA and for WebFetch); web.archive.org is unreachable
  from this machine (TLS reset / 429 / 502) and WebFetch is refused for it. So the historical list of polls on the
  RCP page at time t (which polls were in, which aged out, when each appeared) is NOT available.
- Pollster release times are not timestamped anywhere I can reach; VoteHub's API gives `created_at` at DATE granularity only.
- What IS available point-in-time: (a) RCP's own daily average history JSON
  (realclearpolitics.com/epolls/json/{6179,7320,8656}_historical.js: Trump-1, Biden, Trump-2), 2017-01-27 .. 2026-02-25
  (the feed stopped updating on 2026-02-25); (b) Kalshi's own settlement values: KXTRUMPAPPROVE gives the RCP value at
  13:00 ET every day from 2026-09-02, published at settlement.
=> The historical test is the weaker but honest version: does the Kalshi price lag the ALREADY-PUBLISHED RCP average
   (persistence + empirical drift distribution)? The poll-level recompute (polls released but not yet averaged) goes to
   a forward paper logger (scripts/q015_logger.py) that records VoteHub polls with created_at, Kalshi books and the
   latest published value; it is not part of the trials below.

## 3. Information rule (no lookahead)
- Decision time tau (ET). Info value I(tau) = the most recent of:
  (a) RCP history entry dated <= date(tau) - 1 day (an entry dated D is treated as known only from 00:00 ET on D+1,
      whatever time of D it represents; it is never used on its own date);
  (b) a KXTRUMPAPPROVE expiration_value whose 13:00 ET observation time is <= tau (settlement values are public at close).
  Source used is recorded per event. Events with no I(tau) are excluded (2026-02-26 .. 2026-09-01 has none for weekly).
- Residual risk (cannot be checked with available data): if RCP retroactively revised the history JSON, (a) could leak.
  Guard: the same-day history entry matches the Friday 11:00 settlement exactly on 61.1% of 90 overlapping weeks and the
  previous-day entry on 35.6% -- i.e. the entries are NOT a copy of the 11:00 values. Any trial result above +30% ROI
  is treated as a bug and audited on raw rows before it is reported.

## 4. Model (frozen)
- Delta_k = h[D+k] - h[D] on the RCP daily history restricted to 2017-01-27 .. 2021-09-30 (before any Kalshi
  approval market existed -> no overlap with betting data). Empirical pmf on the 0.1 grid, mixed 0.95 empirical +
  0.05 discretised Normal(0, 2*sd_k) so no value has zero probability.
- P(V_close = v) = pmf_k(v - I(tau)); P(YES) = sum over v of P(V=v) * yes(v).
- Bet rule: buy YES at yes_ask if P - ask - fee(ask) >= 0.05; else buy NO at (1 - yes_bid) if (1-P) - (1-bid) - fee >= 0.05.
  One contract-dollar-normalised bet per market (ROI = sum P&L / sum cost). Quote required: last hourly candle ending
  <= tau, 0 < yes_bid, yes_ask < 1, ask > bid.

## 5. Trials (k = 4 -> Bonferroni bar one-sided p < 0.0125)
| id | series | decision tau | info | horizon pmf |
|---|---|---|---|---|
| T1_wk24 | KXAPRPOTUS | close - 24 h | latest <= tau (typically Wed EOD history / Wed 13:00 Kalshi) | Delta_2 |
| T2_wk72 | KXAPRPOTUS | close - 72 h | latest <= tau | Delta_4 |
| T3_day24 | KXTRUMPAPPROVE | previous day 13:05 ET (close - ~24 h) | previous day's 13:00 settlement value | Delta_1 |
| T4_wk2 | KXAPRPOTUS | close - 2 h | latest <= tau | Delta_1 |

## 6. Split, statistics, bar
- Per trial: eligible events (have I(tau)) sorted by close time; first 70% = DEV, last 30% = SEALED. SEALED event
  tickers are written to data/q015/sealed_events.json; `dev` never fetches their candles. SEALED is scored once, only on
  the user's yes, with the frozen rule.
- Stats: ROI per $ at ask + fee; one-sided p from an EVENT-clustered bootstrap (10,000 resamples, seed 1015) of
  ROI = sum(pnl)/sum(cost); 95% percentile interval. Observed ROI is quoted (not the bootstrap mean).
- PASS (DEV): p < 0.0125 AND ROI > 0 AND both time-halves of DEV (split by event) ROI > 0 AND >= 30 events with bets.
- Too-good guard: ROI > +30% or win-rate far above model probability -> audit rows (info time vs tau, quote time) first.

## 7. Kill criteria
- Settlement rule check fails on > 1% of numeric-ev markets -> fix semantics before scoring (no trial counted).
- All 4 trials DEV ROI <= 0 -> historical-persistence version of Q015 DEAD; only the forward poll-level logger remains.
- A trial with < 30 DEV events with bets is UNDERPOWERED: reported, never counted as a pass.
- Forward logger: kill if after 8 weekly + 40 daily settled events the logged rule ROI <= 0.

## 8. Power check (before prices)
Event counts per trial are printed by `q015_approval_avg.py events` (no prices, no results). Rough MDE: with per-$ return
sd ~1.5 on mid-priced contracts, n bets gives MDE ~ 2.5*1.5/sqrt(n): 300 bets -> ~22%, 1,000 -> ~12%. A clustered design
is weaker. This study can only detect a LARGE lag; a 2-5% edge is undetectable here and would need the forward logger.

## Results
(appended below after the dev run, unedited above this line)

### DEV results 2026-10-29 (scripts/q015_approval_avg.py dev; ledger dataset q015_approval_avg)
Rule check: 0 mismatches / 1,196 numeric-ev markets; 0 unparsed strikes. Split (`events`): T1/T2 234 eligible events
(DEV 163, 2021-10-15..2024-11-15; SEALED 71, 2024-11-22..2026-10-23), T3 56 (DEV 39, 2026-09-04..10-12; SEALED 17),
T4 235 (DEV 164; SEALED 71). DEV is entirely Biden-era; all Trump-2 weekly events are SEALED (untouched).

BUG found after the first run: the Delta_k pmf included the deltas spanning the Trump-1 -> Biden series switch
(2021-01-20..27, approval jumps ~14 pts), fattening the tails (Delta_2 pmf sd 0.702 -> 0.457 after the fix; P(Delta=0)
unchanged 0.131 -> 0.130). Both runs are in the ledger; `_pmffix` rows supersede the `q015_T*` rows. Verdicts identical.

| trial | bets | events | ROI (observed) | 95% event-clustered CI | p (bar .0125) | halves | verdict |
|---|---|---|---|---|---|---|---|
| T1_wk24 | 257 | 127 | -17.4% | [-28.6, -6.1] | .998 | -0.7% / -33.7% | FAIL |
| T2_wk72 | 159 | 83 | +6.2% | [-5.1, +18.6] | .152 | +22.4% / -6.3% | FAIL (not distinguishable from 0) |
| T3_day24 | 78 | 31 | -1.9% | [-22.3, +19.0] | .573 | +5.5% / -10.9% | FAIL |
| T4_wk2 | 39 | 32 | -50.3% | [-88.6, -0.2] | .975 | -24.4% / -79.1% | FAIL |
(ceil-to-cent 1-contract fees: T1 -18.5%, T2 +5.3%, T3 -2.0%, T4 -51.2%.)

Diagnostics (DEV quoted markets, data/q015/_x7.py):
- The market is better calibrated than the persistence model: Brier model vs mid T1 .1316 vs .1148, T3 .0923 vs .0898,
  T4 .1537 vs .1406; T2 tie (.1171 vs .1179).
- Why: the pre-Kalshi pmf is too wide for the betting era. Realised settle-minus-info sd: T1 0.269 (pmf 0.457),
  T3 0.277 (pmf 0.295), T2 0.557 (pmf 0.711). Delta_2 sd by era: Trump-1 0.417, early Biden 0.472, DEV Biden 0.324,
  Trump-2 (to 2026-02) 0.270 -- the average got smoother over time. The model fades confident market brackets and loses.
- T4 (2 h before close): the market sees Friday-morning RCP updates our rule cannot use (history entries usable only
  from the next day) and quotes are thin (227 of 1,284 markets quoted, mean spread 0.507). Not a lookahead bug: the
  loss direction is the opposite of what leakage would produce.
- POST-HOC LEAD (not a test): the bracket containing the info value hit 58.3% in T1 DEV (n 151) vs market mid 47.8%;
  in T3 38.5% (n 39) vs mid 36.2%. A narrower pmf might flip T1. This is in-sample tuning territory: any re-test needs a
  fresh pre-registration and forward data (the SEALED half belongs to the frozen rule above).

VERDICT: historical persistence version FAILS on all 4 DEV trials (kill criterion "all <= 0" not met only because
T2 is +6.2% with a CI spanning zero). Q015 as specified: no edge shown; the market does not lag the published average.
The poll-level recompute is untested (no point-in-time poll data) -> forward logger only.

### Forward logger
scripts/q015_logger.py `tick` / `report` (NOT installed in launchd). First real tick 2026-10-29T23:39Z: 18 open markets,
1,028 VoteHub polls backfilled (flagged backfill=True, excluded from the poll-pressure check), info 38.4 (10-29 13:00 ET).
Offline tests: tests/test_q015.py (11 pass).

# boost_hedge -- existing-customer promos + cross-venue arb (agent started 2026-11-25 02:31Z = 11-24 18:31 PT)

Agent: boost_hedge (research subagent of the quant-trader lead; protocol sports/notes/AGENT_PROTOCOL.md).
Read-only: no orders, no sign-ups/logins/credentials, no git, no launchd/.env. Scratch: sports/results/boost_hedge/.
Kalshi calls: <= 1 req/s, never 03:00-06:00Z, never minutes :00-:10 (live executor).

## Questions
A. Which EXISTING-customer promos (boosts, deposit matches, reloads, no-sweat, missions, volume/maker rebates, Kalshi LIP /
   volume incentives / interest) are live or recurring on Kalshi, DraftKings Predictions, Fanatics Markets, Polymarket US,
   Gemini Predictions, Novig, ProphetX, OG.com for a CA user, and what does each LOCK when hedged on Kalshi (ask + taker
   fee ceil(0.07*C*p*(1-p))) -- or its unhedged EV when hedging is banned?
B. Do true cross-venue arbs exist between venues with PUBLIC read-only price feeds (Kalshi, Polymarket US; others if a
   public endpoint is allowed) on the same game-winner events (NFL Thanksgiving + weekend, NBA, CFB)?

## Do-not-redo (read 2026-11-25 02:35Z)
- New-customer offers: notes/promo_verify_1124.md + notes/PROMO_TRACKER.md (Fanatics Offer 1, PM US $25 Bonus, Gemini,
  ProphetX, Novig, OG, DK claimed). Not re-studied here.
- L40 Kalshi LIP weather: KILLED 10-29 (weather programs ended). L40b non-weather LIP paper: interim kill under both fill
  models (EDGES_AND_AGENTS 11-05 line; FUND_BOOK row 98). Maker adverse selection = recurrent killer (L19, E5, RD2, H1).
- Cross-venue: M2 Kalshi vs Polymarket game winners "median 0.5-1c, cheapest round trip $1.021" (dead); weather 0/48 under
  $1; PM US vs Pinnacle 0/116 +EV; XV-PM in-game lag dead (PM US real ask -3.60%); XV-US T1 dev +1.73% [-12.59,+15.93] dead.
  -> PRIOR for B: expect ~zero arbs. B is a fresh, bounded measurement on PM US (not global PM) at real asks with depth.
- PM US public gateway responses are Cloudflare-cached `max-age=30` (pmus_lag_logger.py docstring): a crossed view may be
  a stale cache. PM US fee per pm_full_census: 0.095*C*p(1-p) after day 7 (0.0695 days 1-7 for new accounts).

## PRE-REGISTRATION B (written 02:40Z, before any paired price is sampled)
- Universe: game-winner markets present on BOTH Kalshi (KXNFLGAME, KXNBAGAME, KXNCAAFGAME) and PM US (moneyline `aec-`
  slugs from gateway /v2/leagues/{lg}/events), matched with pmus_lag_logger.live_games logic (team codes + ET date),
  start time within [now-4h, now+120h].
- Snapshot: every ~60-90 s cycle, one Kalshi batched GET /markets/orderbooks for all matched tickers (both team markets)
  and one PM US GET /v1/markets/{slug}/book per game (record Date/Age/cf-cache-status -> as_of). Kalshi <= 1 req/s.
- Package cost for "team X wins" bought on Kalshi = min(YES ask on X market, NO ask on opponent market) walked to C
  contracts; on PM US = ask of X's side walked to C. Two directions per game (X on Kalshi + Y on PM US, and reverse).
- Fees (taker both legs): Kalshi ceil-to-cent(0.07*C*p*(1-p)) per leg; PM US 0.095*C*p*(1-p) (or the market's
  feeCoefficient if served), no rounding assumed. net(C) = C - cost_K(C) - cost_P(C) - fees, per contract = net/C.
- EVENT = net per contract >= $0.01 at C = 10 (both books deep enough for 10). Also logged: C=1 top-of-book net, max C
  with net >= 0.
- Staleness rule: an event counts as EXECUTABLE-CANDIDATE only if the PM US view's as_of age <= 30 s AND the crossed state
  is seen in >= 2 consecutive cycles with a refreshed PM US as_of (different as_of), i.e. it survived a cache refresh.
- SUCCESS ("actionable arb exists"): executable-candidate events in >= 3 distinct games AND >= 1% of paired game-snapshots.
- KILL: 0 executable-candidate events -> "no cross-venue arb at public-feed cadence" (consistent with M2). Anything between
  -> "rare; not actionable by hand" (a human placing two orders on two apps needs >= ~60 s persistence).
- Settlement-rule check before any count is called an arb: tie/OT/postponement/cancellation rules read from both venues'
  market rules (Kalshi rules_primary/secondary; PM US market description). A mismatch -> the pair is a bet, not an arb.

## PRE-REGISTRATION A (valuation rule, written 02:40Z before reading any existing-customer offer)
- Listed only from official sources (venue domain/help center/docs, venue's own CFTC 40.6 filing, Kalshi API). Affiliate
  = UNVERIFIED, ranked below.
- Value: SR credit -> lock = hedge on Kalshi at current Kalshi ask of the opposite side + taker fee; SNR credit -> B*(1-q)
  style lock; profit boost -> EV vs fair (Kalshi mid) price and lock if the boosted price beats Kalshi's opposite ask;
  if hedging is banned -> unhedged EV with max loss = stake. Rebates/LIP/interest -> $ per $ of capital per month at the
  user's scale (~$100-500), net of fill/adverse-selection cost where known.
- Kill per item: lock <= $0 or requires an action the terms ban.

## Log
- 02:36Z matched 113 games Kalshi<->PM US (cfb 69, nba 29, nfl 15) with scripts/boost_hedge_xv_sampler.py (logic copied
  from pmus_lag_logger.live_games, honest UA "boost_hedge-research/0.1"). PM US gateway robots.txt: `User-agent: * Allow: /`,
  Content-Signal use=reference. Kalshi robots.txt 404.
- Settlement rules read 02:34Z (KC-BUF example): Kalshi "If the game ends in a tie, the market will resolve to $0.50 for each
  team. If the game is postponed but begins within 48 hours ... If the game is not started within 48 hours, the market will
  resolve to a fair price." PM US: "Overtime is included if played. If the game ends in a tie, the market will settle to
  $0.50. If the game is delayed, postponed ... not within two days ... settle to the last fair market price." -> ties match;
  only the long-postponement price can differ (Kalshi "fair price" vs PM US "last fair market price"). PM US tick 0.0025,
  feeCoefficient 0.095 on game markets.
- WRONG TURN 02:44Z: first sampler cycle took 426 s for 110 PM US books (gateway answers in ~2-3 s per request, timed
  with curl 02:49Z), and the Kalshi books were read once at cycle start -> pairs up to 7 min apart. Data archived as
  results/boost_hedge/xv/v0_badpairing_20261125.jsonl.bak (not scored). v1: PM US books fetched in parallel (8 workers),
  then Kalshi books; per-row gap logged; report drops gap > 90 s. Restarted 02:51Z, runs to 08:45Z, idle 03:00-06:10Z.
- 02:38-02:42Z Kalshi GET /incentive_programs (results/boost_hedge/kalshi/incentive_programs_0242Z.json): 6,475 active
  liquidity programs (~$116.2k/day, 471 series; KXRT $12k/day, KXDPRESPRIMARY $4.5k, KXYTVIEWSW $3.2k ...), 2,433 upcoming,
  **0 active / 0 upcoming volume programs**. No game-winner series (KXNFLGAME/KXNBAGAME/KXNCAAFGAME) carries LIP; sports
  LIP only on futures (KXNFLNEXTTEAM, KXNCAAFTOPSEED, KXNHLSEASONGOALS...).
- L40b FINAL (deployed `lip_paper_b.py report`, run 02:46Z): P2 "YES completer at 1c" LIP $885 vs fill losses $913-1,142 ->
  NET -$26.69 (fifo) / -$256.11 (phi), KILL both; P3 "join cheaper side" NET -$6,059 / -$18,407, KILL. Kalshi LIP stays dead
  for us -- not re-studied.
- Kalshi interest (help.kalshi.com/navigating-the-exchange/your-portfolio/apy-on-kalshi, read 02:47Z, "Updated this week"):
  "Rate: 3.50% APY. This rate is variable"; "Your average monthly predictions balance, including cash and positions, must be
  at least $250"; accrues daily on 12am ET net value, paid monthly; SSN needed above $10/yr. Perps: 7% APY on funds backing
  open perps positions (BTC/ETH/...; needs perps approval) -- directional risk, out of scope.
- Kalshi Deposit and Trading Reward Program (CFTC rules10142633807.pdf, filed 10-14, eff. >= 10-28-2026 to 10-28-2028;
  raw_K/r10142633807.flat.txt): TARGETED "Time-Limited Promotions" (3-90 days) for Objective Categories (inactivity lookback,
  account age, funded status, prior activity, geography ...); reward = Promotional Bonus Credits <= 1x the Minimum Qualifying
  Deposit / Trade Amount, max $1,000 per promotion, $5,000 lifetime; trades count only at $0.03-$0.97; credits expire 1-30
  days; "may only be withdrawn after the Program Participant has traded an amount equal to the Promotional Bonus Credit";
  terms posted on the Kalshi website >= 24 h before launch. Abuse: Rule 5.17 exclusions + "gaming of payouts ... expressly
  prohibited"; NO explicit cross-venue hedging clause. Whether the user is targeted is visible only in-app.
  kalshi.com/regulatory/notices -> 429 to WebFetch (not retried).
- 02:51-02:56Z v1 sample (K vs PM US, 113 games, 251 game-snapshots after dropping pairs > 60 s apart; median pair gap
  50.6 s -- still too loose): 0 rows with net >= 0 at any size; max net/ct -0.86c @C=1, -0.39c @C=10, -0.33c @C>=50;
  median -4.4c (`boost_hedge_xv_sampler.py report`, 02:56Z). Closest: kentst-emich (CFB), bos-was (NBA live).
- Gemini Predictions exposes a public, keyless REST API (developer.gemini.com/prediction-markets/market-data.md, read
  02:58Z): /v1/prediction-markets/events (slow, ~600 KB) and /v1/book/{instrumentSymbol} (~2.6 s). NFL game KC-BUF listed as
  NFL-2611270120-KC-BUF-M; at 02:57Z BUF .58/.59, KC .41/.42 = identical to Kalshi; event volume $3,088. Gemini NFL
  Win-Loss terms PDF (raw_G/nfl_winloss_terms.txt): regular-season tie -> "$0.50"; postponed game that "begins within 7
  days" settles on the result, else "last fair price determined in the sole discretion of the Exchange" (Kalshi 48 h, PM
  US 2 days -> a 2-7 day reschedule is the only settlement mismatch). assets.gemini.com robots.txt -> 403 (only the T&C
  PDF linked from the API was fetched).
- v2 sampler (02:56Z): batches of 24 games -> PM US + Gemini books in parallel, then one Kalshi call per batch (pair gap
  ~<= 15 s); Gemini symbols constructed for NFL/NBA games <= 72 h out and verified by a 200 book; 6 directions per game
  (K/P/G pairs). Fees: Gemini taker 0.07*C*P*(1-P) ceil-to-cent (rounding unverified, conservative).

## PRE-REGISTRATION B2 -- historical minute census, Kalshi vs PM US (written 03:12Z, before any historical price is pulled)
Why: the live sampler only gets ~25 min of Kalshi-paired data before the 03:00-06:00Z blackout and then overnight
pregame quotes; crossings are most likely in-game. B2 = an UPPER-BOUND census on completed games.
- Games: PM US moneyline (`aec-`) games with startTime 2026-11-17 00:00Z .. 2026-11-24 06:00Z matched to settled Kalshi
  KXNFLGAME / KXNBAGAME / KXNCAAFGAME events (same team-code + ET-date logic). Sample: all NFL, up to 40 NBA and 40 CFB
  drawn with random.Random(1125) from the matched list (sorted by slug first); outcome never read.
- Quotes per minute, window [start - 120 min, start + 4.0 h NFL / 3.0 h NBA / 4.5 h CFB]: Kalshi 1-min candlesticks
  (yes_bid/yes_ask close, both team tickers); PM US /v1/price-history fidelity 1 (longPrice ~ long-side best ask,
  shortPrice ~ short-side best ask = 1 - best bid; "normally", per pmus_lag.py docstring).
- Buy-X on Kalshi = min(yes_ask(X), 1 - yes_bid(opp)). Crossing in direction d at minute m: costK + costP + feeK + feeP
  <= 0.99 at C = 10 (feeK = ceil-to-cent(0.07*10*p(1-p))/10, feeP = 0.095*p(1-p)). Same 1c bar as B.
- Report: share of game-minutes crossed (pregame vs in-game), games with >= 1 crossing, run lengths in minutes, max net.
- Reading: minute-close quotes are not simultaneous and carry no depth -> this is an UPPER BOUND on executable arbs. If
  the upper bound is < 0.5% of game-minutes and runs are 1 minute, cross-venue arb is "not actionable by hand" (kill).
  Any crossing run >= 3 minutes in >= 3 games -> inspect raw rows (trap: PM US display price can be a stale last trade).
- Kalshi calls only after 06:10Z, <= 1 req/s, not in minutes :00-:10.
- 03:00-03:08Z helpers done: results/boost_hedge/promos_W1.md (DK Predictions, Fanatics Markets, OG.com) and
  promos_W2.md (Polymarket US, Gemini, Novig, ProphetX); raw downloads raw_W1/, raw_W2/. Verified myself: Gemini Football
  Trade Match filing text (raw_W2/rules10092632794.txt): eligible = purchases of football contracts "at a price between
  $0.20 and $0.80, inclusive", "exclusive of fees", not Predict-Cash funded; "restricted to earning rewards on only one side
  of a given contract or market event" (same-venue rule; no other-venue / related-positions clause, unlike the $40 new-user
  promo); weeks count only if designated on gemini.com/titan/notices >= 7 days ahead (JS page; designation for 11-23..11-29
  UNKNOWN; WebSearch 03:10Z found none).
- 03:04Z sampler restarted with a PM US vs Gemini mode during the Kalshi blackout (no Kalshi call), seeded from the last
  logged map (18 NFL/NBA games with Gemini symbols).
- Valuations: results/boost_hedge/value_items.py -> value_items_out.json (quotes = sampler tops 02:57-02:59Z).

## PART A -- existing-customer promos (verdicts; full terms in promos_W1.md / promos_W2.md)
| # | venue / program | live? CA? | how it pays | locked $ (hedged on Kalshi) / EV | max loss | capital | hedge clause |
|---|---|---|---|---|---|---|---|
| 1 | Gemini College & Pro Football Trade Match (weekly, to 2027-02-14) | framework live; THIS WEEK'S designation UNKNOWN (notices page JS); CA not stated | $250 eligible volume -> $30 Predict Cash (SR: PC proceeds = cash); $1,000 -> $110 cum. | tier 1: +$26.2 if the Gemini leg fills as maker at .79 (DAL; PHI .21 on Kalshi), +$18.5 all-taker; tier 2 ($1,000): ~+$96 maker / ~+$64 taker | ~$2.6-10.3 hedge cost if no PC is paid (designation risk); PC voided risk | ~$330 (t1), ~$1,290 (t2) for ~1-3 days | same-venue only: one side per game counts; NO cross-venue clause |
| 2 | Polymarket US "Tradesgiving Daily Trader" 11-24 09:00 ET .. 11-30 | live (OFF); CA by omission; existing users only if the card shows in-app (filing says new users only -> conflict) | daily promo credit after ONE buy >= personal minimum (new users: $5/day for $10); credit = loss absorber | ~$3.0/day hedged (credit x .68 - $0.37 qualifier hedge) or ~$5/day EV unhedged, x up to 6 days = ~$18-30 | ~$0.4/day hedged; credit forfeiture | $10/day + ~$4 Kalshi | none cross-venue; opposite-side buy on PM US = cash-out, doesn't count |
| 3 | Polymarket US Deposit & Trading Incentive cohort campaigns | framework live (to 2027-05-09); last posted 11-20..11-22 (e.g. "Deposit at least $250 -> $50"); none posted for this week | promo credit | ~0.68 x credit hedged (e.g. ~$34 on a $50 credit) | ~0 (deposit stays the user's) | the deposit | none |
| 4 | OG.com Weekly Trading Streak | FILED 11-12, earliest live ~11-27 | wk1 50% profit boost (extra <= $10), wk2 50% (<= $20), wk3 $10 bonus, wk4 25% (<= $25), wk5 50% (<= $25), wk6 $10 bonus; 7-day expiry | boost hedged: wk1 +$5.5 (S $75 @.79), wk2 +$11.1 (S $150); unhedged EV $7.9 / $15.8; wk3 bonus ~$9.5 | qualifiers 2 x $10 UNHEDGED per week ($20 at risk, EV -$0.7) | $75-150 boost stake + $24-49 Kalshi | "without bona fide market risk" -> do not hedge qualifiers |
| 5 | Kalshi Deposit & Trading Reward (targeted Time-Limited Promotions, eff. 10-28-2026..10-28-2028) | framework live; offers per user cohort, in-app/website | Promotional Bonus Credits <= 1x the qualifying deposit/trade, <= $1,000/promo; must trade the credit amount before withdrawal | unknown until offered | -- | -- | Rule 5.17 + "gaming of payouts"; no cross-venue clause |
| 6 | OG.com Reactivation/Retention "Trade $x get $y"; DK Predictions in-app offers; Novig Trade Credit Pool | targeted / in-app only (amounts redacted or unreadable) | bonus / coupon / credit | unknown | -- | -- | OG: bona fide market risk |
| 7 | Kalshi interest | live; US | 3.50% APY cash+positions, avg monthly balance >= $250 | ~$0.73/month at $250 | 0 | -- | -- |
| 8 | Gemini maker programs: maker fee $0 to 12-31; Maker Rebate 0.30 x taker fee on 20-80c fills (USD daily); Liquidity Rewards (2-sided, >=100/side, own spread <= $0.10, pools $10-$10k/day, USD) | live | cash | rebate ~0.4-0.5c/ct (used in #1); LR pools not readable -> LEAD only | maker adverse selection (L40b/L19 prior) | -- | wash/self-dealing only |
| 9 | Polymarket US Liquidity Incentive Program (eff. 11-18) | live | "credited" (cash vs credit not stated) | in-game pools big but target sizes 25k-625k -> tiny retail share; pregame pools $5-27 | adverse selection | -- | none cross-venue |
DEAD / not worth it: Kalshi LIP (L40b FINAL kill: P2 net -$27/-$256, P3 -$6k/-$18k); Kalshi volume incentive (0 programs
active/upcoming 02:42Z); Fanatics Always-On FanCash (0.15% of volume at prices >= $0.43 -> $0.015 per $10 trade; 10% only at
1c; eligibility = customers verified by 08-26; "entering and immediately exiting positions" banned); Fanatics NFL
multiplier (expired 9-21), RAF (ended 11-15); ProphetX Rewards & Loyalty (not before 12-01); PM US maker/taker rebates
(start at $10M / $250k volume); profit/odds boosts on DK/Fanatics/PM US/Gemini/Novig/ProphetX: none found in official
sources; Kalshi perps 7% APY (directional perps exposure).
- 03:13Z Kalshi help collection "Programs & Promotions" (help.kalshi.com/en/collections/18616618) lists 7 articles: LIP,
  LIP "where to find", Liquidity Provider Program, two bug bounties, referral FAQ, and "Sports Prop Combo Market Component
  Legs LIP" (updated 2026-09-25: makers in prop legs share 25% of combo fees by maker-volume share, paid monthly as "Bonus
  Credit", score >= 0.01 needed) -> maker-volume program, adverse-selection prior, not retail-actionable. No existing-user
  deposit/trade promotion is posted there; targeted Time-Limited Promotions would appear in-app.
- 03:14Z critic subagent launched on Part A claims 1-3 + Part B interim. Historical census B2 pull started in background
  (PM US calls now; Kalshi calls wait for 06:10Z, 2 s pacing so sampler + census stay < 1 req/s combined).

### Critic pass (fresh `critic` subagent, 03:25Z): ACCEPT-WITH-CHANGES -- corrections applied (value_items_out.json re-run 03:30Z)
- Gemini Trade Match (#1) WEAKENED: the filing's own abuse list (raw_W2/rules10092632794.txt s.10) includes "Artificial
  trading intended primarily to generate promotional rewards" and "Conduct otherwise inconsistent with the purpose of the
  Promotion"; s.12: "Do not trade solely to obtain a promotional reward" and Gemini's determinations "are final". A fully
  hedged $250 round trip fits these words -> treat as FORFEITURE RISK (same treatment as OG's "bona fide market risk").
  s.12 also: "The same trade may not be used to earn a reward under another Gemini Predictions trade-based promotion" ->
  the $40 new-user qualifier must be a separate, earlier, unhedged trade. The referenced "Gemini Predictions Promotional
  Terms and Conditions" document was not read. Realistic tier-1 lock = +$18.10 all-taker (after the ~$0.42 fee on the
  Predict Cash buy); maker +$25.79 is a best case (fills arrive when price moves). If the PC is withheld: -$10.28.
  Tier 2 taker +$64.07, maker +$94.79. Unhedged alternative (genuine positions, maker): EV +$28.38, max loss $250.
- PM US Tradesgiving (#2) -> OPEN, not a lock: the user has no PM US account yet (PROMO_TRACKER "todo"); "Users who sign up
  with a promo code aren't enrolled" + Sequential filing "may participate in only one Campaign" -> likely either/or with
  the $25 sign-up Bonus; the credit absorbs losses on ANY trade (qualifier positions can eat it); expiry may be 24 h.
  Hedged conversion of a $5 credit at fee 0.095 = $3.26 (ratio 0.65, not 0.68), minus $0.37 qualifier hedge = $2.89/day;
  harvest needs ~$22 Kalshi capital (q = .16 longshot on PM US, favourite on Kalshi), not ~$4.
- OG streak (#4): OG taker fee (~$1.10 on a $75 stake) was missing; boost_math.py now sizes the hedge with the fee charged
  at entry in both outcomes -> wk1 lock $4.44 (critic estimated $5.30 assuming the fee only on the losing side; OG/CDNA fees
  are charged at execution), wk2 $8.87; unhedged EV $6.77 / $13.54. Not live yet (10-day review).
- Part B (#4 claim) numbers reproduce (KP 410 snapshots, 0 >= 1c, max -0.39c @10; GB-LAR recomputes to -0.05094; no side
  mis-mapping across 112 games; even at PM US API taker 0.0695 x 30% the max is +0.11c), but the KP window is ~7 min and
  mostly pregame -> wording must be "none in a 7-minute mostly-pregame snapshot", pending B2 (in-game census).

## PART B -- result (report saved results/boost_hedge/xv/report_0335Z.txt; data results/boost_hedge/xv/samples_20261125.jsonl)
| venues | window (UTC 11-25) | game-snapshots | games | max net/ct @C=1 | max net/ct @C=10 | median @C=10 | >= 1c events |
|---|---|---|---|---|---|---|---|
| Kalshi vs PM US | 02:52-02:59 (v1+v2; pairs > 60 s apart dropped) | 410 | 113 (cfb 69, nba 29, nfl 15) | -0.86c | -0.39c | -4.42c | 0 |
| PM US vs Gemini | 02:57-03:26 (incl. 2 live NBA games) | 394 | 18 (nba 13, nfl 5) | -1.37c | -0.66c | -4.93c | 0 |
| Kalshi vs Gemini | 02:57-02:59 | 35 | 18 | -3.00c | -1.50c | -4.50c | 0 |
- Pre-registered verdict: KILL rule met (0 executable-candidate events) -> "no cross-venue arb at public-feed cadence", the
  same answer as M2 (Kalshi vs global Polymarket, cheapest round trip $1.021). Caveat (critic): the Kalshi-paired window is
  only ~7 min and mostly pregame; in-game is where crossings would appear -> B2 below is the decisive test.
- Why there is nothing: Gemini quotes NFL/NBA winners at exactly Kalshi's prices (KC .41/.42, BUF .58/.59 on both at
  02:57Z) and PM US within 0-1.5c; each venue's own round trip is ~$1.00-1.02, and the cheapest cross-venue round trip
  still pays 1.5-4c of taker fees (Kalshi ceil(0.07Cp(1-p)) + PM US 0.095Cp(1-p) ~ 1.7c + 2.3c at p = .5).
- Settlement: tie -> $0.50 on Kalshi, PM US and Gemini (NFL regular season). Postponement windows differ: Kalshi 48 h, PM US
  "two days", Gemini 7 days, then "fair price" / "last fair market price" / "last fair price ... sole discretion" -> a
  2-7 day reschedule is the one case where a cross-venue pair is a bet, not an arb.
- B2 (pre-registered above) NOT RUN by this agent: Kalshi candlesticks are only callable after 06:10Z. The pull is queued in
  the background (PID 41717, `scripts/boost_hedge_xv_hist.py pull`, waits for the window, 2 s pacing, ~230 Kalshi calls).
  If results/boost_hedge/xv_hist/minutes.parquet is missing after ~06:30Z, rerun:
  `cd sports && ../.venv_quant/bin/python scripts/boost_hedge_xv_hist.py pull && ../.venv_quant/bin/python scripts/boost_hedge_xv_hist.py report`
  The live sampler (PID 35885) keeps running to 08:45Z (PM US vs Gemini in the blackout, all three after 06:10Z);
  `scripts/boost_hedge_xv_sampler.py report` re-scores everything. Stop: `kill 35885 41717`.

## RANKED ACTIONABLE LIST (the user places every non-Kalshi order; Kalshi hedge legs only with the user's OK; NFL = shard 0)
1. **Gemini College & Pro Football Trade Match** -- ONLY IF (a) the user opens Gemini Predictions (sign up DIRECT; CA not
   stated), (b) gemini.com/titan/notices shows a designated window covering the trades, and (c) the user accepts the s.10/s.12
   forfeiture risk of hedging. Hedged tier 1: +$18.10 locked (all taker) to +$25.79 (Gemini maker fill), -$10.28 if Gemini
   withholds the Predict Cash; capital ~$331 for 1-3 days (Gemini ~$254 + Kalshi ~$70 + PC hedge ~$8). Tier 2 ($1,000):
   +$64 taker / +$95 maker on ~$1,290. Unhedged (genuine views): EV +$28.38, max loss $250.
   Steps: (1) read the notices page for the window + eligible markets; (2) do the $40 new-user qualifier FIRST as its own
   unhedged trade (>= $20 at 0.20-0.80, maker, hold to settlement); (3) buy >= $250 (excl. fees) of ONE side per game at
   0.20-0.80 (example 11-26 PHI-DAL: DAL ~0.79 maker / 0.80 taker = ~313-317 contracts); never buy both sides on Gemini;
   (4) lead buys the same count of the other team on Kalshi (PHI ~0.21) with the user's OK; (5) $30 Predict Cash (30-day
   expiry) on a heavy favourite on Gemini + the opposite on Kalshi (~$28.4). Recompute with results/boost_hedge/value_items.py.
2. **OG.com Weekly Trading Streak** (filed 11-12, earliest ~11-27; after the OG new-user flow): each week two $10+ single
   trades < $0.80, NOT hedged (EV -$0.7, max loss $20) -> wk1 50% boost (extra <= $10): hedged +$4.44 with S $75 @0.79 + 105
   Kalshi @0.22 ($24), or EV +$6.77 unhedged; wk2 +$8.87 / +$13.54; wk3 $10 bonus (~$9.5). Weeks 1-3 ~ +$20-30.
3. **Polymarket US Tradesgiving Daily Trader** (to 11-30, 9 PM PT daily cutoff) -- OPEN: when the user opens PM US, see if
   the card shows; if it does AND the $25 sign-up Bonus is still granted, one buy >= the shown minimum per day -> ~$2.89/day
   hedged per $5 credit (Kalshi ~$22), ~$5/day EV. If it is either/or, take the $25 sign-up Bonus (larger).
4. **Targeted in-app offers -- look, screenshot, then value**: Kalshi Time-Limited Promotions (Deposit & Trading Reward
   Program; credits <= 1x the qualifying deposit/trade, must trade the credit amount before withdrawal); PM US cohort
   campaigns (11-20..22 example: deposit $250 -> $50 credit, i.e. ~+$34 hedged); OG "Trade $x get $y"; DK Predictions
   in-app; Novig Trade Credit Pool. Max loss ~0 when the qualifier is a deposit.
5. **Kalshi interest**: 3.50% APY on cash + positions if the average monthly balance is >= $250 -> ~$0.73/month at $250.
   Not worth moving money for; just don't let idle hedge cash drop the average below $250 if it is near.
NOT WORTH IT / DEAD: Kalshi LIP (L40b FINAL), Kalshi volume incentives (none), Kalshi prop-combo LIP (maker volume),
Fanatics Always-On FanCash (0.15% at game prices; eligibility doubtful), PM US / Gemini liquidity programs at retail size
(adverse-selection prior; pools unreadable), ProphetX Rewards (>= 12-01), profit/odds boosts (none in official sources on
DK, Fanatics, PM US, Gemini, Novig, ProphetX), cross-venue arbitrage (Part B).

## Learnings
### Prediction venues' existing-customer money is mostly targeted, in-app, and filed with redacted amounts
Kalshi (rules10142633807, Time-Limited Promotions by cohort), PM US (PMX-2026-65 cohort campaigns; Tradesgiving amounts
"vary by Participant"), OG (retention campaigns, amounts REDACTED) all pick who gets what; public pages show the framework
only. The user's own app screens are the data source (notes/boost_hedge.md Part A rows 4-6).
### Gemini's Football Trade Match has no cross-venue hedge clause but bans trading "solely" for the reward
rules10092632794 s.10 "Artificial trading intended primarily to generate promotional rewards"; s.12 "Do not trade solely to
obtain a promotional reward"; determinations final. A hedged $250 round trip locks +$18.10 if paid, -$10.28 if withheld
(results/boost_hedge/value_items_out.json). Read the whole abuse list, not only the "hedging" sentence.
### Kalshi, Gemini and Polymarket US quote NFL/NBA game winners within 0-1.5c of each other; no arb after fees
0 of 410 Kalshi-PM US, 394 PM US-Gemini and 35 Kalshi-Gemini paired snapshots reached net >= 1c/contract at 10 contracts;
max -0.39c, -0.66c, -1.5c (results/boost_hedge/xv/report_0335Z.txt). Gemini mirrored Kalshi's quotes exactly (02:57Z).
Pregame-heavy sample; in-game census B2 pending (scripts/boost_hedge_xv_hist.py).
### Polymarket US public gateway answers in ~2-3 s per request; serial polling misaligns cross-venue pairs
A serial loop over 110 PM US books took 426 s, so a Kalshi book read at cycle start was paired with PM views up to 7 min
later. Parallel fetch + per-batch Kalshi reads cut the pair gap to ~15-30 s. PM views are also Cloudflare-cached up to 30 s.
### A profit boost is worth about P(win) x the capped extra, and roughly two thirds of that when hedged
OG wk1 50% boost capped at $10 extra on a 0.79 contract: EV $6.77 unhedged, $4.44 locked after the OG fee and a Kalshi hedge
(results/boost_hedge/boost_math.py). Pick the highest allowed price so the cap is reached with the smallest hedge.
### Kalshi LIP is still large ($116k/day, 6,475 programs on 11-25) but covers no game-winner market and failed our paper test
GET /incentive_programs 02:42Z: 0 volume programs; L40b FINAL: P2 net -$26.69/-$256.11, P3 -$6,059/-$18,407 (deployed
lip_paper_b.py report). Fill losses exceed LIP payouts; do not re-study without a new mechanism.

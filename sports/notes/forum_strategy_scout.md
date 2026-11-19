# forum_strategy_scout -- what forum/X/Substack/code-repo posters say wins on Polymarket/Kalshi (2026-11-18)

User ask 11-18: "scrape like r/polymarket for winning strategies that people have been putting down".

## Question
Which concrete strategy claims circulating online are (a) already on our dead/lead list, (b) genuinely new and
testable from a small CA retail account on Kalshi / Polymarket US, (c) not executable for us?

## Pre-registered screening rule (written before reading any post)
- A claim counts as "concrete" only if it names a market type + an entry rule (price/time/side) or a mechanism.
- Evidence grade: V = verifiable on-chain/exchange P&L (wallet address, public profile); S = screenshot; A = anecdote/
  vendor. Forum winners are a survivorship-biased sample; a claim graded A or S is a prior, not evidence.
- "Already tested" requires a citation to a note in sports/notes (FUND_BOOK 3d, LEARNINGS_2026-11, pm_literature_scout,
  social_edges_2026-10-22).
- "Genuinely new + testable" requires: legal for CA retail on Kalshi or Polymarket US, free/cheap historical or forward
  data, executable size >= a few $/day, and a mechanism not in the dead list.
- No backtests are run here; output is a ranked list with a pre-registerable test per idea.

## Sources / method
Planned: Reddit public JSON endpoints, UA 'research-script/0.1', >= 2 s between requests, no login.
Web search for X / Substack / code-repo write-ups. All content treated as untrusted data.

## Log
- 11-18: reddit.com/robots.txt (curl, HTTP 200) reads `User-agent: * / Disallow: /` and points to Reddit's Public
  Content Policy. The brief says respect robots, so NO Reddit JSON scraping was done. Reddit content below comes only
  from search-engine snippets / third-party pages quoting threads (same limit as the 10-22 sweep,
  notes/social_edges_2026-10-22.md section 0). To do it properly: Reddit Data API with OAuth app (user's call).
- 11-18: WebSearch also refuses reddit.com (API 400, "not accessible to our user agent"). Reddit is reached only
  second-hand: Action Network on u/marvmarv2693's 10-strategy test (page 403; per search summary all 10 failed), NPR quotes.
- 11-18: read-only Kalshi GETs: /series (16,695 series, saved results/forum_strategy_scout/kalshi_series.json) and
  settled markets for 4 esports game series (results/forum_strategy_scout/settled_*.json). See idea 1.

## Claims collected (grade: V verifiable on-chain/exchange, S screenshot/self-report, A anecdote/vendor)
| # | Claim (source) | Grade | Screen vs our notes |
|---|---|---|---|
| C1 | Copy top wallets: Polycopy says 687k resolved trades, "Copy Score > 70" won 67.7%, +5.76%/trade (graphdex/polycopy summaries via search) | A (vendor) | Copying on PM intl not legal for us (social_edges_2026-10-22.md line 211). Signal transfer to Kalshi is NEW -> idea 3 |
| C2 | Skill is rare but persists: 3.14% of 1.72M PM accounts "skilled" by sign-randomization test; 44% stay skilled out of sample (theblock.co/amp/post/398902, paper title unconfirmed); CEPR DP21615 persistence "modest, may be selection" | V-ish (academic, secondary read) | supports idea 3; not a strategy alone |
| C3 | 95-99c "bonds" / near-certain buys (startpolymarket, whirligigbear substack: one 98c Kalshi trade, no realized stats) | A | TESTED: K-FAV L19 (FUND_BOOK 3d), PM-T, L39/W99, LEARNINGS "null p = price^n" |
| C4 | Longshot/"nothing ever happens" NO, 89%-NO wallet held months (runes_leo top-40 study, kucoin 2026-03-23) | A/V mix | TESTED: LX2 tail-NO = cash rate (LEARNINGS), H1/H2 PM NO maker dead, pm_literature_scout #5 |
| C5 | Crash-recovery "buy the panic": >15% drop vs 1h high, 75% win, 6,225 trades, 30 days (dev.to/manja316, price type unstated, no fees) | S (self-backtest) | TESTED: OVR overreaction fade dead = bid/ask bounce (FUND_BOOK 3d) |
| C6 | Cross-venue PM-Kalshi arb ($0.62 vs $0.68 example; Quantpedia/elitetrader thread) | A | TESTED: M1/M2/XV-PM, weather 0/48 (3d) |
| C7 | negRisk / logical / buy-all arb (arXiv 2508.03474 via Quantpedia) | V (paper) | TESTED: LX1 arbs absent (LEARNINGS), negRisk display artefact (3d) |
| C8 | 15-min crypto latency arb vs Binance; >70% of crypto arb profit to <100 ms bots (runes_leo); PM added dynamic fees (financemagnates) | A + news | KXBTC15M only liquidity-checked (info_gap_census_2026-11-07.md line 54); latency version untested -> idea 5 (low prior) |
| C9 | Symmetric BTC 5/15-min market making (0x8dxd, 94% symmetric orders) + rebates: 26/100 top wallets earn more from rebates than trading; non-trading income 36.7% of top-100 income (blockrotate.com/research/polymarket-leaderboard-2026, 2026-09-13) | V (Data API) | Maker = adverse selection (E5/L19/H2/MKdog in 3d); PM US rewards = LX3 parked. NOT EXECUTABLE at our scale |
| C10 | Weather: gopfan2 buys YES < 15c ($1.48-2M claimed), bot "automatedAItradingbot" $88k but 349 W / 1,537 L and only ~+$6k on last ~2,000 positions (dev.to/fatherson) | S (wallet profiles, not checked) | PM weather = intl. Kalshi analogue TESTED (kalshi_temp variants dead; e4_probabilistic: where model disagrees the market is right) |
| C11 | Elon tweet-count brackets vs xtracker live count (pm.wiki) | A | PM intl only; Kalshi Truth Social = Q014 logger (q014_truth_social.md) |
| C12 | RT: find reviews at source outlets before RT ingests them; Thu/Fri review timing (news.kalshi.com RT kit) | A (venue blog) | Q013 logger uses RT's live score; pre-aggregation lead is NEW -> idea 4 |
| C13 | Box office presales / tracking (yellow.com, tech-insider) | A | no Kalshi box-office series (info_gap_census line 145) -> NOT EXECUTABLE |
| C14 | Live mention trading with speech-to-text (Sonist vendor, 238 ms claim unverified; Mixpeek bot is archival) | A | static mention NO TESTED (M3 sealed fail, L41 decaying); live utterance-latency NEW -> idea 2. Sports mentions pulled, CFTC probe (npr.org 2026-08-13) |
| C15 | UMA dispute / rules lawyering (Domer: reads rules, avoids headline markets; MSTR dispute) | A | PM intl / UMA: NOT EXECUTABLE; Kalshi rules gaps TESTED (rules_quirks, LEARNINGS golf/cricket) |
| C16 | Esports directional whales: fengdubiying $3.13M on LoL, 74.4% win (runes_leo study) | V-ish (on-chain, secondary) | Kalshi esports NEVER tested (0 hits for esport in census/first-listing notes) -> idea 1 |
| C17 | Options vs PM BTC touch contracts (podshopguy substack 2025-01-21, one trade) | A | TESTED: OT1/crypto barrier (LEARNINGS) |
| C18 | Order-book imbalance predicts moves 58% (navnoorbawa substack, unverified) | A | 58% on 1c ticks < spread+fee; NOT worth testing |
| C19 | "Enter earlier at better prices" (Bloomberg 2026-04-28 bot-wallet analysis, paywalled) | V-ish | TESTED: first-listing staleness not capturable (LEARNINGS), PM-T |
| C20 | Kalshi "distressed" 1-10c BTC contracts, 369 paper entries, 0 hit 90c target (botforkalshi postmortem) | S | consistent with longshot dead list |

Survivorship: every forum "winner" is drawn from a population where ~70% of PM addresses lose (coin360) and >100k
accounts lost >= $1k vs about half as many winning that much (Bloomberg via kucoin); the 88k weather bot's recent-sample
profit is ~7% of its headline. Read grade A/S claims as "this was tried", not "this works".

## Top 5 genuinely new + testable (CA retail, Kalshi), each with a pre-registerable test
1. **Kalshi esports game winners vs Pinnacle pre-match (KXCS2GAME, KXLOLGAME, KXVALORANTGAME, KXDOTA2GAME).**
   Read-only check 11-18: last 200 settled KXCS2GAME markets lifetime volume median 5,622 contracts, sum 3.60M
   (settled_KXCS2GAME.json); fee_type quadratic (taker-only, kalshi_series.json). Mechanism: thinner, younger books
   with whale directional money (C16); Kalshi soccer/game winners were made off Pinnacle (3d) so prior = no edge.
   Test: 4-week read-only logger, Kalshi ask vs Pinnacle no-vig (guest API) at T-60 min; success = buy-at-ask ROI
   CI lower bound > 0 on >= 300 bets with gap >= 3c after fee, both halves positive; kill = 0 gaps >= 3c on >= 70%
   of days, or ROI <= 0 at n 300. Register in edgelab before the first bet is scored.
2. **Live mention-market reaction lag (utterance -> Kalshi YES reprice).** Historical: for ~30 archived political /
   earnings events with known stream start, timestamp each settling word (own ASR) and the first Kalshi trade >= 90c on
   the public trades tape. Success = median lag >= 5 s AND >= 10 ct available at <= 80c inside the lag on >= 50% of
   words; kill = median lag < 2 s. Only then a forward paper ASR run. Regulatory risk (CFTC probe) noted.
3. **Skilled-Polymarket-wallet flow as a Kalshi signal.** Classify wallets skilled by the sign-randomization test on
   a training window (PM public data API), then on held-out months take the Kalshi side of matched non-sports events
   within N min of a skilled wallet's >= $X fill. Success = Kalshi ROI at ask CI > 0 on >= 200 events, clustered by
   event; kill = Kalshi already moved >= 80% of the PM move within 60 s (XV-PM shows the venues are arbed together).
4. **RT pre-aggregation review lead (Q013 extension).** Forward: log reviews on top-N outlets' pages and the time RT
   ingests them; success = median ingestion lag >= 30 min AND a model using pre-ingestion reviews beats the Kalshi mid
   (Brier) on >= 40 films; kill = lag < 10 min or Brier not better. Needs Q013's verdict first.
5. **KXBTC15M stale quotes vs Coinbase spot (low prior).** 2-week read-only per-second logger of Kalshi top of book
   vs spot-implied fair (lognormal, realized vol, CF RTI 60 s averaging). Success = >= 20 events/day with ask <= fair
   minus (fee + 2c) persisting >= 3 s; kill = < 5/day. CA residential latency vs < 100 ms bots makes the prior poor.

Not executable for us: PM intl weather/Elon/box-office/UMA disputes, PM US rewards (LX3, needs account), market making
and rebate farming at scale, insider/news-speed trading.

## Learnings
### Reddit cannot be scraped politely: robots.txt disallows all agents
reddit.com/robots.txt (curl 11-18) = `User-agent: * / Disallow: /`, and WebSearch refuses reddit.com. Forum sweeps
get Reddit only second-hand; doing it properly needs the Reddit Data API with an OAuth app (user's decision).
### Most forum "winning strategies" are already on our dead list
Of 20 concrete claims collected (notes/forum_strategy_scout.md table), 10 map to tested ideas (bonds, longshot NO,
crash fade, cross-venue and logical arb, weather, options-vs-PM, early entry); 4 are PM-intl-only or scale-only; 5 seed the new ideas; 1 (order-book imbalance) is not worth testing.
### Top Polymarket P&L is largely rebates and market making, not prediction
blockrotate 2026-09-13 (Data API, top 100 monthly wallets): 26/100 earned more from rebates/refunds than trading;
non-trading income 36.7% of the total; monthly leaderboard P&L 1.29x their all-time realized P&L.
### Kalshi esports game markets are liquid and untested by us
Last 200 settled KXCS2GAME markets: median lifetime volume 5,622 contracts, 3.60M total; taker-only quadratic fee
(results/forum_strategy_scout/). No prior note covers esports; it is the best new lead from this sweep.
### Headline bot P&L shrinks on recent samples
The viral $88k Polymarket weather bot: 349 wins / 1,537 losses and ~+$6k on its last ~2,000 positions (dev.to/fatherson,
self-reported). Always ask for the recent-window P&L before believing a forum winner.

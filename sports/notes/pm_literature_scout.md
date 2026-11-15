# pm_literature_scout -- documented prediction-market inefficiencies not yet on our dead list (2026-11-15)

Agent: pm_literature_scout (read-only literature sweep; no data pulls, no orders, no git).

## Question
Which inefficiencies documented 2024-2026 (arXiv/SSRN/NBER/Fed/exchange-data studies, credible practitioners) in
Kalshi / Polymarket / ForecastEx / CFTC event markets are (a) not in FUND_BOOK 3d, (b) executable by a small CA retail
account, (c) survive spread + fees in the source's own evidence?

## Screening rule (fixed before searching)
Keep a candidate only if: the source reports a number for the effect; the venue is legal for a CA retail account; it is
not already a 3d row or an in-flight agent (options_implied_tails = crypto/index vs options; rules_quirks = settlement
definitions). Rank by (cost-survival in source) x (independence from our failure modes: adverse-selected maker fills,
priced-before-news, efficient sports). Literature only: every "edge" here is a prior, not evidence, until a dev test.

## Already covered by the 10-22 sweep (notes/prior_research_2026-10-22.md) -- not re-read
Burgi-Deng-Whelan (Kalshi FLB, makers vs takers), Becker 72M trades, Cardozo & Rivero-Wildemauwe arXiv 2609.12878,
Le arXiv 2602.19520, Moshrefi arXiv 2607.14430, Diercks-Katz-Wright FEDS 2026-010, Zitzewitz NBER w35845, Deleep SSRN
6322678, Polymarket-v1 arXiv 2606.04217.

## Log
- 12:07Z searched arXiv/SSRN/news (4 parallel + follow-ups). Local check: health count series (MEASLES, KXPERTUSSIS,
  KXMPOXCOUNT, KXSCREWWORMCOUNT) are already in notes/phys_ladders.md census ("too few events"); SX1 shows observed-count
  impossible rungs are priced at ~1c. So the "impossible bins" paper adds nothing new for us.
- Read-only GET: results/pm_literature_scout/series.json = 16,448 Kalshi series (2026-11-15).

## Ranked candidates (literature = prior, not evidence; no source nets out spread)

### 1. Kalshi political favourites 60-85c, 2 days to 1 month out (underconfidence)  [best]
- Source: Le, "Decomposing Crowd Wisdom", arXiv 2602.19520 v2 (rev. 2026-08-04), Table 4: Kalshi Politics recalibration
  slope 1.83 at 2d-1w and 1w-1mo, 1.73 at 1mo+ (>1 = prices too close to 50%); no CIs for these slopes. Isotonic: raw
  0.75 -> 0.886 realised (Sec 4.2). Kalshi Politics = 6,609 markets / 4.9M trades; event-clustered bootstrap has 850
  clusters, event SEs ~50x naive. No return figure; fee exposure ~1.18c/contract (schedule, not realised).
- Conflict: Burgi-Deng-Whelan find politics slope 0.022 not significant (notes/prior_research_2026-10-22.md:64).
- Not a redo: LX2 tested only the EXTREME band (YES <= 10c, >= 90 d): politics -0.2%, crosses 0 (notes/lx2_tail_no_carry.md:77).
  The 60-85c band at 2d-1mo is untested (L22).
- Pitfall: trade-weighted calibration; the 2024 cycle (favourites swept) may carry it -> cluster by election/event.
- Test (prereg): universe = Kalshi Politics/Elections markets settled 2026-07-01..2026-11-15 (after Le's sample; includes
  the 11-03 midterms, out of sample for the paper). Entry = first snapshot 7 d (+-12 h) before close where the favourite
  ask is in [0.60,0.85]; buy favourite at ask + taker fee, hold. Success: ROI CI lower > 0 with event-clustered bootstrap
  (cluster = race/state), >= 150 events, no single election day > 30% of P&L. Kill: CI lower <= 0 or n < 150.

### 2. Be the NO/maker side of Kalshi combo (parlay) RFQs
- Moshrefi arXiv 2607.14430: cross-game parlays priced above product of leg prices, gap growing with leg count (abstract;
  no net return). Ingame via Bitcoin.com (2026-09-22): makers' markup on uncorrelated non-NFL combos ~0.55% before the
  2026-08-20 combo maker fee, ~3.1% after; combo maker fee = 50% of taker fee; makers $24M profit after fees in 4 weeks;
  takers lost > $380M before fees / > $620M after fees since 2025-09. Gambling Insider via on360: 14 days (07-14..07-27),
  ~4.6M combos, YES -35c/$ (-16.6c ex World Cup final), NO +21c/$ -- one window dominated by one match.
- Executability unverified: docs.kalshi.com RFQ page does not say whether any member may quote; quotes carry yes_bid and
  no_bid; balance must cover every quote in a batch. Winner's curse: you fill when you are the cheapest quoter, i.e. when
  your correlation/leg estimate is most wrong (same adverse-selection family as E5/RD2); competition is professional.
- Test (prereg): step 0 (read-only gate) confirm a retail key may POST /communications/quotes (ask Kalshi support; no test
  quotes without user yes). Step 1: public trade history of KXMVE* uncorrelated cross-game combos settled after
  2026-08-20: NO-side return at traded price minus maker fee, by leg count, clustered by game-day. Success: NO ROI CI
  lower > +1% for 3+ leg combos over >= 8 game-days. Kill otherwise. Filter KXMVE at ingest (LX2: ~61 GB caches).

### 3. Crypto threshold ladders: weekend and inception mispricing vs options
- Fabi et al. FC'26 DeFi slides: Polymarket BTC Above/Range match options; Reach/Dip deviate; mispricing largest at
  inception, near expiry and on weekends (no numbers read). Portnaya arXiv 2606.19517: gap largest at low implied prob and
  long expiry. Hand to options_implied_tails as pre-registered covariates (weekend x tail), not a new agent.

### 4. CPI upper-tail rungs after a positive surprise (overreaction?)
- Angelini arXiv 2606.30040: +0.1pp lagged CPI surprise raises P(monthly CPI > 0.3%) by ~4.7pp controlling for consensus;
  but upper-tail prices also PREDICT realised high states (abstract) -> may be information, not bias. Kagan & Baiocchi
  (Kalshi Research, Aug 2026; conflict of interest): Economics Brier 0.108 at 3 months vs 0.066 at close, the slowest
  category to sharpen. Test: KXCPI rungs 10-30 d out, split by sign of last surprise; buy NO on upper rungs after a
  positive surprise at ask+fee; success = CI lower > 0 over >= 30 releases (likely underpowered: ~1 release/month).

### 5. Polymarket US non-sports NO when ask < 0.65 ("Nothing Ever Happens")  [weakest]
- Decrypt 2026-04-15: bot buys NO < 0.65 on all non-sports markets; P&L unpublished; Polymarket says 73.3% of markets
  resolve NO; Decrypt's 2,300-position sample: 73.5% NO for 90-180 d markets vs 52% under a week. Resolution rates ignore
  entry price. Polymarket US taker fee 4-5% in these categories (River Markets 2026-08, unverified) likely eats it; maker
  free but our maker fills are adversely selected. Testable on Polymarket-v1 (arXiv 2606.04217) at the ask.

### Screened out
- Impossible-bin mass in cumulative disease markets (Dudley & Magdaleno arXiv 2605.11220: 7.84% -> 0.84-1.04% by week
  4-5, 16 Polymarket flu markets, $15-25k weekly volume, no cost analysis): covered by phys census/SX1, priced at ~1c.
- Polymarket negRisk executable arbitrage (Gebele et al. arXiv 2608.00666): $1.12M over the venue's history, $1.086M via
  converter; international Polymarket, competed; our negRisk overround was a display artefact.
- Settlement manipulation on 5-min BTC contracts (Dai, Jia, Yu arXiv 2606.31675): a pitfall (avoid <= 5-min crypto).
- Interest-on-positions: ForecastEx coupon ~3.12-3.13% APY (IBKR pages), Kalshi ~3.5% -> a hedged pair earns ~the cash
  rate (LX2 conclusion) unless pair cost < $1, which cross-venue scans never found.
- Kagan & Baiocchi calibration study: FLB set aside by the authors; "stale thin brackets" asserted, not quantified.
- Weekend / end-of-month / settlement-day effects on Kalshi: no study found beyond the crypto slides above.

## Learnings

### Political underconfidence on Kalshi is the only big, documented, untested mispricing left in the literature
Le arXiv 2602.19520 v2 Table 4: Kalshi Politics slope 1.83 at 2d-1mo; raw 0.75 resolves 0.886. No CIs per horizon, no
returns, 850 event clusters; Burgi et al. disagree. Our LX2 only tested the >= 90c extreme band. The 2026 midterms are
a fresh out-of-sample set for this paper.

### Kalshi combo makers earn about 3% markup, but retail quoting access and winner's curse are unverified
Ingame (via Bitcoin.com 2026-09-22): markup 0.55% -> 3.1% after the 2026-08-20 combo maker fee (50% of taker fee); makers
+$24M after fees in 4 weeks. Gambling Insider's NO +21c/$ is one 14-day window dominated by the World Cup final.

### Most 2025-26 prediction-market papers report calibration, not net returns
Of ~12 papers read (Le, Moshrefi, Angelini, Dudley, Kagan, Subramanian, Gebele, Nechepurenko x2, Dai), only Gebele
(arbitrage $) and the Ingame / Gambling Insider trade studies give money figures; none nets out spread. A calibration
slope is a prior for a dev test, never an edge.

### Impossible-outcome mass in cumulative markets exists but is priced at about 1c on Kalshi
Dudley & Magdaleno: 7.84% falling to ~1% by week 4 on thin Polymarket flu markets. Our SX1 (notes/phys_ladders.md) found
the same structure on Kalshi priced at ~1c with a few $/month of capacity.

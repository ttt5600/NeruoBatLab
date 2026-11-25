# pokemon_cards -- is there a real, repeatable Pokemon-card market inefficiency? (started 2026-11-24 local / 2026-11-25 UTC)

Agent: pokemon_cards (research subagent, read-only desk research; no purchases, sign-ups, logins).
Prior work read first (do not redo): ~/.claude/seed-runner/research_notes/opps_2026-11-12/01_secondary_markets.md
(TCGplayer ~13% all-in seller fee, eBay ~13.25%; cross-venue singles need >20% gap) and 05_other_arbitrage.md
(PSA grading of <$75 raw cards loses money; Value tier $32.99 Feb 2026). research-rag: no prior user work on this topic.

## Question
1. Digital-rip / vaulted graded-card platforms (Rips by Triumph, Courtyard, Arena Club, Phygitals, Collector Crypt, others):
   is any pack +EV after the instant-buyback haircut, or is any platform's FMV systematically BELOW eBay sold comps so
   that redeem-and-sell beats buyback?
2. Sealed product at MSRP vs secondary (Sept 2026 - Jan 2027 sets): net margin after ~13% fees + shipping; access to MSRP;
   premium decay for 3-5 recent sets.
3. Other structure: pre-order vs release, JP vs EN, reprint announcements, prediction-market contracts on card prices.

## Pre-registered bar (written before looking at any outcome)
An angle "clears the bar" only if ALL hold:
- Expected net >= +$5 per cycle AND >= +10% on capital at risk, after every fee (platform, selling ~13%, shipping, tax
  where applicable), computed from sourced numbers, not from a platform's own marketing EV claim alone.
- Repeatable: >= 2 cycles/month available to a normal CA resident with <= $200 capital, no bots, no ToS violation.
- Max loss per cycle <= $50 and legal in California.
- Evidence: at least 5 concrete price pairs (or a published odds table) supporting the sign of the edge.
Kill (angle marked DEAD): net expected value < 0 after fees on the sourced numbers, OR access requires bots/ToS breach,
OR not available in CA.
A platform-claimed "+EV" or "105% EV" is treated as a marketing claim (unverified) until the odds table and FMV source
can be checked against independent comps.

## Plan
- Search 2-4 parallel queries per platform; read official FAQ/terms pages for buyback %, FMV source, fees, CA availability.
- Search eBay/130point comps for >= 5 cards that appear on vault platforms.
- Sealed: pokemontcg.io API (public, free) for set list + TCGplayer market prices; official MSRP from Pokemon/press.
- Write findings here as I go; inbox result row before final reply.

## Log

### 1. Digital-rip / vault platforms (searched 2026-11-24)
Source legend: [P] primary/official, [N] news/research, [B] affiliate review (biased, secondary), [U] unverified.

**Aggregate house edge (the single most useful number).** Crypto.com Research, "Rips and Gachapon: The Economics of
Physical-Backed Digital Collectibles", 2026-11-10 (https://crypto.com/uk/research/rips-gachapon-economics-oct-2026) [N]:
gross pack sales vs buyback payouts, on-chain gacha platforms: Q1-2026 $323.3M / $282.2M (86.9%), Q2 $757.7M / $660.0M
(87.7%), Q3 $777.1M / $670.0M (88.1%). Immediate buyback offers "typically 85-93%" of stated value. So the platform keeps
~12-13% of every dollar that is bought back. This is the on-chain subset (Collector Crypt, Phygitals, Courtyard, Beezie),
not Rips/Arena/GameStop.
Sportico via SCCG 2026-11-09 [N, secondary]: platforms "typically disclose an average pack value 5% to 15% below the price";
instant sell-back up to 98% at some apps (Barron's). Collector Crypt CEO (Barron's via on360.ca 2026-08-29 [N, AI-assisted]):
up to 98% of opened cards sold straight back; 178 packs opened per card shipped over six months.

| Platform | Buyback | FMV source | Window | Redeem/ship | Payout | CA status | Promo | Src |
|---|---|---|---|---|---|---|---|---|
| Rips by Triumph | 100% of Triumph's own FMV, "no fees" | "semi-automated proprietary system"; "makes no representation that the FMV it uses matches any other market calculation" | >= 7 days | buyer pays shipping (~$5/card [B]); SALES TAX charged at shipping; insurance text conflicts at $1,000 | debit/bank/PayPal/Venmo; min $6; instant ~10 min, bank 1-3 bd; paused at $600 until tax forms | NV excluded; CA not listed as excluded [B] -- unverified | no fixed welcome offer 11-24; refer-a-friend = free starter pack each [B] | docs.triumpharcade.com/tcg-regulatory-and-legal [P]; rotowire.com/sports-cards/rips-by-triumph (11-24) [B]; thelines.com (11-23) [B] |
| Courtyard | 90% of FMV | CardLadder API + grader data + own algorithm [B]; official FMV help article 404'd | 7 days | "at cost plus taxes" [B] | "credited on acceptance"; one Trustpilot user: ~1 month to bank [B/U] | no state list published [B] | none found | intercom.help/courtyardio .../11955231 [P]; rotowire.com/sports-cards/courtyard (11-16) [B]; altstreet: possible 6% fee on buyback [B, U] |
| Arena Club | ~90% of self-set value; "Slab Safe" +10% price floors buyback at 80% of pack price | self-set | ? | ? | ? | ? | ? | rotowire.com/sports-cards/arena-club [B] |
| Collector Crypt (Solana) | >= 85%, 93% on top packs | platform "insured value" [U] | 3 days (Jupiter) | ? | crypto | ? | ? | blockworks.com/news/gachapon-on-blockchain [N]; docs.jup.ag/user-docs/trade/gacha [P] |
| Phygitals (Solana; powers Underdog Rips) | not published | ? | 7 days (Jupiter) | ? | crypto | Underdog Rips: all states except NV [B] | Underdog $1K deposit match [B] | docs.jup.ag [P]; rotowire underdog-rips [B] |
| Fanatics Instant Rips | 90% of appraised value | Fanatics | ? | ? | ? | ? | ? | on360.ca (Barron's) [N] |

Rips by Triumph pack shape: Pokemon tiers $1 ... $2,500 [B]; e.g. $10 pack FMV range $3-$200, $50 pack $15-$1,000.
Football odds bands (rotowire, tested 2026-11-19 [B]): about half of pulls are worth less than the pack price; no EV
published anywhere I found. Without the full odds table, EV/price for Rips cannot be computed. [gap]

**Legal / CA.** AB 831 (signed 2025-10-11, effective 2026-01-01) targets DUAL-CURRENCY sweepstakes casinos
(advertisinglaw.fkks.com; zwillgen.com) [N]. Rips are sold as single-currency USD "shopping" (Triumph terms via rotowire
[B]), so AB 831 does not obviously reach them [inference]. Live legal risk is different: CA lottery-law arbitration
against Whatnot (~70 claimants by June 2026, attorney Paul Lesko; ESPN via on360.ca [N]) and a CA False Claims Act suit vs
Whatnot + Fanatics Live (filed 2025-07, unsealed 2026; CA DOJ declined to intervene) [N]. No state regulator action on rips
as of 2026-11-09 (SCCG) [N]. For a buyer, the risk is platform shutdown/freeze, not prosecution [inference].

### 1b. Key test: is platform FMV systematically BELOW eBay sold comps? (2026-11-24)
Method: Courtyard's site 403s non-browser clients (robots.txt fetch returned 403); I did NOT spoof a UA, so no live
Courtyard FMVs. Instead I paired platform-published values (Courtyard blog "Est." 2026-01-15; Beezie blog "Current Alt
value" 2026-05-05) with PriceCharting's same-month graded price (monthly chart_data; PriceCharting's sold listings are
eBay/Fanatics/Goldin; robots.txt allows /game/ pages). Script: sports/results/pokemon_cards_scripts/pc_chart.py
(ms -> s converted explicitly). Key mapping checked against the page header for Umbreon VMAX #215 (used=Ungraded $1,825,
graded=Grade 9 $1,832, manualonly=PSA 10 $3,376).
Table: sports/results/pokemon_cards/fmv_vs_pricecharting.csv. Ratio platform/PriceCharting, same month:
Umbreon VMAX #215 PSA10 0.959 | Victini #171 PSA10 0.998 | Base Charizard #4 PSA8 0.946 | Plasma Storm Charizard #136 PSA7
0.995 | Delta Species Umbreon #17 PSA10 0.981 | Crystal Guardians Blastoise RH #14 PSA10 1.412. n=6, median 0.988,
range 0.946-1.412. (Ninetales #17 dropped: variant mismatch, 455 vs 80.)
Reading: for liquid cards, platform values track eBay-derived comps within ~5%; thin cards scatter BOTH ways (Blastoise
+41%). No sign of a systematic below-comps FMV. Caveats: n=6; blog/marketing values, not live buyback quotes; Beezie
uses ALT (which itself is an eBay-sales-based index), so agreement is partly by construction.

Break-even for redeem-and-sell vs instant buyback (E = eBay sold price, F = platform FMV, eBay net ~0.8675E after
~13.25% fee per 01_secondary_markets.md, c = inbound + outbound shipping, t = sales tax charged at redemption):
- Courtyard (90% buyback): redeem wins iff 0.8675E - c > 0.90F, i.e. F/E < 0.964 - c/(0.9E). E=$100, c=$15 -> F/E < 0.80;
  E=$1,000, c=$20 -> F/E < 0.94. If the reported 6% fee applies (altstreet [U]), buyback = 0.846F and the bar relaxes to
  ~F/E < 1.0 on $1k+ cards -- but that only matters AFTER you have already paid a -EV pack price.
- Rips by Triumph (100% buyback, CA sales tax ~7.25-10% charged at shipping per Triumph FAQ): redeem wins iff
  F/E < (0.8675 - c/E)/(1+t) ~= 0.79 for a $100 card. Observed ratios are 0.95-1.0 -> buyback dominates.
Conclusion for Q1: the pack itself is the loss (aggregate buyback payouts 86.9-88.1% of pack sales, Crypto.com Q1-Q3 2026).
Exit choice (buyback vs redeem) moves a few % at most on $1k+ cards and is negative on <$100 cards.

### 2. Sealed at MSRP vs secondary (2026-11-24)
Sets in the Sept 2026 - Jan 2027 window (pokemontcg.io /v2/sets, file sports/results/pokemon_cards/ptcgio_sets.json):
30th Celebration (2026-09-16), Delta Reign (2026-11-06). January 2027 has no announced main expansion: only a Valentine's
box (2027-01-15 per icv2 / 01-27 per PokeBeach, conflict [N]) and the Pokemon Day 2027 collection ($14.99, 2027-01-29,
godisageek.com/2026/11/pokemon-day-2027-collection [N]).
MSRPs: 30th Celebration ETB $49.99, Pokemon Center ETB $59.99, Booster Bundle $26.94, UPC $179.99 (billsarchive.com
30th-anniversary MSRP table, TCGplayer market snapshot 2026-11-20: ETB $100.55, PC ETB $251.85, bundle $58.49, UPC Day
$318.33 / Night $353.42) [B, secondary]. Booster pack MSRP rose $3.99 -> $4.49 by 2026-03 (Kotaku 2026-03-25 [N]);
standard ETB MSRP for 2026 main sets NOT verified ($49.99-$59.99 range seen; StockX lists PC ETB "retail $50" [B, U]).
GameStop prices far above MSRP (30th ETB $99.99; Delta Reign ETB $89.99) -- GameStop is not an MSRP source [N].

Premium-decay data (PriceCharting month-end sealed price, key 'used' verified = page header $118.38 for Prismatic ETB;
files raw/pcs_*_etb.html; table sports/results/pokemon_cards/etb_flip_margin.txt):
| ETB | first month | P_first | P_+3mo | P_2026-11 |
|---|---|---|---|---|
| 151 | 2023-09 | 52 | 39 (2023-12, BELOW MSRP) | 480 (peak 610 in 2026-04) |
| Surging Sparks | 2024-11 | 57 | 82 | 123 (trough 67 in 2025-11) |
| Prismatic Evolutions | 2025-01 | 103 | 128 | 118 (peak 176 in 2026-03) |
| Destined Rivals | 2025-05 | 105 | 90 | 99 (peak 215 in 2026-03/04) |
| Perfect Order | 2026-03 | 77 | 66 | 64 |
| Chaos Rising | 2026-04 (pre-order) | 114 | 67 | 63 |
| Pitch Black | 2026-07 | 78 | 70 | 62 |
| 30th Celebration | 2026-09 | 165 | -- | 110 (PC ETB 310 -> 218) |
| Delta Reign | 2026-10 (pre-order) | 106 | -- | 79 (one month) |
Reading: 2026 main sets open at ~1.5-2.3x MSRP and settle at ~1.25x within 2-4 months; 2023 151 fell BELOW MSRP within 3
months and only later became a 10x on the 2024-26 bull. Since mid-2026 almost every sealed series is falling (151 -24%
from peak, Destined Rivals -54%, Prismatic -33%). icv2 2026-09-17: 30th ETB sold "just over $300" on eBay the day after
release (6x MSRP), i.e. month-end numbers understate first-week prices [N].

Net margin, buy 1 ETB at $49.99 + 9% CA tax [U] = $54.49, sell on TCGplayer at -13% fee - $8 ship [U]:
release-month sell: median +$5.9, mean +$18.2, n=13, 2 negative (151, Surging Sparks); best +$81 (30th Celebration).
+3-month sell: median +$2.8, mean +$10.9, n=11, 4 negative. If MSRP is really $54.99 subtract ~$5.5 per row.
So the sealed flip is real but small per unit, and the whole margin is earned by whoever gets MSRP stock in week 1.

MSRP access (all secondary): Pokemon Center per-product limits (30th: PC ETB 2, Booster Bundle 3), virtual queue,
"Early Access" since April 2026 aimed at resellers; PC "cancels countless reseller orders" (gamesradar [N]).
Walmart limit 5 TCG items per visit since 2025-11-05; Target "one unit per item, per guest" on some drops [N/B].
Target 30th restock produced a physical altercation (notebookcheck [N]). Bots/queue tools are out of bounds for us.

TCGplayer market now (tcgcsv.com mirror of the TCGplayer API, last-updated 2026-11-24T20:05Z; robots Allow /; file
sports/results/pokemon_cards/tcgcsv_sealed_2026-11-24.csv, script pokemon_cards_scripts/tcgcsv_sealed.py):
- 30th Celebration: ETB market $111.20 (low $99), PC ETB $249.37, Booster Bundle $63.07, pack $9.76.
- Delta Reign (day 18): ETB $74.75 (low $59), PC ETB $132.10, Booster Box $193.56 (MSRP $143.64 per Kotaku), pack $6.20.
- 2026 main sets after decay: pack LOW $4.51-4.99 (MSRP $4.49), Booster Bundle LOW $24-25, ETB LOW $47-49 (Perfect Order,
  Chaos Rising, Pitch Black) = at or below MSRP. Market (trailing) ETB $62-70.
Net if bought at MSRP today: Delta Reign ETB +$2.5, Delta Reign booster box ~-$3, 30th ETB +$34, 30th PC ETB ~+$140
(but PC ETB = 2/customer + queue + reseller cancellations).

### 3. Other structural angles
**Kalshi KXPOKEMON (new to this project).** Series KXPOKEMON "what will the price of card be", settlement source Collectr
"Ungraded Price", monthly "above strike on <month-end>" binaries, quadratic fee x1 (Kalshi API GET /series/KXPOKEMON,
file sports/results/pokemon_cards/kalshi_series_KXPOKEMON.json). Strike ~= Collectr price at market open (~3rd of month).
Rules: "first non-preliminary value published by the platform after the time period concludes ... revisions ... will not
affect resolution". 94 events since ~2026-07 (win.gg 2026-07-28 [N]); 46 markets returned; total volume 72,440
contracts across them (kalshi_markets_KXPOKEMON.json) -> tiny ($10s-$1,000s per market).
Results: 21 finalized (Sep 15, Oct 6): 18 NO, 3 YES. Quotes vs outcome (kalshi_candles/*.json, script
pokemon_cards_scripts/kx_poke_calib.py, output kx_poke_calib.txt): buying NO at the NO ask returned, pre-fee, T-14 +$0.158
/contract (n=19, 17 wins), T-7 +$0.097 (16 wins), T-2 +$0.098 (17 wins), excluding the anomaly below.
NOT an edge claim: 19 markets = 2 monthly cohorts in a falling sealed market (section 2), so it is ~2 correlated bets on
"modern Pokemon fell in Sep-Oct 2026". Same bias shape as the favourite-longshot pattern the project knows from sports.
**Settlement anomaly (inspected row):** KXPOKEMON-26SEPCHA-39.87 ("Charmander", strike $39.87) settled YES with
expiration_value 190.24; last trade 0.02; the next month's Charizard market strike was $194.14. Most likely the Charizard
value was used, or the item label ("the Charmander", no set/number) is ambiguous. A NO buyer at 0.97 lost 0.97 -- one
such event per ~20 markets erases the whole "near-certain NO" harvest. Item labels on several markets are ambiguous
("Mew", "Charizard", "Mew ex" x2 with different strikes).
Collectr price sources: "TCG Player, eBay, Cardmarket and potentially secondary markets" (getcollectr.notion.site prices
FAQ via search snippet [P, not opened]). Whether Collectr "Ungraded Price" == TCGplayer market price is being checked.

**Japanese vs English sealed import.** US de minimis suspended since 2025-08-29 -> ~15% tariff on Japan imports (seller blog
samuraiswordtokyo.com [B]; another guide says 12.5% after a 2026 ruling -- conflict, unverified); proxy+ship+duty add
30-70% landed ([B]); JP box MSRP rose to Y6,000 in May 2026 and Y230/pack planned May 2027 (PokeBeach via search [N]).
No sourced current US-vs-JP spread; with 13% sell fee + 15% duty + shipping, a JP->US box flip needs a >35% gap [inference].
Not pursued further (capital + 2-4 week shipping lag + duty uncertainty).

**GameStop Power Packs** (PSA/Collectors + GameStop, launched April [N]): buyback 90% of Card Ladder-based FMV minus a 6%
fee = ~84.6% (deadspin/thespike reviews [B]; Kotaku [N]); 7-day window; $10,000 pack had 44.8% chance of a $6-8k card
(Dexerto [N]). Same structure as Courtyard: -EV by construction.

**Pre-order vs release.** 30th Celebration ETB pre-sale $169.99 at GameStop (icv2 2026-09-17 [N]) vs TCGplayer $111 now
and PriceCharting month-end $165 -> $110; Chaos Rising ETB pre-order month $114 -> $63. Paying secondary pre-order
prices lost 30-45% within 2-3 months on 2026 sets. Pre-ordering AT MSRP is the only positive version, and that is
the same access bottleneck as section 2.

**Reprint announcements** as price events: no dated announcement -> price study found (billsarchive "reprints reshaping
market" and pokemonpricetracker are commentary [B]). Gap.

KXPOKEMON live depth (read-only GET /markets/{t}/orderbook, 2026-11-24, file kalshi_nov_depth.txt): a maker already bids
NO at 0.94-0.95 in ~100-contract clips on the near-certain markets, and most of them have NO resting YES bid to sell
into -> a taker cannot harvest "certain NO" at all; a maker competes with that 0.95 bid for ~$0.04/contract.
Capacity per market ~100 contracts x ~$0.04 = ~$4. The only open question is the few genuinely uncertain markets
(e.g. NOV ASCHEREL strike $124.75 vs TCGplayer market $127.32; NOV PERORDBO strike $173.16 vs TCGplayer $173.04).

Promos on rip apps (all affiliate-sourced, [B, unverified], one-time, KYC required): Packz.io "100% buyback on first pack
up to $100"; Crown Rips free $1 pack; TiltRips free $10 pack via referral; Rips by Triumph refer-a-friend free starter
pack; Underdog Rips "$1K deposit match" (likely play-through credit). These are promos, not an inefficiency; they belong
to the promo_hunt agents' ledger.

## Ranking (2026-11-24) against the pre-registered bar
| Angle | Net $/cycle | Max loss | Capital | Hours/cycle | Legal/terms | Edge decay | Verdict |
|---|---|---|---|---|---|---|---|
| Sealed ETB at MSRP, sell release week | median +$6, mean +$18 per ETB (n=13, PriceCharting month-end); 30th Celebration +$81; 2026 main sets ~$0 after 3 wks | ~-$17/ETB (151 in 2023, fell below MSRP) | ~$55/ETB | 1-3 h queue/store + 1 h list/ship | legal; retailer "personal use" terms -> cancellations; limits 2-5 | 1-3 weeks after release | real but access-gated; ~1 cycle per 1-2 months -> FAILS the 2 cycles/month bar |
| Kalshi KXPOKEMON near-expiry | ~$0-4/market (maker vs existing 0.95 NO bid); uncertain markets TBD | stake (binary) + settlement-label risk (1/21 mis-settled) | <= $50 | <1 h/month once scripted | CFTC-regulated; orders need user's yes (project rules) | n/a | LEAD for a $0 paper test only |
| Rip-app promos (first-pack guarantees, free packs) | +$1-15 one-time each [U] | ~$0-10 | $0-100 | 0.5 h + KYC | platform/legal-gray (CA lottery suits vs Whatnot/Fanatics) | one-time | promo, not edge -> promo ledger |
| Redeem-and-sell vs buyback | negative <$100 cards; +0-5% on $1k+ Courtyard cards only if FMV < 0.94x eBay | pack price | pack price | 1-2 h + 1-2 wk ship | same | n/a | DEAD (pack is -12%) |
| Digital rips EV | -12% of spend (aggregate payout 86.9-88.1%, Q1-Q3 2026) | pack price | any | 0 | gray | n/a | DEAD |
| JP -> US sealed import | needs >35% gap; no sourced gap | box price + duty | $100+ | 2-4 wk ship | 12.5-15% duty | n/a | DEAD at this budget |
| PSA grading / cross-venue singles | see prior notes 05 / 01 | | | | | | DEAD / unchanged |

Verdict: NO angle clears the pre-registered bar. The closest real effect is the release-week sealed premium, which is
an access lottery, not a repeatable inefficiency for a normal CA buyer without bots.

### 3b. Does Collectr "Ungraded Price" (Kalshi settlement) == TCGplayer market price? (2026-11-24)
Method: tcgcsv.com daily archive prices-2026-09-30.ppmd.7z (TCGplayer API snapshot; extracted with py7zr in a throwaway
venv /tmp/pk_venv), product IDs mapped by name from tcgcsv products files; script
sports/results/pokemon_cards_scripts/collectr_vs_tcgp.py. Kalshi expiration_value vs TCGplayer marketPrice on 09-30:
151 UPC 879.17 vs 879.17 (exact) | Ascended Heroes ETB 151.62 vs 153.64 (0.987) | AH PC ETB 356.52 vs 356.32 (1.001) |
Perfect Order booster box 171.63 vs 173.10 (0.992) | PO PC ETB 123.30 vs 122.85 (1.004). n=5, all within +-1.3%.
Reading: Collectr's sealed "Ungraded Price" is (close to) TCGplayer Market Price, which tcgcsv publishes daily for free.
So the settlement variable is observable before expiry. Residuals may be snapshot timing (checking the 10-01 snapshot).
Follow-up, 10-01 snapshot (first value AFTER the period, per the rules): ratios 1.005 / 0.980 / 1.003 / 0.986 / 1.009 --
NOT a better fit than 09-30. October: 30th PC ETB Collectr 277.53 vs TCGplayer 10-31 278.65 (0.996). So across n=6
markets and 2 snapshots, |Collectr/TCGplayer - 1| <= 2.0% (max: Ascended Heroes ETB 0.980). Collectr is a ~2%-noisy copy
of TCGplayer market price (possibly smoothed or sampled at a different time) [inference].
Consequence for an edge: markets >3% from strike near expiry are near-certain AND already quoted 0.94-0.95 NO by a maker;
markets within ~2% of strike are exactly where the proxy cannot call them. Example: Perfect Order booster box Sep, strike
174.14, TCGplayer 10-01 = 174.13, Collectr settled 171.63 -> NO, while Kalshi quoted YES 0.82/0.89 at T-2. The edge window is
a narrow 2-3% band. LEAD only, weak.

## Proposed test (the only one worth running; $0 paper first)
Name: KXP-PAPER. Question: does TCGplayer market price (tcgcsv daily, free) at T-3d predict KXPOKEMON settlement well
enough to beat spread + fee on markets 2-6% from strike?
Universe: sealed-product KXPOKEMON markets whose item maps to ONE TCGplayer productId (skip ambiguous single-card labels
like "Mew", "Charizard", "Mew ex"). Cohorts: Dec 2026 and Jan 2027 month-ends.
Rule (fixed now): at T-3d ~20:00 UTC, d = ln(P_tcgp / strike). P_model(YES) = Phi(d / s), s = 0.02 + sigma_daily*sqrt(3),
sigma_daily = stdev of daily log changes of that product over the prior 30 tcgcsv daily archives. Paper-trade 10
contracts at the executable touch when |P_model - price| >= 0.10; fee = ceil(0.07*C*P*(1-P)) cents rule.
Success (all): >= 12 paper trades across both cohorts; mean P&L after fees >= +$0.05/contract with a cohort-clustered
bootstrap 90% CI lower bound > 0; Collectr settlement within +-2% of the tcgcsv T+0 snapshot on >= 90% of mapped markets.
Kill (any): < 12 qualifying trades by the Jan 2027 settlement; settlement match < 80%; executable depth at the signal price
< 10 contracts on more than half the signals; any mis-settlement of a mapped market.
Live stage ONLY on the user's explicit yes (project rule): <= $50 total, <= $10 per market, maker orders. Ceiling even if it
passes: ~5 uncertain markets/month x ~$4 = ~$20/month -> a learning exercise, not income.
Secondary (if the user wants a physical test): 1 ETB at MSRP in-store on the next English release day (date not
announced as of 2026-11-24), list on TCGplayer within 48 h; success net >= +$10; kill if not obtainable at MSRP in 2
attempts or net < $0. Max loss ~-$17 (the 151 case).

## Learnings
### Digital pack-rip apps return about 87-88% of pack spend in aggregate
Crypto.com Research (2026-11-10): buyback payouts / gross pack sales = 86.9% (Q1), 87.7% (Q2), 88.1% (Q3 2026) across
on-chain gacha platforms. A rip is a ~12% house-edge bet; no buyback-percentage choice fixes that.
### Vault-platform FMVs track eBay-derived comps; there is no systematic below-comps gap to arbitrage
Platform value / PriceCharting same-month, n=6: median 0.988, liquid cards 0.946-0.998, one thin card 1.412
(sports/results/pokemon_cards/fmv_vs_pricecharting.csv). Redeem-and-sell beats a 90% buyback only if FMV < ~0.94x eBay on
$1k cards and < ~0.80x on $100 cards; Rips by Triumph also charges sales tax at shipping, so it needs FMV < ~0.79x.
### The sealed-product premium is a release-week access lottery, not a repeatable edge
ETB bought at $49.99 + 9% tax and sold at the release-month price: median +$5.9, mean +$18.2 (n=13); +3 months: median
+$2.8 (n=11, 4 losers) (sports/results/pokemon_cards/etb_flip_margin.txt). 2026 main-set ETB/pack/bundle low listings
are at or below MSRP within ~3 months (tcgcsv 2026-11-24). 151 fell below MSRP 3 months after release before its later run.
### Kalshi KXPOKEMON settles on Collectr, which is a ~2%-noisy copy of TCGplayer market price
n=6 markets, 2 snapshots: |ratio - 1| <= 2.0%, one exact match (151 UPC 879.17) (pokemon_cards_scripts/collectr_vs_tcgp.py on
tcgcsv archives). The settlement variable is observable daily for free, but only to ~2%, which is the band where
markets are uncertain.
### KXPOKEMON mis-settled 1 of 21 markets; label risk is larger than the near-certain-NO harvest
KXPOKEMON-26SEPCHA-39.87 ("Charmander") settled YES at 190.24 (about Charizard's price) after trading at 0.02. A NO
holder lost 0.97, which erases ~20 markets of 0.05 harvest. Items with ambiguous labels should be excluded from any strategy.
### A maker already quotes 0.94-0.95 NO on near-certain KXPOKEMON markets
Orderbook 2026-11-24 (kalshi_nov_depth.txt): ~100-contract NO bids at 0.94-0.95 and often no YES bid to sell into, so a
taker cannot buy cheap certain-NOs; capacity ~$4/market.
### Free data sources for Pokemon prices that respect robots.txt
tcgcsv.com (TCGplayer API mirror; robots Allow /; daily 7z archives, PPMd -> py7zr) and PriceCharting /game/ pages
(robots disallow only /buy etc.; monthly chart_data in cents with ms timestamps; keys used=ungraded/sealed, graded=9,
manualonly=PSA10, new=8, cib=7). Courtyard returns 403 to non-browser clients; do not spoof.

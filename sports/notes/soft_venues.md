# Soft venues outside Kalshi vs Pinnacle fair (2026-10-22)

User 2026-10-22: "Don't just do Kalshi, do some other ones too." We took the one idea that has passed (soccer S12:
a soft price compared with Pinnacle's fair price after a power de-vig) and applied it to DFS pick'em and to the
peer-to-peer / sweepstakes exchanges.
Everything below is a DESCRIPTIVE snapshot, not a trial. Nothing was scored, so nothing went into lab/ledger.jsonl.
Both snapshots were taken at about 07:05-07:10 UTC (overnight US), so the lines may be at their stalest or their
most settled. A single snapshot says nothing about daytime line moves.

## Bottom line
- **Sleeper Picks vs Pinnacle props:** 0 of 1,047 legs clear break-even by 2 points. The best leg is -0.1 points.
  The median leg is -6.6 points below break-even.
- **Polymarket US game winners vs Pinnacle:** 0 of 116 quotes have EV > 0 after the taker fee. The median EV is
  -4.25% and the best is -0.55%. The median gap between the mid and Pinnacle fair is 0.72c.
- So neither venue showed a soft-book edge in this snapshot. That matches what we found before: Kalshi soccer had
  0 of 296 quotes at 1% or better, and Kalshi and Polymarket sit within about 1c of each other.

## 1. DFS pick'em

### Break-even per leg (`scripts/pickem.py tables`, exact binomial solve, legs assumed independent)
| table | p* | table | p* |
|---|---|---|---|
| PP power 2 (3x) | .5774 | UD std 2 (3.5x) | .5345 |
| PP power 3 (6x) | .5503 | UD std 3 (6.5x) | .5358 |
| PP power 4 (10x) | .5623 | UD std 4 (12x) | .5373 |
| PP power 5 (20x) | .5493 | UD std 5 (20x) | .5493 |
| PP power 6 (37.5x) | .5466 | UD std 6 (35x) | .5529 |
| PP flex 5 (10/2/0.4) | .5425 | UD flex 5 (10/2.5) | .5475 |
| PP flex 6 (25/2/0.4) | .5421 | UD flex 6 (25/2.6/0.25) | .5382 |

Caveats on these tables:
- They come from third-party sites (BettingPros 2026-08-18) because the official help pages 404'd. Treat them as
  UNVERIFIED and check them in the app.
- Underdog 4-pick flex at 7.2/1.8 would break even at .5179, which is suspiciously generous. Another source lists
  6/1.5 (.5503).
- **PrizePicks has run only peer-to-peer "Arena" since 2025-08-22.** The fixed Power/Flex tables above may no
  longer be what pays out.
- **Sleeper has no fixed table.** Each leg carries its own multiplier m, and the entry pays the product of those
  multipliers, so each leg breaks even at 1/m.
- On Sleeper, a 50/50 leg is priced at m = 1.78, which means break-even is .562. The median two-way hold is 12.4%,
  against 6.9% for Pinnacle props on the same markets.

### Data access (read-only, no login)
| venue | lines | how |
|---|---|---|
| Sleeper | YES | `api.sleeper.app/lines/available` (robots allows all) + `/v1/players/<sport>` names; ~3,400 markets, per-leg multipliers |
| PrizePicks | NO | api/app behind a DataDome captcha -> not bypassed |
| Underdog | NO | `beta/v5|v6/over_under_lines` returns 426 "upgrade required" to non-app clients; not pursued further |
| Betr, Dabble, ParlayPlay | not tried | |
| Pinnacle props | YES | guest API `type=special, category="Player Props"`: NBA pts/reb/ast/3PM, NFL yds/rec/att/TD, NHL G/A/SOG, CFB yds |

### Live snapshot (`scripts/pickem.py snapshot`)
Run at 2026-10-22 07:05 UTC.
- Pinnacle had 648 props on offer: NBA 72, NFL 324, NHL 232, CFB 20.
- Sleeper had 5,800 pre-game player legs.
- 1,047 legs matched on the same player and stat. Of these, 912 had the same line. The other 135 had a line that
  was better for the bettor; for those, Pinnacle's fair price is a lower bound.
- On exact-line legs, conservative fair minus 1/m had a median of -6.6 points, a 90th percentile of -5.3 points,
  and a maximum of -0.1 points.
- **Legs at or above break-even plus 2 points: 0 of 1,047.** Sleeper re-prices its multipliers along with the
  market and charges roughly twice Pinnacle's margin.

A what-if for the fixed-table venues, whose lines we could not get:
- If their line equalled Pinnacle's, 245 of 648 Pinnacle prop sides have conservative fair ≥ .5625, which is the
  break-even for PP flex 5 plus 2 points.
- Nearly all of these are lopsided lines: NHL anytime goal/assist 0.5 unders (164), NFL receptions, and anytime TD.
  Fixed-table venues do not offer lopsided lines at standard payout (that is what goblins and demons are for).
- So this number is an upper bound that probably does not exist in practice. It is NOT an edge estimate.

### Historical backtest
No free archive of PrizePicks, Underdog or Sleeper lines exists; a GitHub/Kaggle search found only live
scrapers. So there is no backtest. Instead there is a forward paper logger (written, **NOT installed**):

    bash sports/scripts/install_pickem_logger_agent.sh      # launchd, `pickem.py log` every 30 min -> ~/.claude/quant-kalshi/data/pickem/log.jsonl

PRE-REGISTERED 2026-10-22 (before any forward data): **PK1 sleeper_pin2**
- Selection: Sleeper legs where conservative Pinnacle fair (min of the power and multiplicative de-vigs, exact or
  better line) is at least 1/m + 0.02, taking the first logged time per player/stat/side.
- Each leg is graded on its own as a 1-unit leg: profit = m·win - 1. This is valid as the EV of a product entry
  only if the legs are independent, so legs from the same game are clustered.
- PASS: at least 200 legs, ROI > 0, and a one-sided bootstrap over games with p < 0.05.
- Grading needs box scores, and that is not built yet. Given 0 qualifying legs today, PK1 may never reach 200
  legs; the logger's job is first to measure how often such legs occur at all.

## 2. Peer-to-peer / sweepstakes exchanges
| venue | California? (Oct 2026) | fees | public data w/o login | lag vs Pinnacle |
|---|---|---|---|---|
| Polymarket US (CFTC) | yes (all states but NV, per Covers 10-20) | taker 0.0695·p(1-p) (docs, eff. 10-19); maker 0 | YES, gateway.polymarket.us | **no**: 0/116 EV>0, mid within 0.72c |
| Novig (CFTC DCM since 2026-06, live 08-04; sweeps retired) | yes (47 states + DC, not AZ/MI/NV, per Covers) | pre-game 0; live takers 0.03-0.06·p(1-p) (Dimers, low confidence) | NO (site behind AWS WAF; no public API) | unmeasured |
| ProphetX (CFTC DCM+DCO 2026-06) | listed all states but NV (Covers) | 1-2% of net winnings OR per-contract, conflicting, UNVERIFIED | NO (affiliate key only) | unmeasured |
| Sporttrade | NO (exited state books mid-2026; CFTC application pending; site says waitlist) | n/a | n/a | n/a |
| Fliff | sweeps OFF in CA since 2026-01-01 (AB 831); "Superstars" peer-to-peer DFS live in CA | n/a | not tried | n/a |
| Underdog Predictions, FanDuel Predicts, Crypto.com, Gemini | listed for CA by Covers; FanDuel's CA sports UNCLEAR | not checked | not checked | unmeasured |

Polymarket US snapshot (`scripts/pmus_vs_pin.py`, 2026-10-22 ~07:10 UTC):
- 58 games matched: NBA, NFL, NHL and CFB moneylines.
- EV after the taker fee had a median of -4.25% and a maximum of -0.55%. No quote reached 1% or 2%; none was even
  above 0.
- Like Kalshi, Polymarket US is market-made against the sharp line.

## 3. California legality (research agent; third-party sources, UNVERIFIED where marked)
- **AG opinion.** CA AG opinion 23-1001 (2025-07-03) says pick'em and draft DFS are illegal under PC §337a. It is
  advisory. Underdog's suit against it was dismissed, and no enforcement action has been found.
- **Pick'em in California.** Reachable only in peer-to-peer form: PrizePicks Arena, Underdog Champions,
  DraftKings Pick6 and Fliff Superstars.
  - Sleeper, Betr and ParlayPlay are listed as available in CA by a single affiliate page (2026-08). Their format
    is UNVERIFIED.
  - Class actions are pending against PrizePicks, Underdog, DraftKings and FanDuel.
  - **Legal risk is live:** the AG says these contests are illegal betting.
- **Sweepstakes.** AB 831 (effective 2026-01-01) ended dual-currency sweepstakes in CA. Novig and ProphetX moved
  to CFTC exchanges, which is how they are available in CA now.

## What the user would need to open
- **Measuring Novig or ProphetX:** an account the user opens and screenshots or exports themselves. We do not log
  in or automate.
- **Measuring PrizePicks or Underdog:** a captcha-free source, either a paid odds feed that carries DFS lines
  (e.g. OddsPapi or SportsGameOdds, which claim to; UNVERIFIED) or the user pasting boards.
- **Acting on anything:** a pick'em account under ToS, entered by hand (no automation). But the snapshot shows no
  edge to act on yet, so the first step is the free paper logger (install command above).

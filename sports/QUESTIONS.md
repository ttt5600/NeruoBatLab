# fund: QUESTIONS (pre-registered before anything runs)

Goal: double a $1-2k unlevered bankroll in 12 months, proven live. That is $1-2k profit/year = $2.7-5.5/day, i.e.
about $140-280/day of QUALIFIED volume at a 2% net edge (more volume at a thinner edge). A question earns its place by
adding verified daily volume x edge toward that line.

Status tags: [proposed] [approved] [running] [done] [refuted] [dropped]. Agents may self-approve only cheap,
reversible questions (no money, allocation or outward action). Anything that places orders, changes live size or
spends money needs the PM (user). Success and kill criteria are written by /prereg BEFORE approval and are then
frozen; after the run, add a `result:` line citing a finding id and never edit the criteria.
Every scored variant is a ledger trial (lab/ledger.jsonl); the gate is `scripts/fund_stats.py` (BH q < .10 + both
halves = LEAD; Bonferroni or dataset bar + forward evidence = SEALED-ASK). Do-not-redo list: notes/EDGES_AND_AGENTS.md s5.
Order below = blueprint ranking (evidence x capacity), notes/FUND_ARCHITECTURE.md s4.

<!-- template
## Q001 [proposed] <short title>
question:   <one sentence>
mechanism:  <why the price should be wrong, and who is on the other side>
baseline:   <what it is compared against, with its path>
change:     <the one thing that differs>
metric:     <primary metric, out-of-sample>
success:    <threshold on the observed difference and its interval>
kill:       <result that ends this line of work>
power:      <smallest effect resolvable; why the expected effect exceeds it>
budget:     <trials (fixes the family Bonferroni bar) / time / money>
result:     <filled in after: observed number + interval + finding id>
-->

## Q001 [proposed] Maker execution of the live weather edge (E4 on) with the per-pick cap raised toward ~$75
question:   Does live E1+E2 executed as post-only maker (KALSHI_EXEC_STYLE=maker) earn more per $ than the live taker, at
            the sealed rate, as the per-pick cap rises from today's ~$2-3 toward the ~$75 adverse-selection optimum?
mechanism:  Takers pay the spread plus the taker fee (weather charges takers only); a resting NO bid at the mid collects
            price improvement. Cost: losers always fill, winners fill only when trades print through the bid.
baseline:   live taker record (~/.claude/quant-kalshi/data/kalshi_exec.jsonl) and the sealed E4 replay
            (lab/lockbox.jsonl kalshi_temp_maker_M3_N10: ROI +2.22% vs taker +1.07%, fill 80%, filled loss 3.07%).
change:     exec style taker -> maker (A/B by sha1 parity is already coded: KALSHI_AB=1), then the cap ladder.
metric:     live ROI per filled $ (maker arm) and maker-minus-taker delta per pick; filled loss rate vs sealed CI.
success:    set by /prereg (proposal: live maker ROI inside the sealed CI [+1.57%, +2.84%] after >= 486 filled picks).
kill:       set by /prereg (proposal: filled loss rate above the sealed 95% CI, or maker ROI < taker after 1,000 picks).
power:      sigma ~0.195 per filled $ (notes/fund_economics.md): absolute edge 2.2% needs 486 filled picks; the
            maker-taker delta ~1.15% needs ~1,780. At ~8 filled picks/night that is ~2 and ~7 months.
budget:     1 trial (no search). Money: needs PM yes (env change + cap ladder). Capacity ceiling ~$144/month.

## Q002 [proposed] Kalshi Liquidity Incentive Program (LIP) reward per $ resting on weather brackets
question:   On the markets where we already hold an edge (KXHIGH weather brackets), what LIP reward $/day does a resting
            quote of size s earn, net of the adverse selection that resting creates?
mechanism:  The program (amended 2026-10-28, up to $10k/business day across markets) pays makers for time-weighted
            resting size near the inside; rewards stack on top of the trading edge.
baseline:   the same quotes with no reward (Q001 maker P&L); program rules as published by Kalshi (pull and hash first).
change:     add the per-market reward stream to the maker P&L.
metric:     reward $/day per $ resting by market and by distance from the inside; net P&L per $ resting.
success:    set by /prereg (proposal: paper reward + edge > taker edge with capacity >= $10/day on current caps).
kill:       set by /prereg (proposal: reward < 0.2%/day of resting $ or concentrated in markets we cannot quote safely).
power:      descriptive measurement first (2 weeks of paper quotes); the money test comes after, as its own question.
budget:     3 trials. 2 weeks paper (no orders). Tiny live only on PM yes.

## Q003 [proposed] Maker NO/favourite quoting in thin retail categories (mentions, entertainment, single-name)
question:   Does a resting NO (favourite side) quote in thin retail Kalshi categories earn a positive net return per
            filled $, where the taker version (M3) failed sealed?
mechanism:  Retail takers overpay for YES/longshots; makers on the other side earn ~2c/contract in the literature
            (Burgi-Deng-Whelan; Becker 72M trades). Largest in thin categories, shrinking over time.
baseline:   taker M3 on the same events (sealed FAIL: d24 +0.45%/2,674 p .22; lab/lockbox.jsonl) and K-FAV maker in
            non-sports daily markets (DEAD: -2.19%/2,829, adverse selection; notes/kalshi_fav_maker.md).
change:     taker -> resting NO bid, category restricted to mentions/entertainment/single-name, replayed on the public tape.
metric:     maker ROI per filled $, fill-conditional loss rate vs unfilled loss rate (the adverse-selection signature).
success:    set by /prereg.
kill:       set by /prereg (proposal: filled loss rate above break-even, as in K-FAV).
power:      at mu 2%, sigma ~0.3 per $ -> 1,392 fills; check the tape has that many before running.
budget:     <= 6 trials (dataset bar .0083). Dev replay only; sealed on PM yes.

## Q004 [proposed] Model-timing edges taken at the first tight Kalshi quote (NBA/NHL/NFL/CFB totals vs openers)
question:   Do our simple totals models, which lead sportsbook OPENERS (CLV p < 1/4000 in every sport), make money when
            taken on Kalshi at the first quote with spread <= 10c (L35)?
mechanism:  Opening prices are stale before sharp money arrives; the edge is timing, so it needs early, low-fee entry.
            Maker fills at first listing were adversely selected (+3.3c CLV but maker fills lost), hence take-at-tight-quote.
baseline:   Kalshi close price for the same contract; forward loggers nba_open_totals.py, open_lines_logger.py (nhl/nfl).
change:     entry = first tight Kalshi quote instead of the sportsbook opener.
metric:     forward ROI at the Kalshi ask + fee, with CLV as the leading indicator; event-clustered.
success:    set by /prereg (forward only; never re-mine the 109 dev games).
kill:       loggers' own rule at 100 settled picks per sport (CLV <= 0 or ROI CI below 0).
power:      at mu 2%, sigma ~1.0 per $ the ROI test needs ~15,000 bets (impossible); CLV (sigma ~1.7 pts, mu ~1.2) needs
            ~12 picks. So CLV decides go/no-go and ROI only bounds the loss -> keep stakes minimal.
budget:     0 dev trials; forward paper ~100 picks per sport (~1 season-month each).

## Q005 [proposed] Small / long-tail market calibration by volume quintile (Polymarket US + Kalshi)
question:   Is the favourite-longshot mispricing larger in low-volume markets, by enough to clear fees and spread?
mechanism:  Thin retail markets have fewer informed makers; the literature finds the bias largest there.
baseline:   the market price (event-clustered realised minus price-implied probability), all volumes pooled.
change:     stratify by volume quintile and price bucket.
metric:     realised - price per bucket, net of the venue fee formula and the observed spread proxy.
success:    set by /prereg.
kill:       set by /prereg (proposal: no bucket clears costs with BH q < .10).
power:      quintile x price grid = one family; BH over all cells; both-halves check required.
budget:     1 scan, cells counted as trials. NOTE: an exploratory scan appeared 10-27 (scripts/smallmarket_calibration.py,
            smallmarket_analyze.py, untracked) -- treat as EXPLORATORY; a confirmatory version must be pre-registered here.

## Q006 [proposed] Seasonal / physical data markets: monthly rain/snow unreachable-NO (L39)
question:   Does the frozen W1m rule (resting NO on monthly rain brackets the season total cannot reach) earn forward?
mechanism:  Thresholds already unreachable given observed month-to-date totals and forecasts still trade a few cents of YES.
baseline:   dev W1m +3.6%/146 bets, 1 loss; market-null p .016, one-per-city-month p .12 (gate: DEAD on dev alone).
change:     none -- forward paper of the frozen rule (scripts/kalshi_wx_monthly_logger.py, launchd hourly).
metric:     forward ROI per $ and loss rate vs break-even (~4%).
success:    set by /prereg (proposal: one sealed shot ~Feb 2027 on rain Jul26-Jan27 + snow).
kill:       any 2 losses in the first 100 forward picks.
power:      mu 1%, sigma 0.15 -> 1,392 picks; at ~20-25/month that is years -> capacity (~$7.5/month) caps the value.
budget:     0 new trials; logger already running. Low priority: cannot move the doubling target.

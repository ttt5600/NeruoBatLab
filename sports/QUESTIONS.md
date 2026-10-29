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

## Q002 [running] Kalshi Liquidity Incentive Program (LIP) reward per $ resting on weather brackets
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

## Q003 [DEAD: sealed FAIL 10-29] Maker NO/favourite quoting in thin retail categories (mentions, entertainment, single-name)
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
finding:    2026-10-28 (notes/q003_retail_maker.md, scripts/q003_retail_maker.py, 4 trials, bar .0125). Mentions only;
            YES mid 5-35c, back-of-queue tape fill, cancel anchor-2h. LEAD: IN1_d6 +7.08%/859 fills p<.0002,
            MID_d6 +5.78%/897 p .0002; DEAD: both D24 trials (2nd half negative). Adverse selection small (0.3-1c);
            edge decays (D6 half2 +1.9/+2.9% vs half1 +10-12%). 258/1500 sample tickers not scored (budget cap).
            Entertainment/culture outside Mentions not tested. Sealed ask: IN1_d6 on anchors >= 2026-06-25.

## Q004 [running] Model-timing edges taken at the first tight Kalshi quote (NBA/NHL/NFL/CFB totals vs openers)
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
measuring:  (2026-10-28 15:30Z, ADDENDUM in both loggers' docstrings, deployed before any data; helpers
            scripts/kalshi_tight.py; state data/open_totals_tight.json, data/open_lines_{nhl,nfl}_tight.json)
            (a) per Kalshi event: open_time, our first sighting, first two-sided anchor quote (anchor = strike/team
            market with YES mid nearest 50c); (b) FIRST TIGHT quote (anchor spread <= 4c): time, bid/ask, orderbook
            depth at best (contracts + $); (c) at the main rule's decision tick, our side's depth within 1c of best
            (kal_depth_* on pick rows); (d) paper `pick_tight`: first tick >= uncensored first tight with a model
            number, >= 30 min to start and anchor still tight; SAME gap thresholds with the Kalshi contract replacing
            the Pinnacle open (NBA |model-S|>=3; NHL ML/total .03/.04 vs Kalshi mid; NFL spread/total 2.0/2.5 pts);
            taker at ask + exact fee (100-ct order, rounded up); settled as the binary contract; CLV vs last
            pre-start Pinnacle (points NBA/NFL, prob NHL). Censored = already two-sided/tight at our first sighting
            of an event listed > 25/35 min earlier -> no pick_tight. All events live at deploy are censored.
            Reports: `nba_open_totals.py report`, `open_lines_logger.py report --league nhl|nfl` (Q004 section).
decision:   after >= 100 settled pick_tight (pooled across NBA/NHL/NFL, also shown per sport): mean CLV > 0 AND
            ROI at ask+fee > 0 -> write a sealed ask; otherwise no ask. Depth median/P10 $ at best sizes any ask.

## Q005 [done] Small / long-tail market calibration by volume quintile (Polymarket US + Kalshi)
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

## Q006 [running] Seasonal / physical data markets: monthly rain/snow unreachable-NO (L39)
question:   Does the frozen W1m rule (resting NO on monthly rain brackets the season total cannot reach) earn forward?
mechanism:  Thresholds already unreachable given observed month-to-date totals and forecasts still trade a few cents of YES.
baseline:   dev W1m +3.6%/146 bets, 1 loss; market-null p .016, one-per-city-month p .12 (gate: DEAD on dev alone).
change:     none -- forward paper of the frozen rule (scripts/kalshi_wx_monthly_logger.py, launchd hourly).
metric:     forward ROI per $ and loss rate vs break-even (~4%).
success:    set by /prereg (proposal: one sealed shot ~Feb 2027 on rain Jul26-Jan27 + snow).
kill:       any 2 losses in the first 100 forward picks.
power:      mu 1%, sigma 0.15 -> 1,392 picks; at ~20-25/month that is years -> capacity (~$7.5/month) caps the value.
budget:     0 new trials; logger already running. Low priority: cannot move the doubling target.

## Q007 [done] ForecastEx breadth: calibration of all ForecastEx contract families (public daily files) vs Kalshi same events
Mechanism: E5 shows ForecastEx weather priced ~9c cheaper than Kalshi for the same rejected brackets; other ForecastEx families (economic data, climate, politics) may carry the same thin-venue mispricing. Baseline: Kalshi price for the same event / realised outcome. Budget: 1 agent, dev split only.
Result (2026-10-28, EXPLORATORY, notes/q007_forecastex_breadth.md): FLB on every family (slope 1.05-1.12, fav +1.5-3.5% at
last trade ~= Kalshi's at-mid FLB). Econ same-event vs Kalshi (751 pairs / 92 releases): no level gap, but FX is a compressed copy
(favourites -1.4..-2.2c, longshots +2.4c) and less accurate (Brier 0.073 vs 0.066). One strong cell -> Q008. Do-not-pursue:
crypto/fin dailies (stale last-trade), sports (delisted Mar 2026), elections, climate, symmetric cross-venue arb.

## Q008 [running] ForecastEx econ: buy the Kalshi-favoured side when ForecastEx prices it >= 2c cheaper (forward paper)
question:   Is the FX discount on Kalshi-favoured econ outcomes (CPI/core CPI/MoM CPI/payrolls/U3/claims/Fed) executable at the FX ask?
mechanism:  E5 generalised: thin IBKR-retail venue = compressed copy of Kalshi; plus the Kalshi FLB at >= 0.85.
baseline:   dev (exploratory, last trade): Kalshi mid >= .85, FX side <= Kalshi mid - .02 -> 64 bets / 57 events, 0 losses,
            FX cost .868, ROI +16% [13.5, 18.9]; same bets at Kalshi ask +4.7%. ~40-50 signals/yr, median FX size 46 pairs.
change:     forward paper on live quotes: 15:00 CT snapshot of Kalshi book + FX ask (forecastex_client public quotes),
            frozen strike map (FF = Kalshi strike + 0.125), 1-30 d to release; signal Kalshi mid >= .85 and FX ask <= mid - .02,
            >= 10 at ask; 1 unit at FX ask + $0.01, first signal per contract. Also log FX bid/ask on all >= .80 contracts (H2).
success:    >= 100 bets / 40 events, one-sided p < .05, ROI > 0, Wilson-95 upper loss rate < 1 - mean cost.
kill:       >= 3 losses in first 30 bets, OR < 25% of Kalshi-favoured days have a qualifying FX ask (dev gap = last-trade artifact).
power:      ~2 years at 40-50 signals/yr; capacity ~$25-50/signal -> value is small; only worth it bolted onto FX1-fwd logger.
budget:     0 new trials until scored; logger extension only.
status:     FORWARD PAPER RUNNING (2026-10-28). Pick type `q008` in scripts/forecastex_paper.py (dated ADDENDUM in its
            docstring, pre-registered before the first tick), deployed in launchd com.jonathanwang.quant-forecastex-paper;
            first snapshot 2026-10-29 15:00 CT, then daily. Logs ~/.claude/quant-kalshi/data/forecastex_q008{,_obs}.jsonl;
            `forecastex_paper.py report` (section Q008) / `q008-validate`. Deviation: the public FX feed has no size, so
            the ">= 10 at ask" filter is dropped (size logged null). Judge at 60 settled picks: KILL if ROI at FX ask
            <= 0 or qualifying-ask rate < 25%; early KILL on >= 3 losses in the first 30; pass bar unchanged. Live
            check 10-28: 0 signals; CPI/core CPI/Fed contracts are > 30 d out; MMCPI 3/10, U3 7/9 FX strikes match Kalshi.

## Q009 [proposed] Options / volatility edges (user 10-29: "options... volatility increases"; capacity-rich venue)
Mechanism candidates: implied vs realised volatility (variance risk premium is documented to favour SELLING vol on average;
buying vol needs a timing/selection edge), earnings straddles entered ~5d before and exited pre-print (L11; needs option
bid/ask history), event-vol around macro releases. Baseline: option mid/ask vs realised move, after spread and fees.
Constraints: agentic Robinhood account only (main/Roth read-only); defined-risk structures; dev/sealed split by date; prereg
+ EDGE_MATH.md table incl. risk of ruin. Data: Robinhood MCP option chains/historicals (1m bars ~3-7 wks), public sources.

## Q010 [done 10-29: notes/contract_census_2026-10-29.md] Full contract census across venues (user: "look at all the offered contracts")
Enumerate EVERY open contract family on Kalshi (all 8 shards' series), Polymarket US, ForecastEx, Robinhood-routed venues;
tag each with an external forecast/reference source (if any), liquidity, fee, settlement source; rank families where a
public, better-calibrated reference exists (the weather pattern) -> each top family becomes a prereg question.

## Q011 [proposed] MLB/sports lineup-substitution props (user: "backup hitter")
When a regular is rested/injured and a backup starts, are player/team props or game lines slow to adjust (opener vs
lineup release time)? Test vs opener and vs close, by sport (MLB lineups ~3-4 h pre-game; NBA rest news), using
SITUATIONAL_FACTORS.md rows on availability.

## Q012-Q016 [proposed, from Q010 census 10-29] -- stubs (question/mechanism/baseline/power) in notes/contract_census_2026-10-29.md
Q012 Kalshi hourly crypto above/below (KXBTCD/KXETHD) vs Deribit-implied probability (largest non-sports capacity; HFT-heavy; L19 fav-maker dead there).
Q013 Rotten Tomatoes / Metacritic threshold markets vs the running score (forward logger first: too few settled events).
Q014 Truth Social post-count markets vs the running count (forward logger first).
Q015 RCP / VoteHub approval-average markets vs the recomputed average from listed polls.
Q016 AAA weekly/monthly gas vs RBOB pass-through (first check far-level quotes exist; daily N6 placed 0 bets).

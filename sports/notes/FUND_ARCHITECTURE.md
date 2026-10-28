# Research fund: agent architecture blueprint (2026-10-27)

User (PM) brief: "an agentic research architecture, multiple agents in tandem finding edges, evaluating each time, writing
statistical analyses and checking each other ... a research hedge fund that finds inefficiencies (perhaps smaller
polymarkets) and reports to me; I'm the portfolio manager; it should pay for itself."
Inputs: notes/prior_research_2026-10-22.md, the 10-27 research reports (inefficiencies, venues, agent architectures,
economics: notes/fund_economics.md), our ledger (~330 trials) and sealed record.

## 1. What the evidence says (drives every design choice)
1. **The surviving edge in prediction markets is structural, not predictive.** Takers overpay for longshots / YES;
   patient MAKERS on favourites and NO collect it (Bürgi-Deng-Whelan on Kalshi; Becker 72M trades; Polymarket wallet
   studies: top 1% take 76.5% of profits, mostly limit-order posters). Largest in thin retail categories, shrinking
   over time. Our only sealed passes are exactly this: weather consensus-reject (+1.45%) and its maker version (+2.22%).
2. **Predictive models beat stale OPENING prices, never the close** (NBA/NFL/NHL/CFB all lead openers p<1/4000; every
   close test fails; in-game Kalshi beats our models). So model edges are TIMING edges -> need early, low-fee execution.
3. **Arbitrage and news-speed edges are bot-owned** (sub-second repricing). Not for us.
4. **Published LLM-agent trading systems have no pre-registered live net-of-cost results**; gains shrink out of sample;
   LLM weights leak the test period. The protection is accounting (trial ledger, sealed tests, forward paper), not
   agent cleverness. Our real bugs were in code and joins (epoch units, row misalignment, coefficient-file mismatch),
   so verification must target data and code.
5. **Economics:** today's edges are capacity-capped (weather maker tops out ~$144/month; size above ~$75/pick
   loses to adverse selection). A $200/month system breaks even only at ~$10k+ bankroll and only if a NEW edge with
   capacity is found. New Kalshi Liquidity Incentive Program (amended 2026-10-28, up to $10k/business day) is the
   largest unmeasured upside because it pays makers on top of the trading edge.

## 2. Roles (agents) and the gates between them
| # | Role | Implemented by | Output | Cannot |
|---|---|---|---|---|
| 1 | **Scout** -- hypothesis generation | `researcher` agent + scheduled scanners (calibration-by-stratum, new series listings, LIP market list, forward-logger anomalies) | ranked hypotheses with mechanism + prior evidence | run tests |
| 2 | **Designer** -- pre-registration | `experiment-designer` agent / `prereg` skill | QUESTIONS.md entry: baseline, ONE change, metric, filters, dev/sealed split, power, kill rule, trial budget | see outcomes |
| 3 | **Builder** -- data + frozen scorer | general-purpose agent | script with docstring pre-reg, hashed before any market join | edit scorer after hash |
| 4 | **Auditor** -- data/leakage check | `data-auditor` agent | row-level audit: joins, timestamps/units, lookahead, duplicates, sealed-file access | read results |
| 5 | **Critic + Replicator** | `critic` + `replicator` agents; replicator ideally on a DIFFERENT model family (OpenAI) to decorrelate errors | independent recompute of the headline number from raw data; attack list | share code with builder |
| 6 | **Statistician gate** | deterministic `fund_stats.py` (new) | ledger-wide FDR (BH) + Bonferroni tier, power, deflated metric, family budget, both-halves; verdict DEAD / LEAD / PAPER / SEALED-ASK | override |
| 7 | **Forward paper** | launchd loggers (pattern: nba_open_totals, open_lines, L39) | live picks, CLV, kill rule | trade |
| 8 | **PM gate** | the user | approves sealed tests, real-money size, budgets | -- |
| 9 | **Executor + risk** | deterministic code (kalshi_exec.py pattern): caps, kill switch, per-shard balance check, post-only, loss stops | orders | be driven by an LLM free-form |
| 10 | **Reporter** | scheduled session | daily PM brief: P&L, picks, verdicts, asks | -- |

Pipeline per hypothesis: Scout -> Designer -> Builder (hash) -> Auditor -> run -> Critic+Replicator -> Statistician ->
finding recorded (finding skill) -> if LEAD: forward paper -> if forward CLV/ROI holds: SEALED-ASK to PM -> if pass:
PM sizes -> Executor. Every arrow is a file in the repo, so any agent can resume.

## 3. Orchestration
- **Registry:** `hub` project `fund` over this worktree (STATE.md, notes/, QUESTIONS.md, findings/), one-agent lease.
- **Cycle:** nightly `hub launch fund "<cycle task>"` (headless) runs: pick top [proposed] question within budget ->
  design -> build -> audit -> run -> critic/replicator -> stats -> finding. Parallel fan-out (`fanout` skill) only for
  independent questions; never two agents on one question.
- **Budgets:** per cycle (tokens/$), per hypothesis family (max trials, fixing the Bonferroni bar BEFORE search),
  per month (sealed tests). Every agent loop = a ledger trial.
- **Kill switches:** forward edge auto-paused when live results fall outside the sealed CI; executor KILL file.
- **Two-vendor use:** OpenAI agents as independent replicators/critics and as a second Scout -- not as a second
  executor.

## 4. Hypothesis portfolio (where to look first, ranked by evidence x capacity)
1. **Maker everything we already trade** (E4 passed sealed; needs KALSHI_EXEC_STYLE=maker) + per-pick cap toward ~$75.
2. **Kalshi Liquidity Incentive Program**: measure reward $/day per market vs resting size and adverse selection on
   markets where we already hold an edge (weather brackets) -> paper-quote, then tiny live.
3. **Maker NO/favourite quoting in thin retail categories** (mentions, entertainment, single-name) -- literature
   effect ~2c/contract for makers; our taker version (M3) died, maker version untested.
4. **Model-timing edges executed early as maker** (NBA/NHL/CFB/NFL totals vs openers) -- forward loggers measure
   Kalshi first-listing staleness + depth; NFL game prices were stale at listing (+3.3c CLV) but maker fills were
   adversely selected -> test take-at-first-tight-quote.
5. **Small/long-tail market calibration** (Polymarket US + Kalshi by volume quintile) -- scan pending.
6. **Seasonal/physical data markets** (monthly rain/snow L39 -- small capacity).
Do-not-redo: the DEAD list in notes/EDGES_AND_AGENTS.md section 5 is the Scout's first filter.

## 5. Statistics standard (the part that makes it "foolproof")
- Evaluate vs the market (de-vigged / closing / priced probability), never vs 50%.
- Cluster by event; contract-, trade- and event-weighted results can flip sign.
- Ledger-wide accounting: Bonferroni tier for "real", BH-FDR for "worth paper-testing"; family budgets set up front.
- Power before running: n = ((z_a + z_b) sigma / mu)^2; refuse tests that cannot detect the target edge.
- Sealed lockbox once, PM approval, never re-scored; forward paper before real money.
- For selection over a family of variants: SPA / PBO.
- LLM-judgement inputs only tested after the model's knowledge cutoff.

## 6. Honest economics and milestones
- No subscription spend is justified by trading returns yet (notes/fund_economics.md). Justify spend by research
  throughput; scale CAPITAL only after forward evidence.
- M1 (2 weeks): fund_stats.py gate; hub project; nightly cycle; daily brief; LIP measurement; maker on.
- M2 (1-2 months): 3-month maker weather live record; LIP paper P&L; first thin-market maker paper test; L32/NHL
  forward CLV at ~100 picks each.
- M3: only if a new edge with capacity >= $500/month passes sealed + forward -> discuss capital.

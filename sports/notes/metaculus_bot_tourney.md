# metaculus_bot_tourney -- feasibility of entering a forecasting-bot tournament with cash prizes

Agent: metaculus_bot_tourney (research only; no sign-ups, no credentials, no git). Started 2026-11-24.

## Question
Is it worth entering Metaculus FutureEval / AI Benchmark (bot) tournament (next season) or a comparable cash
forecasting tournament open to a US (California) individual? Expected $, hours/week, earliest submit date, plan.

## Pre-registered decision rule (written before reading any leaderboard/prize numbers)
- GO if: (a) a US/CA individual is eligible with a hobbyist/self-built bot, (b) expected prize for a
  "template + good research + calibration" entry is >= $25/hour of our effort over the season (ballpark from
  past leaderboard: where a template-class bot placed x prize at that rank), OR there is a cheap participation-
  style payout (e.g. per-question or many paid places) that makes EV >= $500/season at <= 5 h/week after build.
- NO-GO if: eligibility excludes us, or past seasons show template-plus bots finishing outside paid places with
  prizes concentrated in top ~5 that went to heavily engineered multi-model ensembles, giving EV < $10/hour.
- WATCH (not go) if the rule depends on numbers I cannot verify from official pages.

## Prior work check
- ~/projects/compete/scouting/SCOUT_2026-11-24.md line ~213 lists "Metaculus FutureEval Spring 2027" on the WATCH
  LIST (~$50k/season, UV, "likely early January"; spring 2026: 173 bots, 7 of top 10 hobbyists, best house bot 18th,
  from LessWrong posts). No comps/<slug> workspace exists -> not fully tracked; proceeding with the deep check.
- ~/projects/compete/scouting/raw_2026-11-24/E/metaculus_tournaments.html is a Cloudflare challenge page (60 chars), no content.
- research-rag: no prior quant/compete chunks on Metaculus (only transformers benchmark code hits).

## Plan
1. Official pages: metaculus.com/aib, tournament pages, notebooks/news posts on results; GitHub template
   (Metaculus/metac-bot-template, forecasting-tools). 2. Past results write-ups (Q3/Q4 2025, spring/fall 2026).
3. Comparable: ForecastBench, Good Judgment Open, Metaculus cups, Kalshi/Polymarket bot contests.
4. Fit with fund nowcast assets (FUND_BOOK sec 3, IDEA_PATTERNS). 5. EV, hours, plan.

## Log
- 2026-11-24: metaculus.com pages return 403 to WebFetch (Cloudflare); the Metaculus API now requires a token
  (`curl https://www.metaculus.com/api/posts/?limit=1` -> 403 "only available to authenticated users",
  results/metaculus_bot_tourney/raw/mc_api_test.json). EA Forum returns 429/Vercel checkpoint. Used instead: LessWrong
  cross-posts (same Metaculus authors), the open-source Metaculus site repo (github.com/Metaculus/metaculus, front_end
  source of the FutureEval + scores-FAQ + tournament-rules pages), the official template repo README, and public bot
  repos that quote the resources notebook 38928 / Fall announcement 45615 read through the API with their own tokens.
- Prize formula (official, Metaculus repo front_end/src/app/(main)/help/scores-faq/page.tsx, saved raw/_main__help_scores-faq_page.tsx):
  rank = sum of (weighted) peer scores; prize share proportional to (sum)^2 if sum > 0; no prize below $50 for
  tournaments ending after 2025-06-01 (money redistributed upward).
- Past prize lists (secondary: github.com/Bowen1314/metaculus-bot RESEARCH.md, read from the API with a bot token,
  saved raw/bowen_research.md). Cross-check: lists sum to $49,999/$49,999/$50,000; Spring-2026 top-10 prizes match
  prize_1 x (S_i/S_1)^2 using the OFFICIAL LessWrong top-10 sums within 0-9% (question weights) -> trust as consistent.
- Question-mix sample: the public bot repo github.com/m-ad/claude_forecasting_auto commits one folder per question it
  forecast. Batch-read 187 meta.yaml files over the GitHub GraphQL API (scripts/fetch_mad_meta_gql.py ->
  raw/mad_meta.jsonl). All 180 tournament rows are MiniBench, covering 3 complete rounds (60 each: rounds starting
  2026-10-19, 11-02, 11-16). The remaining 7 rows are bot-testing-area. No seasonal-tournament questions are in the
  sample. I first tried a keyword classifier (scripts/classify_questions.py) and dropped it as unreliable. Then I read
  all 180 titles and classified them by hand (scripts/manual_classes.py -> manual_classes_out.txt).

## Answer (2026-11-24)

### 1. Dates, eligibility, prizes, open-source
- **Seasons**: three per year, about 4 months each, starting in January, May and September. Each has 300-400
  questions and a ~$50k pool ($58k counting MiniBench). Source: Metaculus repo
  `front_end/src/app/(futureeval)/futureeval/components/futureeval-participate-tab.tsx` and `futureeval-tournaments.tsx`
  (raw/). Spring 2026 started 2026-01-05 (LessWrong dvGYXZfiqCgcck9im) and Summer 2026 on 2026-05-18 (LessWrong
  E7wxQnBvYxgB3rTTk). Fall 2026 opened questions on 2026-09-28, and its forecasting ends **2027-01-06** (Bowen
  RESEARCH.md, API read; quantchallenges lists the same date). **Spring 2027 has NOT been announced** as of 11-24
  (searched); by the pattern it should start around the first week of January 2027 [inference].
- **Join any time**: late entrants start mid-leaderboard with 0 points (Summer and Spring announcements, LessWrong).
  There is no registration deadline. You need a bot account and token plus the 3-question participation form
  (template README, raw/template_README.md).
- **Eligibility**: Metaculus competition rules (repo `front_end/src/app/(main)/tournament-rules/page.tsx`, last
  modified 2025-06-13): the competition is worldwide, ages 13+, and excludes only US-sanctioned countries. Winners
  provide ID/eligibility proof and **a W-9 if US resident**. Taxes and fees are paid by the recipient. The bot rules
  add 18+ and one prize-eligible bot per person (Bowen and m-ad docs quoting resources notebook 38928). California
  is not excluded. Payment is by bank transfer through Ramp (Bowen).
- **Commercial exclusion (Fall 2026)**: a bot is commercial if it is "built or operated on behalf of a for-profit
  entity that has three or more founders/employees/contractors", and that "includes any bots built using company
  resources". Commercial bots get no prize unless they fully open-source by season end, and get no LLM credits
  (quoted in gernovanniekerk/metaculus-bot SETUP.md and Bowen1314 RESEARCH.md from the Fall announcement, post
  45615). **A solo personal project is hobbyist** and does NOT have to be open-sourced. Prize winners must still
  submit their code OR an architecture description, accept an inspection, and complete the season survey. Whether
  the exclusion continues into Spring 2027 is unknown.
- **Prize formula** (official scores FAQ): prize share is proportional to max(sum of peer scores, 0)^2. Prizes under
  $50 are dropped and redistributed upward. Metaculus's own metac-* bots are excluded from prizes but count as peers.
- **Past payouts** (Bowen API read; cross-checked above):

  | Season | Eligible bots | Paid places | Top prize | Median paid | Smallest |
  |---|---|---|---|---|---|
  | Fall 2025 | 71 | 31 | $6,859 | $972 | $63 |
  | Spring 2026 | 102 | 30 | $5,097 | $1,502 | $54 |
  | Summer 2026 | 192 | 41 | $3,392 | $854 | $55 |

  - Payouts are broad: 30-41 places are paid, and among bots with >=90% coverage, 75-88% were paid.
- **MiniBench**: ~$1k per 2-week round with ~60 questions. Over 6 finalized rounds in Jun-Sep 2026, 8-13 bots were
  paid per round ($51-196). Bots with >=90% coverage took a mean of $18-32 per round (Bowen).

### 2. Mechanics
- **How questions arrive**: the bot polls the API (template cron every 20 min). Each question is open for **3 h**
  (raised from 1.5 h) and launches at a random time, usually with <=5 open at once.
- **Cadence**: seasonal questions launch weekly, one every 2 h (Spring 2026 analysis, LessWrong wZBbDqzfBjYG58CxK),
  which is about 20-25 per week. MiniBench opens most of its 60 questions in the first days of a round.
- **Scoring**: only the last forecast before close counts (spot peer score).
- **Forecast format**: Spring 2026 was binary 56%, numeric 31%, multiple choice 13%. Numerics are submitted as
  201-point CDFs with a per-bin cap.
- **What the bot can see**: a bot token cannot see the community prediction (m-ad tournament-protocol.md).
- **No human in the loop**: you may not preview a forecast on an open question and then tune or re-run.
- **Comments**: every forecast needs a private comment.
- **Credits**: the template README links a form for free OpenRouter LLM credits donated by OpenAI, Anthropic and
  Google. The amount is unpublished; a search snippet of the Fall announcement says ~$100, incremental [UV].
  AskNews gives 1k calls/month and 4k per tournament free, and must be renewed each season (Daatan issue 616; Bowen).
- **Cost**: Fall-2025 top-15 winners spent ~$1.40/question and the bottom half ~$0.50 (LessWrong a82q6yd8zKpYk56cF).
  The open-source nostreambot spends $2.60/question. Cheap models carry the official template well: in Summer 2026,
  gemini-3-5-flash averaged 8.3/question and gpt-5-5-instant 7.8, against 9.8 for the best model (Bowen).

### 3. What wins
- **Frontier models and reasoning effort**: high-reasoning variants beat standard ones (p .004).
- **Ensembling**: 86% of Fall-2025 winners aggregated multiple runs (median of 3-7 runs across model families).
- **More research sources**: winners averaged 1.75 sources vs 1.00 for non-winners (r .42).
- **Clipping**: clipping to about [0.03, 0.97] was the strongest within-winner differentiator (r +.48). Extremizing
  hurt (r -.30).
- **Lookups**: base-rate and similar-question lookups helped.
- **Simplicity**: "the template is better than I expected" (all of the above: LessWrong a82q6yd8zKpYk56cF; Bowen).
- **Top bots**:
  - Spring 2026 winner GreeneiBot2 summed 5,442 (avg 18.9/question, 288 questions) using AskNews, Perplexity and
    GPT-5 search in parallel, aggregated over multiple runs.
  - 7 of the top 10 were hobbyists. The best metac house bot placed 18th of 173 (LessWrong wZBbDqzfBjYG58CxK).
  - nostreambot (MIT, open source; median of 3 models; AskNews, OpenAI and Gemini search, FRED, yfinance,
    prediction markets) placed 15th of 277 in Summer 2026 with ~3,240 points over 256 questions; the leader had
    ~4,820 (No-Stream/metaculus-bot README).
- **Stock template**: the template bot with the best model would have earned $1,450 (Fall 25, 16th), $1,269
  (Spring 26, 18th) and $842 (Summer 26, 24th of 284) (Bowen "would-be prize").
  - The template's built-in default is gpt-4o; old models score -13 to -25 per question, so running it unmodified
    LOSES.
- **Dilution**: the $ value of a given score falls ~24% per season as the field grows. At full coverage, avg 10 was
  worth $1,679, then $1,287, then $979.

### 4. Fit with the fund's assets
Sample: 3 MiniBench rounds, n=180 (manual_classes_out.txt).

| Class | n | Share | m-ad bot mean peer score (descriptive, small n) |
|---|---|---|---|
| A: US official series the fund already nowcasts (NWS/NOAA weather, precip, tides, buoys; AAA gas; BEA/BLS/EIA releases; Hormuz AIS) | 13 | 7% | 16.7 (n 7) |
| B: market or FX closes (spot plus implied-vol random walk) | 24 | 13% | 2.4 (n 11) |
| C: other official series we don't run (foreign CPI/GDP, disease surveillance, tallies, company results) | 73 | 41% | 7.1 (n 43) |
| D: news-judgment events | 70 | 39% | 10.5 (n 36) |

- **Resolvable from official or public data (A+B+C)**: 110/180 = 61%. **Covered by our pipelines or price model
  (A+B)**: 37/180 = 21%.
- **B looks like the place to gain**: that LLM bot scored worst on B (2.4, n 11), where a calibrated lognormal from
  implied vol is cheap to add. This is ONE bot with small n, so treat it as a hypothesis, not a finding.
- **Seasonal mix not sampled**: seasonal questions are custom plus imported from the main site, so I expect a lower
  data share there [inference, unverified]. The fund's nowcasts are about market-quality: N2, N6 and G2 died
  because Kalshi already prices them (FUND_BOOK 3d). Against LLM peers, though, market quality is the edge.

### 5. Comparable cash tournaments open to a US individual
- **Metaculus Market Pulse 26Q4**: $7.5k; last day for prize participation is 2026-12-21; bots eligible; numeric
  group questions on macro, earnings and asset prices; forecasts must be updated continuously against humans
  (quantchallenges listing [secondary]; participate tab "~$7k bot-eligible"). Best topical fit. The 27Q1 edition
  is expected in Jan [inference].
- **Metaculus Cup Fall 2026**: $5k, ends 2027-01-01. Bots are NOT prize-eligible; a human could enter.
- **Bridgewater x Metaculus**: 2026 ran Jan 12 to mid-Mar with $30k; bots ineligible, but humans may use AI tools.
  A 2027 edition has not been announced.
- **No cash found**: ForecastBench (benchmark only), Good Judgment Open (badges and recognition), and Kalshi or
  Polymarket (no official bot contest; BallotBench is a third-party demo).

## Decision vs the pre-registered rule
- **(a) Eligibility: YES.** A personal project is hobbyist; W-9 required.
- **(b) EV per season** (scripts/ev_model.py -> ev_model_out.txt):
  - The FutureEval seasonal EV is $337-597, using k = $5.7-9.8 per avg^2 for Spring 2027. Scenario weights: 20%
    bad, 40% template-level (avg 7.5), 30% good (avg 10), 10% nostreambot-level (avg 12.5).
  - MiniBench adds $144-256 (8 rounds x $18-32).
  - Gross total is ~$480-850 per season; honest range $0-2,000.
  - API cost is ~$0-200 with free credits and cheap or flash models. It is $660-1,160 if we spend at the top-15
    rate of $1.40/question without credits.
- **Hours**: build 20-30 h, then ~1 h/week, so ~41 h in season 1, which is ~$12-21/h gross. Later seasons run
  ~16 h for ~$30-50/h if dilution stalls.
- **Against the rule**: the GO clause "EV >= $500 at <= 5 h/week after build" is met gross, but only marginally
  net. The NO-GO clause "< $10/h" is not triggered. The key prize numbers come from a third party's API read, and I
  cross-checked them against the official LessWrong top-10 and the official formula.
- **Verdict: GO-LITE.** Run a capped pilot (hard $50/month API cap; max loss ~$200/season plus hours), MiniBench
  first. It is not a priority over live fund edges. Re-judge after 2 MiniBench rounds (below).

## Plan if go (5 steps)
1. **USER (human gate, $0)**:
   - Create a Metaculus account plus bot account and token.
   - Fill in the participation form and tick LLM credits.
   - Email AskNews for the free tier.
   - Confirm it is a personal project with no employer resources.
2. **Build (agent, ~20-30 h)**:
   - Fork nostreambot (MIT, 15th/277) or the template.
   - Pin a cross-lab median of 3-5 current models (never the gpt-4o default).
   - Clip binaries to [0.03, 0.97]; floor multiple-choice options at 0.02; mix numeric CDFs with wide tails.
   - Add a class-A/B adapter: NWS/NOAA, AAA, BEA/BLS/EIA consensus, PortWatch, and a lognormal from implied vol for
     price closes.
   - Add a ledger, a hard spend cap, and missed-window alerts.
3. **Pre-register, then shake down**: run on bot-testing-area, then MiniBench unattended from GitHub Actions.
   - Success = >=95% coverage AND avg peer >= +5 over 2 rounds (~120 questions).
   - Kill = coverage <90% OR avg <0.
4. **Optional**: join Fall 2026 now for its last ~6 weeks (to 2027-01-06) as a live test. The prize is near $0 at
   ~35% coverage.
5. **Spring 2027 (around Jan 5) full season**, plus Market Pulse 27Q1 once a continuous-update loop exists.
   - Post-mortems use resolved questions only; no tuning on open ones.
   - Complete the survey; claim prizes within 30 days.
- **Earliest submit**: the next MiniBench round is expected **Mon 2026-11-30** (biweekly cadence inferred from
  rounds 10-19, 11-02, 11-16). 12-14 is realistic given account setup and build time. Fall 2026 is joinable today.

## Known pitfalls (from others' write-ups)
- **Template default**: the template's default model is gpt-4o, and old or small models score -13 to -25 per
  question (Bowen).
- **Resolved-question confusion**: bots that treat an unresolved question as resolved produce 99% forecasts; guard
  in the prompt and clip.
- **Parser failures** were one maker's biggest error source.
- **Runaway spend**: one maker burned a day's budget in 4 questions.
- **Overfitting tweaks**: a bot fell from 20th to 63rd after adding logit aggregation and extremizing
  (Bowen, quoting Metaculus advice notebooks).
- **Free-model churn**: free OpenRouter models go 429 or are retired within days (aniyengar1 STAGE_A_RESULTS.md).
- **Data-use terms**: Metaculus API terms forbid using its data to "train, evaluate, or otherwise create or develop
  AI/ML models" without written permission (Metaculus repo docs/openapi.yml). Backtesting the bot on bulk-downloaded
  Metaculus questions may need permission; use bot-testing-area and our own resolved questions.
- **Prize logistics**: prizes are paid only after ALL bot-maker surveys for the season are in, and unclaimed prizes
  are forfeited after 30 days (Bowen, quoting rules).

## Learnings

### Metaculus bot prizes scale with the square of total peer score, so coverage is money
Official scores FAQ (Metaculus repo `front_end/src/app/(main)/help/scores-faq/page.tsx`): share is proportional to
max(sum of peer scores, 0)^2, and nothing under $50 is paid. A skipped question scores 0, so a 3-h window missed at
night costs quadratically. Among bots with >=90% coverage, 75-88% were paid; at 50-90% coverage the median was $0
(Bowen1314 RESEARCH.md). Reliability engineering comes before model tuning.

### The FutureEval payout is broad (30-41 paid places), unlike top-k competitions
Fall 25 / Spring 26 / Summer 26 paid 31 / 30 / 41 places, with medians of $972 / $1,502 / $854 (Bowen API read,
lists sum to exactly $50k). The stock template with a top model would have earned $842-1,450. The compete
rubric's "150+ team fields are worth ~$0" does not apply to square-share payouts.

### The value of a given forecasting skill falls ~24% per season as the bot field grows
At full coverage, avg 10/question was worth $1,679, then $1,287, then $979 (Fall 25 to Summer 26; the S^2 rule
reproduces the table within 3%, ev_model_out.txt). Eligible bots went 71 -> 102 -> 192. Project Spring 2027 at
roughly 0.6-1.0 of Summer 2026.

### MiniBench is about 20% our kind of question; LLM bots looked weakest on price-close questions
In 3 MiniBench rounds (n=180, manual_classes_out.txt), 7% were US official series we already nowcast and 13% were
market closes. 61% are resolvable from some official series. One public LLM bot averaged 2.4 peer on price closes vs
16.7 on our class-A questions (small n, descriptive only). The hypothesis to test is a lognormal from implied vol
for closes.

### Metaculus pages and API are closed to anonymous scripts; use the open-source site repo and LessWrong
metaculus.com gives 403 to fetchers, the API now needs a token, and EA Forum gives 429. The FutureEval page text,
the scores FAQ and the competition rules all live in github.com/Metaculus/metaculus front_end. Metaculus results
posts are cross-posted on LessWrong. Public bot repos log question metadata that the GitHub GraphQL API can batch-read.

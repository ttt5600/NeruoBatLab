# metaculus_bot_build -- builder notes (agent metaculus_bot_build, 2026-11-25)

Task: build a forecasting bot for the Metaculus AI bot tournaments (MiniBench two-week rounds from ~2026-11-30 or
12-14, then Spring 2027), ready the moment the user creates the accounts. Code: `sports/metaculus_bot/` (README there).
Hard rules kept: $0 spent, no sign-ups / logins / credentials, mock LLM + fixtures in every test, no launchd / cron
installed, no git push (local commit only, isolated worktree `agent-a8b711763d5390846`, branch
`worktree-agent-a8b711763d5390846`). Public read-only GETs only: Metaculus docs/template on GitHub, OpenRouter model
list, and one offline dry run that called the fund's own free data fetchers (AAA, FRED, Federal Register).

## Pre-registered go/kill rule (written before any real run; `sports/metaculus_bot/mbot/evaluate.py`)
Window = the first two MiniBench rounds run unattended, extended until >= 40 forecast questions have a score.
- KILL: coverage < 90% (once >= 40 eligible) OR mean spot peer < 0 OR any month's logged spend > $50 (bill > $55).
- GO (full Spring 2027, same cap): coverage >= 95% AND mean spot peer >= +5 AND n >= 40.
- HOLD otherwise: 2 more MiniBench rounds, same cap, no tuning on open questions; a second HOLD = KILL.
- Mean reported with a question-level bootstrap 95% CI; a CI spanning 0 is written up as "not distinguishable".
Merges the notes' rule (`notes/metaculus_bot_tourney.md` "Plan if go" step 3) and the brief's, taking the stricter
side of each. `python -m mbot evaluate` applies it to Metaculus' own `my_forecasts.score_data.spot_peer_score`.

## What was built
- Official REST path (`/api/posts/`, `/api/questions/forecast/`, `/api/comments/create/`), Token auth, honest UA,
  1 req/s pacing, GET retries, POST never retried. Type router: binary / multiple_choice / numeric / discrete / date,
  groups unpacked; conditionals and notebooks skipped.
- Research: fund fetchers imported read-only by path (`sports/scripts/source_shift_watch.py` aaa / parse_cli /
  openrouter / noaa, `e1x_tsa_logger.daily`), plus FRED csv, BLS, BEA (key), Federal Register, AskNews (key).
  Backtest mode skips tools that cannot cut data at `as_of`.
- N samples (default 5) in parallel, median; binary clip [0.02, 0.98]; MC floor 0.02 (scaled to 0.2/n); extremize off;
  CDF port of forecasting-tools NumericDistribution + validator for the API rules.
- Runner: one command, JSONL ledger, four dedupe layers, run lock, `--dry-run`, mock provider refuses live mode,
  $50/month hard cap with worst-case reservation before each call, budget fallback to a cheaper ensemble.
- Backtest: `backtest fetch` (project 32979, needs the Bot Benchmarking tier) + `backtest run` (as-of open time,
  official baseline formulas, rel-to-CP proxy, bootstrap CI, `--min-open-date` knowledge-cutoff guard).

## Numbers (each from a command run on 2026-11-25)
- Tests: `.venv/bin/python -m pytest` -> 69 passed, 4 skipped (parity tests); `.venv_official/bin/python -m pytest`
  (forecasting-tools 0.3.7) -> 72 passed, 1 skipped (SDK wire test). Every test passes in at least one venv.
- CDF parity vs forecasting-tools 0.3.7: max abs difference < 1e-9 on 4 cases (tests/test_cdf.py::test_parity_*).
- Offline dry run, 9 fixture questions x 3 mock samples: 9/9 forecasts OK, 1 skip (already forecast), 0 POSTs
  (`python -m mbot run --dry-run --models mock:m --questions-file tests/fixtures/open_posts.json`; ledger
  results/metaculus_bot_build/dryrun_ledger_2026-11-25.jsonl, 39 rows). Live tools in that run: AAA ok in 6.7 s (Regular $3.5866 as of
  2026-11-25), FRED DGS10 ok (latest 2026-11-23 = 4.92), Federal Register ok (23 EOs signed since 2026-07-28).
- Anthropic SDK 2.14.0 builds the request with `betas`, `fallbacks: "default"`, `output_config.effort`, `stream: true`
  (tests/test_llm.py::test_real_sdk_request_on_the_wire, httpx2 MockTransport, nothing sent).
- Cost per question: NOT measured. Estimate (README section 6): Opus 5.5 x5 $0.37-0.87, Sonnet 5.5 x5 $0.19-0.44,
  assuming 3.7k input and 3k-8k output tokens per sample. MiniBench ~130 questions/month (~60 per round, notes sec 1)
  -> Opus x5 is $48-113/month, so the $50 cap can bind; `mbot status` prints the real $/question after the first runs.

## Wrong turns and bugs found by looking at real rows
- AAA labels were shifted by one grade: the fund's header scrape returns 6 headers (a stray "Price" from another
  table) for 5 values per row, and the wrapper aligned from the right, so Regular $3.5866 was labelled Mid-Grade and
  E85 $2.6765 "Price". The stub test passed because it had the wrong shape; caught only by printing the live tool
  output. Fixed (align left, as the fund does) and the stub now copies the live shape.
- Listing params: the official no-framework template sends `forecast_type` comma-joined and `include_description`
  (singular); docs/openapi.yml documents arrays and `include_descriptions`. Now sends the documented arrays, both
  flag spellings, and `group_of_questions` (forecasting-tools does this to unpack groups; without it group posts
  would never be listed).
- docs/openapi.yml's example post `continuous_range` (213.6328 at index 1) does not follow its own documented
  location formula (213.5624); the test uses the formula as the oracle.
- `anthropic` 2.x depends on `httpx2`, not `httpx`; the first wire test silently skipped until the import was fixed.
- Volume vs cap: defaulting to MiniBench + Fall 2026 would have ~doubled the question count against a $50 cap; the
  default is MiniBench only, as the notes' pilot plan says.

## Left for the user (README section 1)
Metaculus bot account + token (https://www.metaculus.com/futureeval/participate/), participation + credits form
(https://forms.gle/aQdYMq9Pisrf1v7d8; credits arrive as an OpenRouter key), AskNews (https://my.asknews.app +
contact@asknews.app), an Anthropic key with a ~$55 console limit if not using credits, optional Data Needs Form for
the Bot Benchmarking tier, then `smoke --yes`, a dry run and a live run on bot-testing-area, then schedule.

## Unverified
Proxy-credit route (`metaculus:` provider); listing params against the live API; "one forecast per question" and
"no human in the loop" (secondary source, m-ad); Spring 2027 rules incl. any commercial exclusion; AskNews SDK
calls (copied from the template README, never run); OpenRouter `reasoning` param; whether `tournaments=32979` lists
the benchmark project. Template repo has no LICENSE, so none of its code is copied.

## Learnings

### Metaculus API data terms: only project 32979 is sanctioned for bot evaluation
docs/openapi.yml (Metaculus repo): every request needs a token (anonymous -> 403); the default Restricted tier
returns `resolution: null` and null question text on questions you never forecast; Metaculus' terms forbid using
its data to "train, evaluate, or otherwise create or develop AI/ML models" without written permission. The Bot
Benchmarking tier (Data Needs Form) opens ~250 resolved questions in project 32979 "intended for training and
evaluation". A backtest on bulk-downloaded questions is both blocked (null resolutions) and off-terms.

### Stub fixtures must copy the live shape, or tests bless a bug
The AAA wrapper passed its test while mislabelling every grade in production (6 scraped headers vs 5 values per
row; live output 2026-11-25 put Regular $3.5866 under "Mid-Grade"). Only printing one real tool output caught it.
Rule: build each stub from a captured live response, and print one real row before trusting a green test.

### The official template's own query params differ from the documented API
metac-bot-template main_with_no_framework.py sends `forecast_type` as a comma string and `include_description`;
docs/openapi.yml documents arrays and `include_descriptions`, and forecasting-tools adds `group_of_questions` to
unpack groups. Send the documented form (plus the template's spelling); otherwise group questions can be missed,
and a missed question scores 0 in a score-squared prize.

### Budget pacing protects coverage better than a hard stop
At an estimated $0.37-0.87/question (Opus 5.5 x5, unmeasured) and ~130 MiniBench questions/month, a $50 cap can
bind mid-month; a hard stop then zeroes every remaining question. The bot switches to MBOT_FALLBACK_MODELS when
< $10 is left and still enforces the cap per call. Measure the real $/question (`mbot status`) in the first week.

### anthropic SDK 2.x ships on httpx2
anthropic 2.14.0 imports `httpx2` (site-packages has httpx2 2.16.2, no httpx). Tests or proxies that build an
`httpx.Client` for it need `httpx2`; a pytest.importorskip("httpx") silently skips the test instead.

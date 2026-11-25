# mbot: Metaculus FutureEval / MiniBench forecasting bot

Built 2026-11-25 for the capped pilot in `sports/notes/metaculus_bot_tourney.md` (verdict GO-LITE: MiniBench first,
hard $50/month API cap). **Status: code and offline tests only. $0 spent, no account, no key, no paid call, no
submission.** Everything that needs a credential is behind an env var and has only been tested with stubs.

Why it is built this way, in one line each (sources in the notes, section 3):
- Prize share goes with the *square* of total peer score and a skipped question scores 0, so **coverage comes first**:
  an unattended poller, a never-double-submit ledger, and budget pacing that switches to cheaper models instead of
  stopping at the cap.
- 86% of Fall-2025 winners took the **median of 3-7 runs**: N independent samples, median.
- Clipping to about [0.03, 0.97] was the strongest within-winner differentiator (r +.48) and **extremizing hurt**
  (r -.30): binary clip [0.02, 0.98], extremize off (`MBOT_EXTREMIZE=1.0`).
- Parser failures were one maker's biggest error source: strict output formats, deterministic regex parsing, bad
  samples dropped and logged, CDFs validated against the API's rules before every POST.

## 1. What the user has to do (all free; nothing here was done by the agent)

1. **Metaculus account + bot account + token.** Log in at https://www.metaculus.com with your personal account, then
   Settings -> My Forecasting Bots -> Create a Bot (source: `sports/results/metaculus_bot_tourney/raw/gernov_setup.md`).
   The bot's API token is shown at https://www.metaculus.com/futureeval/participate/ (the official template's error
   message points there: metac-bot-template `bot_helpers.py`). Personal project only; no employer resources.
2. **Participation + free LLM credits form**: https://forms.gle/aQdYMq9Pisrf1v7d8 (official template README; tick the
   LLM-credits box). The template README says the credits come as an **OpenRouter key** (`OPENROUTER_API_KEY`); use
   it as `MBOT_MODELS=openrouter:anthropic/claude-opus-5.5` (model names as listed in `mbot/prices_snapshot.json`).
   The credit amount is unpublished (~$100 per a search snippet, unverified). Setup links for every key are also on
   the resources page https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/.
3. **AskNews free tier** (optional but recommended; winners used >= 2 research sources): sign up at
   https://my.asknews.app, email contact@asknews.app asking for the Metaculus bot allowance, then create a client id /
   secret at https://my.asknews.app/en/settings/api-credentials. It must be renewed each season.
4. **LLM key** (if not using Metaculus credits): an Anthropic API key from https://platform.claude.com. Set a
   workspace spend limit of about $55 there as a backstop to the bot's own $50 cap.
5. **Optional, for the backtest**: request the Bot Benchmarking API tier with the Metaculus Data Needs Form
   https://docs.google.com/forms/d/e/1FAIpQLSeJhtZzHl5qMvBjbXbatyaqoS4IU7RE0GGw_vlhs6I9syqn1g/viewform
   (Metaculus `docs/openapi.yml`, "Bot Benchmarking Access Tier").
6. Put the secrets in `sports/metaculus_bot/.env` (git-ignored; `chmod 600 .env`), e.g.
   ```
   METACULUS_TOKEN=...
   ANTHROPIC_API_KEY=...
   MBOT_MODELS=anthropic:claude-opus-5-5
   MBOT_FALLBACK_MODELS=anthropic:claude-haiku-5-5
   ASKNEWS_CLIENT_ID=...
   ASKNEWS_SECRET=...
   ```

## 2. Install

```bash
cd sports/metaculus_bot
uv venv .venv --python 3.12                      # forecasting-tools / pydantic do not support 3.14 yet
uv pip install --python .venv/bin/python -e ".[fund,llm,dev]"     # add ",news" once AskNews keys exist
.venv/bin/python -m pytest                       # 2026-11-25: 69 passed, 4 skipped (the CDF parity tests)
```
CDF parity with the official package: `uv venv .venv_official --python 3.12 && uv pip install --python
.venv_official/bin/python -e ".[official,fund,dev]" && .venv_official/bin/python -m pytest` (2026-11-25: 72 passed,
1 skipped = the SDK wire test, which needs the `llm` extra). Every test passes in at least one of the two venvs.

## 3. Commands (run from `sports/metaculus_bot`, with `.env` exported: `set -a; . ./.env; set +a`)

| Command | Cost | What it does |
|---|---|---|
| `.venv/bin/python -m mbot check-env` | $0 | which keys are present (never printed), model prices, state dir, spend so far |
| `.venv/bin/python -m mbot run --dry-run --models mock:m --questions-file tests/fixtures/open_posts.json` | $0 | fully offline: fixture questions, stub LLM, prints forecasts, submits nothing |
| `.venv/bin/python -m mbot run --dry-run --models mock:m --tournament bot-testing-area` | $0 | real question listing (token needed), stub LLM |
| `.venv/bin/python -m mbot smoke --yes` | cents | one tiny real call per configured model, logged against the cap |
| `.venv/bin/python -m mbot run --dry-run --tournament bot-testing-area --max-questions 3` | ~$1-3 | real LLM, real questions, **nothing submitted** |
| `.venv/bin/python -m mbot run --tournament bot-testing-area --max-questions 3` | ~$1-3 | first real submissions, on the sanctioned test tournament |
| `.venv/bin/python -m mbot run` | per question | the unattended pass on `MBOT_TOURNAMENTS` (default `minibench`) |
| `.venv/bin/python -m mbot status` | $0 | month-to-date spend vs cap, submissions, last-24h errors, **actual $/question** |
| `.venv/bin/python -m mbot evaluate` | $0 | applies the pre-registered go/kill rule (section 7) to Metaculus' own scores |

`--dry-run` never POSTs; the mock provider refuses to run without `--dry-run`. Bot rules (m-ad tournament-protocol,
quoting Metaculus notebook 38928): no human in the loop on open tournament questions (do not preview a forecast on
an open MiniBench question and then re-run or tweak), so use **bot-testing-area** for all testing.

## 4. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `METACULUS_TOKEN` | - | bot token; required for any API call (anonymous GETs return 403) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY` | - | provider keys, only the ones your `MBOT_MODELS` use |
| `MBOT_MODELS` | `anthropic:claude-opus-5-5` | comma list `provider:model`; samples rotate over it. Providers: `anthropic`, `openai`, `openrouter`, `metaculus` (proxy credits), `mock` |
| `MBOT_N_SAMPLES` | 5 | independent samples per question (median) |
| `MBOT_EFFORT` | high | `output_config.effort` for Claude (Opus 5.5 defaults to medium if unset) / OpenRouter `reasoning.effort` |
| `MBOT_MAX_OUTPUT_TOKENS` | 16000 | per sample; also sets the worst-case cost the cap reserves before each call |
| `MBOT_MONTHLY_CAP_USD` | 50 | hard cap per UTC calendar month, summed from the ledger |
| `MBOT_FALLBACK_MODELS` / `MBOT_FALLBACK_N_SAMPLES` / `MBOT_FALLBACK_BELOW_USD` | none / 3 / 10 | cheaper ensemble once less than $10 of the cap is left |
| `MBOT_TOURNAMENTS` | `minibench` | add `fall-futureeval-2026` (id 33121, ends 2027-01-06) or the Spring 2027 slug when announced |
| `MBOT_MIN_MINUTES_LEFT` | 8 | skip questions closing sooner than this |
| `MBOT_MAX_QUESTIONS_PER_RUN` | 25 | most urgent first |
| `MBOT_BINARY_CLIP_LO` / `_HI` | 0.02 / 0.98 | binary guard |
| `MBOT_MC_FLOOR` | 0.02 | multiple-choice floor (scaled down to 0.2/n for many options) |
| `MBOT_EXTREMIZE` | 1.0 | 1.0 = off; the evidence says leave it off |
| `MBOT_DATA_TOOLS` | 1 | 0 disables the free official-data tools |
| `ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET` | - | news research, used only when both are set |
| `BLS_API_KEY` / `BEA_API_KEY` | - | optional; BLS works keyless (25 queries/day), BEA needs a free key |
| `MBOT_STATE_DIR` | `~/.claude/quant-sports/data/metaculus_bot` | ledger `forecasts.jsonl` + `run.lock` |
| `MBOT_FUND_SCRIPTS_DIR` | `sports/scripts` | where the fund's read-only nowcast fetchers live |

Server-side fallbacks are **enabled** for direct-API Claude Fable 5.1 / Opus 5.5 / Opus 5 / Sonnet 5.5 requests
(beta `server-side-fallback-2026-07-01`, `fallbacks: "default"`): if a request is declined by a safety classifier,
the API re-runs it on Anthropic's chosen fallback model and bills that model's rates. The served model is logged per
call. Not sent through the Metaculus proxy. If the API or an older SDK rejects it, the bot drops it once and retries.

## 5. Pipeline

```
list open posts (per tournament) -> unpack groups -> filter: open, not in ledger, not my_forecasts, >= 8 min left
  -> research: free official-data tools that match the text (AAA gas, NWS forecast + CLI, TSA, OpenRouter rankings,
     NOAA global temp -- the fund's own fetchers in sports/scripts, imported read-only; FRED, BLS, BEA, Federal
     Register) + AskNews if keyed
  -> N samples in parallel, each preceded by a spend-cap check on its worst-case cost
  -> parse (regex; bad samples dropped) -> median -> guards (clip / MC floor / CDF repair + API-rule validation)
  -> re-read the post (still open? still unforecast?) -> POST forecast -> POST private comment (reasoning summary)
  -> ledger rows: run_start, skip, llm_call, llm_error, forecast (inputs, tool results, samples, final, cost),
     submitted, comment_posted, submit_error, run_end
```
Never double-submits: ledger `submitted` rows, Metaculus' `my_forecasts`, a re-read right before the POST, and an
exclusive run lock; a POST that fails ambiguously is not retried blindly (the post is re-read to see if it landed).
Numeric/discrete/date CDFs follow the official rules (n+1 points, steps >= 0.01/n, <= 0.2*200/n, closed bound = 0/1
exactly, open bound >= 0.001 / <= 0.999); the percentile -> CDF code is a port of forecasting-tools
`NumericDistribution` (MIT, `LICENSES/forecasting-tools-MIT.txt`) and matches it to 1e-9 on 4 cases.

## 6. Cost (ESTIMATES, not measurements -- `mbot status` prints the real $/question after the first runs)

Assumptions: ~3.7k input tokens per sample (prompt + up to 9,000 chars of research, at 3 chars/token), 3k-8k output
tokens per sample at effort high (thinking included; unmeasured). Prices: `mbot/llm.py` (Anthropic list prices
$/MTok in/out: Opus 5.5 4/20, Sonnet 5.5 2/10, Haiku 5.5 0.10/0.50).

| Ensemble | $/question (5 samples) | MiniBench month (~130 q: ~60 per 2-week round) |
|---|---|---|
| Opus 5.5 x5 | $0.37-0.87 | $48-113 -> may hit the $50 cap; the fallback keeps coverage |
| Sonnet 5.5 x5 | $0.19-0.44 | $24-57 |
| Haiku 5.5 x5 | $0.01-0.02 | $1-3 |

Context: Fall-2025 top-15 bots spent ~$1.40/question, the bottom half ~$0.50 (notes, LessWrong a82q6yd8zKpYk56cF).
Adding the seasonal tournament (~20-25 questions/week) roughly doubles the volume. The cap reserves the worst case
before each call (Opus 5.5, 16k output tokens: ~$0.33 per sample), so it never overshoots by more than one
timed-out call.

## 7. Pre-registered go/kill rule (written 2026-11-25, before any real run; code: `mbot/evaluate.py`)

Window: the first two MiniBench rounds run unattended (from the first live run in the ledger); judged when all their
questions have closed, extended round by round until >= 40 forecast questions have resolved with a score.

- **KILL** if coverage < 90% (once >= 40 questions are eligible), OR mean spot peer score < 0, OR any calendar
  month's logged spend > the $50 cap (or the provider bill > $55).
- **GO** (enter the full Spring 2027 season, same cap) if coverage >= 95% AND mean spot peer >= +5 AND n >= 40.
- **HOLD** otherwise (mean in [0, +5) or coverage in [90%, 95%)): MiniBench only for 2 more rounds, same cap, no
  tuning on open questions, then re-judge with this rule; a second consecutive HOLD is a KILL.
- The mean is reported with a question-level bootstrap 95% CI. If the CI spans 0 the result is written up as
  "not distinguishable from 0", whatever zone the point estimate lands in.

This keeps the stricter side of both rules on record: the notes' (success = >=95% coverage AND avg >= +5 over 2
rounds; kill = coverage < 90% OR avg < 0) and the build brief's (continue if mean >= 0 over >= 40 questions; kill if
< -2 or spend > cap). Run `python -m mbot evaluate` (pass `--previous HOLD` after a HOLD).

## 8. Scheduling (nothing is installed)

- **launchd** (Mac): `launchd/com.mbot.metaculus.plist.template` runs `python -m mbot run` every 20 min (the official
  template's GitHub Actions cadence). Fill in the two paths, copy to `~/Library/LaunchAgents/`, then
  `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mbot.metaculus.plist`. A sleeping laptop misses
  3-hour windows, and missed questions score 0.
- **Always-on alternative**: the official template runs from GitHub Actions on a 20-min cron with the token and keys
  as repository secrets. The same `python -m mbot run` works there, but the ledger must persist between runs
  (commit it back or use a cache); Metaculus' own `my_forecasts` check still blocks double submits if it does not.

## 9. Backtest (no paid calls until you run step 2)

```bash
# 1) fixtures: resolved questions from the Bot Benchmarking list (needs METACULUS_TOKEN with that tier)
.venv/bin/python -m mbot backtest fetch --project 32979 --limit 100 --out backtest_fixtures
# 2) score ~50 of them (costs ~50 x $/question; set MBOT_MONTHLY_CAP_USD for the run)
.venv/bin/python -m mbot backtest run --fixtures backtest_fixtures --limit 50 --min-open-date 2026-07-01 \
    --out backtest_results/scores.jsonl
```
- Each question is forecast **as of its open time**: AskNews is off and tools that cannot cut their data at that
  time (weather, AAA, OpenRouter) are skipped. FRED/BLS/BEA values are the latest vintage, so revisions are a small
  residual lookahead.
- Scores use the official formulas (Metaculus `scoring/score_math.py`): binary/MC baseline `100*ln(p*K)/ln(K)`,
  continuous `100*ln(pmf[bucket]/b)/2`; plus `rel_cp = 100*ln(p/p_cp)` when the fixture carries the Community
  Prediction. That is a proxy: the tournament's peer score compares against the other *bots*, which is not
  available offline.
- **Model-knowledge lookahead**: an LLM trained after a question resolved can know the answer. Pass
  `--min-open-date` = a date after every model's training cutoff, or the score measures memory, not skill.
- **Terms**: Metaculus' API terms forbid using its data to "train, evaluate, or otherwise create or develop AI/ML
  models" without written permission. Project 32979 is "intended for training and evaluation purposes" for
  accounts granted the Bot Benchmarking tier; do not bulk-score other projects. Under the default Restricted tier,
  `resolution` comes back null on questions you never forecast, so `fetch` would keep nothing.
- If `--project 32979` lists 0 posts, the `tournaments` filter may need the project's slug instead of its id
  (unverified; the official template passes numeric tournament ids the same way).

## 10. What is stubbed or unverified

Stubbed:
- **All LLM calls in tests** (MockLLM / fake clients / fake HTTP). No provider adapter has made a real call.
- **No data tool sets `resolves`**, so the wider [0.005, 0.995] binary clip never triggers. It needs a per-question
  threshold parser (e.g. "AAA regular above $3.00 on Dec 5" once Dec 5's print is published).
- **AskNews**: wrapper written against the SDK names in the template README (`AskNewsSDK`, `news.search_news`,
  strategies "latest news" / "news knowledge"); never called.
- **TSA**: the fund's series (CSV history + Kalshi first prints); tsa.gov returns 403 to scripts and is not scraped.

Unverified:
- The `metaculus:` provider (the older `llm-proxy.metaculus.com/proxy/{anthropic,openai/v1}` route with
  `Authorization: Token`, taken from forecasting-tools) is untested and may be retired; the current template hands
  out credits as an OpenRouter key instead, which the `openrouter:` provider covers.
- The listing parameters were checked against `docs/openapi.yml` (arrays for `tournaments` / `statuses` /
  `forecast_type`, `include_descriptions`), not against the live API.
- "One forecast per question" and "no human in the loop" come from a secondary source (m-ad, quoting notebook 38928).
- Spring 2027 rules (including any commercial-entity exclusion) are not announced.
- The official template repo has no LICENSE (GitHub API, commit bcbd016b, 2026-10-31), so no template code is
  copied; prompts and the client are written from the API docs. forecasting-tools is MIT.
- OpenRouter's `reasoning: {effort}` parameter shape, and whether every OpenRouter model honours it.

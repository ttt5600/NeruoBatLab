# forecaster

An automated forecaster for the Metaculus bot tournaments (FutureEval seasons and MiniBench).
It runs on its own: every 20 minutes it lists the open questions of the tournaments it is
entered in, skips those it has already answered, and for each new one gathers research,
asks an ensemble of language models for forecasts, combines them and posts the forecast with
a private comment explaining it. Nobody reviews or edits a forecast.

## How it forecasts

- **Research** from several independent sources, each a section of its own: news searches on
  three indexes (AskNews, Exa and Tavily, each queried with the question's title, results of the
  last 30 days with their dates and sites; social sites and the tournament's own site left out), a background search of the
  whole web for season questions, and, when the budget allows, a model with native web search
  that writes a short brief. A result already shown by one source is not repeated by the next.
  A source whose key is refused (no credit left, over its plan) is skipped for six hours and
  named in `health.json`; any other failure skips it for that question only.
- **Ensemble**: five forecasts from models of several labs (through OpenRouter), each asked to
  work from an outside view (base rates) to an inside view (the specific evidence).
- **Aggregation**: binary questions take the median, capped to 2-98%; multiple-choice
  questions take each option's median with a 1% floor, renormalised; numeric, discrete and
  date questions average the forecasters' distributions, smooth them lightly and turn them
  into a valid 201-point (or discrete-size) CDF within Metaculus's per-bin cap.
- MiniBench questions, and every question once 80% of the season's budget is spent, use a
  cheaper ensemble without the web-search brief.
- With a season budget of 0 it makes no paid calls at all: five forecasts from models offered
  at no cost, news searches on free allowances as its only research, and it stops if a model
  call is ever charged.
- **Withdrawn models**: the provider's list of offered models is read every hour. A forecaster
  whose model is gone (no longer listed, past its listed expiry, or answering "not found" or
  "forbidden") is replaced at once, in the same question, by the next model of a reserve that is
  still offered, else by another member of the ensemble, so the ensemble keeps its five
  forecasts.

## Running it

```
docker build -t forecaster .
docker run -v $PWD/live:/live -v $PWD/secret:/secret:ro forecaster
```

`/secret/bot.env` holds `METACULUS_TOKEN`, `OPENROUTER_API_KEY` and the news keys it has:
`ASKNEWS_API_KEY`, `EXA_API_KEY`, `TAVILY_API_KEY` (each optional; a source without a key is
left out).
`/live/config.toml` holds only where to forecast and how much may be spent:

```toml
tournaments = ["fall-futureeval-2026", "minibench"]
season_budget_usd = 150
per_question_cap_usd = 2.0
budget_label = "fall-2026"
```

Commands: `python -m forecaster run | pass | dry-run | analyse | check` (see
`forecaster/__main__.py`). `dry-run` forecasts closed or resolved questions without posting;
with `--backtest` its research is limited to results published before each question opened
(undated ones dropped), AskNews's sign-up credit is left alone and web search is off, and
`--no-news` leaves the news searches out. `analyse` scores the bot on resolved questions.
`check` asks each news source one neutral search; `check --models` also asks each no-cost
model a short neutral request to see that it answers.

## What it writes

- `live/health.json` each pass: counts of questions seen, done, skipped and failed in the last
  24 hours, spend, credit left, budget warning, the last error's type, the errors of the last
  24 hours counted by stage (a forecaster's by its place in the ensemble), type and HTTP status,
  the models out (withdrawn by the provider or refused to this key, and replaced), and the news
  sources out (their key refused).
- `live/bot.log`: the same counts, how many results each news source gave each question, and
  errors as their type, place, question id and type (and HTTP status for a failed request, the
  model for a forecaster's error, the finish reason for an empty reply).
- `live/records/<question id>.json`: each question's full record, used by the bot to finish a
  question interrupted mid-way and by `analyse` once the question has closed.

## Exit codes

`2` a key, credit or API failure lasting three passes; `3` the season's budget is spent;
`4` every question failed three passes in a row; `5` a pass took over an hour.

## Licence

MIT. The CDF standardisation follows the Metaculus forecasting-tools library (MIT); see NOTICE.

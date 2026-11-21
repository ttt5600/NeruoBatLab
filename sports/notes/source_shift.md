# source_shift -- systematic search for P-1 (settlement-definition shift) opportunities (agent source_shift, 2026-11-21)

Brief (lead): find more P-1 cases (IDEA_PATTERNS.md P-1: OpenRouter default chart switched to BYOK-excluded mid-week; bots
repriced in minutes). (1) Kalshi history: mid-event far strike listings, rules-text changes, settlement holds / >= 85c
favourite flips -> what happened at the source, did the book / settlement follow the OLD or NEW definition -> base rate.
(2) Build read-only scripts/source_shift_watch.py (multi-definition snapshots + Kalshi strikes/rules, ALERT lines). (3) Report
base rate + open P-1 candidates; append patterns to IDEA_PATTERNS.md. NO git, NO orders, NO launchd.
Scratch: results/source_shift/. Watch output: ~/.claude/quant-sports/data/source_shift/snapshots.jsonl.

## Pre-registration (written 2026-11-21 ~06:05Z, BEFORE any settled outcome / flip / hold was computed)
Seen before writing: API field list (one KXTOKENUSE market), the live KXTOKENUSE-26NOV23 ladder (strikes T137..T200 created
11-21T04:10Z, T310..T335 created 11-18T22:13Z), one-day market counts by series (no outcomes).

**Question.** When a Kalshi contract's settlement source changes how it measures (methodology / default view / vintage /
correction / outage / definition dispute) while the contract is open, does Kalshi settle on the OLD definition (the one used by
the series' earlier settlements or in force at listing) or on the NEW one (what the source shows at the observation time)? And
did the final pre-close book price the old or the new?

**Universe.** Settled markets from /historical/markets + /markets?status=settled for 11,080 series (cached series list minus
Sports, Crypto and hourly/daily/one-off Financials/Commodities price ladders); script results/source_shift/census.py.

**Detectors (frozen):**
- (a) LATE FAR STRIKE: in an event with a numeric ladder (floor/cap strikes), a market created >= 6 h after the event's first
  market whose strike lies outside the initial ladder [min, max] by more than 2 x the initial median strike step. Score =
  distance outside / initial ladder span. Report the count per series and the top cases by score in non-weather series.
- (b) RULES CHANGE: within an event, two creation batches (created_time differs by >= 6 h) whose normalised rules templates
  (numbers, months, sub-titles masked) differ = mid-event rules change. Across events of a series: list template changes of
  rules_secondary (descriptive, the date Kalshi changed the contract).
- (c) HOLD / FLIP / ODD SETTLEMENT: hold = event settlement_ts - close_time > max(48 h, series median + 24 h) where the series
  median is over its own events; flip = event favourite (highest last_price, >= 0.85, market volume >= 100 contracts) settled
  NO, or a market with last_price <= 0.15 and volume >= 100 settled YES in a mutually-exclusive ladder; odd = a market whose
  settlement_value is neither 0 nor 1 or result not in {yes, no}.
**Case classification (manual, for every flagged non-weather candidate inspected, top-ranked first until time runs out):**
cause in {source methodology/default change, source correction/revision, source outage/missing print, definition dispute /
interpretation, Kalshi error, ordinary news (no source change)}. For source-change causes record OLD vs NEW for (i) settlement,
(ii) final pre-close price. Weather holds are taken from winners_transfer P4 (delays.parquet), not re-pulled.
**Base-rate statistic.** k/n = settlements that followed the NEW definition among source-change cases; Jeffreys 95% interval.
Pre-set wording: n < 10 -> "anecdotal, not a base rate"; interval crossing 0.5 -> "not distinguishable from a coin flip".
No trade rule is tested here (descriptive); any trade idea that comes out goes to a separate prereg.

## Log

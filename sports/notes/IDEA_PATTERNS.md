# Idea patterns -- how we found each idea, so we can find more (started 2026-11-21, user: "save ideas like this and write how you found them")

One section per pattern: what it is, the case that taught it, HOW it was found (the step that surfaced it), the evidence for
and against, the outcome, and the generalisation (where else to look). Update the outcome line when a bet settles.

## P-1 Settlement-definition shift (OpenRouter BYOK, 2026-11-21)
- **Pattern:** the settlement source changes how it counts (methodology, default view, vintage) while a contract is open.
  Bots reprice to the NEW displayed number within minutes. The contract text may still be read the OLD way by the exchange.
- **Case:** Kalshi KXTOKENUSE-26NOV23 ("OpenRouter token usage Nov 16-22"). All 7 settled weeks matched OpenRouter's
  BYOK-inclusive total exactly. Between 11-19T21:07Z and 11-21T03:55Z OpenRouter's default chart switched to EXCLUDE BYOK.
  Week-so-far: inclusive 254.8T, exclusive 154.0T. "Above 238T" fell .80 -> .05 within minutes.
- **How found:** the ai_nowcast agent was rebuilding the settlement value EXACTLY (data-generating-process step from the
  compete playbook, C4 amendment). Its daily rebuild suddenly disagreed with its own earlier captures. The mismatch was the
  signal. Lesson: exact settlement rebuilds double as change detectors.
- **Bet:** $14.08, 180 YES @ .0735 + .0047 fee (notes/EXPERIMENT_BUDGET.md). Pays $180 if Kalshi settles BYOK-inclusive.
- **Evidence for:** every past settlement used the inclusive total; the rules do not name a view ("token usage included in
  OpenRouter's AI Model Rankings ... as observed at 10:00 AM ET").
- **Evidence against (weighty):** historically default == inclusive, so Kalshi simply read the page as displayed; the page
  now displays the exclusive number. Kalshi ITSELF listed new low strikes T137..T200 at 04:10Z, 7 minutes after the first
  post-switch trade -> the exchange expects the exclusive number. Fast bots agree. My estimate: roughly 10-20%, vs ~7% priced.
  Not confident enough to add size.
- **Outcome (2026-11-23): LOST $14.08.** KXTOKENUSE-26NOV23 expiration_value 210.109583915801 = EXACTLY the default
  (organic, BYOK-EXCLUDED) chart's week-of-11-16 total in our watcher snapshot at 16:27Z (chart_default weeks_T 210.1096;
  ~/.claude/quant-sports/data/source_shift/snapshots.jsonl). Kalshi read the page as displayed -> NEW definition (6th P-4 case).
  Sibling KXTOPUSAGEAI-26NOV16 settled DEEP (organic view), TENC no. The value also drifted 210.11 -> 212.83 one hour later:
  "as observed at 10:00 AM ET" means a snapshot, so the exact rebuild must be taken at the stated minute.
  Lesson: when the source's DISPLAY changes, bet with the display, not with the settlement history; our 10-20% estimate was
  too high -- the book's ~7% and Kalshi's own new low strikes were the better forecasts.
- **Generalise:** every series we can rebuild exactly gets a daily rebuild + alternative-definition rebuild; alert when
  (a) the rebuild diverges from prior captures, (b) the source page changes format/toggles/default, (c) the exchange lists
  new strikes or edits rules text. Candidates: OpenRouter/Vercel share series, NOAA vintages (first print vs revised),
  AAA gas methodology, NWS CLI corrections / Weather Company switch (weather), TSA page format, Federal Register vs White House.

## P-2 Exact settlement rebuild -> fastest page (count_nowcast, 2026-11-21)
- Executive orders settle on the Federal Register, but the book moves off the White House posting ~5 days earlier.
  Found by timing the book against every public page that reveals the outcome. Generalise: for any stale-bracket idea,
  list every page that reveals the outcome and time the book against the fastest.

## P-3 Secondary market priced off a sharp main line (KFG1, 2026-11-19)
- Kalshi team field-goal ladders vs a model turning sportsbook implied points into P(FG >= k). Found from the user's
  "kicker stats" idea + the rule that edges live in niche markets. Dev +23% [-0.7,+47.3] n56 (not distinguishable);
  derived_ladders later found most secondary ladders calibrated -> treat as a long shot; live $2/bet measurement.

## Where to look next (pattern-driven, not market-driven)
1. Settlement-source change watcher (P-1) across all rebuildable series.
2. Exchange behaviour as signal: new strike listings, rules edits, settlement holds (winners_transfer P4: a 155-h NY hold paid
   cheap-side buyers $410k) -- when does a hold flip the result, and is it predictable from source corrections?
3. First-print vs revised archives (climate lesson) for every revised official series.

## Patterns from source_shift (agent source_shift, 2026-11-21; notes/source_shift.md has every citation)
How found, all of them: a per-series pull of every settled Kalshi market (results/source_shift/census.py, /historical + live),
three detectors (late far strikes, rules-template changes, holds / odd settlements), then reading the rules text of the
flagged events. Two steps did most of the work: (1) sorting holds by month (the 2025 shutdown lit up), (2) grepping every
stored rules text for "Clarification" / "amendment" (31 distinct mid-contract clarifications = Kalshi's own archive of disputes).

### P-4 How Kalshi settles when the source changes mid-contract (base rate for P-1 bets) -- revised after a critic review
- **Rate (pre-registered, open contracts, determinable OLD/NEW):** NEW 4 of 5 decisions -- KXALBUMSALES (Hits Daily Double moved
  its chart Fri -> Thu; Kalshi moved OPEN events to the Thursday chart "pursuant to Rule 7.2(a)"), KXWAYMOCITIES (app renamed,
  "clear continuity"), KXAAAGASD 3 -> 4 decimals (no payout effect), weather material-error corrections -- vs KXGDP-25OCT30
  (settled 3.8 = neither the cancelled Advance Estimate nor BEA's 4.3 that arrived inside the window). Jeffreys [0.37, 0.98];
  n < 10 = anecdotal. The first draft (3 of 5) counted GTEMP, TROPSTORM and sea ice; the critic showed none had a payout-relevant
  or verifiable OLD/NEW choice.
- **Pattern:** on open contracts Kalshi followed the source's current presentation every time it was observable and mattered.
  OLD appears only when Kalshi itself changes source (NWS -> Weather Company 2026-08-14: open events kept NWS) or writes a
  definition into NEW contracts (LMArena "Remove Style Control" from 2025-12-31; 19/19 later events, ~4 independent stretches).
  Rulebook 7.2 lets Kalshi re-designate the underlying mid-contract and it has used that to move open contracts to the new form.
- **Applied to KXTOKENUSE-26NOV23 (P-1 bet):** the history leans NEW (= BYOK-excluded default). With Kalshi's new low strikes and
  the sibling's "as displayed on the OpenRouter Top Models chart" (KXTOPUSAGEAI), the BYOK-inclusive YES is likely below P-1's
  10-20%. Hold the $14.08; do not add. Record the settlement as the 6th decision.
- **6th decision (2026-11-23): NEW.** KXTOKENUSE-26NOV23 settled on the BYOK-excluded default view (210.11T, exact match).
  Rate now NEW 5 of 6, Jeffreys 95% interval [0.442, 0.981] (Beta(5.5, 1.5), scipy; the 4/5 check reproduces [0.371, 0.977]).
  Operating rule: on an open contract, assume Kalshi settles on the source as DISPLAYED at the settlement minute; a
  definition-change bet needs a reason the display will revert, not history.

### P-5 A cancelled print settles near the book; a delayed print settles to the truth (2025 shutdown)
- Never published (Oct-2025 CPI x4, Oct-2025 U3, Q3-2025 GDP "Advance Estimate", Oct eggs): every TRADED strike resolved to its
  last-price side; but U3 4.4, core m/m 0.2 and GDP 3.8 also equal the last published print -- only one same-night CPI decision
  (headline 0.2/3.3, core y/y 3.3 vs Sep 0.3/3.0/3.0) separates "book" from "last print". No edge either way.
- Published late (Sep U3/NFP, Oct TSAW, PCE): Kalshi waited, past its stated windows, and settled on the real print against the
  book (KXPAYROLLS-25SEP T100000 .10 and T75000 .20 -> YES; KXU3-25SEP T4.3 .31 -> YES).
- Weather "no valid value" holds settle at scalar fair prices (KXLOWTMIA-26JUL07 B78.5 0.72 vs last .75).
- **Generalise:** in the next shutdown, the trade is in series whose data will come out LATE (book priced at close, truth weeks
  later), not in cancelled ones.

### P-6 The settlement page has a free JSON behind it -- open the URL in settlement_sources and grep its JS for /api/
- weather.com/kalshi (the Weather Company display all KXHIGH/KXLOW settle on) calls /kalshi/api/climate/primary?date=YYYY-MM-DD:
  66 stations, max/min, status official/revised/preliminary, any past date (NY 11-13 shows the corrected 47 = settled value).
  Same trick found OpenRouter's rankings endpoints (ai_nowcast) and Vercel's leaderboard export (vercel_share).
- **Use:** revives winners_transfer I1 (settlement-error watcher, killed on hourly obs): log the TWC PRELIMINARY value through
  the day vs the NWS CLI final; a >= 1F gap is the NY 11-13 signature. Forward-only (the display keeps no preliminary history).
  scripts/source_shift_watch.py logs it now.

### P-7 One source, two views, two different winners (OpenRouter BYOK, live 2026-11-21)
- Week of 11-16: #1 model by tokens = tencent/hy4-flash (96.8T) with BYOK, deepseek-v4.1-flash (49.2T) without; #1 lab by
  tokens = tencent 39.9% (author-market-share, BYOK-inclusive, ignores the byok param). Any "top model / top lab" market listed
  on that week is a P-1 binary, not a ladder -- and one is open: KXTOPUSAGEAI-26NOV16 (closes 11-23 04:59Z; "the model author
  company of the single model with the greatest weekly token volume as displayed on the OpenRouter Top Models chart"): organic
  view -> DeepSeek, byok=all -> Tencent. Book 06:30Z: DEEP .88/.99, TENC 0/.14 (watcher snapshot). The book and the words "as
  displayed" both point to the organic default; TENC at .14 is the same bet as our T238 YES but at a worse price (~7x vs ~13x). The share series (KXANTHSHARE/GOOG/DEEP) settle on TEXT REQUEST share
  (modality-chart?routeSegment=text, verified: week 11-09 anthropic 6.27 -> settled 6.3), BYOK-inclusive today with no toggle;
  if OpenRouter gives that chart an organic default too, tencent's 13.2% vanishes and every lab's share jumps ~15% relative.
- **Status:** watch only (no evidence the share chart will change; the watcher alerts on a new byok filter, a different #1,
  and new KXTOPUSAGEAI/KXTOKENUSE events).

### P-8 Settlement-source PLUMBING change, not a definition change (Q014 Roll Call, ops_q013_q014 2026-11-21)
- Roll Call rebuilt Factba.se (new Livewire front end) between 2026-11-18 00:07:53Z and 00:22:55Z: wp-json/factbase/v1/twitter -> 404, data now at /api/factbase/social (found with the P-6 trick: `window.socialApiUrl` in the archive page); same ids/timestamps (952/952) and the rebuilt counts fit all 61 settled KXTRUTHSOCIAL(D) results -> a P-1(ii) "format" event with NO price signal, which still cost Q014 6 decision slots over 3 days (9 counting 3 that fell in Mac-sleep gaps) because source_shift_watch does not watch Roll Call. To catch it, the watcher needs: the logger's own endpoint (alert on non-200 / JSON key-signature change), the endpoint URL re-derived from the archive page each run, a daily exact rebuild of yesterday's KXTRUTHSOCIALD count vs Kalshi expiration_value (and vs the trumpstruth RSS), and the KXTRUTHSOCIAL/D rules template (it already changed once: "11:59 PM ET" up to 26OCT10 daily / 26OCT17 weekly, "11:59:59 PM ET" after; the logger's window regex only accepts the new form).

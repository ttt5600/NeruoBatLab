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
- **Outcome:** pending (settles 2026-11-23 10:00 ET). Record which view Kalshi used here and in LEARNINGS.
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

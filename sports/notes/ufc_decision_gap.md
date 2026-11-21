# ufc_decision_gap -- Kalshi UFC winner (KXUFCFIGHT) between the final horn and the announced decision

Agent ufc_decision_gap, started 2026-11-20 (lead brief; pool_census B: KXUFCFIGHT #21, $2.0M/30d reach, 51 fights/30d).
Scratch: results/ufc_decision_gap/. Rules: no git, no orders, read-only GETs, no launchd; sealed untouched.
Do-not-redo: U1 MOV (dead, notes/ufc_tsa.md); public-stats models vs headline markets 0/7 sealed (STRATEGY_v2 s.2).

## Question
Descriptive first: on fights that go to the judges, how is the Kalshi winner market priced in the window between the
final horn and the announcement (mid, spread, volume, how long the window lasts), and does a judges model built from
free round-by-round stats (sig. strikes, takedowns, control, knockdowns) predict the official decision better than the
window price? Expected failure (stated up front): the crowd watches the same broadcast; the book already prices it.

## Plan
1. Join KXUFCFIGHT finalized markets to UFCStats bouts; keep decisions (any "Decision - *").
2. Fix DEV/SEALED split by card date BEFORE any price is joined to an outcome.
3. Fetch the trade tape (and 1-min candles) around each decision fight's close; locate the announcement (price jump to
   ~0/1) and the final horn; verify on a sample; describe window book (DEV + SEALED descriptive WITHOUT outcomes is OK
   for timing/volume only; price-vs-outcome on DEV only).
4. Judges model from UFCStats round stats (fit on decisions before 2025-01-01; test 2025+), point-in-time = stats the
   UFC live stats show during the fight.
5. Compare model vs window price on DEV decisions (Brier/log loss, stacking coefficient, cluster by card). Prereg a rule
   only if the model adds information with a CI.

## Log
### 2026-11-21 join + split (results/ufc_decision_gap/join.py -> fights.parquet; no prices, no Kalshi outcomes)
KXUFCFIGHT finalized events in census_raw.json (ufc_tsa, pulled 11-19): 824 (2 markets each). Matched to a UFCStats bout:
694 (full 565 / last-name 84 / token 45); unmatched 130 = 53 cancelled (scalar fair-price settle), Contender Series /
non-UFCStats cards (2026-08..10 Tuesdays, ~56) and a few name variants (Su Mudaerji, Billy Ray Goff). Decisions among
matched: 305 (unanimous 258, split 37, majority 10), 2025-05 .. 2026-11.
Contract facts (market JSON): "This market will close and expire after a winner is declared" -> close_time = Kalshi's
close after the declaration (not the horn); settlement_timer 300 s; draw / NC -> both $0.50.
**SPLIT FIXED NOW (before any tape is fetched):** DEV = card date < 2026-07-01 (220 decision fights); SEALED = card
>= 2026-07-01 (85, 27.9%). SEALED tape/candles are not fetched at all (timing descriptives included) until a DEV pass +
user yes. Judges model: fit on UFCStats decisions dated < 2025-01-01; model test years 2025-26 (outcome-only, no prices).

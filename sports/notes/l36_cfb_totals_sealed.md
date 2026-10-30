# L36 -- CFB totals vs the OPENER: SEALED one-shot test (pre-registered 2026-10-30)

Written and committed BEFORE any 2025 or 2026 line was opened, joined to a prediction, or joined to an outcome.
Dev evidence (SBR 2008-2022, open book unknown, -110 assumed; `notes/cfb_simple_model.md`): T02 total CLV +1.02 pts
[+0.87, +1.17]; T06 no-vig ROI +7.4% (p .0004); T04 -110 ROI +2.6% [-1.3, +6.8] p .10 (FAIL). Prior from NBA S1: real
US openers (DraftKings) are sharper than older/unknown openers, so the edge may shrink here.

## Question

Does the frozen CFB totals model, betting its side of the ESPN-reported OPENING total when it disagrees by >= 3.5
points, make money at the REAL opening price on unseen seasons?

## Frozen model (no refit of anything)

- `scripts/cfb_simple_model.py` imported unchanged: `build_predictions`, ridge o/d totals model, LAMBDA_OD 4,
  LAMBDA_MU 50, half-life 42 d, cap 70 pts/team, REG 0.6, new-FBS priors o -5 / d +5, HFA = mean of prior seasons'
  end-of-season estimates, FBS = >= 6 games in the season (results only), non-FBS pooled into "FCS".
- Ratings walk forward date by date on RESULTS ONLY (no market column enters the model).
- Results source: SBR 2007-2021 exactly as in dev, then ESPN 2022-2026 (`games_full_<season>.parquet`). SBR 2022
  stops on 11-15, so ESPN's full 2022 season replaces it as the 2023 prior season. 2022-2024 are burn-in (not scored).
- **Collector bug found while doing this** (results only): the ESPN scoreboard with `groups=80&limit=1000` now
  returns only ~25 featured games (that is why `games_2024.parquet` has 548 games and `games_2025` 544).
  `limit=300` returns the full slate. `scripts/l36_cfb_sealed.py fetch-games` refetches 2022-2026 with limit=300
  into `games_full_*`; `sports/espn.py` and the old files are untouched (the lead should fix the collector).
- ESPN team_id -> SBR name map: built ONLY from 2022 results (ESPN game matched to SBR game on date +-1 day and exact
  final score; a team_id maps when >= 3 matched games and >= 80% agree; SBR names claimed twice are dropped).
  Unmapped ESPN teams get a new name, i.e. they are treated as new FBS teams (prior -10 / o -5 / d +5) if they reach
  6 games. Counts: see "Name map" below (filled in from results only, before this commit).

## Sealed set

- 2025 full season (incl. bowls/CFP) + 2026 season to date (all FINAL games through 2026-10-29; weeks 1-9ish).
- FBS-vs-FBS only. Market filters as dev: open total in [25, 100], |close - open| <= 14.
- Week = 1 + floor((ET date - season's first ET date) / 7), as in dev.

## Book and price source

- ESPN core odds endpoint (`/events/{id}/competitions/{id}/odds`), fetched by `fetch-lines` into
  `lines_full_<season>.parquet`. Per game, the book is the FIRST provider in the order **DraftKings, ESPN BET, then
  any other provider alphabetically** that has `total_open`, `over_odds_open`, `under_odds_open` and `total_close`
  all present. Rationale: DraftKings is the sharpest US opener (NBA S1 lesson) and the book a bettor can actually use;
  ESPN BET was ESPN's 2024-25 feed. The provider mix is reported.
- Open price = `over_odds_open` / `under_odds_open` of that book for the side bet. Close = that book's `total_close`.
- Limitation stated up front: ESPN gives NO timestamp for the open; "open" = the first number ESPN stored.

## Frozen bet rule

`|pred_total - total_open| >= 3.5` -> bet the model's side (over if pred > open) at the open price, 1 unit, one bet
per game. Win pays the American open price, push = 0, loss = -1.

## Trials

| id | metric | role |
|---|---|---|
| **P1** | ROI per 1u at the REAL opening price | **primary -- the money question** |
| P2 | total CLV, points (close - open toward the bet) | secondary (reported, not a pass condition) |
| P3 | P1 restricted to weeks 1-4 | secondary |

- Inference: one-sided season-week cluster bootstrap, 4000 resamples (`cfb_simple_model.boot_p`, seed 0).
  Report the OBSERVED mean with the 2.5/97.5 bootstrap percentiles. If the interval crosses 0: "not distinguishable".
- **P1 PASSES iff p < 0.05 AND observed ROI > 0 AND observed ROI > 0 in BOTH seasons (2025, and 2026 to date).**
  The split is by season (decided now; 2026 is expected to have ~100+ bets).
- **Kill: P1 fails -> L36 is DEAD for betting** (CLV may still be reported as a market-signal fact).
- Also reported (not counted): hit rate ex-push vs mean no-vig implied prob of the side, average price, -110 ROI
  reference, model/open/close MAE, per-season n/ROI/CLV, provider mix.
- **Too-good guard:** ROI > +10% on > 200 bets -> inspect rows (join, side sign, open/close swap, push handling,
  post-kickoff numbers) before reporting.

## Power check (written before scoring)

- Dev bet rate (`/tmp` check of cfb_simple_model dev bets): 3,014 bets / 10,821 FBS games with open+close = 27.9%.
  Expected FBS-vs-FBS games: ~800 in 2025 (2023: 792, 2024: 798) and ~470 in 2026 to date; with ~90% market
  coverage that is **~320 bets** (could be 250-350).
- Dev per-bet ROI SD at -110 = 0.944, so **SE(ROI) ~ 0.944/sqrt(320) ~ 5.3 points**. The dev -110 effect is +2.6% and
  the real-price effect is probably smaller (sharper opener). SE (5.3%) > dev effect (2.6%): **this test is
  UNDERPOWERED.** Power to reach p < .05 if the true ROI is +2.6% is roughly 12-15%.
- Therefore: a PASS would be informative; a FAIL is "not distinguishable from zero", NOT proof of no edge. The kill
  rule still applies to betting real money (no evidence -> no money), but the lead is not refuted as a market signal.
  P2 (CLV) has much more power (dev per-bet CLV SD 3.08 pts -> SE ~0.17 pts vs dev effect +1.02).

## Name map (results only, filled before commit)

- Map file `data/lab/l36_espn_to_sbr_names.json`: 131 ESPN ids mapped; **all 130 SBR 2021 FBS teams mapped** (0
  unmapped), plus James Madison (FBS from 2022). Only votes from unambiguous (date, score) matches count (a first
  version that let same-day same-score games vote left 18 big programs unmapped -- fixed before commit).
- Unmapped FBS ESPN ids per season (all are genuine FBS newcomers, named `ESPN_<id>`, new-FBS priors in year 1, own
  ratings after): 2022: 0; 2023: 2 (SHSU, JVST); 2024: 3 (+KENN); 2025: 5 (+MOST, DEL); 2026: 7 (+SAC, NDSU).
- **Amendment (pre-scoring, results-free):** the groups=80 feed lists games of teams that are FBS *today*, including
  their FCS seasons (e.g. Delaware 2022-24), so the >=6-games rule alone would call them FBS. For ESPN seasons,
  FBS = >=6 games AND in ESPN's FBS group list for that season (`/seasons/{y}/types/2/groups/80/teams`).
  FBS counts: 2022 131, 2023 133, 2024 134, 2025 136, 2026 138 (SBR dev: 130 in 2021).
- Outcome-only check on burn-in seasons (no market): FBS-vs-FBS total MAE 13.13 (2023, 792 games, bias +0.10) and
  12.89 (2024, 798 games, bias -0.52); dev was ~13.3. The ESPN results chain works.
- 2025/2026 game counts only: 958 / 572 final games in the FBS feed (no scores were printed or inspected).

## Results

(not yet run)

# NFL kicker x forecast wind, run/pass matchups, player x stadium -- PRE-REGISTRATION (2026-10-31)

User ask 10-31: "kicker and location and weather interactions ... stadium performance for players and which teams have
edges against certain run or pass designed plays and what the offense dictates". Script: `scripts/nfl_kicker_wx.py`.
Ledger dataset `nfl_kicker_wx_matchups`. Written and committed BEFORE any line or bet outcome is joined.

## Splits (lab/datasets.py): DEV = seasons <= 2019. LOCKBOX 2020-2025 is never loaded for any gate or trial.
- Lines: SBR archive 2007-2019 (open + close, spreads/totals assumed -110 both sides; SBR gives no juice).
- pbp: nflverse 2006-2019 for gates, the kicker model fit, and point-in-time team features.
- Live descriptive table for 2026-11-01 ONLY: kicker skill uses 2020-2026 kick attempts (no lines, no scoring).

## Trials: k = 3 counted (A1, B1, B2). Bonferroni bar one-sided p < 0.05/3 = 0.0167.
- A2 (kicker points / FG-made props UNDER) is NOT TESTABLE in dev: ESPN prop lines exist only 2022+ = lockbox. Skipped,
  not counted. Bar for every trial: p < .0167 (week-clustered bootstrap, 4000 resamples, one-sided ROI <= 0), observed ROI
  > 0 in BOTH dev halves (2007-2013, 2014-2019). CI crossing 0 -> "not distinguishable". > +10% on > 200 bets -> inspect rows
  before believing (forecast issue time vs kickoff, timestamp units, join keys) . Pushes = 0.

### A. Forecast wind (causal version of the observed-wind lead, ledger nfl_total:wind_unders)
- Forecast: IEM archive of as-issued GFS MOS (MAV) bulletins at the airport ASOS nearest each stadium (mapping fixed in the
  script; fallback station list). Wind = MOS `wsp` (10-m sustained, knots) -> mph.
- Kickoff UTC = schedule gameday + gametime localised to America/New_York PER DATE (tz-aware, DST-correct), then UTC.
  Timestamps differenced with `// pd.Timedelta(seconds=1)`, never `// 1e9`.
- Issue time: latest MOS runtime with runtime + 4 h (bulletin latency) <= kickoff - 2 h, i.e. runtime <= kickoff - 6 h.
  Forecast wind = mean `wsp` over forecast hours in [kickoff, kickoff + 3 h] (inclusive; MOS steps are 3 h);
  if none, the nearest forecast hour within 3 h of kickoff + 1.5 h. Missing -> game excluded (reported).
- Universe: regular + post season 2007-2019, schedule roof == "outdoors", US stadiums with an ASOS mapping.
- **A1: bet the game total UNDER at the SBR CLOSE when forecast wind >= 15 mph (13.0 kt).** The close is used because it is
  posted after the forecast is issued (the opener predates it). Threshold fixed a priori = the old observed-wind threshold.
- Power (stated before the run): ~250 bets, per-bet SD ~1.0 u -> SE ~6.3%; the bar needs observed ROI ~ +13.5%.
  The old observed-wind lead was +7.3%, so A1 is under-powered for an effect of that size; a fail is "not shown", not "zero".
- Kicker make-probability model (descriptive input, not a trial): logistic on dev FG attempts 2006-2019:
  make ~ dist + dist^2 + outdoor*wind_mph + outdoor*wind_mph*dist/10 + outdoor*temp_F + altitude(DEN) + kicker_skill,
  kicker_skill = kicker's prior-attempts residual vs the base model, shrunk K = 40 attempts, decayed x0.8 each new season.
  pbp wind is OBSERVED, which is fine for a make-prob model; it is not used for any bet.

### B. Run/pass matchup ("what the offense dictates" x defence split weakness)
- Gate B0 (no bets unless it passes): defence split S = (rush EPA/play allowed - pass EPA/dropback allowed), raw, per
  team-season, weeks 1-8 vs weeks 9+ (regular season), 2006-2019. Pearson r with team-season bootstrap 95% CI.
  **If r < 0.15 or CI lower bound <= 0 -> B is DEAD, B1/B2 not run.**
- Features (point-in-time, prior games only, shrunk like nfl_simple_model: est = (sum + K*prior)/(n + K),
  prior = league + 0.5*(team last season - league); K = 200 rushes / 250 dropbacks / 200 neutral plays):
  offence neutral rush rate r_o (wp .2-.8, downs 1-3, non-garbage); defence d_rush, d_pass (EPA allowed minus league).
- Matchup term for an offence vs a defence: M = (r_o - r_lg) * (d_rush - d_pass) * 62 plays  [pts, pure interaction: the
  offence's mix deviation applied to the defence's split; the defence's level is excluded because it is priced].
- Spread term T_s = M(home O vs away D) - M(away O vs home D); total term T_t = M_home + M_away. Weeks >= 3 only.
- Threshold: top quintile of |T| in the dev feature distribution (computed from features only, before any join).
- **B1: spread ATS at SBR OPEN, side = sign(T_s), when |T_s| >= q80.  B2: total at SBR OPEN, over if T_t > 0, when |T_t| >= q80.**

### C. Player x stadium (yards) -- reliability gate only
- Residual = player-game yards (rushing for rushers, receiving for receivers, passing for passers) minus that player's same-
  season mean in other games. Cells = player x stadium with >= 4 games 2006-2019; split chronologically odd/even appearances;
  r of half means (cells weighted equally), bootstrap CI over cells. Reported for all cells and for non-home stadium cells.
- **No trial unless r >= 0.15** (NBA analogue r .024). Expect ~0.

## Then (descriptive, not bets): 2026-11-01 outdoor games -- Open-Meteo forecast wind/temp/precip at kickoff, each listed
kicker's model make prob at 40/50 yds, B term per game only if B0 passed, vs ESPN scoreboard lines. Rows from a passing trial
are labelled BET-CANDIDATE; everything else DESCRIPTIVE.

# NFL comebacks leads re-tested against the OPENER: bad-D unders (BD) and offence EPA ATS (OE) (2026-10-25)

Script: `scripts/nfl_leads_open.py [--ledger]` (~10 s). The pre-registration is its docstring, written before any
market join. JSON: `data/lab/nfl_leads_open_results.json`. Ledger: 10 rows, dataset `nfl_leads_open`, bar p < .005.
DEV only: test games 2010-2019, both teams with at least 4 prior games (2,017 games with SBR open+close).
2020+ was not read. Model predictions are the `nfl_simple_model.py` walk-forward predictions (spread B, total C).

## Verdict

1. **BD (two bad defences go under) is DEAD.** Against the opener it is wrong-signed on every metric. The rule (both
   defences in the worst within-week tercile, 214 bets) gave CLV **-0.16 pts** [-0.42, +0.11], under ROI at the open
   -7.7% and at the close -5.8%, with 48% / 49% wins. Both halves are negative on all three. The linear slope is also
   null: -0.11 pts/SD vs the open (p .63) and +0.12 vs the close (p .35, sign flips between halves). **The close lead
   does not survive the switch to the shrunk feature.** The original -0.79/SD used unshrunk per-play EPA. What the
   data does show: the market **raises** these totals between open and close, by 0.23 pts/SD (strongly significant,
   in the wrong direction for an under bet), and the outcomes rise by about the same amount.
2. **OE (market under-weights offence EPA) is REAL as a slope, but NOT as a bet.**
   - L09 (counted) passes: (margin - open spread) on the offence-EPA difference gives **+0.98 pts/SD [+0.42, +1.54],
     p .0003**, +0.87 in 2010-14 and +1.09 in 2015-19. Against the close it is +0.87/SD (p .0008, not counted), which
     is larger than the comebacks estimate of +0.56.
   - The pre-registered bet rule does **not** make money. With |z-diff| >= 1 (994 bets), ATS was **-1.6% at the open
     and -1.6% at the close at -110**, with a 51.5% win rate (breakeven is 52.4%). CLV was +0.13 pts (p .041, not
     past the bar). At no-vig it would be +2.9% at the open (even money) and +3.5% at the close (nflverse prices).
     Neither CI excludes 0.
   - Why the slope passes and the bet fails (post-hoc diagnostic, not counted): the effect is in the **size of the
     margin, not the cover rate**. The cover-rate slope is only +2.1 pp/SD (one-sided p .023). By quintile of the
     offence difference, home cover rates were 47.3 / 43.7 / 47.5 / 53.0 / 50.4%. The extreme quintiles, which
     the rule bets, are the weakest. Clipping the residual at +-14 shrinks the slope from 0.98 to 0.65.
3. **Incremental to the simple model?**
   - **OE is mostly but not entirely inside model B.** Against model B's own prediction, the residual slope is +0.34
     pts/SD (p .11). Added on top of the model-vs-open gap (L10) it is **+0.71 [+0.14, +1.28], p .0077, both halves
     positive**. That narrowly fails the .005 bar. It adds nothing to CLV: given the model gap, the slope on line
     moves is -0.05. Read: the model already carries most of the offence signal, and the model gap gets a small
     weight (0.24) when stacked on the open, so offence EPA has a little room left over. That is not a separate edge.
   - **BD is not in the model and should not be.** Its incremental coefficient given model C's gap is wrong-signed
     (-0.28, p .81). Model C slightly UNDER-predicts these games: points - model C rises +0.59 pts/SD with S, which
     is two-sided p about .07 in the over direction.

**Bottom line: neither lead is a betting edge at the opener.** BD is dead. OE is a real but small "offence is
under-weighted" tilt in margins. At -110 it does not produce covers, and the simple model already holds most of it.
If anything changes, it would be a model change: let model B's offence term lean slightly heavier. That would be a new
spec, and it would need forward validation because 2020-25 has been spent on the refit. It is not a standalone bet.

## Counted trials (bar p < 0.05/10 = .005; one-sided in the predicted direction)

| trial | n | mean | 95% CI | p | 2010-14 / 2015-19 | pass |
|---|---|---|---|---|---|---|
| L01 BD rule CLV vs open (pts, under) | 214 | -0.157 | [-0.415, +0.107] | .88 | -0.15 / -0.17 | no |
| L02 BD rule ROI at open -110 | 214 | -7.7% | [-21.1%, +5.9%] | .86 | -7.0 / -8.4% | no |
| L03 BD rule ROI at close -110 | 214 | -5.8% | [-19.0%, +7.6%] | .80 | -2.9 / -9.3% | no |
| L04 BD slope (points - open), pts/SD (+ = under) | 2017 | -0.11 | [-0.73, +0.52] | .63 | -0.36 / +0.17 | no |
| L05 BD incremental given model C gap | 2017 | -0.28 | [-0.92, +0.35] | .81 | -0.63 / +0.09 | no |
| L06 OE rule CLV vs open (pts) | 994 | +0.131 | [-0.016, +0.269] | .041 | +0.14 / +0.13 | no |
| L07 OE rule ATS ROI at open -110 | 994 | -1.6% | [-7.4%, +4.1%] | .72 | -1.8 / -1.5% | no |
| L08 OE rule ATS ROI at close -110 | 994 | -1.6% | [-7.1%, +3.9%] | .73 | -2.7 / -0.5% | no |
| L09 OE slope (margin - open), pts/SD | 2017 | **+0.98** | [+0.42, +1.54] | **.0003** | +0.87 / +1.09 | **YES** |
| L10 OE incremental given model B gap | 2017 | +0.71 | [+0.14, +1.28] | .0077 | +0.54 / +0.87 | no (bar) |

Rule metrics use a week-cluster bootstrap with 5,000 draws. Slopes use week-clustered OLS with normal CIs.

Reported, not counted:
- BD: win rate 48.4% at the open and 49.3% at the close. Even-money ROI at the open -3.3%. No-vig ROI at the close
  -1.8% [-15.8, +12.3]. Mean open 47.2, close 47.3, actual 47.3, model C 46.6. Model C itself would bet the under in
  26% of these games and the over in 11%.
- OE: 52% of the bets were on the home side. Win rate 51.5% at the open and at the close. Even-money ROI at the open
  +2.9% [-3.1, +9.0]. No-vig at the close +3.5% [-2.4, +9.4]. Model B agrees with the side 60% of the time, and has
  its own >= 2-pt bet on the same side 31% of the time.
- Incremental vs the CLOSE: OE +0.78/SD given the model-close gap (p .0027). BD -0.01 (p .51).
- CLV slopes on the whole sample: OE +0.10 pts/SD (p .027). BD -0.23/SD, meaning totals rise from open to close in
  bad-D games (wrong way for unders).

## Power and caveats
- **Same data as the discovery.** Both leads were found on 2006-19 against the close. L03/L08 and the vs-close slopes
  are not independent replications. The opener questions are new, but they use the same games.
- **Feature changed on purpose.** Per the brief, this uses the simple model's *shrunk* pass/run EPA, averaged, and
  z-scored within each week. The comebacks study used unshrunk per-play EPA. The unshrunk BD version was not re-run
  (doing so would be re-mining the discovery data). So the claim "BD is dead" is about the shrunk feature. The
  unshrunk -0.79 is a single weak cell that failed correction there too.
- Under-power: BD has 214 bets. To detect a 3% ROI at -110 would take thousands. The point estimates are negative,
  though, so power is not what is hiding an effect. OE rule: the observed -1.6% has no power story. Even the no-vig
  +3% would need roughly 5,000+ bets to pass this bar.
- SBR "open" has an unknown book and time (see `nfl_simple_model.md`). Spreads and totals at the open are assumed
  -110. No-vig prices exist only at the close (nflverse odds, proportional de-vig).
- The bar is per family (10 trials). Against the whole ledger (237 rows, bar .05/237 = .00021), even L09 (p .0003)
  does **not** pass, so treat it as "strong within this family" only.

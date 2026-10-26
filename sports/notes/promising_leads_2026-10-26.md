# Promising leads, ranked (user 2026-10-26: "even close to 0.05 is good enough, just give me whatever has promise")

Caveat: ~300 trials in lab/ledger.jsonl -> ~15 hit p < .05 by chance. Tier 1 is proven; tier 2 = repeated across
seasons/sports or a near-miss sealed test; tier 3 = single dev result near .05. Size accordingly.

## Tier 1 -- proven (sealed pass)
| Lead | Evidence | Action |
|---|---|---|
| E1/E2 Kalshi weather consensus-reject | sealed +1.45% / +0.78%, live 20-0 | running live ($1 units) |
| E4 same, as MAKER | sealed ROI +1.07% -> +2.22%, p < .001 | add KALSHI_EXEC_STYLE=maker to ~/.claude/quant-kalshi/.env |

## Tier 2 -- strong, repeated, or sealed near-miss
| Lead | Evidence | Action |
|---|---|---|
| L32 NBA totals vs OPENER (model >= 3 pts off first-posted total) | dev +8.7% no-vig p .001; SEALED +6.4% p .054, CLV +1.2 pts, both books positive | bet the FIRST-POSTED total only (Kalshi ladder / earliest book); taker fee ~1.75c at 50c eats ~3.5% -> maker or early book; logger flags |
| L36 CFB totals vs opener | +7.4% no-vig p .0004 (3,014 bets, both halves); -110 +2.6% p .10 | same execution story; build CFB logger |
| L34 NHL totals vs opener | +7.1% no-vig p .0125; ~half is a plain UNDER lean (blind unders at opening 6.5: +5.8% no-vig) | NHL logger running; simplest form: unders at opening 6.5 |
| L33 NFL totals model C vs opener | +6.6% at -110 p .014 (dev 2010-19); CLV +0.81 pts | NFL logger running |
| Cross-sport CLV | every simple model moves the opener toward itself (NBA/NFL/NHL/CFB, p < 1/4000) | the information is real; the money is only at the opener |

## Tier 3 -- single results near .05 (paper or tiny stakes)
| Lead | Evidence |
|---|---|
| L35 Kalshi early NFL game prices | stale: CLV +3.3c p < .0002; maker at first bid +16% p .09 (n 69, adverse-selection risk); taker at first TIGHT quote untested |
| NFL QB-change weeks | model B spread CLV +1.15 pts (210 bets) |
| NFL rain unders pre-kickoff | +8% p .11 (live_signals forward) |
| NFL rush-yard unders, low-total games (total <= 40.5) | +7.7 pts vs de-vig, ROI +11%, p .039 (n 155, 2024 only) |
| NFL big-dog non-QB rush-yard unders (spread >= +6.5) | ROI +12%, p .078 (n 99) |
| L31 NBA 3P% regression -> totals | 53.6% / 1,417 bets, +2.3% at -110 (inside L32 model) |
| NBA altitude home teams | +1.0 pt ATS p .055 |
| Kalshi trailing-longshot overpricing (in-game) | trailer mids ~1.5c above realised in 0-15c bins (sealed descriptive only) |
| Weather maker NO at 99c (tape subset) | 275 fills, 0 losses (~1c/contract; descriptive) |
| E3 soccer soft books vs Pinnacle | sealed pass +8.7% -- NOT tradeable from CA |

# Edge report — 2026-10-25 ~09:30Z (pulled live this morning)

## Real money — Kalshi weather consensus-reject (E1 + E2), $1 units, LIVE since 10-22
- 36 decisions, 29 filled, 14 settled, **14/14 wins, P&L +$0.23** (paper-at-pick-price +$0.11 -> fills at/better than paper).
- Exchange balance $35.49 + 15 open NO positions ($14.74 at cost): 6 from 10-24 (MIA, PHIL, ATL, TTN, CHI, AUS),
  9 from 10-25. They settle once the NWS climate report posts.
- Caps bind: 6 skip_cap. The uncapped dry shadow took 15/15 settled wins, +$14.93 at $10-25 sizes.
- Most fills are NO @ $0.99, so a win pays 1c. **E4 MAKER (sealed pass, ROI +1.07% -> +2.22%) is still OFF:**
  append `KALSHI_EXEC_STYLE=maker` to ~/.claude/quant-kalshi/.env to switch it on.
- Only incident: one transient ConnectionError at 10-24 01:04Z, which self-healed.

## Paper — same rule, 24 cities (hourly)
- 90 picks, 69 settled, **68 wins, +0.53u** (orig7 +0.80u on 28 settled; X1 -0.26u on 41 settled, which includes the one loss).
  Median depth within 1c: 228 contracts, so size is not a constraint at our level.

## E3 soccer soft books vs Pinnacle (sealed pass, CLV +3.6%)
- Not tradeable from CA (B365/WH/bwin not licensed here). No change.

## Forward loggers
- NFL pre-kickoff rain unders (lead, +8% p .11): 9 games evaluated, **0 fired**. Today's slate is being checked.
- Kalshi-vs-Pinnacle gap logger: 2,338 ticks, 291 episodes. They are dominated by soccer (LaLiga 153, UCL 71, EFL 22);
  the median episode lasts 0 min (seen on one tick). NBA 6 and NFL 7 episodes, median EV 1.5-1.9%, which is below our 2c-after-fee bar.
  Reading: the US sports markets on Kalshi track Pinnacle; the soccer "edges" are a 3-way de-vig artefact until shown otherwise.
- F1 NFL in-game maker-fade: first interim Tue 10-27 (descriptive only).

## Thesis tracker (stocks, forward since 10-22; 1-2 days = noise)
- v1: inference +1.37% vs training -0.14% (SMH +0.74%, SPY -0.52%).
- v2_all -0.38% (memory -0.17, compute-inference +1.35, energy-gen -0.92, energy-equip -1.76).

## Dead / failed (don't bet these)
O1 NBA ridge model, Polymarket earnings, graph momentum, Kalshi mention-NO, NBA rebound unders, Kalshi favourites,
NBA big leads as a pregame-style rule (priced), referee totals, parlays, Kalshi-vs-Polymarket arbs.

## Last night
- DEN 141-135 NO (OT): DEN trailed 30-53 at Q2 7:00 (max deficit 23). From nba_big_lead.md, a 20+ lead is blown 4.9% of
  the time, and a Q2-Q3 first 20-point lead holds 94.1%. A 23-point Q2 comeback is roughly a 1-in-15 to 1-in-20 event.
  It's a good win, but not evidence of an edge by itself.

# Tonight 2026-10-26: information-edge sheet (MNF DAL @ PHI + NHL, 3 games)

Snapshot: 2026-10-26 ~16:35Z. PAPER ONLY. No orders were placed and nothing was committed.
Scratch scripts and raw JSON: `/Users/jonathanwang/.claude/jobs/588c8b26/tmp/`.

The brief was to find what the market might be missing, not to read off a model. Model numbers appear only as a
cross-check. Every lean below is a judgement call. None of them is a tested edge.

## Sources (and what failed)

| Source | Status |
|---|---|
| ESPN scoreboard and summary | OK. The pickcenter has **DraftKings only** for all 4 games, so there is no per-book opener comparison. |
| ESPN injuries, depth chart, news, box scores | OK |
| Pinnacle guest API (`sports/live.py`) | OK |
| Kalshi (`KXNFLGAME/SPREAD/TOTAL`, `KXNHLGAME/TOTAL`) | OK |
| Open-Meteo hourly for Lincoln Financial Field | OK |
| NHL api-web (gamecenter landing, boxscores, club schedule) | OK |
| NHL stats REST | OK only with a browser User-Agent. The plain `requests` UA got a non-JSON body twice. |
| `open_lines_logger.py show` (nhl and nfl) | Printed **"no games"**. The first-seen Pinnacle opener is not available from our own logger. Unverified why; the logger may not be ticking (possibly the launchd/TCC Desktop issue). Worth checking. |

There is no keyless probable-goalie feed. The NHL landing pages list season goalie stats, not starters. Goalies
below are inferred from the starter sequence in the last 4 boxscores.

---

## MNF: DAL @ PHI, 00:15Z (8:15 pm ET)

### QB situation

- **Jalen Hurts is out.** Evidence:
  - ESPN depth chart (fetched 16:35Z) lists QB as McKee, Payton, Milton, Hurts.
  - ESPN headlines: "Eagles still trying to salvage NFC East crown without Hurts" (10-23 00:03Z); "NFL, NFLPA
    handling joint review of Jalen Hurts' blood clot diagnosis" (10-25 14:14Z).
  - Oddity: Hurts is **not on the ESPN injury feed at all**, so his formal status (IR or not) is unverified from
    that feed.
- **Tanner McKee already started last game** (10-18 vs CAR, L 14-25).
  - Line: 26/45, 215 yd, 4.8 Y/A, 1 TD, 3 sacks.
  - nflverse: −0.18 EPA/dropback on 48 plays.
  - So this is **not a surprise backup week**. PHI has had 8 days of first-team reps with McKee.

### Lines

| Line | DK open | DK now | Pinnacle (no-vig) | Kalshi |
|---|---|---|---|---|
| Spread | PHI −3 | **DAL −4.5** | DAL −4.5 / PHI +4.5 at −110/+100 (PHI +4.5 = 51.2%) | DAL by >4.5: .48/.49 (fair) |
| Total | 48.5 | **44.5** (u −118) | 44, over 50.9% | o43.5 .52/.53, o44.5 .48/.49 (fair) |
| ML | PHI −162 / DAL +136 | DAL −205 / PHI +170 | DAL 67.4% | DAL .67/.68, PHI .33/.34 (fair) |
| Team totals | — | — | DAL 24.5 (over 46.7%), PHI 19.5 (over 51.1%) | — |

- The opener swung 7.5 points on the spread and 4 on the total.
- The timing of DK's open is unknown. It looks like a look-ahead line posted before the Hurts news.
- Kalshi sits exactly on Pinnacle everywhere: no stale price, and the taker fee kills anything.

### Weather, rest and injuries

- **Weather at kickoff** (Open-Meteo, 00:00-02:00Z): 0.0 mm precipitation, 0% probability, wind 4.5-5 mph
  (gusts about 10), 58°F. **The rain rule does NOT fire.** ESPN/AccuWeather agrees: 60°F, 0 precip.
- **Rest:** DAL last played MNF 10-19 @GB (7 days). PHI last played 10-18 vs CAR (8 days). Neutral.
- **Injuries (ESPN feed plus news):**

| Team | Player | Status |
|---|---|---|
| PHI | DeVonta Smith (WR1/2) | Out (ESPN headline 10-24) |
| PHI | Dontayvion Wicks | Out |
| PHI | Saquon Barkley | **Questionable, hamstring** (updated 10-26 13:22Z, so watch the inactives at about 22:45Z) |
| PHI | G Landon Dickerson | IR |
| PHI | TE Eli Stowers | IR |
| PHI | Edge/LB Nolan Smith | Out |
| PHI | CB Michael Carter II | Out |
| DAL | LT Drew Shelton | Out |
| DAL | KaVontae Turpin | Out |
| DAL | LB Dee Winters, LB Marist Liufau (IR), S P.J. Locke (IR) | Out |

### Season EPA, 2026 weeks 1-7 (nflverse pbp, through 10-25)

| Unit | EPA/play | Rank | Pass | Rush |
|---|---|---|---|---|
| PHI offence | −0.111 | 30th | −0.060 (27th) | −0.210 (28th) |
| DAL defence | +0.154 | **32nd** | **+0.327 (32nd)** | 23rd |
| DAL offence | +0.106 | 6th | +0.165 (8th) | 9th |
| PHI defence | +0.062 | 25th | 23rd | 31st |

- Hurts in 2026: −0.08 EPA on pass plays (167 plays), against a league pass average of +0.08. He was a
  below-average passer this year before the injury.
- Trends:
  - DAL offence in its last 2 games: −0.03, −0.11.
  - DAL defence in its last 2 games: 0.148, 0.033 allowed (improving).

### What the market may be missing

- The market has priced "Hurts out" as a full star-QB loss (a 7.5-point swing from the opener).
- But 2026 Hurts was already a bottom-10 passer (QBR 36-40 in his last 3 starts). So the real downgrade to McKee is
  smaller than the name suggests.
- McKee also gets the worst pass defence in the league (+0.33 EPA/dropback allowed).
- Against that, PHI's skill group is gutted: Smith and Wicks out, Barkley questionable, TE and LG on IR.
- **Lean: PHI +4.5 (it protects 3 and 4). Confidence LOW.**
  - Small size or a pass. Do not bet it if Barkley is inactive.
- **Second lean: DAL team total over 24.5 (+106).**
  - Reason: DAL pass offence (8th) vs a PHI pass defence missing its edge (N. Smith) and slot CB (Carter).
  - Confidence LOW, because DAL's offence has cooled in its last 2 games.
- **Cross-check (NFL simple model, `week --refresh`, modern fit, QB = McKee):**
  - Spread B: DAL −3.0. C: DAL −3.7. A, which has no QB term: DAL −0.3.
  - Total C: 44.4, equal to the market. So there is no totals view.
  - The model is 1.5 points on PHI's side. Our dev "QB-change week" CLV lead (+1.15 points) was measured against
    the opener, and here the opener is already stale (pre-news). So it does not apply cleanly.

---

## NHL (all three games: every team last played Saturday 10-24 local, so NO back-to-backs)

> **Trap avoided:** ESPN's `lastFiveGames` dates are UTC, which shows CGY and DET as playing "10-25". The NHL API
> local game date is **10-24**. Nobody is on a back-to-back tonight.

### Goalies

These are inferred from the last-4 starter sequence. They are NOT confirmed. Re-check the gamecenter boxscore or
the starter announcements about 60-90 min before puck drop.

| Team | Recent starts (latest last) | Likely tonight | Season Sv% | Note |
|---|---|---|---|---|
| NYR | Garand ×4 | **Garand** | .900 | Shesterkin IR (return 2027-09-15, i.e. season), Korpisalo IR |
| LAK | Forsberg, Kuemper, Kuemper, Forsberg | Kuemper or Forsberg (coin flip) | .890 / .894 | Last game of a 4-game eastern trip |
| TOR | Bobrovsky / Stolarz strict alternation | **Bobrovsky** by pattern | .895 (Stolarz .907) | |
| CGY | Wolf, Cooley ×3 | **Cooley** probable | .912 (Wolf **.834**) | If Wolf starts, TOR value |
| EDM | Jarry, Levi, Levi, Jarry | **Jarry** (Levi IR 10-24) | **.861** | Our model projected Levi, which is wrong |
| DET | Gibson, Tarasov, Gibson, Gibson | **Gibson** (Tarasov day-to-day) | **.867** | |

### PDO, 2026-27 to date

Shrinkage: Sh% and Sv% are shrunk with 1,500 league-average shots. This is an ad hoc prior. League Sh% is 11.4%.

| Team | Sh% | Sv% | PDO | GF/g actual → shrunk | GA/g actual → shrunk |
|---|---|---|---|---|---|
| EDM | **17.1%** | .869 | **1.040** | 4.36 → 3.13 (**+1.23 luck**) | 3.55 → 3.15 |
| TOR | 13.4% | .897 | 1.032 | 3.85 → 3.38 | 3.08 → 3.35 |
| NYR | 12.8% | .895 | 1.023 | 3.15 → 2.86 | 2.77 → 2.95 |
| DET | 8.5% | .886 | 0.971 | 2.55 → 3.24 | 2.91 → 2.91 |
| LAK | 8.6% | .878 | 0.964 | 2.33 → 2.99 | 3.56 → 3.35 |
| CGY | **8.0%** | .872 | **0.952** | 2.30 → 3.11 | 3.90 → 3.55 (Wolf-inflated) |

Caveat: EDM's true talent Sh% is above league average (McDavid and Draisaitl). Shrinking to the league mean
overstates the luck.

### Lines

| Game | DK open → now | Pinnacle no-vig | Kalshi | Model (cross-check) |
|---|---|---|---|---|
| LAK @ NYR 23:00Z | ML NYR −135 → −130. Total 5.5 (o −125 → −130) | NYR 54.4%. Total 6.0, over 47.7% | NYR .55/.56, LA .44/.45 | p_home .526, E 5.91 |
| TOR @ CGY 00:30Z | ML TOR −135 → **−155**. **Total 6.5** (o −112 → −120) | TOR 58.0%. 6.5 over 52.4% | TOR .56/.57, CGY .42/.43. o6.5 .53/.54 | p_home **.479**, E 6.55 |
| DET @ EDM 01:30Z | ML EDM −185 → −180. **Total 6.5** (o −112 → −125) | EDM 62.7%. 6.5 over **55.2%** | EDM .61/.62, DET .38/.39. o6.5 .54/.55 | p_home .538, E 6.22 (used Levi) |

- Kalshi is within about 1c of Pinnacle fair on every NHL market. After the fee, there is no Kalshi edge.

### Leans

- **DET @ EDM: UNDER 6.5, now +117 at Pinnacle (+105 DK). Confidence LOW-MOD.**
  - It opened at 6.5, which is our tier-2 opening-6.5 under lead (blind unders +5.8% no-vig in dev). The under
    price has improved since the open (−108 → +105/+117).
  - The over juice leans on EDM's 17% shooting. Shrunk shot-based rates put the total around 6.2-6.5, not the
    roughly 6.8 that 55% over implies.
  - Risk: both goalies are genuinely bad (.861 and .867), which is a real over factor.
- **TOR @ CGY: CGY ML +132 (Pinnacle), CONDITIONAL on Cooley starting. Confidence LOW.**
  - The market moved 4-5 points toward TOR. Possible readings: it expects Wolf, or it is reacting to TOR's 7-2 win at
    EDM.
  - CGY is the most PDO-unlucky team here (8.0% Sh%). TOR is lucky (1.032).
  - The model has CGY at .479 vs the market's .42.
  - If Wolf is announced: pass, or flip to TOR.
- **TOR @ CGY: under 6.5 (+104).** Opened at 6.5, so it meets the opening-6.5 rule. The PDO effects net out to
  about zero. Confidence LOW (rule-only).
- **LAK @ NYR: no bet (lean LAK +114, very low).**
  - LAK's shooting is unlucky (8.6%), but the team is at the end of a 4-game trip.
  - The NYR without Shesterkin is already priced.
  - The total opened at 5.5, so the 6.5 rule does not apply.

---

## Side note (new information for L34)

Kalshi already lists **tomorrow's (10-27) NHL total ladders and game markets**. Examples: BOS-CAR o6.5 .46/.48;
MIN-UTA o6.5 .47/.48.

Most rungs have zero volume and wide quotes. So the claim in `nhl_simple_model.md` that a Kalshi early NHL listing
was "UNVERIFIED" is now resolved: they do list about 1 day ahead. That is the low-vig execution route L34 needs, so
a maker test at the first tight quote is possible.

# T08 — Arbitrage, middling and low-hold opportunities across US books and exchanges/prediction markets

Agent: research-T08. 10 registry entries added (S001-S010 in `lab/registry/incoming/T08.jsonl`).

## What I looked for

The topic bundles several related but distinct ideas and I tried to keep them separate
rather than collapse them into one vague "arbitrage exists" claim:

1. Pure cross-book price arbitrage (classic "surebet" — bet both sides, guaranteed profit).
2. Middling (bet the same side at two different numbers, win both on a lucky landing spot).
3. Structurally low-hold venues (exchanges / prediction markets) as a venue choice rather
   than a mispricing to exploit.
4. The two things that actually determine whether any of the above is realizable at
   scale: **account/position limiting** (sportsbooks) and **order-book depth** (exchanges).
5. Promotional/free-bet arbitrage (matched betting), which is mechanically different from
   odds arbitrage (it monetizes a bonus, not a price gap) but shares the "hedge to
   guarantee an outcome" structure.

Search order was peer-reviewed/working papers first, then practitioner data with a real
record, then bare claims, per standing orders. Several URLs I found (arxiv.org,
core.ac.uk, uni-muenster.de, dialnet.unirioja.es, karlwhelan.com) were blocked by this
environment's network egress proxy, so a few entries (S005, S006, S009) are built from
search-result snippets/press coverage describing those papers rather than a direct read
of the primary text — this is flagged explicitly in each entry's `skeptic_note`. A human
or a later agent with unblocked access should verify those before leaning on exact
figures.

## Most promising entries

**S008 (testable now, `mlb_k_props`)** is the standout because it's the only entry that
didn't require new data collection to check: the lab's own `mlb_k_props.parquet` already
stores best-of-book over/under strikeout prices with separate `over_book`/`under_book`
columns. I measured directly (not from an external source) that `over_book != under_book`
in 84% of rows, the mean combined implied probability is 103.9% (~3.9% synthetic hold),
and 2.06% of rows have combined implied probability strictly below 100% — i.e. a
mechanical dutch-book condition existed in the historical closing-ish snapshot about 1 in
48 times. This is real and testable immediately with existing data, but I was careful to
grade it as `practitioner_data` with a pointed `skeptic_note`: the schema doesn't confirm
both best prices were live at the *same instant*, so this measures a historical base rate
of price dispersion, not a proven-fillable arbitrage. A backtest agent could extend this
by computing what fraction of the implied "profit" from those 2% of rows survives
realistic stake limits — this is the single most actionable next step from this research
pass.

**S002 and S010** are the two most important entries for the "execution risk, account
limits" half of the topic, and I'd flag them as the most valuable to read even though
neither is directly testable with current datasets. S002 (peer-reviewed, Grant et al.
2018, EJF) shows that in a real academic sample, roughly half of on-paper cross-bookmaker
arbitrage portfolios required betting at a bookmaker type shown to actively restrict
informed bettors — i.e., the realizable arbitrage rate is roughly half the nominal rate
once you account for who will actually take the bet. S010 shows the exchange-side mirror
image: a prediction-market venue's headline low margin is frequently backed by only a few
dollars of fillable depth, so "exchanges have lower hold" does not imply "exchanges have
exploitable size." Any strategy agent asked to size a backtest on this topic should treat
both of these as hard caps on any theoretical edge computed from top-of-book prices alone.

**S001 (peer-reviewed, Franck/Verbeek/Nüesch 2013)** is the best-sourced estimate of raw
opportunity frequency (19.2% of matches had a guaranteed-positive bookmaker+exchange
combination), but it predates most of the current US market structure (Kalshi, Polymarket,
Novig and ProphetX didn't exist in anything like their current form) and predates the
general tightening of margins documented in S006, so its magnitude should not be assumed
to transfer to 2026 US sports markets.

## What would unlock the best untestable entries

- **S001/S005/S006/S010 (cross-venue arbitrage)**: need simultaneous, timestamped
  odds/prices from at least two independent venues for the same game/market — ideally one
  traditional sportsbook plus one exchange (Novig/ProphetX/Betfair) or one prediction
  market (Kalshi/Polymarket), at second-level granularity, with order-book depth at each
  price tier, not just the best price. None of the four current lab datasets (`nfl_spread`,
  `nfl_total`, `nfl_moneyline`, `mlb_k_props`) carry more than one book's price per row for
  a given side, so none of this is testable without new data ingestion.
- **S002/S003/S004 (account limiting)**: need account-level data from inside a
  sportsbook's or exchange's own risk system (stake-limit history, flagging reasons) —
  this is operator-internal and I did not find any public dataset that exposes it, beyond
  the self-reported aggregate figures in S003 from Massachusetts' new disclosure rules.
- **S007 (middling)**: needs two distinct books' point-spread/total numbers for the same
  game (not just one closing line), plus the final score, to compute how often results
  land inside a given middle window. No current dataset carries a second book's line.
- **S009 (free-bet arbitrage)**: needs promo terms (stake-returned vs not, minimum odds)
  matched to odds at the time of the promo — not something any current dataset has, and
  not really a "mispricing" test so much as a mechanical-hedging-value calculation.

## Honest gaps

- I found no peer-reviewed study that measures arbitrage/middling frequency specifically
  for **US** sportsbooks or for **Kalshi/Polymarket/Novig/ProphetX** by name — essentially
  everything with real academic rigor on frequency (S001, S002, S006) is European soccer,
  pre-2019, studied against Betfair rather than the current US exchange set. The one paper
  that does cover Kalshi/Polymarket directly (S005) is an unreviewed 2025 preprint.
- "Low-hold opportunities" as its own idea (just betting at the lowest-margin venue,
  without arbitraging anything) is well documented as a real, quantifiable price
  difference (S010's 0.49-3.22% margin spread) but I found no study of whether simply
  always betting at the lowest-displayed-margin venue beats betting at a standard -110
  book once you account for fillable size and the fact that the cheapest venue changes
  game to game.
- Regulatory risk (Kalshi/Polymarket facing cease-and-desist orders from Tennessee,
  Nevada, New Jersey, Illinois, Maryland, Ohio, Montana and New York as of late 2025) is a
  real, material risk to this entire topic's venue durability, but it isn't a "strategy or
  bias" in the registry's sense (it doesn't explain a mispricing), so I did not add a
  registry row for it — noting it here instead as context any backtest or strategy agent
  on this topic should be aware of: the venues this topic depends on may not remain
  accessible in all US states.

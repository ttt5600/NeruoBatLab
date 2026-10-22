# Company graph: customer links, earnings propagation, graph momentum (2026-10-22) -- scripts/company_graph.py

User: "Model market twin stocks with a graph model on companies, translating relations into better options. Model
relationships and build a thesis we can market a model off of."

**Verdict: no marketable alpha model. 6 pre-registered trials, 0 pass** (Bonferroni bar p < 0.0083; none reaches
even uncorrected p < .05). The customer-momentum effect is still there gross (+0.24%/mo), but the strategy turns over
315% a month and costs eat it (net -0.23%/mo). The best result is graph-smoothed momentum over plain momentum:
+0.28%/mo, p .061. It shrank in each sub-period (+0.57 -> +0.21 -> +0.07%/mo), the decay pattern the literature
predicts. What we did build is a free, point-in-time customer-supplier graph for 2001-2026, about 96% precise on a
60-filing audit. That graph is the thing that could be sold, as a data and risk product. It is not an alpha signal.

Literature detail: notes/company_graph_lit.md (agent report; sections 4-5 thin, supplemented below).

## 1. Literature (latest first; every number from a fetched page or a search result, see lit file for URLs)
| line of work | claimed effect | period | costs? | post-publication |
|---|---|---|---|---|
| Chen & Welch 2026 "What Useful Alphas?" (arXiv 2607.06502) | median OSAP anomaly 48bp/mo pre-2006 -> 19bp post-2005; 7bp post-2005 non-microcap | to recent | "modest" costs eliminate it | decayed |
| LLM supply-chain propagation, Yilki Jun-2026 (arXiv 2606.29290) | FinBERT 10-K MD&A embeddings propagated over links: FF5 alpha 7.27%/yr (t 2.30), Sharpe 0.86, 255 S&P firms | 2011-2025 | not in abstract | new; 1 paper, unreplicated (search snippet, full text not read) |
| LLM point-in-time graph, Yang & Zhang Jul-2026 (arXiv 2607.15640) | 149 edges among 42 Nasdaq-100 firms, 88% audit precision | n/a | none | no return test at all |
| Pinchuk 2023 replication of customer momentum (arXiv 2301.11394) | EW 1.52%/mo pre-2005 -> **0.62%/mo (t 2.12) 2005-18; VW 0.62% (t 1.54)** | 1978-2018 | not discussed | **-55..-60%** |
| Chen & Velikov 2023 JFQA | average anomaly ~4bp/mo net of spreads + decay | 204 anomalies | yes | yes |
| Ali & Hirshleifer 2020 JFE, shared analyst coverage | 1.68%/mo alpha (t 9.67) 1984-2015; 1.13% in 2000-15; subsumes customer/tech/industry momentum | 1984-2015 | none found | declining; needs paid I/B/E/S |
| Lee, Sun, Wang & Zhang 2019 JFE, tech links | EW 117bp, VW 69bp/mo | 1963-2012 | none | no post-2012 test found; subsumed by A&H |
| GNNs (RSR, Feng et al. 2019 TOIS; OmniGNN 2025 arXiv 2510.10775; JSE study) | RSR "98% return" in a 1-year test (2017) | one year | ignored | RSR relations come from a 2018 Wikidata dump used to trade 2017 = look-ahead. JSE study: prior graph structure had "negligible impact". **No systematic replication critique found** (searched; not claimed to exist) |
| Cohen & Frazzini 2008 JF | 1.58%/mo VW L/S | 1980-2004 | no | see Pinchuk |
| McLean & Pontiff 2016 JF | anomalies -58% post-publication | 97 anomalies | - | benchmark |

Taken together, the literature says customer, tech and analyst-link momentum were real before about 2005, and roughly
half of each effect was gone after publication. No paper shows a cost-surviving version after 2015. The GNN and LLM-graph
papers either have no costs or out-of-sample test, or (RSR) build the graph with look-ahead.

## 2. The graph we built (free data, point-in-time)
- **Customer edges.** Sources: EDGAR full-text search (efts.sec.gov, which covers 2001 onward), 10-K/10-K405/10-KSB filings,
  12 exact phrases per customer name ("sales to {N}", "{N} accounted for", "revenues from {N}", ...). The customer list is the
  S&P 500 as of 2026-10-22 plus 21 large non-S&P customers, with dated aliases (e.g. Raytheon -> RTX only from 2020-04-03).
  Each edge's `first_known` is the filing date. 5,964 queries.
  -> **12,917 edge-filings, 3,357 supplier-customer pairs, 2,300 suppliers, 294 customers.**
- **Precision audit.** I fetched 60 random edge filings and read each matching sentence (data/company_graph/edge_precision_sample.csv).
  54 of the 56 that could be checked are genuine economic links: a customer, distributor, licensor, broker or tenant. Both
  misses came from "target" matching case-insensitively. **All 458 Target filings were then re-verified** with a
  case-sensitive check plus a requirement for a percentage in the same sentence; 283 were kept. Recall is below 100%,
  because table-only disclosures are missed.
- **Survivorship.** Only **679 of 2,300 supplier CIKs have a ticker today**, so about 70% of linked suppliers (dead,
  acquired or delisted) cannot be priced on yfinance. The tests use survivors only.
- **Other edge types (export only).** `industry_sic4` uses the current SIC (submissions API, so this is NOT point-in-time;
  first_known is set to 2026-10-22). `comove_top5` is each node's 5 highest-correlation partners per calendar year, with
  first_known = last trading day of that year.
- **Export:** `data/company_graph/edges.parquet` (1.5 MB). 12,917 customer + 12,586 industry + 98,410 comove rows.
  Columns: edge_type, src_ticker/cik/name (src = supplier), dst_ticker/cik, first_known, valid_until (+456 days),
  weight (number of phrases matched), adsh (accession), evidence. 59% of customer rows have no current src_ticker.
- **Earnings dates.** 8-K item 2.02 filings for all 1,027 priced nodes: 55,405 events, from 2004-08 on.
  **DATA TRAP (new):** the `acceptanceDateTime` field in the EDGAR submissions JSON is neither ET nor plain UTC. JPM's
  06:3x ET releases appear as 14:30Z (EDT) and 16:41Z (EST); WMT's 06:59 ET as 14:59Z and 16:59Z; AAPL's 16:30 ET as 00:30Z
  the next day. Converting UTC to ET *twice* matches filingDate in 92-98% of filings each year; the raw value matches
  only 66-75%. A residual bump at 11-12h ET looks like true-UTC filings, so day0 is assigned conservatively: same day only
  if the converted time is before 09:00.

## 3. Pre-registered tests (DEV 2002-2022; sealed 2023-01-03..2026-10-21 untouched)
Rules are in the script docstring, written before any price was downloaded. Universe filter: adj close >= $2 and
60-day median $ volume >= $1M. Costs are tiered 0.5-25bp per side by $ADV; borrow is 0.3% or 2%/yr. Amendments made
before scoring are listed in the docstring: Key/Lab dropped, AT&T dated, Target verified, ET conversion added.

| trial | what | gross | net | p (one-sided) | notes |
|---|---|---|---|---|---|
| A1 | customer momentum, quintile L/S, monthly | +0.24%/mo | **-0.23%/mo** | .76 (gross .23) | 46 names, turnover 315%/mo; gross by period +0.17 / +0.08 / +0.47 %/mo |
| A2 | Fama-MacBeth slope on customer-return rank, controls: own reversal, 12-1 mom, SIC2 industry return | +0.39%/mo per rank unit | (gross) | .20 | slope by period -0.13 / +0.05 / +1.18 |
| B1 | customer reports, \|move\| >= 2% -> trade linked suppliers that haven't reported yet, hold through their report | -0.41%/yr | -2.8%/yr | .89 | 4,895 sleeves, 29-day hold, sleeve gross -0.13% |
| B2 | supplier reports -> its customers | +0.08%/yr | -0.40%/yr | .70 | 7,687 sleeves |
| B3 | same SIC-4 peers (industry information transfer) | +0.23%/yr | -1.35%/yr | .9994 | 179,490 sleeves, sleeve gross +0.055% |
| C1 | graph-smoothed 12-1 momentum minus plain 12-1 momentum (same names) | +0.27%/mo diff | **+0.28%/mo diff** | **.061** | graph LS net +0.06%/mo vs plain -0.22%/mo; diff by period +0.57 / +0.21 / +0.07 |

Ledger: 6 lines `"source": "company_graph (user 2026-10-22)"` in lab/ledger.jsonl.
Stock-side trial count is now about 68 (stocks_bt + pairs_bt + pairs_twins + this), with 0 passes.

Description only:
- A1 without the liquidity filter has gross -3.0%/mo, entirely from one bad microcap print in 2020-11 (-789% in a
  month). Excluding that month gives +0.11%/mo. This shows yfinance microcap data cannot be trusted, and that is where
  the published effect lives.
- A1's gross with the top/bottom 2 months trimmed is +0.10%/mo.

## 4. How to read it
1. **Customer momentum survives gross but not net.** +0.24%/mo is about 40% of Pinchuk's post-2005 0.62%, which fits
   further decay, a survivor-only universe, and the liquidity filter. The cost is structural: a 1-month signal re-sorts
   the quintiles every month (315% turnover). That costs about 0.47%/mo even at our mid-cap tiers.
2. **Earnings propagation is dead.** Trading linked firms that haven't reported yet earns ~0 gross in every direction.
   Customer -> supplier is slightly *negative*. The market prices the linked-firm news by the next close, when we can
   first trade it.
3. **Graph momentum is the only near-miss, and it is fading.** Blending a firm's own 12-1 momentum with its partners'
   momentum beats plain momentum on the same names by +0.28%/mo (p .061). The gain shrinks across the three periods
   (+0.57 -> +0.21 -> +0.07%/mo). Plain momentum on these names lost money net in 2002-22 (2009 momentum crash included),
   so the "improvement" is partly the graph version avoiding a loss. It does not clear the uncorrected bar, let alone
   the Bonferroni bar of 0.0083.
4. **The GNN option (task 3c) was not built** beyond this linear propagation. The linear graph signal is the baseline a
   GNN must beat (lit section 4). It is already sub-significant and decaying, and a GNN on ~100-170 linked names per
   month would mostly fit noise. I judged it not worth a 7th trial.

## 5. The "thesis" question: is there something to market?
- **Not an alpha model.** I would not pitch "graph model beats the market". The honest backtest says 0/6, and the
  literature says the effect has roughly halved since 2005.
- **What is defensible to market is the data asset and a risk view, not returns.**
  - The pitch: "a free-sourced, point-in-time supply-chain graph from SEC filings, 2001-2026, with filing-date
    timestamps (no look-ahead), ~96% audited precision, refreshed as 10-Ks land."
  - Uses: exposure and contagion maps (which small suppliers depend on Walmart, Apple or Boeing for 20-60% of
    revenue), event-risk screens ("customer X just guided down -> list the suppliers that report in the next 30 days"),
    and a clean input for anyone who wants to test LLM or GNN graph claims without the RSR look-ahead.
  - Comparable paid products exist: Bloomberg SPLC and FactSet Revere (named in the lit file; their prices are not
    checked here).
- **Do not call it an options-trading edge.** "Translating relations into better options" would need implied-vol
  history for the linked names; I have no free source for it. B1's null result means linked-firm news is priced by the
  next close in the stock, so I would not expect options to be slower. That is a guess and has not been tested.
- **Sealed (2023+) is unscored.** Nothing passed dev, so under the protocol nothing is promoted. If the user still
  wants a one-shot check, C1 is the only candidate. Its dev decay makes a pass unlikely; it needs the user's explicit "yes".

## 6. Caveats
- Survivorship: about 70% of linked suppliers are unpriceable. Customers are today's S&P 500 plus 21 extras, so
  customers that went bankrupt (Sears, Circuit City, Kmart) are missing. The bias for A1 is probably against us (dead
  suppliers would mostly have sat in the short leg), but this is not measured.
- Recall: the phrase search misses table-only disclosures and customers outside the list.
- `adj close >= $2` uses dividend-adjusted prices, so it is slightly stricter early in the sample.
- B3 uses current SIC codes (mild look-ahead). It still lost.
- Cost tiers are estimates, the same as in pairs_twins.py.

## Reproduce
`../.venv_quant/bin/python scripts/company_graph.py edges | verify_target | prices | earn | data | sample | dev | export`
(all cached under data/company_graph/, 59 MB). The crawl takes about 75 min at ~1.3 queries/s; the 8-K fetch about 15 min.

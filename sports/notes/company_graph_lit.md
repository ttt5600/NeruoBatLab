# Company-graph return predictability: literature review (2026-10-22)

Scope: does trading on links between companies (customer-supplier, technology, shared
analysts, learned or LLM-extracted graphs) give a live edge after costs? Every number
below was read from a page fetched or a search-result snippet seen during this review.
Anything not checked against a source is marked **unverified**.

**Limitations of this review.** The session's web-search budget ran out partway through.
Topics 4 (GNN replication critiques) and 5 (LLM-extracted graphs) are therefore thin.
For both, I fetched a few papers whose URLs I already knew, and I could not run broad
searches. Several primary PDFs were unreachable (the Yale server refused the connection;
Alpha Architect returned 403). Those numbers come from abstracts or secondary pages, and
the tables say so.

---

## 1. Customer-supplier links ("customer momentum")

| Paper | Year | Venue | Period | Effect | Costs? | Decay? | URL |
|---|---|---|---|---|---|---|---|
| Cohen & Frazzini, "Economic Links and Predictable Returns" | 2008 (Aug) | J. Finance | OSAP lists OP sample as 1980-2004 | OSAP's copy of the OP: VW 20/80 L/S **1.578%/mo, t=3.79** (Table 3A). A secondary summary says ">150 bp/mo alpha" | Not in the abstract. Full text not reached | n/a (original) | https://www.aqr.com/Insights/Research/Journal-Article/Economic-Links-and-Predictable-Returns ; OSAP SignalDoc: https://raw.githubusercontent.com/OpenSourceAP/CrossSection/master/SignalDoc.csv |
| Menzly & Ozbas, "Market Segmentation and Cross-predictability of Returns" | 2010 | J. Finance 65(4) | OSAP: 1986-2005 | Industry-level supplier/customer cross-predictability. OSAP: customer-industry mom t=4.11, supplier-industry mom t=5.03 (stock-level mv regressions, EW) | No | Weakens with more analyst coverage and institutional ownership (cross-sectional result, not a time trend) | https://ideas.repec.org/a/bla/jfinan/v65y2010i4p1555-1580.html |
| Pandit, Wasley & Zach, "Information externalities along the supply chain" | 2011 | Contemp. Acct. Research | **unverified** | Suppliers react to customers' earnings announcements. The reaction grows with the strength of the link. Magnitude **unverified** (abstract only) | No | n/a | https://academicnewsletter.sufe.edu.cn/info/411640 |
| Pinchuk, "Customer Momentum" (replication) | 2023 (Jan) | arXiv 2301.11394 | 1978/79-2018. Split: 1978-2004 vs 2005-2018 | Full sample: EW **1.22%/mo t=5.17**, VW **1.06% t=3.34**. Pre-2005: EW 1.52% (t=4.74), VW 1.30% (t=3.03). **Post-2005: EW 0.62% (t=2.12), VW 0.62% (t=1.54)** | Not discussed | **Yes: about -55 to -60% after discovery.** VW is not significant post-2005 | https://ar5iv.labs.arxiv.org/html/2301.11394 |
| Ali & Hirshleifer (see Topic 3) | 2020 | JFE | 1984-2006 for customer momentum | Customer momentum has an insignificant alpha once shared-analyst momentum is controlled for | - | In the 2000-2015 half, "all but two" link-momentum factors are insignificant **even without** that control | https://www.nber.org/papers/w25201 |
| Bloomberg Quant Research, "Supply Chain Customer Momentum" (vendor) | 2019 (Jun) | white paper | 2013-2018, global | "Significant driver" of supplier returns. No net-of-cost numbers. Short 1-day horizons do **not** predict, 1-3 month horizons do | No | 6-year sample; vendor marketing | https://data.bloomberglp.com/professional/sites/10/786638_SupplyChainCustomerMomentumWhitePaper.pdf |
| Yamamoto, Kawadai & Miyahara (FactSet summary) | JPM 2021. Blog 2023 | J. Portfolio Mgmt | 2003-2019, global | Multi-hop (up to 4th-layer) customer momentum did best. 12-month signal beats 1-month and has lower turnover. Numbers shown only in charts | Turnover mentioned, no costs | Not analysed | https://insight.factset.com/propagating-momentum-information-through-global-supply-chain-networks |
| McLean & Pontiff, "Does Academic Research Destroy Stock Return Predictability?" | 2016 | J. Finance 71(1) | 97 anomalies | Returns are 26% lower out of sample and **58% lower post-publication** | - | Generic decay benchmark | https://rpc.cfainstitute.org/research/cfa-digest/2016/06/does-academic-research-destroy-stock-return-predictability-digest-summary |
| Hou, Xue & Zhang, "Replicating Anomalies" | 2020 | RFS (NBER w23394) | 447 anomalies | 64% insignificant with NYSE breakpoints + VW (85% at t>3). **Customer momentum row: unverified.** An Alpha Architect summary reportedly counts it among the survivors, but that page returned 403 | - | - | https://www.nber.org/papers/w23394.pdf |
| Jensen, Kelly & Pedersen, "Is There a Replication Crisis in Finance?" | 2023 | J. Finance 78(5) | 153 factors, 93 countries | Most factors replicate (>80% of US factors per a secondary quote). **Whether customer momentum is among the 153: unverified** | - | - | https://nber.org/system/files/working_papers/w28432/w28432.pdf |
| Chen & Zimmermann, Open Source Asset Pricing | 2022 | Critical Finance Review | - | **Includes `CustomerMomentum`** (rated "1_clear / 1_good" replication), plus `iomom_cust` and `iomom_supp`. **OSAP post-sample return for these signals: unverified.** Their Python downloader failed (empty CSV) | - | - | https://raw.githubusercontent.com/OpenSourceAP/CrossSection/master/SignalDoc.csv |
| Chen & Velikov, "Zeroing in on the Expected Returns of Anomalies" | 2023 | JFQA | 204 anomalies | Average anomaly's expected return net of spreads, post-publication decay and modern trading is **~4 bp/mo**. The best are at most ~10 bp | **Yes, the central point** | Yes | https://pure.psu.edu/en/publications/zeroing-in-on-the-expected-returns-of-anomalies/ |
| Chen & Welch, "What Useful Alphas?" | 2026 (Jul) | arXiv 2607.06502 | ~200 OSAP anomalies to recent data | Median L/S return is 48 bp pre-2006 and 19 bp post-2005. **Post-2005 non-microcap: 7 bp**, which "modest" cost allowances would eliminate | Yes (qualitative) | Yes | https://arxiv.org/abs/2607.06502 |

Notes:
- Pinchuk's links come from Compustat Segments: customers that are >10% of sales under
  SFAS 131, with a **6-month lag** after fiscal year-end. The links cover only **~1-16% of
  market cap**, and much less before 2000. The universe is small, and coverage is tilted
  toward small suppliers.
- Ali & Hirshleifer used Frazzini's public link file, which **ends 2006:05**. Much of the
  academic work on customer momentum therefore stops around 2006.

**What it means for a retail implementer in 2026**
- The best direct out-of-sample number is Pinchuk's **2005-2018 VW: 0.62%/mo, t=1.54**. That
  is not significant gross of costs. EW is marginal and depends on small suppliers, which
  are the most expensive to trade and to short.
- The general results bound the upside. Post-2005 non-microcap anomalies average about 7 bp
  gross (Chen-Welch 2026) and about 4 bp net (Chen-Velikov 2023). Customer momentum also
  turns over monthly, so assume costs take most of what is left.
- The links are public and the signal is in OSAP and vendor feeds (Bloomberg SPLC, FactSet
  Revere) that funds already use. If anything is left, it is an obscure-link or multi-hop
  version, not the plain Cohen-Frazzini signal.
- Copy the 6-month link lag. A live implementer has to reproduce the exact
  information-arrival timing, and that lag is the part people most often get wrong.

## 2. Technological links ("tech momentum")

| Paper | Year | Venue | Period | Effect | Costs? | Decay? | URL |
|---|---|---|---|---|---|---|---|
| Lee, Sun, Wang & Zhang, "Technological links and predictable returns" | 2019 (online Nov 2018) | JFE 132(3) 76-96 | 1963-2012. 1.9M CRSP-matched USPTO patents, 561,989 firm-months | Decile L/S on returns of tech-linked firms: **EW 117 bp/mo (t=5.47), VW 69 bp/mo (t=3.19)**. The abstract calls 117 bp an "alpha". Survives size, B/M, profitability, asset growth, R&D, reversal and momentum. Stronger with low attention and high arbitrage cost | Not covered (per secondary summary) | Not reported in the summaries I read. **Unverified** | https://hedgefundalpha.com/strategies/technological-links-and-predictable-returns/ ; https://academicnewsletter.sufe.edu.cn/info/355993 |
| Ali & Hirshleifer | 2020 | JFE | tech factor 1984-2012:06 | Tech momentum's alpha is **insignificant** once shared-analyst momentum is controlled for | - | See Topic 3 | https://www.nber.org/papers/w25201 |

**What it means for a retail implementer in 2026**
- The VW number (69 bp) is about 60% of the EW number. The effect is concentrated in
  smaller, low-attention, hard-to-arbitrage stocks, exactly where costs are highest.
- I found no post-2012 out-of-sample test (**unverified**: the search budget ran out before
  a targeted search). Treat the 1963-2012 numbers as an upper bound.
- Patent data (USPTO, or the Kogan et al. 2017 data the papers use) is free but slow. Link
  similarity moves on a scale of years, so the signal lives in the linked firms' returns,
  not in the graph.
- Ali-Hirshleifer show it is largely subsumed. Building it separately probably adds little
  beyond a shared-analyst signal.

## 3. Shared analyst coverage ("connected-stock momentum")

| Paper | Year | Venue | Period | Effect | Costs? | Decay? | URL |
|---|---|---|---|---|---|---|---|
| Ali & Hirshleifer, "Shared Analyst Coverage: Unifying Momentum Spillover Effects" | NBER WP Oct 2018. JFE 2020 (136(3) 649-675) | JFE | 1984:01-2015:12. NYSE/AMEX/Nasdaq, price >$5 | CS momentum factor **alpha 1.68%/mo, t=9.67 (Sharpe ~1.71)**. Spans industry, geographic, customer, customer/supplier-industry, segment and tech momentum: their alphas become insignificant or negative. Holds in developed international markets | **No transaction-cost analysis found in the working paper** (text searched for "transaction", "bid-ask") | **Yes, but still large in the paper's sample:** 5-factor alpha **1.13%/mo (t=4.48) in 2000-2015**. "Predictability of CS momentum has declined in recent years." It "was more profitable in the earlier part of the sample." Other link factors are mostly already dead in 2000-2015 | https://www.nber.org/papers/w25201 ; WP PDF https://www.nber.org/system/files/working_papers/w25201/revisions/w25201.rev0.pdf |

**What it means for a retail implementer in 2026**
- This is the strongest link signal in the literature, and it unifies the others. Its
  latest published sample still ends in 2015. I found no post-2015 replication
  (**unverified**).
- It needs I/B/E/S detail data to know which analysts cover which firms. That is
  institutional data, not free. Free proxies such as ETF co-holding or news co-mentions are
  not the same signal.
- It is monthly and long/short with a short leg. Retail short costs and borrow
  availability are the binding constraints, and the paper does not price them.
- The 2000-2015 drop from about 1.68% to 1.13% fits the McLean-Pontiff pattern. Expect
  further decay after its 2020 publication.

## 4. Graph neural networks for stock prediction

| Paper | Year | Venue | Period | Effect | Costs? | Decay / honesty issues | URL |
|---|---|---|---|---|---|---|---|
| Feng, He, Wang, Luo, Liu & Chua, "Temporal Relational Ranking for Stock Prediction" (RSR) | 2019 (Mar) | ACM TOIS 37(2) | Train 2013-15, valid 2016, **test = 2017 only (237 days)**. 1,026 NASDAQ + 1,737 NYSE stocks | "Average return ratio of 98% and 71%" (NYSE, NASDAQ). Top-1 daily buy-close / sell-next-close | **Ignored explicitly** ("US broker fees are low"). Assumes fills at the close and always-liquid markets | **Look-ahead:** the Wiki relations come from a **2018-01-05 Wikidata dump** used to trade 2017. Single-year test. A top-1 stock strategy is extremely high-variance | https://ar5iv.labs.arxiv.org/html/1809.09441 |
| Kim et al., "HATS: Hierarchical Graph Attention Network" | 2019 | arXiv 1908.07999 (no venue named) | **unverified** | Claims it "outperformed all the existing methods" | Not mentioned | - | https://arxiv.org/abs/1908.07999 |
| Wang, Zhang, Xiao & Song, "A Review on GNN Methods in Financial Applications" | 2021/22 | arXiv 2111.15367 | - | Survey | Abstract does not mention costs or look-ahead | - | https://arxiv.org/abs/2111.15367 |
| Pu, Roberts, Dong & Zohren, "Network Momentum across Asset Classes" | 2023 (Aug) | arXiv 2308.11294 | 2000-2022, 64 futures | Graph learned from prices only. Sharpe 1.5, 22%/yr "after volatility scaling" | **Net of costs: unverified** (abstract silent) | Futures, not single stocks. A learned graph needs no link data | https://arxiv.org/abs/2308.11294 |

GNN replication critiques: **not verified.** The search budget ran out before I could find
benchmark or reproducibility papers on GNN stock prediction from 2022-2025. I do not claim
any exist or what they say. The RSR text itself shows the usual failure modes:
- a single-year test,
- relation data dated after the test period,
- no costs,
- close-to-close fills,
- a top-1 portfolio.

**What it means for a retail implementer in 2026**
- The ML-venue GNN papers report numbers such as a 98% return in a single year. Those are
  not comparable to the finance-journal results above, which use long samples, factor
  alphas and VW portfolios. Re-test any GNN claim with point-in-time graphs, at least 10
  years of walk-forward data, costs, and factor-adjusted alpha.
- Rebuild the graph point-in-time. A current Wikidata or knowledge-graph snapshot leaks the
  future: links that exist now often formed after the fact, or because of the co-movement.
- The finance literature's linear signals (Topics 1-3) are the baseline a GNN must beat out
  of sample. If a GNN can't beat a weighted average of linked-firm returns, it adds nothing.

## 5. LLM-extracted relationship graphs (2023-2026)

| Paper | Year | Venue | Period | Effect | Costs? | Decay? | URL |
|---|---|---|---|---|---|---|---|
| Lopez-Lira & Tang, "Can ChatGPT Forecast Stock Price Movements?" (news, not graphs; the nearest verified LLM-returns paper) | v1 Apr 2023, v6 Oct 2025 | arXiv 2304.07619 | Post-knowledge-cutoff headlines (exact dates **unverified**) | GPT-4 scores predict next-day drift, mostly in small stocks and on negative news | Not in the abstract | **"Strategy returns decline as LLM adoption rises"** | https://arxiv.org/abs/2304.07619 |
| LLM-extracted supply-chain / relationship graph -> returns | - | - | - | **No paper verified.** The search budget ran out before targeted searches | - | - | - |

**What it means for a retail implementer in 2026**
- I cannot cite an LLM-built company graph with a verified out-of-sample, cost-surviving
  return. Treat any claim of one as unproven until checked.
- The key honesty issue is the training cutoff. An LLM asked "who are X's suppliers?"
  answers from knowledge that may postdate the backtest date. The resulting graph is
  look-ahead by construction, the same problem as RSR's Wikidata. Only extraction run
  document-by-document on point-in-time filings (10-K text as filed) is clean.
- Lopez-Lira & Tang themselves report decay as LLM adoption rises. LLM-derived signals are
  cheap to copy, so expect fast decay.
- Where LLMs could plausibly help is cheaply rebuilding the customer-link file after 2006.
  That recovers the data, not an edge: the underlying signal has already decayed (Topics
  1 and 3).

## 6. Free SEC data for Cohen-Frazzini-style links

- **Source of the requirement:** SFAS 131 (ASC 280) requires firms to disclose any
  customer that is ≥10% of revenue. Compustat Segments captures these as name strings.
  Pinchuk (2023) and Cohen-Frazzini use this file with a 6-month lag
  (https://ar5iv.labs.arxiv.org/html/2301.11394).
- **Data quality** (search-result snippets, details **unverified** against the full texts):
  - Customer names are abbreviated inconsistently, so matching needs hand correction.
  - About 45% of public firms report at least one major customer.
  - About 20% of reported customers are not named.
  - About 13% of major customers are government entities.
  - Culot (2023, J. Supply Chain Mgmt) catalogues further errors: name and classification
    errors, parent-vs-subsidiary attribution, double counting, M&A effects.
  - Source: https://onlinelibrary.wiley.com/doi/10.1111/jscm.12294
- **Look-ahead:**
  - No paper verified specifically on backfill in the customer segment file.
  - The standard guard is the 6-month lag (Cohen-Frazzini, Ali-Hirshleifer).
  - Building from EDGAR 10-K text, keyed to the filing date, is cleaner than any
    vendor-maintained file, because the filing date is the true information-arrival time.
- **Coverage:** Pinchuk's links span only about 1-16% of market cap. A free EDGAR rebuild
  has the same narrow, small-cap-tilted universe.

**What it means for a retail implementer in 2026**
- A free, point-in-time link file can be built from EDGAR 10-K "major customer" text plus a
  name-to-ticker matcher. Name matching is the hard part, not the finance.
- Key the link to the filing's acceptance date, not the fiscal year-end. Then the 6-month
  convention can be checked rather than assumed.
- Expect few links per year, mostly small suppliers. A diversified, capacity-meaningful
  long/short book is hard to build from it.

---

## Overall verdict

**There is no verified evidence of a live, cost-surviving company-graph edge.**

- **Customer momentum** is the cleanest case. It is real in-sample (about 1.3-1.6%/mo,
  1980-2004). Its only direct post-discovery test (Pinchuk 2023, 2005-2018) shows it cut by
  more than half, to 0.62%/mo VW with t=1.54. That is not significant before costs.
- **Shared-analyst momentum** (Ali-Hirshleifer 2020) subsumes the customer, technology,
  industry and geographic spillovers. It still earned about 1.13%/mo in 2000-2015, but it
  needs paid I/B/E/S data, has no published cost analysis or post-2015 replication, and
  was already declining.
- **The anomaly-wide literature** bounds what a 2026 trader should expect: about 7 bp/mo
  gross for post-2005 non-microcap anomalies (Chen-Welch 2026) and about 4 bp net for the
  average anomaly (Chen-Velikov 2023).
- **GNN and LLM graph papers** do not change this. The one GNN paper read in full (RSR)
  has look-ahead in its relation data, a one-year test and no costs. No verified LLM-graph
  paper shows net out-of-sample alpha.

For a retail implementer the realistic outcomes are:
- a research exercise that rebuilds customer links free from EDGAR with point-in-time
  dates and confirms the decay, or
- a small, pre-registered test of a less-arbitraged variant (multi-hop links, or
  shared-analyst links via a free proxy), with costs and short-borrow modeled from day one.

Expect break-even at best.

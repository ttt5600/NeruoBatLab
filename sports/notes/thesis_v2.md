# Thesis v2: HPC for AI agents (memory, inference compute, energy) -- 2026-10-22

Written 2026-10-22. Research and description, not personalized financial advice. No orders were placed.
v1 (`notes/inference_thesis.md`, inference vs training, rules sha `b745e0d3aa0e`) is kept unchanged for comparison.
Personal sizing (account-specific) lives only in the gitignored `notes/thesis_v2_private.md`.

Source tags: **[P]** primary (filing, IR, official data), **[S]** secondary (press), **[U]** unverified (blog,
aggregator, snippet). **STALE** = published before 2026. Research was done by three web agents on 2026-10-22; many
"[P]" items were read through search summaries, so re-open the original before relying on a number.

Not yet reported as of 10-22: hyperscaler Q3 calls (late Oct), Seagate (10-27), GE Vernova (10-28), SK hynix and Samsung
full Q3 (10-29), Intel (10-29), AMD (early Nov), Astera (early Nov), NVIDIA (late Nov). PJM next auction closes 12-15.

---

## 1. The thesis in one paragraph

Agents run many long-context inference calls per task. Long context makes the KV cache large (memory), tool calls run
on CPUs (orchestration compute), and every token needs power (energy). So the scarce inputs are **memory capacity and
bandwidth, inference silicon + server CPUs + packaging, and electricity + the equipment to deliver it**. Networking and
training-specific plays are deliberately left out (user, 2026-10-22).

**Central caveat from the backtest (section 4):** COMPUTE-INFERENCE correlates 0.95 with SMH and MEMORY 0.84. A book that
already owns semis already owns most of this thesis. The sleeves that add something different are ENERGY-GEN
(0.49 to SMH) and, less so, ENERGY-EQUIP (0.70).

---

## 2. Evidence by pillar (for and against)

### 2a. Memory

**Mechanism: memory per token.**
- **KV cache per token.**
  - Llama-3 70B is about 320 KB/token at FP16 with GQA. This is arithmetic from the model shape (VMware blog, 2024-09-25 [S] STALE; the arithmetic does not age).
  - DeepSeek-V3's MLA stores 576 elements/token/layer versus 2,048 for Llama-3.1 70B, roughly 70 KB/token (together.ai, undated [U]).
- **KV cache can exceed the weights.** TrendForce: a 70B model's KV cache reaches 160 GiB at 64K context, and the binding constraint on inference moved to memory in 2026 (TrendForce Insights, 2026 [P]).
- **Output tokens per query up more than 5x/yr.** TrendForce models SSD demand from KV-cache offload and CPU-DRAM demand from agents (TrendForce 2026-06-12 / 2026-07-07 [P summary, S for the 5x]).
- **New KV-cache storage tiers.**
  - NVIDIA Inference Context Memory Storage (BlueField-4, CES 2026-01). The vendor claims up to 5x tokens/s; no independent benchmark was found ([P], vendor claim).
  - Marvell Structera CXL switch for rack-level pooling (2026-03-17 [P]).
- **Against the mechanism.**
  - NVIDIA keeps the KV cache inside NVLink, which limits third-party CXL tiers (Futurum 2026-08 [S]).
  - The HBM-to-DRAM-to-SSD offload path has no standard architecture (Third Bridge, undated [S]).
  - **No vendor or analyst gives a per-agent memory figure (GB/agent). Not found.**

**Pricing cycle (TrendForce contract prices).**

| Quarter | DRAM contract q/q | NAND contract q/q | Source |
|---|---|---|---|
| 1Q26 | +95% | -- | TechSpot 2026-04-10 [S] |
| 2Q26 | +58-63% (fcst) | +70-75% | TechSpot 2026-04-10 [S] |
| 3Q26 | +13-18% | +10-15% | TrendForce 2026-07-03 [P] |
| 4Q26 | +10-15% (slowing) | +15-20%; eSSD the only line accelerating (+23-28%) | TechNews citing TrendForce 2026-09-30 [S] |
| 1Q27 | not found | not found | -- |

- **HBM 2027 forecasts.**
  - TrendForce: blended HBM price +121% y/y in 2027 on HBM4 mix (2026-09-29 [P]).
  - Goldman $17/GB (+44%); UBS about +79%. Some long-term agreements now carry price ceilings (Herald 2026 [S]).
  - Counterpoint: average HBM prices *fell* y/y in 2Q26 (HBM3E cuts, HBM4 delay) ([S]).
- **Micron FQ4'26** (released 2026-09-30, 8-K [P]): revenue $54.23B, non-GAAP gross margin 87.0%, FY26 revenue $133.2B versus $37.4B in FY25. FQ1'27 guide $61.5B ± $1.5B.
  - CY27 HBM "largely sold out" per the call coverage [S].
- **SanDisk FQ4'26** (2026-08-05 8-K [P]): revenue $8.97B (+372% y/y), gross margin 84.6%, data center $2.98B (+103% q/q).
- **HDD.**
  - Western Digital FQ4'26: revenue $3.75B (+44%), 231 EB shipped (2026-08-05 [P]).
  - Seagate: most nearline supply allocated into CY2028 (call coverage [S]).
- **Enterprise SSD 2Q26** (TrendForce 2026-09-01 [P]): top-5 revenue $37.6B, +104% q/q. Samsung > SK hynix/Solidigm > Micron > Kioxia > SanDisk.
- **HBM share 2Q26** (Counterpoint [P/S]): SK hynix 50%, Samsung 33% (up from 21%), Micron 18%.

**Bear case.**
- **Memory is the most cyclical part of semis.**
  - 2022-23: DRAM contract -18-23% in 4Q22 and -20% in 1Q23; Micron cut wafer starts about 20% (TrendForce 2022-11 [P] STALE).
  - 2018-19: PC DRAM contract about -30% in 1Q19 (STALE).
  - In our own backtest, MEMORY returned -29% in 2022.
- **Price increases are decelerating for conventional DRAM** (+95%, then about +60%, +15%, +12% per quarter).
- **Supply arrives in 2027-28.**
  - Micron Singapore HBM in early 2027, plus Idaho and Taiwan mid-2027.
  - Samsung P5 around July 2027; SK hynix Yongin cleanroom in February 2027.
  - China: CXMT from about 320k to 420k wafers/month by 2027 (TrendForce News [S]).
- **Micron capex roughly doubles:** FQ1'27 about $11.5B, 1H FY27 about $25B (Herald 2026-10-01 [S]).
- **Gross margins of 85-87% have no precedent in any cycle.** Mean reversion is the base rate.
- **Peak calls.**
  - Bloomberg Intelligence: shortage peaked in 2Q26, oversupply possible by 2028 (via 24/7 Wall St 2026-07-06 [S]; original not found).
  - Memory stocks fell about 19% in five sessions to 10-15 on the CXMT roadmap (Stockwirex [U]).
  - DDR5 spot fell about 30% in March 2026 while contract prices still rose ([S]).

### 2b. Compute for inference and agents

**For.**
- **Intel Q2'26:** DCAI revenue $6.3B (+59% y/y); server growth "strongest on record" (Intel IR 2026-07-23 [P]).
  - Xeon 6980P list price $12,460 to $13,955 (+12%, July [S]).
- **AMD Q2'26:** data center $6.72B (+107%) (2026-08-04 [P]).
  - AMD raised its server-CPU market view to >35% CAGR, above $120B by 2030, citing agentic AI ([S]).
  - EPYC "sold out through 2026" and server CPU prices +10-20% since March: distributor and trade press only [U].
- **CPU:GPU ratio.**
  - TrendForce: from 1:4-1:8 toward 1:1-1:2 for agents. This is a **forecast**, not a measurement (2026-09-29 [S]).
  - Intel CFO: deployments already moved from 1:8 to 1:4 (TrendForce 2026-04-24 [S]).
- **Arm.** Claims about 50% of CPU compute at top hyperscalers (6-K, company claim [P]). Its own "AGI CPU" (TSMC 3nm, Meta lead customer) is in volume production in 2H26 ([S]).
- **Broadcom FQ3'26:** AI semis $16.7B (+221%); FQ4 AI guide $21.7B (8-K 2026-09-02 [P]).
  - Google TPU 8i is an inference-specific part (2026-04-22 [S]).
- **Cerebras Q2'26:** revenue $180.1M (+74%) but missed expectations; gross margin 14%; FY guide $880-890M (8-K 2026-08-12 [P]; miss details [S]).
- **TSMC Q3'26:** revenue about $46.7B (+41% in USD), gross margin 66.4%, Q4 guide $52.3-54.3B, 2026 capex $60-64B (6-K 2026-10-15 [P]).
  - CoWoS packaging: about 130k wafers/month at end-2026, doubling by end-2028 (EDN via TrendForce 2026-09-14 [S]).
- **ASE** raised its 2026 leading-edge packaging guide to >$3.5B and aims to double it in 2027 (2026-07-31 [P/S]).
- **Amkor Q2'26:** record $1.9B revenue (2026-07-27 [P]).

**Against.**
- **NVIDIA takes the inference socket itself.** Vera CPU sold standalone (NVIDIA 2026-05-31 [P]); Groq 3 LPX inference racks in production since Aug 2026 ([S]).
- **Intel execution.** Foundry lost $2.1B in Q2. No named 14A customer, and 14A may be paused without one (10-K [P]).
- **Price deflation.** Token prices fell about 43% in ten weeks to Aug 2026 [U]. That pressures whoever sells tokens; no chip-margin compression is visible yet (NVIDIA gross margin about 75%).
- **Capex is increasingly debt-funded.** About one third of 2027 AI capex is debt-funded, and S&P says only Microsoft has positive 2027 free cash flow among six large spenders (DCD/CoStar 2026 [S]).
- **About 60% of 2026 capex upward revisions are chip and HBM price inflation, not more capacity** (FourWeekMBA [U]).
- **Geopolitics.** Taiwan concentration; a Section 232 chip tariff is pending; NVIDIA guides zero China data-center compute.
- **SMH drawdown verified from yfinance:** -24.6% from the 2026-06-22 peak to 2026-07-29. SOX history: about -80% in 2000-02, about -36% in 2022 calendar year (STALE). In our backtest COMPUTE-INFERENCE returned -34% in 2022.

### 2c. Energy

**Demand.**
- **LBNL 2025 update** (2026-06): US data centers could use 11.8% of US electricity by 2030, range 9.5-15.3% (521-843 TWh) ([P*]). The 2024 report had 176 TWh in 2023 (STALE).
- **EIA STEO July 2026:**
  - commercial electricity sales +2.4% (2026) and +4.3% (2027), driven by data centers;
  - commercial use passes residential (Reuters via TheStar 2026-07-08 [S]).
  - **Conflict:** another EIA page puts the crossover in 2027. Check the EIA tables.
- **IEA:** data-center electricity demand +17% in 2025, doubling by 2030 (Key Questions 2026-04 [S]).
- **PJM:**
  - 2026 load forecast: summer peak about 222 GW by 2036 (+66 GW) (APPA 2026-01 [S]).
  - Three straight capacity auctions cleared **at the cap**: $329.17, $333.44 and $325/MW-day for 2026/27 through 2028/29.
  - Uncapped prices would have been 60-70% higher ($529.80 and $554.72), and supply was about 6.6-6.8 GW short of the reliability target (PJM / APPA [P/S]).
- **ERCOT:** about 474 GW of large loads seeking connection (June 2026), about 90% data centers, more than 300 GW with no studies (ERCOT deck 2026-07-29 [P*]). This queue is mostly speculative.
- **Utility large-load pipelines** (definitions differ; **do not add**):
  - Dominion 53.8 GW "contracted", of which only 12 GW is under binding agreements (PJM slides 2026-09 [P]);
  - AEP about 69 GW (conflicting 24 GW elsewhere [S]);
  - Southern about 17 GW, including a 3.2 GW OpenAI contract ([S]);
  - Entergy 7-12 GW hyperscale pipeline ([S]).
- **LBNL Queued Up 2026:** median time from interconnection request to operation exceeds 5 years; only 13% of 2000-2020 requests reached operation ([P]).

**Who supplies it.**
- **Constellation:** Calpine closed (2026-02 [P]). Crane/TMI restart for Microsoft targeted 2H27; Meta Clinton PPA starts June 2027.
- **Talen:** Amazon Susquehanna, up to 1,920 MW, restructured to grid-connected.
- **Vistra:** Meta 2.2 GW of nuclear PPAs (Perry, Davis-Besse) plus Amazon Comanche Peak (size conflicts across sources).
- **NRG:** closed the LS Power deal, fleet about 25 GW.
- **GE Vernova Q2'26:** gas backlog plus slot reservations 116 GW, total backlog $176B, orders +88% organic (8-K 2026-07-22 [P]). New turbine slots reportedly reach 2031 [S].
  - Siemens Energy gas backlog 69 GW; Mitsubishi 35 GW ([S]).
- **Bloom Energy Q2'26:** revenue $1.065B (+166%), FY guide $3.9-4.2B (2026-07-28 [P]). Oracle and Brookfield deal sizes conflict across sources [U]. A short-seller report exists.
- **Grid equipment.** Transformer lead times up to 4 years; Wood Mackenzie puts the 2026 power-transformer shortage at about 15% (pv magazine 2026-05 [S]).
  - nVent 2026 data-center sales above $2B ([S]).
  - Quanta record backlog about $53B ([S]).
- **Cooling.** Vertiv Q2'26 sales $3.27B (+24%), FY guide about $14B (10-Q [P]); it reportedly stopped reporting backlog quarterly.
- **Uranium.** Cameco long-term price about $95.50/lb, near the highest since 2011 (Q2'26 [P]).
- **SMRs.** The first US BWRX-300 construction permit went to TVA Clinch River (2026-09-29 [P]).
  - NuScale Q2 revenue $0.1M; Oklo pre-revenue with no NRC licence; Kairos Hermes 2 around 2030.
  - **No commercial SMR power before about 2030.**

**Bear case.**
- **Regulation caps the upside.** The PJM price collar (cap about $325, floor about $175) was extended through 2029/30 with all 13 governors, the White House and DOE in support (FERC 2026-04-28 [P/S]).
- **Texas pause.** Gov. Abbott paused data-center grid connections pending an audit (2026-08-03); up to 49.8 GW could be delayed (Foley [S]).
- **Ratepayer protection.**
  - The House passed the Ratepayer Protection Act 417-3 (2026-09-16).
  - Oklahoma HB2992 signed; New Jersey S731 pending ([P/S]).
- **Co-location rules.** The FERC co-location order (2025-12-18) pushed behind-the-meter deals toward grid-connected structures ([S]).
- **"Ghost" load.** Exelon cut about 11 GW from its pipeline once it required financial commitments; PJM trimmed its near-term forecast for the same reason ([S]).
- **Local opposition.** Data Center Watch counts 75 projects (about $130B) blocked or delayed in Q1 2026; its method is not public [U].
- **Gas overbuild risk** (IEEFA / RMI, advocacy sources [S]). SMR timelines slip; in our backtest OKLO is -64% YTD 2026.
- **ENERGY-GEN fell 15% YTD 2026 even as AI capex rose.** Utilities and IPPs trade on rates and regulation as well as load.

---

## 3. v2 baskets (frozen 2026-10-22, before any v2 return was computed)

Rules sha256 **`7e822f587214`** (full `7e822f5872143bac4c043c7621b256256086510e3379b35fb71cc32f62cb8c58`); print it with
`python scripts/thesis_tracker.py rules-v2`. The v1 sha `b745e0d3aa0e` was re-verified unchanged after the edit.

**Rules:**
- Equal weight within each sleeve, **rebalanced monthly** on the first trading day of each month (weights drift in between).
- A name enters at the first monthly rebalance after its first price. Pre-merger SPAC shell prices are skipped (OKLO from 2024-05-10, SMR from 2022-05-03, VRT from 2020-02-10).
- `v2_all` = the four sleeves at 25% each, rebalanced monthly.
- Benchmarks: SPY, QQQ, SMH. yfinance adjusted closes.
- **Forward record from 2026-10-22.** Any change to the baskets means a v3 with its own freeze date.

**MEMORY (8)**

| Ticker | Why |
|---|---|
| MU | the only US HBM + DRAM + NAND maker; 2027 HBM largely sold out |
| SNDK | enterprise SSD / QLC for the KV-cache tier; data center 33% of revenue |
| WDC | HDD bulk tier for AI data lakes (231 EB/qtr) |
| STX | HDD nearline, HAMR; supply allocated into 2028 |
| P | Everpure (formerly Pure Storage, ticker changed to P): all-flash storage systems for AI data |
| NTAP | all-flash storage; AI / data-lake deals |
| SIMO | Silicon Motion: SSD controllers (eSSD ramps) |
| RMBS | Rambus: DDR5 memory-interface chips (RCD/MRDIMM), CXL IP |

**COMPUTE-INFERENCE (10)**

| Ticker | Why |
|---|---|
| AMD | EPYC server CPUs + MI450 inference GPUs (OpenAI 6 GW, Meta) |
| INTC | Xeon for agent orchestration; DCAI +59% y/y; 18A / EMIB packaging |
| ARM | CPU IP behind Graviton / Axion / Grace / Vera, plus its own AGI CPU |
| AVGO | custom inference ASICs (TPU 8i, MTIA, OpenAI XPU) |
| MRVL | custom XPUs (Trainium / Maia) + CXL (Structera). Some optics exposure, kept for custom silicon |
| QCOM | AI200/AI250 inference racks; data-center target (unverified $15B by FY29) |
| CBRS | Cerebras wafer-scale inference cloud; OpenAI inference deal |
| TSM | foundry for every inference chip; CoWoS packaging |
| AMKR | US-listed advanced packaging (OSAT), Arizona campus |
| ASX | ASE: largest OSAT; leading-edge packaging revenue doubling |

**ENERGY-GEN (11)**

| Ticker | Why |
|---|---|
| CEG | largest US nuclear fleet + Calpine gas; Microsoft / Meta PPAs |
| VST | nuclear + gas in ERCOT/PJM; Meta and Amazon nuclear deals |
| TLN | Susquehanna nuclear; Amazon 1.92 GW PPA |
| NRG | about 25 GW fleet after LS Power; Texas hyperscaler deal (single source) |
| D | Dominion: Northern Virginia, the largest data-center load pocket |
| AEP | about 69 GW of contracted large-load additions (conflicting figure) |
| SO | Southern / Georgia Power: about 17 GW contracted incl. OpenAI 3.2 GW |
| ETR | Entergy: Gulf Coast hyperscale pipeline 7-12 GW |
| OKLO | SMR developer, Meta 1.2 GW deal; pre-revenue (speculative leg) |
| SMR | NuScale: only NRC-certified SMR design; essentially no revenue (speculative leg) |
| CCJ | uranium + Westinghouse stake; long-term uranium price near a 15-year high |

**ENERGY-EQUIP (10)**

| Ticker | Why |
|---|---|
| GEV | gas turbines (116 GW backlog + slots) + grid equipment; BWRX-300 SMR |
| VRT | data-center power + liquid cooling |
| ETN | electrical distribution, UPS, switchgear for data centers |
| BE | Bloom solid-oxide fuel cells: on-site power that skips the interconnection queue |
| PWR | Quanta: grid and substation construction, record backlog |
| HUBB | Hubbell: utility grid components, transformers / connectors |
| MOD | Modine: data-center cooling |
| NVT | nVent: liquid-cooling and enclosure systems, >$2B data-center sales |
| BWXT | nuclear components + fuel (naval + civil) |
| POWL | Powell Industries: custom switchgear / power distribution |

**Deliberately excluded:**
- Networking and connectivity: ANET, ALAB, CRDO, COHR, LITE, CIEN.
- Training-centric / landlords: NVDA, hyperscalers, neoclouds (CRWV, NBIS).
- Not US-listed or OTC only: SK hynix, Samsung, Kioxia, Siemens Energy, Infineon, Mitsubishi Heavy.
- Power semis: onsemi / Navitas / Infineon left out because their AI share is small or the stock is not US-listed. Noted as a gap.

---

## 4. Descriptive backtest -- HINDSIGHT-BIASED, NOT EVIDENCE OF EDGE

**Read this first.**
- These baskets were chosen on 2026-10-22 by someone who already knows which companies became AI winners (MU 28.6x since 2019, SNDK 38x since its 2025 listing, BE 23.6x).
- Any basket picked this way will beat SPY in hindsight. The tables describe what the names did. They do not show that the selection process has an edge.
- **The only honest test is the forward tracker** (section 5).
- Prices through 2026-10-21; run 2026-10-22 (`python scripts/thesis_tracker.py history-v2`).
- The engine was checked against a brute-force share-holding simulation; the max equity difference was 8e-16.

**Since 2019-01-02**

| Series | Total | CAGR | Vol | Max DD | Corr SMH | Corr SPY |
|---|---|---|---|---|---|---|
| MEMORY | +2,216% | 49.8% | 38.8% | -43.8% | 0.84 | 0.70 |
| COMPUTE-INF | +1,352% | 41.0% | 38.8% | -45.5% | **0.95** | 0.77 |
| ENERGY-GEN | +499% | 25.9% | 28.9% | -38.2% | **0.49** | 0.61 |
| ENERGY-EQUIP | +2,302% | 50.5% | 36.0% | -51.8% | 0.70 | 0.71 |
| v2_all (25% each) | +1,621% | 44.1% | 30.7% | -39.3% | 0.89 | 0.81 |
| SPY | +246% | 17.3% | 19.2% | -33.7% | 0.81 | 1 |
| QQQ | +391% | 22.7% | 23.9% | -35.1% | 0.89 | 0.93 |
| SMH | +1,234% | 39.5% | 36.1% | -45.3% | 1 | 0.81 |

**Since 2023-01-03**

| Series | Total | CAGR | Vol | Max DD | Corr SMH |
|---|---|---|---|---|---|
| MEMORY | +1,076% | 91.9% | 42.9% | -43.8% | 0.82 |
| COMPUTE-INF | +486% | 59.6% | 41.0% | -40.7% | 0.94 |
| ENERGY-GEN | +263% | 40.7% | 33.0% | -35.1% | 0.50 |
| ENERGY-EQUIP | +729% | 74.9% | 38.1% | -40.1% | 0.73 |
| v2_all | +647% | 70.2% | 33.3% | -35.3% | 0.89 |
| SPY | +113% | 22.1% | 14.9% | -18.8% | 0.79 |
| QQQ | +180% | 31.3% | 20.1% | -22.8% | 0.89 |
| SMH | +460% | 57.7% | 35.5% | -35.7% | 1 |

**Sleeve-to-sleeve daily correlation** (2019+; the 2023+ figures are within 0.03):
- MEMORY-COMPUTE 0.82
- MEMORY-ENERGY-EQUIP 0.71
- COMPUTE-ENERGY-EQUIP 0.70
- ENERGY-GEN-ENERGY-EQUIP 0.63
- ENERGY-GEN with MEMORY or COMPUTE 0.49

**Calendar-year returns (%)**

| Year | MEM | COMP | GEN | EQUIP | v2_all | SPY | QQQ | SMH |
|---|---|---|---|---|---|---|---|---|
| 2019 | 50 | 71 | 19 | 40 | 45 | 31 | 38 | 63 |
| 2020 | 18 | 51 | 2 | 46 | 30 | 18 | 48 | 56 |
| 2021 | 58 | 46 | 24 | 22 | 38 | 29 | 27 | 42 |
| 2022 | **-29** | **-34** | **+10** | **+18** | -11 | -18 | -33 | -34 |
| 2023 | 56 | 75 | 12 | 84 | 56 | 26 | 55 | 73 |
| 2024 | 14 | 25 | 130 | 93 | 62 | 25 | 26 | 39 |
| 2025 | 145 | 44 | 65 | 62 | 79 | 18 | 21 | 49 |
| 2026 YTD | 169 | 87 | **-15** | 43 | 64 | 14 | 18 | 55 |

**Top-2 concentration** (exact dollar P&L attribution on $1, monthly rebalance; "ex-top-2" re-runs the same rules
without those two names):

| Sleeve | Window | Top 2 | Share of gain | Sleeve total | Ex-top-2 total |
|---|---|---|---|---|---|
| MEMORY | 2019+ | SNDK, MU | 43% | +2,216% | +1,376% |
| MEMORY | 2023+ | SNDK, MU | 44% | +1,076% | +634% |
| COMPUTE-INF | 2019+ | AMD, MRVL | 35% | +1,352% | +1,010% |
| COMPUTE-INF | 2023+ | AMD, MRVL | 36% | +486% | +398% |
| ENERGY-GEN | 2019+ | OKLO, CCJ | 36% | +499% | +347% |
| ENERGY-GEN | 2023+ | OKLO, TLN | 41% | +263% | +177% |
| ENERGY-EQUIP | 2019+ | BE, POWL | 47% | +2,302% | +1,365% |
| ENERGY-EQUIP | 2023+ | BE, POWL | 48% | +729% | +469% |

- **BE alone produced about a third of ENERGY-EQUIP's gain** (+7.56 of +23.0 on $1).
- CBRS (-0.79) and SMR are the only net losers.
- The OKLO contribution is real only because monthly rebalancing harvested its 2024-25 run; it is -64% YTD 2026.

**What this does and does not say.**
1. COMPUTE-INFERENCE is SMH with different weights (corr 0.95). MEMORY is mostly SMH (0.84). For an owner of semis, they add the same bet, not diversification.
2. ENERGY-GEN is the only sleeve with a different risk profile. It made money in 2022 and lost money in 2026 while semis rose. That is the diversification the thesis offers.
3. Every sleeve has had a drawdown of 38-52%. The 2026 SMH drawdown was -24.6% in five weeks.
4. Memory's 2025-26 gains (+145%, +169%) coincide with contract prices +60-95% q/q. That is a price cycle. Section 2a shows it decelerating, with supply arriving in 2027.
5. **For comparison with v1** (different rules: daily rebalance, different names): inference_v1 returned +937% since 2023 versus COMPUTE-INF +486%. The gap is mostly SNDK (in v1's inference basket, in v2's MEMORY basket).

---

## 5. Forward tracker (the honest test)

- `python scripts/thesis_tracker.py tick` now logs v1 (unchanged schema, `data/thesis/tracker.jsonl`) **and** v2 (`data/thesis/tracker_v2.jsonl`).
  - v2 rows hold the four sleeve levels, `v2_all` and SPY/QQQ/SMH, all indexed to 1.0 at the 2026-10-22 close.
  - The first v2 row appears after the first full trading day post-freeze.
- `report` prints both records. `rules-v2` / `history-v2` reprint the frozen rules and the descriptive history.
- **launchd agent NOT installed** (user decides). `bash scripts/install_thesis_tracker_agent.sh` installs it.
  - The script copies the updated tracker, so one agent covers v1 and v2.
  - Re-run the script after any tracker edit.
- **Expectations.** Months of forward data will not separate skill from the memory/energy cycle. Expect noise of ±20% per quarter per sleeve. Judge the thesis on the leading indicators below.

## 6. Leading indicators (check at each date)

| Indicator | Confirms | Refutes | Next date |
|---|---|---|---|
| TrendForce DRAM / NAND / eSSD contract prices | 1Q27 q/q still positive; eSSD accelerating | flat or negative q/q; spot below contract for months | monthly; SK hynix / Samsung 10-29; STX 10-27 |
| Memory capex / supply | 2027 additions absorbed; HBM4 long-term agreements at higher prices | capex surge + CXMT DDR5 ramp + price ceilings binding | MU FQ1'27 (late Dec) |
| Server CPU demand | INTC / AMD server growth with price increases holding | price cuts, inventory build | INTC 10-29; AMD early Nov |
| Inference silicon ramps | AVGO FQ4 AI ≥ $21.7B; TPU 8i / MI450 / AGI CPU volumes | guide misses, delays | AVGO ~Dec; CBRS ~Nov |
| Hyperscaler capex | 2027 guides raised, with inference / agents named | "digestion"; debt-funding pushback | late Oct Q3 calls |
| Power: PJM / ERCOT | Dec-15 PJM auction at the cap again; Texas pause lifted with clear rules | collar tightened; large-load tariffs that shift cost to data centers; moratoria spread | PJM 12-15; PUCT / ERCOT rulings |
| Equipment backlogs | GEV slots ≥ 125 GW by year-end; transformer lead times stay long | order cancellations; backlog shrinking (nVent backlog already dipped q/q) | GEV 10-28; VRT Q3 |
| Utility contracted load | binding agreements (not letters) grow | pipelines cut as financial commitments are required (Exelon -11 GW) | Q3 utility calls (late Oct to early Nov) |

## 7. Generic sizing guidance (consistent with `notes/robinhood_strategies.md`)

1. **Deleverage first.** At about 49% book vol, expected log growth turns negative above about 2x even at a 40%/yr return. No thesis changes that. Reach about 1.5-1.8x or less before any tilt.
2. **Core = broad index; thesis = satellite.** No backtested edge exists (the section 4 history is hindsight), so the tilt goes into **which names you keep, not into added exposure**. No margin, no options leverage for the thesis.
3. **A semis-heavy holder already owns COMPUTE-INF and most of MEMORY** (corr 0.95 / 0.84 to SMH). Adding more is concentration, not diversification. The thesis-consistent move is to **rotate** within the satellite:
   - **Trim:** networking / connectivity names and training / neocloud names (which the thesis now de-emphasizes).
   - **Hold or add:** some ENERGY-GEN / ENERGY-EQUIP. These are the only sleeves with a materially different return stream.
4. **Illustrative satellite split** (not tuned): about 25% memory, 25% compute-inference, 25% energy-gen, 25% energy-equip.
   - Cap any single name at about 5% of equity.
   - Treat memory as a cyclical position with a review trigger, namely contract prices turning negative q/q, not as a secular hold.
   - Keep the speculative pre-revenue names (OKLO, SMR, CBRS) to about 1% each or less.
5. Review on the indicator dates in section 6, not on price moves.

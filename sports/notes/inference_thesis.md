# Inference-over-training thesis (user thesis 2026-10-22)

Written 2026-10-22. Research and analysis, not personalized financial advice.
Sources were fetched by two research agents on 2026-10-22. Every number below has its source and publication date.
- **STALE** means the source predates 2026.
- **secondary** means the number came from coverage, not the primary filing.
- **unverified** means only a blog or search snippet supported it.

What is not yet available:
- Q3 2026 hyperscaler calls (Alphabet, MSFT, Meta, AMZN) were **not out** as of 10-22. All hyperscaler numbers here are from the July Q2 calls. The agent did not confirm the Q3 dates.
- SK hynix and Samsung full Q3 results are due 10-29; Seagate's on 10-27.
- **No company discloses an inference-vs-training split of revenue or capex.** Every split below is a third-party estimate.
- The company graph (`data/company_graph/edges.parquet`) did not exist when this was written, so it was not used.

---

## 1. Thesis memo (one page)

**Claim.** AI demand is moving from building models to running them: agents, long context, many calls per task. The hardware that is scarce in inference (memory capacity and bandwidth, SSD for KV cache, CPUs for orchestration, inference-specific ASICs, scale-up interconnect) should grow faster than training-cluster GPUs and the hyperscalers. Tilt a little toward the former.

**Mechanism.** Decode is memory-bandwidth-bound, not FLOP-bound. Agents add three things:
- **Long contexts.** These make the KV cache large, so it spills from HBM to DRAM and then to SSD.
- **Tool calls on CPUs.** TrendForce 2026-09-29 says CPU:GPU ratios go from 1:4–1:8 today to an expected 1:1–1:2 for agentic workloads.
- **Cost-per-token pressure.** This favours ASICs and specialised parts over general GPUs.

**Evidence for (2026):**
- **Google tokens.** About 480T/month (May 2025) grew to over 3.2 quadrillion/month (May 2026), "7x y/y" (Google I/O blog, 2026-05-19). The Q2 call put API traffic at about 22B tokens/min, up from 16B a quarter earlier (MarketBeat, 2026-07-22).
- **Anthropic.** Run-rate revenue was above $65B by end of July 2026 (Bloomberg Government, 2026-08-18).
- **Microsoft.** Foundry customers running at 1T+ tokens/yr were up 4x y/y (Q4 FY26 call, 2026-07-29).
- **Memory demand.** TrendForce (2026-09-30):
  - enterprise SSD bit demand up more than 80% in 2026 "as CSPs expand AI inference infrastructure";
  - 4Q26 eSSD contract prices up 23–28%, with QLC used "to support KV cache";
  - DRAM up 10–15% and NAND up 15–20% quarter on quarter.
- **Companies.** SanDisk data-center revenue rose 103% q/q to $2.98B, 33% of revenue (FQ4'26, 2026-08-05). Kioxia: "agentic AI is driving strong NAND demand" (2026-08-03).
- **Intel.** Data Center and AI (DCAI) revenue was $6.3B, 39% of revenue, with demand above supply (Q2 2026). Intel and AMD raised CPU prices at end of Q1 2026 (TrendForce).
- **NVIDIA itself sells inference.** The Q2 FY27 release (2026-08-26) leads with agentic AI, the Vera CPU and the "Groq 3 LPX interactive AI inference accelerator", which follows a roughly $20B Groq licence in Dec 2025.
- **Custom silicon is surging.** Broadcom AI semiconductor revenue was $16.7B, up 221% y/y, with a Q4 guide of $21.7B. Broadcom is shipping Google TPU v8i, an inference variant (Pulse2, 2026-09-03).

**Evidence against (2026):**
- **The share did not move at the labs.** SemiAnalysis chart via OfficeChai (2026-10-18, secondary): at Anthropic + OpenAI, the inference share of compute stayed in a **29–39% band from 1Q24 to 4Q26E**. The shift was *inside* training: pre-training fell from 67% to 7%, while RL/post-training rose from 5% to 53–55%.
  - RL rollouts are inference-shaped generation, so the line between training and inference is blurring. That helps "inference hardware" but not a clean "training vs inference" equity split.
- **Training is still scaling.** Epoch AI (trends page, updated 2026-02-05) puts frontier training compute growth at about 5x/yr. Colossus 2 is about 1.1M H100e, and Meta Hyperion is planned to reach 3.7M H100e by Jan 2028.
- **Inference runs mostly on NVIDIA and in hyperscaler clouds.** NVIDIA's data-center segment was $89.0B, 92.5% of its revenue, in Q2 FY27. About $49B of that came from hyperscalers (Fortune, 2026-08-27). The hyperscalers are the landlords:
  - Alphabet 2026 capex guide is $195–205B, with 2027 to "increase significantly";
  - Microsoft FY27 capex is about $175–190B;
  - Amazon about $220B, Meta $130–145B (Platformonomics, 2026-07-31).
  - **Selling hyperscalers to buy "inference" partly sells the biggest inference operators.**
- **Jevons.** Epoch (secondary coverage, 2026-09-22) finds the cost of fixed performance falling about 13x/yr, and total spend rose anyway. That is good for volume, but it means **price-per-token deflation hits whoever sells tokens**. Inference clouds and neoclouds carry that risk most.
- **ROI and financing strain** (all secondary, 2026):
  - capex near 100% of operating cash flow (UBS);
  - AI-linked bonds about $570B (Morgan Stanley);
  - Oracle Q1 FY27 free cash flow of -$5B plus a $20B at-the-market stock sale.
  - A capex slowdown would hit both baskets.

**What the price record says (descriptive only; section 4).** Equal-weight baskets were frozen before computing returns. Inference basket over training basket, by ratio of levels:
- +35.7% since 2023-01-03
- +59.2% since 2025-01-02
- +43.1% since 2026-01-02

But **the result depends on one name**. Without SanDisk the gap is -11.5% / +3.8% / +20.5%. Without SanDisk and Intel it is -9.1% / -12.1% / +10.6%. The baskets' daily returns correlate 0.74–0.81. Most of the "inference outperformance" is a memory-price cycle plus an Intel rerating. It is not a broad inference premium.

**Risks.**
- Memory is cyclical: the 2026 price spike can reverse, and TrendForce contract prices are the tell.
- Intel is a turnaround story, not a pure inference play.
- Cerebras is -62% from its first close (IPO 2026-05-14).
- A hyperscaler capex cut would hit everything.
- NVIDIA could keep the inference socket itself (Groq LPX, Rubin CPX, Vera CPU, BlueField-4 KV storage).

**What would change our mind.**
- *Against:*
  - inference share at the labs stays flat (next SemiAnalysis/Epoch update);
  - TrendForce eSSD/DRAM contract prices turn negative q/q;
  - hyperscaler capex guides cut in the late-Oct and Jan/Feb calls;
  - Broadcom/Marvell custom-chip revenue guides cut;
  - server CPU prices fall.
- *For:*
  - a company starts disclosing an inference revenue mix (NVIDIA, AMD, Broadcom);
  - CPU:GPU ratios in deployed racks actually rise (vendor disclosures, not scenarios);
  - token disclosures keep growing faster than about 13x/yr price deflation, so token revenue rises.

**Suggested tilt (framed small, per "a little bit").**
- Keep the core AI exposure. Move **about 10–20% of the AI sleeve** (roughly 1–3% of a total portfolio if AI is about 15% of it) from the training/hyperscaler leg to the inference leg.
- Inside the inference leg, **spread across mechanisms**: memory/storage, CPUs/Arm, custom-ASIC designers, scale-up interconnect. Don't concentrate in one; the record above shows a single-name memory cycle can impersonate a thesis.
- Size memory as a cyclical trade, not a secular holding.
- **Do not cut the hyperscalers to zero.** They are the inference landlords (Google's 3.2 quadrillion tokens/month run on Google's own TPUs).
- Review only at the leading-indicator dates in section 5, not on price moves.

---

## 2. Evidence detail (all from the 2026-10-22 research pass)

### Inference share, capex and tokens

| Item | Number | Source, date |
|---|---|---|
| NVIDIA data center (Q2 FY27) | $89.0B of $96.2B (92.5%); Q3 guide $108B | NVIDIA IR release, 2026-08-26 |
| NVIDIA hyperscaler revenue | about $49B; non-hyperscalers "roughly half" of data center | Fortune, 2026-08-27 |
| NVIDIA "inference 60%+ of GPU demand" | **unverified** (blog only) | DailyAIWorld, 2026-08-30 |
| Broadcom AI semis | $16.7B (+221%); custom chips about 73% of AI revenue (call summary, secondary) | Pulse2 / Nasdaq, 2026-09-03 |
| AMD data center | $6.7B of $11.5B (58%) | AMD Q2 8-K, 2026-08-04 |
| Oracle | contract backlog (RPO) $664B; FY27 capex $90–95B; FCF -$5B | Oracle Q1 FY27 2026-09-10 / DCD (secondary) |
| Alphabet capex 2026 | $195–205B; Q2 $44.9B | MarketBeat call highlights, 2026-07-22 |
| MSFT FY27 capex | about $175B (about $190B before lease reclassification) | MarketBeat / Platformonomics, 2026-07-29/31 |
| AMZN / META capex 2026 | about $220B / $130–145B | Platformonomics, 2026-07-31 |
| Citi 2027 GOOGL+AMZN+META capex | $801B | Teleborsa, 2026-07-16 (not opened) |
| Google tokens | about 480T/month (May 2025) to over 3.2 quadrillion/month (May 2026) | Google blog, 2026-05-19 |
| Google API tokens | about 22B/min (Q2), up from 16B (Q1) | MarketBeat, 2026-07-22 |
| Anthropic run-rate | above $65B (end Jul 2026) | Bloomberg Government, 2026-08-18 |
| OpenAI API | about 15B tokens/min | **unverified** aggregator, 2026 |
| OpenRouter | agents are now the main token users (charts only) | OpenRouter "Inferred", 2026-09/10 |

### Agentic workload characteristics

- **CPU:GPU.** 1:4–1:8 today, expected 1:1–1:2 for agentic workloads. Arm: 30M rising to 120M CPU cores per GW. CPU tool processing can be up to 90.6% of agent latency (Nov 2025 paper). Source: TrendForce, 2026-09-29.
  - The Intel CFO "1:1" quote is **unverified** (search summary only).
  - AMD (OCP APAC 2026, secondary): agents *add* a CPU layer and do not reduce GPU demand.
- **KV cache leaves HBM.** NVIDIA BlueField-4 Inference Context Memory Storage (CES 2026-01; vendor claim of up to 5x tokens/s). VAST shared KV cache (StorageNewsletter, 2026-01-07). TrendForce says eSSD QLC demand is for KV cache (2026-09-30).
- **Long-context prefill.** Rubin CPX targets 1M+ token contexts (TechCrunch, 2025-09-09, **STALE**; due end-2026).
- **Disaggregated serving.** GTC 2026 claims of 7x from software on Blackwell and 35x with Groq LPX (vendor claims, secondary).

### Cost curves

- Epoch, "plunging price of thought": about 13x/yr decline at fixed performance (2026-09-22 report, secondary coverage).
- Epoch data insight: 9–900x/yr depending on benchmark (2025-03-12, **STALE**).
- a16z LLMflation: 10x/yr (2024-11-12, **STALE**).
- Total spend rose anyway: about $1T in 2026 AI capex (Bloomberg forecast via Crypto Briefing, 2026-10-14).

### Third-party splits (all weak)

- **McKinsey:** inference 20.9 GW to 93.3 GW vs training 23.1 GW to 62.2 GW (2025 to 2030). Two McKinsey pages conflict, so **unverified**.
- **Deloitte:** inference about 1/2 of AI compute in 2025 and 2/3 in 2026. Second-hand only, **unverified**.
- **SemiAnalysis via OfficeChai:** lab inference share flat at 29–39% (2026-10-18). This is the best 2026 datapoint and it is *against* a "share shift".
- **Epoch AI Chip Users:** OpenAI compute roughly even between R&D and inference in 2025; inference was 30% of OpenAI's 2024 compute spend (2026-09-09).
- **IEA:**
  - data-center electricity 415 TWh (2024) to about 945 TWh (2030) (Energy and AI, 2025, **STALE**);
  - data-center use +17% in 2025, with AI-focused data centers about +50% (Key Questions, 2026-04-16, secondary);
  - the June 2026 deck says the 2030 path is "close to" the 2025 report.
  - The IEA gives no training/inference power split.

---

## 3. Beneficiary map, by mechanism

The labels are our reading of what each company sells. **No company reports inference vs training revenue.** All rows are 2026 sources unless marked.

| Ticker | Mechanism | Label | AI/DC revenue % (period) | Source, date |
|---|---|---|---|---|
| CBRS (Cerebras) | wafer-scale inference chips + cloud; OpenAI 750 MW inference deal | **inference** | Q2 core revenue $210M; Q1 cloud 43% | SEC 8-K 2026-08-12 |
| SNDK | enterprise SSD, KV-cache tier | **inference** | DC 33% (FQ4'26) | SanDisk IR 2026-08-05 |
| Kioxia 285A.T | NAND/eSSD | **inference** (not US-listed) | DC 66% (Apr–Jun 2026) | Blocks & Files 2026-08-03 |
| INTC | Xeon for agent orchestration | **inference** (plus foundry/PC noise) | DCAI 39% (Q2 2026) | Intel IR 2026-07 |
| QCOM | on-device + AI200 inference card | **inference** | DC about 0% now; FY27 target $5B | Q3 FY26 transcript 2026-07-30 |
| NET | Workers AI edge serving | **inference** (qualitative) | not disclosed | MarketBeat 2026-08-08 |
| AVGO | custom chips (TPU v8i, MTIA, OpenAI) + Ethernet | both, inference-tilted | AI 56% (Q3 FY26) | Pulse2 2026-09-03 |
| MRVL | custom chips (Trainium/Maia) + optics DSP | both, inference-tilted | DC % not found (Q2 FY27 revenue $2.74B) | Marvell IR 2026-08-27 |
| ALAB | scale-up PCIe/UALink switches (Scorpio) | both, inference-tilted (our reading) | about 100% DC | Astera IR 2026-08 |
| ARM | CPU IP (Grace/Vera, Graviton, Axion) | both, inference-tilted | DC % not disclosed | Arm 2026-07-29 |
| AMD | Instinct GPU + EPYC | both | DC 58% (Q2 2026) | AMD 8-K 2026-08-04 |
| MediaTek 2454.TW | Google TPU partner (press understanding) | likely inference | DC target over $2B in 2026 | MediaTek prepared remarks 2026-07-31 |
| Alchip 3661.TW / GUC 3443.TW | ASIC design services | unclear | AI share not stated | 2026-08 |
| TSM | foundry for everything | both | HPC 68% (Q3 2026; HPC is not AI) | TSMC 6-K 2026-10-15 |
| MU | HBM + DRAM + SSD | both | DC 63% (FQ4'26); HBM line not reported | Next Platform 2026-10-02 |
| SK hynix / Samsung | HBM | both, training-tilted (tied to GPU packages) | Q3 due 10-29 | S&P 2026-08-05 / TrendForce 2026-10-08 |
| WDC / STX | HDD bulk storage | weak link to inference | WDC cloud 89% (single source) | 2026-07/08 |
| NVDA | GPUs + NVLink + networking + Groq LPX | both | DC 92.5% (Q2 FY27) | NVIDIA IR 2026-08-26 |
| ANET | scale-out Ethernet | both, training-tilted | AI target ≥ $3.5B of $12.6B (about 28%, target) | ANET 8-K 2026-08-04 |
| COHR / LITE / FN | 800G–1.6T optics | both, training-tilted | COHR DC&Comms 79%; FN DC 51%; LITE not found | 2026-08 |
| CIEN / CSCO | data-center interconnect, switching | both | CIEN cloud 53%; CSCO hyperscaler AI about $4B FY26 | 2026-08/09 |
| CRDO | active electrical cables (in-rack) | both | split not disclosed | 2026-09-01 |
| CRWV / NBIS / IREN | neoclouds | both (NBIS Token Factory = inference tilt) | about 100% / 98% / 51% AI cloud | 2026-08 |
| ORCL | AI cloud ("training and inferencing") | both | IaaS 38%; RPO $664B | Oracle 2026-09-10 |
| MSFT / GOOGL / AMZN / META | landlords; TPU/Trainium/MTIA are inference-first chips | both | Cloud 44% / 21% / 21% / n.a. | Q2 calls 2026-07 |
| VRT / ETN / GEV / BE / CEG / VST | power and cooling | both (power does not care which workload) | DC % mostly not disclosed | 2026-07/09 |

- **Purest public inference exposure:** CBRS, the eSSD/NAND names (Kioxia, SNDK), and server CPUs (INTC, AMD EPYC, ARM). The 1:1 CPU:GPU ratio is a forward scenario.
- **Training-tilted:** HBM and scale-out optics/switching. "Pure training" does not exist as a reported revenue line.

---

## 4. Testable version

### Rules (written before any return was computed; `scripts/thesis_tracker.py`, rules sha256 `b745e0d3aa0e`)

- **inference_v1** (9 names, equal weight, rebalanced daily):
  - names labelled "inference": CBRS, SNDK, INTC, QCOM, NET;
  - names labelled "both, inference-tilted": AVGO, MRVL, ALAB, ARM.
  - Kioxia is excluded because it is not US-listed.
- **training_v1** (8 names): the leg the user wants to trim, "pure training GPUs and hyperscalers": NVDA, MSFT, GOOGL, AMZN, META, plus training-tilted scale-out ANET, COHR, LITE.
- **Mechanics:** US listings, yfinance adjusted closes, equal weight with daily rebalance. A name enters on its first trading day. Benchmarks are SMH and SPY.
- **Start dates stated up front:** 2023-01-03 (post-ChatGPT), 2025-01-02, 2026-01-02.
- **Freeze 2026-10-22.** Only returns after this date are out-of-sample.

### Realized history (prices through 2026-10-21; run 2026-10-22 07:14Z)

| since | inference_v1 | training_v1 | SMH | SPY | inf/trn level ratio | corr | months spread > 0 |
|---|---|---|---|---|---|---|---|
| 2023-01-03 | +936.9% (vol 44%, maxDD -38%) | +663.9% (vol 32%, maxDD -36%) | +460.4% | +112.8% | **+35.7%** | 0.79 | 46% of 46 |
| 2025-01-02 | +260.0% | +126.1% | +128.1% | +34.8% | **+59.2%** | 0.81 | 55% of 22 |
| 2026-01-02 | +98.3% | +38.6% | +49.1% | +14.1% | **+43.1%** | 0.74 | 50% of 10 |

**Robustness: the ratio after removing names**

| Removed | since 2023 | since 2025 | since 2026 |
|---|---|---|---|
| CBRS | +49.5% | +75.3% | +57.6% |
| SNDK | **-11.5%** | +3.8% | +20.5% |
| SNDK + INTC | -9.1% | **-12.1%** | +10.6% |

**Per-name returns since 2026-01-02**

| Basket | Name | Return |
|---|---|---|
| inference | SNDK | +401% |
| inference | MRVL | +174% |
| inference | INTC | +136% |
| inference | ARM | +98% |
| inference | ALAB | +62% |
| inference | NET | +57% |
| inference | QCOM | +5% |
| inference | AVGO | -2% |
| inference | CBRS | -62% (since 2026-05-14 first close of $311) |
| training | LITE | +120% |
| training | ANET | +38% |
| training | COHR | +22% |
| training | NVDA | +18% |
| training | AMZN | +16% |
| training | MSFT | +13% |
| training | META | +8% |
| training | GOOGL | +6% |

**How to read this:**
- **This is not a backtest of a forecastable edge.** The baskets were picked in Oct 2026 by someone who already knows which companies became AI names. That is selection bias, and several names did not exist for the early windows: SNDK listed 2025-02, CBRS 2026-05, ALAB 2024-03, ARM 2023-09.
- One cyclical name (SNDK, +4,000% since listing at $36) produces most of the 2023 and 2025 spread.
- Monthly hit rates of 46–55% look like a coin flip with fat tails, not a steady premium.
- The 2026 YTD gap survives removing SNDK and INTC. That is the most supportive piece, but it covers 10 months during a memory and CPU price spike.

### Forward tracker

- `scripts/thesis_tracker.py tick` logs both basket levels plus SMH/SPY, indexed to 1.0 at the 2026-10-22 close, to `data/thesis/tracker.jsonl`.
- `report` prints the forward record.
- To run it daily after the US close, install with: `bash scripts/install_thesis_tracker_agent.sh`. **Not installed.** It installs launchd agent `com.jonathanwang.quant-thesis-tracker`, weekdays 17:30, from `~/.claude/quant-kalshi`, and pip-installs yfinance into that venv.
- Basket changes must be a new version (`inference_v2`) with its own freeze date. Never edit v1.
- A few months of forward data will not separate skill from the memory cycle. Judge the thesis on the indicators below.

---

## 5. Leading indicators (check at each date; these confirm or refute faster than prices)

| Indicator | Confirms | Refutes | Next datapoint |
|---|---|---|---|
| Hyperscaler capex guide + token/inference commentary | 2027 capex raised, with inference/agents named as the driver | guide cuts; "digestion" language | GOOGL/MSFT/META/AMZN Q3 calls, late Oct 2026 (dates unconfirmed) |
| TrendForce DRAM / NAND / eSSD contract prices | q/q increases continue into 1Q27 | flat or negative q/q | TrendForce monthly; SK hynix/Samsung 10-29; STX 10-27 |
| Inference revenue mix disclosed | NVDA/AMD/AVGO start quantifying it | silence, or a training-led narrative | NVDA Q3 FY27 (about late Nov) |
| Custom chip (ASIC) ramps | TPU v8i / MTIA / Trainium / OpenAI XPU volumes; AVGO Q4 AI ≥ $21.7B guide | guide misses, delayed tape-outs | AVGO FQ4 (about Dec); MRVL FQ3 (about early Dec) |
| CPU:GPU attach in deployed racks | INTC/AMD/ARM report agent-driven server CPU growth plus price increases holding | CPU prices cut; inventory build | INTC/AMD Q3 calls (late Oct / early Nov) |
| Lab compute split | SemiAnalysis/Epoch show the inference share rising above about 40% | stays at 29–39%, or RL rollouts take the growth | next SemiAnalysis / Epoch Chip Users update |
| Token volumes vs price deflation | token growth (Google 7x y/y) above price decline (about 13x/yr at fixed capability), so token revenue rises | token revenue flat while volumes grow | Google I/O and earnings; OpenRouter "Inferred" |
| Inference specialists | CBRS revenue at or above its $855–865M FY26 guide; OpenAI 750 MW ramp on schedule | guide cuts | CBRS Q3 (about Nov) |

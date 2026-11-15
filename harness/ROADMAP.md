# Zebra finch encoder roadmap (canonical)

Updated 2026-11-15. This file replaces docs/roadmap.html (frozen at Oct 17, superseded). The report doc
https://claude.ai/artifact/Q3Bf46TNUyRHh8cZZEdH1M has a "Roadmap" section last synced 2026-11-05 (NOT yet updated with the
2026-11-15 results below). On resume the user wants: **where we are, possible leads, next steps**, in under 15 lines.

Goal (hub goal zf): specialise a HuBERT-architecture encoder to zebra finch, judged on 11-class call type
(CPU board, leave-birds-out). BEANS and other species are out of scope (user, 2026-10-15). Shipping a non-HuBERT model
(Whisper, Perch, an ensemble) would change the goal: the user's call.

## Where we are (2026-11-15)
- **E36 done (finding 090):** vanilla HuBERT on our 116 h, lr 1e-4, two seeds, mean 0.8388. No recipe arm is distinguishable
  from it: run16 -0.0067, run19 (AVES labels) +0.0114, E13 (BirdAVES-large labels) +0.0138, E30 -0.0126 (birds + seeds; L3 and
  10-split agree). The stock lr 5e-4 does not train at our batch size. The AVES-family arms are positive in every seed pair but
  unresolved.
- **E37 done (088 shuffle, 092 randproj):** shuffled labels collapse to 0.633. Random-projection labels: NOT MET as
  pre-registered — tie run16, but resolve ABOVE run23 (MFCC k-means) at layer 3 (+0.0186, df-1 seed term; borderline under the
  pooled seed sd). No evidence k-means beats a random quantiser of the same MFCC frames. Forward rule: pre-register the seed
  variance for one-seed arms.
- 090 is scoped to four from-scratch arms; run20, run29, randproj and the DAPT groups are not yet contrasted with vanilla.
- **New: a probability-mean ensemble beats every single encoder (finding 091, confirmed).** Whisper base + Whisper small +
  BirdAVES bioxn-large 0.8816 vs best single 0.8646, +0.017 Bonferroni [+0.002, +0.030]; with layers chosen inside training
  folds +0.021 [+0.009, +0.035]. Concatenating features does not resolve. Our run19 is as good a partner for Whisper as
  BirdAVES-large. Not a single HuBERT, so it does not meet the goal by itself.
- **The test (zfeval/experiments/power, exploratory, not critic-checked):** interval variance = birds 50%, seeds 29%, fold split
  20%. Today's 2-seed arm test has 43% power for a true 0.02, 14% for 0.01. Mean p(true class) over 10 splits resolves 14/66
  recipe pairs vs 4/66 for accuracy (0/14 null pairs either way); with it, 4-5 seeds per arm give ~80% power for 0.02. A true
  0.01 is out of reach with 48 birds. Clips wrong for every model: 41 (not 174); no labelling ceiling binds.
- **Data (scout, 2026-11-15):** no other lab has released ZF call-type labels in Elie & Theunissen's scheme; every public
  11-type set is our release. Independent evaluation can come from caller identity (V33DA, Hahnloser lab, 10 birds, HF;
  D'Amelio 2017 Gahr lab, Dryad 4g8b7) — not yet verified by us. Biggest pretraining source: Rockefeller song library
  (Tchernichovski, 280 males; license unknown, ask the lab). Published baselines on our release: Riad et al. 69%, Elie 2024 ~60%.
- Point-estimate leaders (single models): Whisper base 0.8646, Whisper small 0.8599, BirdAVES bioxn-large 0.8590; ours
  run19 s2 0.8546, E13 s2 0.8538.

## In flight
- 40654080 E36 stock seed 2 (paid A40, ~860-1,500 SU). Cannot change E36's verdict; user asked whether to cancel / move to lowprio.

## Next steps, in order (cheap first; nothing below spends credits without the user's yes)
1. Pre-register the sharper test (mean p(true), 10-split, fixed layer rule, pooled seed sd) via /prereg + critic; re-judge
   090/092 and the AVES-family question with it (CPU, free).
2. X1 frozen screen of new model classes vs Whisper-small: Perch 2.0, Dasheng-base, BEATs, Whisper-medium (Mac CPU, free).
   Smoke test: Dasheng end to end.
3. X2 supervised fine-tuning, nested leave-birds-out, fine-tuned vs frozen on the same folds: run19 arm on Mac CPU (~3 h),
   Whisper-small arm on CPU (~1 day) or free lowprio. 036's old fine-tuning result was MPS (pre-076) -> new reason.
4. Identity probe on V33DA / D'Amelio (frozen embeddings): the only independent-bird test available.
5. If the AVES-family gap matters: 2-3 more seeds per arm (run19/E13 vs vanilla) under the new statistic (~1,000 SU each paid, or free lowprio).
6. X3 / E31 (continued SSL on Whisper-small) only if X2 resolves >= +0.02.
7. Savio cleanup (user's yes): E13 relabel features (~165 GB), 14 GB BEANS copy.

Designs for X1-X3: see 2026-11-15 notes in harness/pi_notes.md (experiment-designer drafts).

## Decisions waiting on the user
- Goal scope: may the shipped model be non-HuBERT (Whisper/Perch) or an ensemble?
- 40654080 cancel / keep / move to free queue.
- E36b, E35, E32, run23 seed 2, E15, E26, E33 as before (lower priority now that recipe arms cannot be ranked).

## Leads (ideas, not pre-registered)
- Ensemble distillation: train one HuBERT student on the 091 ensemble's soft call-type targets? (043 found distillation did
  nothing on call type with a weaker teacher; the ensemble is a stronger one.)
- Tet-focused contrast: AVES-label arms +0.06 Tet recall (zfeval/experiments/ensemble_tet, exploratory, 3/10 resolved); a
  confirmatory test needs new seeds, and birds dominate its variance.
- Where Whisper wins (081: Tet and Nest).

## Closed lines (do not reopen without a new reason)
Log-mel front end (087) · stronger teacher among AVES-family models (089) · compute past 4x (054) · AVES's own recipe knobs
(073) · continued pretraining from AVES/BirdAVES for call type (075, 084) · iteration-2 own labels (056) · BEANS (retired
2026-10-22) · stronger probe heads (042; dropped by designer 2026-11-15).

## Pointers
STATE: harness/STATE.md · notes: harness/pi_notes.md · register + frozen rules: harness/experiments.yaml · findings:
knowledge/findings/ (index knowledge/index.md) · analysis JSONs: ~/zf_labelset/zf_detection_dataset_v1/analysis/ · scorer:
harness/score_run.sh · report doc: https://claude.ai/artifact/Q3Bf46TNUyRHh8cZZEdH1M (doc id ba903e67-b3f0-47aa-9de1-fd7903adac44).

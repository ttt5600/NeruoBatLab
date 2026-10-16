# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-16 09:20 (lead: session_01EGDWEvYFd6zSmWQfUugJnH)

## USER PRIORITY (2026-10-15): ZF specialization judged on CALL TYPE; skip BEANS. Wants DAPT-for-call-type work and
visual per-call-type notebooks. Memory: feedback_specialization_not_beans.md; PI.md backlog item 0.

## !! MPS BUG (found 2026-10-16 08:40): torch 2.8 MPS SDPA wrong for 2-8 frame inputs (T=1, T>=9 fine; CPU fp32 = fp64
to 1e-5). 1,940/3,412 call-type clips mis-encoded in EVERY ct11_<tag>_emb.npy (MPS). AVES-bio: MPS 0.8453 L3 -> CPU
0.8438 L0. Memory: reference_mps_sdpa_short_sequence_bug.md. Never encode short clips on MPS.

## In flight (Mac)
- calltype_cpu_rescore.py (task b3qhkia2w, log $CLAUDE_JOB_DIR/tmp/cpu_rescore.log): 55 models, ~140 s each (~2.5 h,
  started 09:03) -> features/ct11_<tag>_cpuemb.npy + analysis/calltype_cpu_rescore.json. Resumable (skips done tags).
- Savio: E8 39945354, E9 39945376 PENDING. Cert to 10:44 today.

## Next action (in order, when the re-score ends)
1. Check FAILED tags in the log (HF wrappers). Then: calltype_errors.py (now CPU default) -> dapt_calltype_by_type.py
   (rewritten: downloads control, L3-vs-L3, round1 on CPU, error-removal view) -> rerun notebook 08 (sections 8/9/11
   rewired to the JSON; conclusions STILL MPS-era wording: rewrite per critic + CPU numbers) -> rebuild + republish the
   browser (same file path; url 8vsjrxjswggK2eZpuBxUbu).
2. Finding 076 (MPS correction): board before/after per model, rank changes, scope audit of other MPS users
   (chick_holdout_variants, finetune_calltype, synth_timing, attentive_probe, distill_*, randinit, beans_*: check input
   lengths). Critic. Then rewrite 075 on CPU numbers per its critic (findings/075 caveats list the changes). Critic.
3. E33 pitch pre-test on CPU (script ready; reads CPU caches + calltype_cpu_rescore.json best layers).
4. Rebuild model board / comparison with CPU call type BESIDE the MPS column (never replace past numbers).

## Waiting on the user
- E33 go/no-go after the CPU pre-test. Cancel E9 (FSD-only) for a call-type DAPT slot?
- sv login before E8 starts (~Oct 17 05:00). Older: E30-32, E26 donors, resume fix, label audit.

## Tasks (TaskList): #1 re-score (running) #2 notebook+browser on CPU #3 076 + 075 #4 E33 pre-test. Digest done.

# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-17 02:00 (lead: session_01EGDWEvYFd6zSmWQfUugJnH)

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS. Wants visual notebooks, and tasks HANDED OFF to
agents with only summaries pasted back (2026-10-16). Memory: feedback_specialization_not_beans.md.

## MPS BUG (076 draft, critic running): torch 2.8 MPS SDPA wrong for T=2..8 frames. CPU re-score done for 47/55
(analysis/calltype_cpu_rescore.json; features/ct11_<tag>_cpuemb.npy). USE ONLY CPU call-type numbers from now on.
Board moved -0.013..+0.014, ranking changed (bioxn-large 0.8590 now top; E20 daptv3aves_2e4s2 0.8570; E23 0.8482).

## Delivered since 2026-10-16
- Savio `lab eval MODEL` (eval_model.py + slurm/eval_model.sh, MODEL= at top; INTERN.md s7; pack lab_data/eval/
  calltype11_pack.npz; shared leaderboard lab_data/eval/results). Job 40036098 (E23) queued -- result unseen.
- Notebook 07 + model facts (notebooks/model_facts.py/.csv); notebook 08 + browser v2 on CPU (8vsjrxjswggK2eZpuBxUbu).
- Agents' summaries pasted to user: paper 2610.16382 (BeVo; split leaks; AFDD descriptors useful), lit review
  (WiSE-FT etc.), size-matched comparison (same-size bar biox-base = aves-all 0.8476; ours level with it).

## In flight
- Agents: HF re-score (8 tags, transformers env), cohort audit (duplicates / Ne-Wh label conflict / parser drops ->
  analysis/cohort_audit.json), critic for 076.
- Mac: E33 pitch pre-test on CPU (task brmq5yq1s, log $CLAUDE_JOB_DIR/tmp/e33_pretest.log -> analysis/e33_pitch_pretest.json).
- Savio: BRC CERT EXPIRED 10:44 Oct 16 (notified). E8 39945354 / E9 39945376 / eval 40036098 status unknown.

## Done 01:50-02:00: 076 CONFIRMED (MPS bug); 075 CONFIRMED (DAPT = any AVES-family shift vs AVES-bio; E20 gain is the
board split's best case, 1/10 splits); cohort audit (analysis/cohort_audit.json): duplicates harmless (<=0.0033), seed-0 split
favourable -> propose a 10-split-mean column BESIDE the board (user decision; never replace).

## Next action
1. E33 pre-test result (AVES-bio half: pitch +-1 st cuts bird-ID 0.73->0.59-0.62 but ALSO call type 0.8438->0.8057 at +1) -> E33 entry + tell user. HF re-score -> add to 076 table.
4. Notebook 07 call-type panel -> CPU numbers (build_layer_sweeps reads <tag>_calltype.json = MPS; switch to cpu JSON).
5. When the user logs in: sv status; check eval job 40036098 (CPU vs CUDA agreement), E8/E9.

## Waiting on the user
- sv login. E33 go/no-go after pre-test. Cancel E9? Hand off BeVo split comparison + cohort fix (asked 10:20 Oct 16).
- Older: E30-32, E26 donors, resume fix, label audit.

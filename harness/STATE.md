# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-15 23:06 (lead: session_01EGDWEvYFd6zSmWQfUugJnH)

## USER PRIORITY (2026-10-15 ~22:55): ZF specialization judged on CALL TYPE; skip BEANS / other animals.
Wants: DAPT work aimed at call type + visual per-call-type notebooks (heatmaps, errors, spectrograms, confidence,
distance to cluster). Memory: feedback_specialization_not_beans.md. PI.md backlog item 0.

## Delivered this session
- notebooks/07_layer_sweeps.ipynb (user: "this is great"); 08_calltype_errors.ipynb (11 sections); error browser
  https://claude.ai/artifact/8vsjrxjswggK2eZpuBxUbu (docs/calltype_error_browser.html, gitignored, 12 MB; rebuild with
  notebooks/build_calltype_error_browser.py). Per-clip data: analysis/calltype_errors.npz (calltype_errors.py).
- Key results (notebook 08 outputs): no DAPT checkpoint beats its parent overall; per type DAPT trades Te/Tu down for
  Ag/Th up in nearly every checkpoint; Tu recall ~0.40 leave-birds-out vs ~0.72 random split -> errors are about new
  birds; AVES full fine-tune ~0.860 = best frozen probes -> small headroom.

## In flight
- Mac: round-1 DAPT (ZF-only) call-type scoring, 12 ckpts (task bkw35zii1, log $CLAUDE_JOB_DIR/tmp/score_r1_calltype.log);
  then E33 pitch pre-test (task bpm0vv91w, -> analysis/e33_pitch_pretest.json).
- Savio: E8 run19 seed 2 job 39945354 PENDING; E9 run20 seed 2 job 39945376 PENDING. Cert valid to 10:44 Oct 16.

## Next action (in order)
1. When scoring ends: re-execute notebook 08 (section 9 = round-1 trajectory), write section 9's reading, commit.
2. When the pre-test ends: read e33_pitch_pretest.json; add to notebook 08 / E33 entry; tell the user.
3. Candidate finding 075 (DAPT per-type trade + bird generalisation) -> critic before confirming.
4. E8 done: score_run.sh run19_seed2 (call type + detection only; NO BEANS per user).

## Waiting on the user
- E33 voice-invariant DAPT go/no-go (needs code: --pitch-shift-semitones). Pre-test result first.
- E9 (FSD50K-only seed 2) is anti-specialization: cancel to free a slot for a DAPT call-type run? User's call.
- Older: E30/E31/E32 go/no-go, E26 donor source, resume-fix, 174-clip label audit (browser "All three wrong" view helps).

## Clocks
- BRC cert to 2026-10-16 10:44; E8 est. start Oct 17 05:00 -> needs sv login again before then.
- Next digest after 09:00 Oct 16.

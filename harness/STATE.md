# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-17 20:35 (lead: session_01EGDWEvYFd6zSmWQfUugJnH; written as the resume point before a /clear)

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS. Deliver visually (notebooks/artifacts). HAND OFF
tasks to agents/sessions and paste back only summaries. Memory: feedback_specialization_not_beans.md.

## Settled since Oct 15 (all committed)
- 076 CONFIRMED: Mac MPS attention wrong for 2-8 frame inputs -> every Mac call-type cache was wrong. ALL 55 models now
  re-scored on CPU: analysis/calltype_cpu_rescore.json (+ _summary.json); USE ONLY these. Whisper/AST unchanged (0 clips).
  CPU top: whisper-base 0.8646, whisper-small 0.8599, bioxn-large 0.8590, E20 DAPT 0.8570; AVES-bio 0.8438.
- 075 CONFIRMED: DAPT does not raise call type beyond what any AVES-family model shows vs AVES-bio; E20's resolved gain
  is the board split's best case (1/10 splits). Soft calls Tu/Th/Ne/Te = ~50% of clips, ~80% of errors.
- Cohort audit (analysis/cohort_audit.json): 33 duplicate copies, 1 Ne/Wh double label, 21 parser drops -> harmless
  (<=0.0033); board's fold split (seed 0) is favourable -> PROPOSED: 10-split-mean column beside the board (user's call).
- E33 pre-test done (experiments.yaml E33 pretest_result): pitch = voice axis but also separates tuck vs thuk/tet ->
  recommended NOT to run E33 at +-2 semitones.
- Savio `lab eval MODEL` built + pushed (INTERN.md s7; pack lab_data/eval/calltype11_pack.npz; leaderboard
  lab_data/eval/results). First job 40036098 (E23) queued Oct 16 -- RESULT NEVER CHECKED (cert expired).
- Notebooks (kernel "ZF analysis (Python 3.10)" = zf-py310): 07_layer_sweeps (CPU call type + model facts table,
  notebooks/model_facts.py/.csv), 08_calltype_errors (CPU). Error browser v2: https://claude.ai/artifact/8vsjrxjswggK2eZpuBxUbu
- Jupyter server (Anaconda, port 8888, token in $CLAUDE_JOB_DIR/tmp/jupyter_token) may still be running.

## NEW USER REQUEST (not started; hand off to an agent)
Embedding automation (Savio `lab embed` + Mac): MODEL + wav folder/list -> embeddings, options: (A) detection windows of a
chosen length/hop (e.g. 1 s or 0.5 s; virtual chunking by offsets, no per-chunk wavs; one output file + index TSV);
(B) frame-level (20 ms) embeddings of continuous files for onset/offset decoding. Reuse eval_model.py's loader + CPU/GPU
check. Also ANSWER the user's question: how speech systems get word onsets/offsets from continuous audio (VAD frame
classification + smoothing/hysteresis; keyword spotting sliding windows; CTC spikes + forced alignment (torchaudio /
MFA / WhisperX); DCASE SED thresholds + collars) and map it to ours (findings 018, 022, 023, 032; temporal-resolution memory).

## Waiting on the user
- sv login (cert expired 10:44 Oct 16): then `sv status`, `sv eval list` (job 40036098: CUDA vs CPU agreement), E8/E9.
- Decisions: E33 (rec: no at +-2 st), 10-split column, cancel E9?, cohort fix (versioned), BeVo split comparison (agent
  plan in chat: our probe on their split; map at analysis/bevo/), AVEX as a baseline, WiSE-FT weight-average test.
- Older: E30-32, E26 donors, resume fix, label audit.

## Resume: /loop follow harness/PI.md  (first wake: read PI.md, AGENT.md, this file, knowledge/BRIEF.md, pi_status.sh)

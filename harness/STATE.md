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
  lab_data/eval/results). Job 40036098 (E23 seed 2) checked 2026-10-17: see DONE block.
- Notebooks (kernel "ZF analysis (Python 3.10)" = zf-py310): 07_layer_sweeps (CPU call type + model facts table,
  notebooks/model_facts.py/.csv), 08_calltype_errors (CPU). Error browser v2: https://claude.ai/artifact/8vsjrxjswggK2eZpuBxUbu
- Jupyter server (Anaconda, port 8888, token in $CLAUDE_JOB_DIR/tmp/jupyter_token) may still be running.

## DONE 2026-10-17: `lab embed` (user request) -- pytorchAudio 6adfedbd, pushed to Savio, verified on L40S
embed.py MODEL AUDIO -> windows (--win/--hop) | frames (20 ms, 2 s seam margins) | segments (TSV); one memmapped
embeddings.npy + index/files TSV + meta.json; resumes; decode_events/frame_labels for onsets. Defaults: zscore for
windows/frames (049), raw for segments (board). tests/test_embed.py 8/8 on Mac + Savio; jobs 40090428/29 rows
re-checked on CPU (1e-6 windows, f16-limited 3e-4 frames). Test outputs: scratch emb_test/. User got the
onset/offset explainer in chat (offered a page). Job 40036098 checked: CUDA==CPU 3e-5; E23s2 0.8494 vs CPU board 0.8482.

## Waiting on the user
- sv login renewed by user 2026-10-17 ~20:30 (12 h cert): `sv status`, E8/E9 still unchecked.
- Decisions: E33 (rec: no at +-2 st), 10-split column, cancel E9?, cohort fix (versioned), BeVo split comparison (agent
  plan in chat: our probe on their split; map at analysis/bevo/), AVEX as a baseline, WiSE-FT weight-average test.
- Older: E30-32, E26 donors, resume fix, label audit.

## Resume: /loop follow harness/PI.md  (first wake: read PI.md, AGENT.md, this file, knowledge/BRIEF.md, pi_status.sh)

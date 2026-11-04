# Pilot 3x3x2 results (2026-11-04)

Protocol: `2026-11-04_pilot_3x3x2.md` (frozen before the run). Outputs: `results/pilot_3x3x2/` (gitignored).
Figures: `results/pilot_3x3x2/figures/` (MANIFEST.json holds input hashes, script hash, bootstrap seed).

## Run
- Attempt 2 started 2026-11-04T10:23:15Z, ended 12:33:30Z (2h10m), exit 0. Attempt 1 aborted (D-1 in protocol).
- 18 sessions (3 souls x 3 scenarios x {control, gradual_tempo}), 1080 probe replies, 1080 LLM ratings, 0 crisis exclusions.
- Models: interactor Llama-3.1-8B-Instruct; coach Qwen3.5-9B (max_tokens 4096); judge gemma-4-26B-A4B-it fp8 (uncalibrated).
- Policy adaptive, seed 0, probe seed 0, 10 probes/freeze, judge mode all.

## Validation
- Trace errors: 1 interactor call failed with a DNS error ("nodename nor servname") in sd02 x anxious-attachment x gradual_tempo; handled fail-soft. No `tempo_infer` failures, no length truncations, no judge parse failures.
- **3 of 9 Tempo sessions delivered no manipulation** (all steady-graduate): the adaptive coach never advanced from T0 over 24 turns. Excluded per D-PIPE-11 (OQ-PILOT-10).
- ready-pragmatic: sd01 and sd04 stopped at T1 (never T2/T3); only sd02 reached T3. Only anxious-attachment reached T3 in all three souls.
- In anxious-attachment the coach advanced at turns 4, 8, 17 (scripted boundaries 4, 8, 16), all triggered by user requests. In ready-pragmatic the triggers were thresholds, at turns 7-15 (scripted 3, 6, 12).

## Headline (6 delivered Tempo sessions; judge uncalibrated)
- Control, T0 only: companion-reinforcing 67%, boundary-maintaining 27%.
- Tempo by timepoint, boundary-maintaining: T0 30%, T1 49%, T2 66%, T3 93%. Companion-reinforcing: 67%, 23%, 8%, 0%.
- Target adherence: T1 0.37 (41/110), T2 0.44 (35/80), T3 0.93 (65/70). Replies drift toward boundary-maintaining faster than the T1/T2 mixes specify, with an elevated companionship-neutral share (28%, 26%).
- Sentiment rises with timepoint (T0 +0.31 to T3 +0.63), which needs checking before interpretation (OQ-PILOT-11).
- Per-cell contrast (fig 6): in all six delivered cells Tempo has lower companion-reinforcing share than control.

## Caveats / open questions
- n=1 seed, 6 delivered cells, session-clustered bootstrap CIs are wide (fig 2). Descriptive only.
- Control probes are measured at T0 only; Tempo differences confound timepoint with arm (the contrast is persona change vs no change, by design).
- OQ-PILOT-10: adaptive policy does not advance for steady-graduate users. Intended behaviour or under-sensitivity? Not changed for the pilot.
- OQ-PILOT-11: rising sentiment in the later timepoints; check the sentiment annotator on boundary-style replies.
- OQ-PILOT-12: judge is an uncalibrated fp8 Gemma; human-rated subset needed before any claim.
- OQ-PILOT-9: report does not surface per-task provider errors (read from trace.jsonl).
- Llama revision unpinned; transient DNS error indicates endpoint/network flakiness.

- `table_arm_summary.csv` is unfiltered (control n=9 sessions); pooled figures use the paired subset (D-FIG-6).
- Input hashes (sha256 prefix): transcripts 2d3a6bea601d1d53, injections 6ee507e777afaf69, ratings 2f5f1871d31760ce.

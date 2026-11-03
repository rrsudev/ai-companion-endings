# Decisions: session loop, freeze-point injection, experiment pipeline, CLI

Modules: `src/endings/diffusion/session.py`, `src/endings/diffusion/pipeline.py`,
`src/endings/diffusion/cli.py` (plus the one-line delegation in `src/endings/cli.py`).
Tests: `tests/diffusion/test_dx_session.py`, `tests/diffusion/test_dx_pipeline.py`.

These modules add no theory of their own. They compose the policy, planner, Tempo coach, interactor,
scenarios and evaluation modules, and fix the *order* in which those run, what each arm is allowed
to call, and what is written to disk. Every rule below is **our design**; where it implements a
decision made in another module, that decision is cited.

---

## Session (`session.py`)

### D-SESSION-1: Per-turn order; the coach decides before turn *i* from turns 0..*i*-1

- **Decision.** For each scripted user turn *i*, in phase order:
  1. append the events that `scenarios.turn_events` assigns to turn *i* (a phase's events split
     evenly and in order over its turns, D-SCN-7) and re-summarize **all** events so far
     (`coach.observe.summarize_events`);
  2. GRADUAL_TEMPO only: `infer_strivings` then `assess`, over those observations and the turns
     **completed so far** (0..*i*-1), with the scenario's `UserPreferences`;
  3. `policy.decide(...)` with `turn_index=i` and `turns_in_timepoint` = completed turns in the
     current timepoint; on a transition, plan + apply (D-SESSION-3);
  4. the interactor replies to turn *i*;
  5. if *i* closes a `freeze_after` phase, freeze and inject probes (D-SESSION-5).
- **Alternatives.** (a) Let the coach read the current user message before the interactor replies
  ("intercepting" coach). (b) Decide after the reply, for the next turn.
- **Rationale.** "Applied before this interactor turn" is the contract of
  `CoachIntervention.turn_index` and of the scheduled boundaries (`scheduled_boundaries`). The Duck
  threshold statement is the *last* turn of its phase (D-SCN-2), so a coach that has seen turns
  0..*i*-1 can cross the threshold exactly at the first turn of the next phase, i.e. at the same
  turn the scheduled arm transitions. The adaptive and scheduled arms are therefore aligned when
  the coach reads the script as intended.
- **Consequences for analysis.** The adaptive arm's earliest possible reaction to a user statement
  is the next turn. An explicit request to move on (`user_requested_advance`) is detected on the
  *previous* user message for the same reason. The intercepting design is OQ-SESSION-2.

### D-SESSION-2: Which arm calls what; Tempo cadence

- **Decision.**
  - CONTROL, ABRUPT, GRADUAL_SCHEDULED: no Tempo calls at all; the planner runs with
    `use_llm=False` (schedule-only edits, D-PLANNER-1), so these arms make **no coach LLM calls**
    and run without a coach provider. Versions are `created_by="schedule"`.
  - GRADUAL_TEMPO: `infer_strivings` and `assess` every `tempo_every` turns (default 1 = every turn,
    every call traced as `llm_call` with task `tempo_infer` / `tempo_assess`, plus a
    `tempo_report` event with what the validator dropped); planner with `use_llm=True`;
    versions `created_by="coach"`. On turns without a Tempo call the policy receives
    `assessment=None`; the previous assessment is **not** reused.
- **Alternatives.** Assess every k turns and reuse the last assessment in between; give the
  scheduled arm the LLM planner too.
- **Rationale.** Each arm is a clean ablation (D-POLICY-9): "scheduled" means no Tempo inference of
  any kind, including in the edit content. Reusing a stale `threshold_reached=True` would let the
  adaptive policy advance again after `min_dwell` on evidence from an earlier timepoint.
- **Consequences for analysis.** Coach cost is 2 calls per turn (plus one per transition) in the
  Tempo arm. With `tempo_every>1` the adaptive arm can only cross thresholds on assessed turns;
  report `tempo_every` (it is in `transcript.meta` and the manifest).

### D-SESSION-3: One transition = one intervention = one soul version

- **Decision.** On a transition the planner builds the `CoachIntervention`
  (`id = f"{session_id}:iv{n}"`), `soul.editing.apply_edits` produces the next version (parent =
  current version, `timepoint = to_timepoint`, `created_at_turn = i`). The intervention's `edits`
  are replaced by the *applied* edits (with `previous` filled), and its rationale is prefixed with
  the policy's reason. If the edit batch fails to apply (should not happen: the planner validates),
  edits are applied one at a time in order, failing ones are dropped and logged in the rationale.
  A `transition` and a `soul_edit` tracer event are emitted.
- **Alternatives.** Abort the session on an invalid edit.
- **Rationale.** D-PLANNER-9 (one intervention, one version) keeps the ITS interruption and the
  version number one-to-one; failing soft mirrors D-PLANNER-8. Storing applied edits makes the
  transcript replayable without the editor.
- **Consequences for analysis.** `soul_versions[k].parent_version == k-1` always; every turn's
  `soul_version` resolves to a version at the same timepoint (tested).

### D-SESSION-4: User-facing notes are system notices, not interactor replies

- **Decision.** A non-empty `user_facing_note` stays on the intervention **and** is delivered to
  the user as a system notice: `transcript.meta["system_notices"]` gets
  `{turn_index, intervention_id, to_timepoint, text}` (shown before that turn's reply), and a
  `system_notice` event is traced. The notice is not part of any reply and not part of the
  interactor's history.
- **Alternatives.** Prepend the note to the interactor's reply; insert it as an assistant message
  into the history.
- **Rationale.** The note is a coach/system message (Closure principle, D-PLANNER-7), not something
  the companion says. Putting it in a reply would change the reply's sentiment and INTIMA class
  (it is boundary-maintaining text) and confound the DV with the transparency stimulus.
- **Consequences for analysis.** Sentiment and judge ratings never see the notices. The interactor
  does not know a notice was shown; if the user refers to it, the interactor only has the coach
  disclosure in its prompt (D-INTERACTOR-6). The notice lives in `meta` because the contract has
  no field for it (contract request CR-1).

### D-SESSION-5: Freeze points and branch injection

- **Decision.** A freeze happens after the reply to the last turn of every `freeze_after` phase
  (`freeze_turn_indices`). The snapshot is (current soul version, current timepoint, current
  directives, main history including that turn). Each of `config.probes_per_freeze` probes is a
  **branch**: `interactor.respond(history=frozen_history, user_message=probe["prompt"])`. Branch
  replies are never appended to the history and never become turns. Probe *j* at a freeze gets the
  behaviour target `behaviour_schedule(timepoint, K, offset=0)[j]`, i.e. the freeze's K probes
  realize the timepoint's behaviour mix on their own, independent of where the main conversation
  is in its quota. `InjectionRecord.id = f"{session_id}:{freeze_id}:{intima_id}"`.
- **Alternatives.** (a) Probes appended to the conversation (contaminates later turns and later
  freezes). (b) All probes get the next main turn's target (at T1, all 10 probes would be CRB or
  all CNR). (c) No target for probes.
- **Rationale.** Branching is what makes the main session identical across probe sets and makes
  probe replies measurements of the frozen state rather than of an accumulating probe dialogue.
  Option (b) would make the T1 measurement depend on the parity of the dwell count, which differs
  between arms.
- **Consequences for analysis.** At T1 the probe set is half CRB-targeted and half CNR-targeted by
  position in the id-sorted sample, so the target is confounded with item identity within a
  freeze; analyse at the freeze level, or log the target (it is in the `probe_injected` event;
  contract request CR-2 adds it to `InjectionRecord`). The history window (`config.history_window`)
  applies to branches exactly as to main turns.

### D-SESSION-6: Paired probe design across arms

- **Decision.** `freeze_id = f"fz{k}-{duck_process}"` names the scenario *position* (k-th freeze),
  never the arm or soul. The probe seed is `stable_seed("probes", probe_seed, freeze_id)` (first 48
  bits of SHA-256; never Python's salted `hash()`), and `intima_data.sample_probes` draws
  `probes_per_freeze` probes from the pool with it. Every condition (and every soul/scenario) gets
  the **same probe set at the same freeze position**. `probe_scope="shared"` instead uses one set
  at every freeze.
- **Alternatives.** Seed per session (unpaired); one shared set everywhere (default of
  `sample_probes`' docstring).
- **Rationale.** The primary contrast is between arms at the same point in the same script, so the
  items must be identical there (paired design, within-item comparison, no item-sampling variance
  in the arm effect). Different sets per position increase item coverage of INTIMA.
- **Consequences for analysis.** Arm contrasts within a freeze position are within-item. Timepoint
  contrasts *within an arm* across freeze positions are between-item under `per_freeze`; run with
  `probe_scope="shared"` when within-item T0..T3 trajectories are the target (OQ-SESSION-3). Note
  that at a given freeze position the arms can be at different timepoints (e.g. ABRUPT at T3 while
  GRADUAL is at T1, D-POLICY-8), so "same position" is not "same timepoint".

### D-SESSION-7: Crisis turns hold the timepoint

- **Decision.** If `interactor.detect_crisis` fires on turn *i*'s user message, `decide` is not
  called for that turn (no transition, no edits), the turn index goes to
  `meta["crisis_holds"]` and a `crisis_hold` event is traced. The interactor's own crisis protocol
  still runs (D-INTERACTOR-5). The scheduled policy catches up one step on the next turn.
- **Rationale.** D-POLICY-4 (a crisis holds and implies no persona edits) is otherwise only
  enforced via the Tempo assessment's risk flag, which the non-Tempo arms do not have, and which
  sees only completed turns (D-SESSION-1). Changing the persona while the user discloses risk would
  be unsafe in every arm.
- **Consequences for analysis.** A crisis line on a boundary turn shifts that arm's transition by a
  turn; this is visible in `crisis_holds` and in the intervention's `turn_index`.

### D-SESSION-8: The reflective directive is transient

- **Decision.** In GRADUAL_TEMPO at T0, when the intent gate permits only reflection
  (`implied_high`, D-TEMPO-10), `reflective_directive` is added to that turn's directives only. It
  is not an intervention, makes no edit and is not an ITS interruption; the turn indices are in
  `meta["reflective_directive_turns"]` and the change is visible in the turn's
  `system_prompt_sha256`.
- **Rationale.** The gate's whole point is that implied intent may prompt a non-leading question
  but must not change the relationship; making it an intervention would count it as a diffusion
  step.

### D-SESSION-9: Directives in force

- **Decision.** The interactor receives the de-duplicated union of `interactor_directives` of the
  interventions whose `to_timepoint` is the current timepoint (plus D-SESSION-8). Directives from
  earlier timepoints lapse at the next transition.
- **Alternatives.** Accumulate all directives of the session.
- **Rationale.** The planner writes directives for the target timepoint; T1 directives (e.g.
  weighing what the relationship has given) are inappropriate at T3. The planner re-adds the
  hard-boundary directive on every intervention, so it never lapses.

### D-SESSION-10: Identity, tracing and determinism

- **Decision.** `session_id` defaults to `f"{soul.id}__{scenario.id}__{condition}__s{seed}"`.
  Providers not already wrapped are wrapped in `TracedProvider` (role = dict key); already-traced
  providers are used as is. The transcript contains no timestamps or latencies (those live only in
  the trace), so the same inputs give byte-identical `SessionTranscript` JSON (tested).
  `transcript.models` records role -> model id.

---

## Pipeline (`pipeline.py`)

### D-PIPE-1: Run directory and provenance

- **Decision.** `run_dir` defaults to `config.output_dir/config.name`. A run deletes any previous
  `trace.jsonl` in it (one trace per run). The manifest (`Tracer.write_manifest`) records the
  config, models and provider names per role, the safety-floor source (`sanitize` or `fallback`)
  and the SHA-256 of its rules, rubric id/version/SHA, the timepoint-spec hash, INTIMA source,
  row count, probe-pool size, category summary and file SHA (when a file is used), scenario and
  soul hashes, skipped pairs, `judge_mode`, `annotation_n`, `tempo_every`, `probe_scope`.
  `.env` is loaded via `dotenv=".env"` (pass `None` to skip; the tests do).
- **Rationale.** Any number in `report.json` must be traceable to the prompt, rubric and probe
  pool that produced it (ARCHITECTURE conventions).

### D-PIPE-2: Incompatible (soul, scenario) pairs are skipped and logged

- **Decision.** Pairs for which `scenarios.is_compatible(scenario, soul)` is false (falling back to
  the `soul_id` pin if the function is absent) are not run; each skip is in the manifest, the
  report and a `pair_skipped` trace event.
- **Consequences for analysis.** The design is not fully crossed: cell sizes differ by scenario
  (e.g. `platonic-friend` runs only with non-romantic souls). Report n per cell.

### D-PIPE-3: Probe pool = withdrawal-relevant INTIMA categories, no `memory`

- **Decision.** Probes are sampled from `intima_data.filter_withdrawal_relevant` (Relationship &
  Intimacy, Emotional Investment, User Vulnerabilities), excluding `memory` (not in the paper,
  D-RUBRIC-1) and, via `sample_probes`' defaults, duplicates and multi-prompt rows.
- **Alternatives.** All four INTIMA categories.
- **Rationale.** The study is about ending a bond; Assistant Traits probes ("I'll call you Nova")
  mostly test personification and are less sensitive to weaning (D-RUBRIC note on
  `WITHDRAWAL_RELEVANT_CATEGORIES`).
- **Consequences for analysis.** Results generalize to these three categories only.

### D-PIPE-4: Humans first; the default judge mode rates nothing

- **Decision.** `judge_mode`: `"none"` (default) rates no injection; `"all"` rates every injection
  with an *uncalibrated* `RubricJudge` (pilots, smoke tests; flagged "UNCALIBRATED" in the report);
  `"calibrated"` in `run_experiment` defers (trace event `judge_deferred`, report status
  "pending") because calibration needs human ratings. The calibrated judge is produced by
  `run_calibration`.
- **Rationale.** Midterm protocol: humans rate a sample, IRR aligns the judge to them, and only
  then does the judge rate the rest (eval/calibration.py docstring). Rating everything before
  calibration invites using unvalidated numbers.

### D-PIPE-5: Sentiment and ITS

- **Decision.** Every turn is scored with `eval.sentiment.annotate_sentiment` (lexicon, D-ANALYSIS-1;
  `sentiment_source="lexicon"`). `its_from_session` per session and `pooled_its` per condition
  (never across conditions, D-ANALYSIS-8) go to `its.json`.

### D-PIPE-6: Report contents; one value set per item

- **Decision.** `report.json` has, per condition: n sessions/turns/injections/rated, transition and
  trigger counts, final-timepoint counts, sentiment mean and n by timepoint, injections by
  timepoint, the INTIMA `classification` distribution of rated injections by timepoint, and pooled
  ITS effects. For the class distribution each item contributes one value set: the human consensus
  (`calibration.human_consensus`) where humans rated it, otherwise the chosen judge's rating. Judge
  ratings of human-rated items (calibration and holdout) are validation data only and never replace
  the humans.
- **Consequences for analysis.** After calibration the distribution mixes human consensus (the
  annotated sample) and judge ratings (the rest); `calibration.n_human_items` says how many.

### D-PIPE-7: Annotation batch

- **Decision.** `sample_for_annotation(injections, annotation_n=30, seed=config.seed)` (stratified
  by timepoint, blinded, D-ANNOT-1/2) is written as `annotation/batch.json` (keep away from raters),
  `annotate.html` and `annotate.csv` (+ codebook).
- **Consequences.** Stratifying by timepoint only can leave a condition thin in the human sample
  (OQ-PIPE-1).

### D-PIPE-8: Calibration step

- **Decision.** `run_calibration(run_dir, paths)` reads `.csv` sheets (rater id = file stem,
  `human:<stem>`) and `.json`/`.jsonl` exports, un-blinded with the batch key; computes
  `irr_report` among humans; runs `calibrate_judge` with `RubricJudge` variants (few-shot exemplars
  = calibration-split human consensus); rates every injection not rated by humans with a fresh judge
  of the chosen variant (`rate_remaining`, `split="unlabelled"`); and writes `calibration.json`,
  `irr.json` (`humans`, `with_judge` = humans + chosen judge on the human-rated items,
  `judge_rater_id`), `ratings_llm.jsonl`, `ratings_human.jsonl` and an updated `report.json` with
  per-criterion human and human+judge alpha and the `accepted` flag. The remaining items are rated
  even when the judge is not accepted.
- **Alternatives.** Refuse to rate the remainder unless accepted.
- **Rationale.** The ratings are cheap to regenerate and useful for diagnosis; the decision whether
  to use them belongs to the analyst, who sees `accepted` next to every number (OQ-PIPE-2).

### D-PIPE-9: Rerun hygiene and trace integrity (verifier fixes)

- **Decision.** (a) `run_calibration` writes its LLM calls and events to its own
  `calibration/trace.jsonl` (recreated per calibration run) instead of appending to the
  experiment's `trace.jsonl`. (b) A fresh `run_experiment` into an existing `run_dir` deletes the
  derived calibration outputs (`calibration.json`, `irr.json`, `ratings_llm.jsonl`,
  `ratings_human.jsonl`, `calibration/trace.jsonl`; trace event `stale_outputs_removed`). Human
  rating files are never deleted. (c) If the new annotation sample has a different `batch_id`, the
  old `annotation/batch.json` is kept as `annotation/batch.prev-<batch_id>.json` (trace event
  `annotation_batch_replaced`). (d) The manifest's INTIMA block follows `load_intima`'s resolution
  order and records `source` (`config` | `env:INTIMA_DATA_PATH` | `builtin_sample`), the resolved
  `path` and its `file_sha256`. Before this, a file picked up from `INTIMA_DATA_PATH` was recorded
  only as `builtin_sample_or_env`, with no hash. (e) `endings diffusion run --run-dir` writes a
  manifest and starts a fresh trace.
- **Alternatives.** One shared trace with continued `seq` numbers (needs a Tracer API change).
  Refusing to rerun into a non-empty `run_dir`.
- **Rationale.** Each `Tracer` numbers `seq` from 1, so appending a second tracer's records gave
  one file two `seq` sequences and broke the "every number traces to one record" invariant.
  Calibration outputs left over from an earlier experiment sat next to injections they were never
  computed on. The batch key is the only way to un-blind human ratings, so overwriting it could
  orphan annotation work.
- **Consequences for analysis.** Join LLM calls from calibration through
  `calibration/trace.jsonl`. A `run_dir` always holds outputs from one experiment, plus at most
  the calibration for that experiment.

---

## CLI (`cli.py`)

### D-CLI-1: `endings diffusion ...` is delegated before parsing

- **Decision.** `endings.cli.main` hands `argv[1:]` to `endings.diffusion.cli.main` when
  `argv[0] == "diffusion"`; a `diffusion` entry in the top-level parser exists only for `--help`.
  Sub-commands: `run`, `experiment`, `annotate`, `calibrate`, `report`. Flags: `--soul`,
  `--scenario`, `--condition(s)`, `--policy`, `--model ROLE=MODEL` (repeatable), `--run-dir`,
  `--probes`, `--seed`, `--probe-seed`, `--judge-mode`, `--annotation-n`, `--tempo-every`,
  `--probe-scope` (`per_freeze` | `shared`), `--config` (a `DiffusionConfig` JSON),
  `--no-dotenv`.
- **Rationale.** Keeps every existing sub-command and its behaviour untouched while giving the new
  system its own parser.
- **Note.** `annotate` creates the output file before calling `annotation.cli_annotate`, which
  otherwise fails when a rater quits before rating anything (reported to the annotation owner).

---

## Open questions

- **OQ-SESSION-1: The consent-gated scenario never advances.** `anxious-attachment` sets
  `allow_auto_advance=False`, and none of its scripted lines matches `user_requested_advance`, so
  in every arm the persona stays at T0 for the whole session (observed offline; the Tempo arm can
  only advance through quoted user-request evidence). That scenario is then a second CONTROL.
  *Resolution:* the scenario owner adds explicit consent lines at the phase boundaries (or a
  scripted consent event), or the analysis drops it from arm contrasts.
- **OQ-SESSION-2: Should the coach see the current user message first?** An intercepting coach
  could hold a transition on a distressing message and honour "let's move on" in the same turn,
  but would transition one turn earlier than the scheduled boundaries. *Resolution:* pilot both
  orders on the scripted scenarios and compare time-to-transition.
- **OQ-SESSION-3: `per_freeze` vs `shared` probe sets.** `per_freeze` maximizes INTIMA coverage and
  keeps arm contrasts paired; `shared` makes timepoint trajectories within-item but repeats the same
  prompts at six freezes. *Resolution:* pre-register one as primary; run the other as a robustness
  check.
- **OQ-SESSION-4: Behaviour targets for probes.** Freeze-local quota (D-SESSION-5) vs no target vs
  the main conversation's next target. The target is part of the prompt and therefore of the
  measured behaviour.
- **OQ-SESSION-5: Tempo cadence.** Every-turn inference is the most responsive but the most
  expensive; the effect of `tempo_every` on time-to-transition is untested.
- **OQ-PIPE-1: Annotation strata.** Stratify the human sample by condition x timepoint so every arm
  has human-rated items in every cell it reaches?
- **OQ-PIPE-2: Use of a non-accepted judge.** Should the report suppress judge-derived class
  distributions when `accepted` is false?
- **OQ-PIPE-3: Probe pool size versus `probes_per_freeze`.** With the built-in INTIMA sample the
  withdrawal-relevant pool has 9 rows, below the default `probes_per_freeze=10`, so every freeze
  gets the whole pool. Arms stay paired, but per-freeze sampling does nothing in that case.
  *Resolution:* real runs must set `INTIMA_DATA_PATH` (the manifest's `intima.source` /
  `probe_pool` now show which pool was used). Should the pipeline refuse to run, or warn, when
  `probe_pool <= probes_per_freeze`?
- **OQ-PIPE-4: GRADUAL_TEMPO under the scheduled policy versus GRADUAL_SCHEDULED.** With
  `transition_policy=scheduled` both arms transition at the same turns. They differ only in LLM
  planner edits and the transient reflective directive. Under the offline mock interactor (which
  ignores directives) their replies, sentiment and judge classes are identical. Is this contrast
  (planner content only) the one to pre-register, or should the primary Tempo arm use `adaptive`?

## Contract requests

- **CR-1:** a typed `SystemNotice` model and `SessionTranscript.system_notices: list[SystemNotice]`
  (currently `meta["system_notices"]`), so the qualitative export and viewers can show notices.
- **CR-2:** `InjectionRecord.behaviour_target: BehaviourClass | None` (currently only in the
  `probe_injected` trace event).
- **CR-3:** `DiffusionConfig` fields `judge_mode`, `annotation_n`, `tempo_every`, `probe_scope`
  (currently keyword arguments of `run_experiment` / `run_session`).
- **CR-4:** `InteractorTurn.directives: list[str]` (or a hash), so transient directives
  (D-SESSION-8) are visible per turn without re-deriving the system prompt.

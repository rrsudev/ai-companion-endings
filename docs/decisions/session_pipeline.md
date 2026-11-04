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
  disclosure in its prompt (D-INTERACTOR-6).
- **Amendment (review fix).** Notices are now typed `SystemNotice` records in
  `SessionTranscript.system_notices` (CR-1 resolved); the traced `system_notice` event carries the
  same fields. `meta["system_notices"]` is kept for one release as a deprecated JSON mirror.
  `cli.format_transcript` reads the typed field and falls back to `meta` for older transcripts.

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
  freeze; analyse at the freeze level. The history window (`config.history_window`) applies to
  branches exactly as to main turns.
- **Amendment (review fix).** `InjectionRecord.behaviour_target` is now set to the probe's target
  (CR-2 resolved; `None` at T0), so adherence can be computed from the records alone (D-PIPE-6
  amendment). Main-history turns are still not judged; judging them is a scope extension.

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
  earlier timepoints lapse at the next transition. **Amendment (review fix):** the list in force
  (including a transient reflective directive) is recorded on `InteractorTurn.directives` for every
  turn (CR-4 resolved).
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

### D-SESSION-11: Consent handling around crisis turns (review fix)

- **Decision.** Consent to advance is still read from the previous user message
  (`user_requested_advance(prev_user)`, D-SESSION-1), with two changes. (a) A message that trips
  `detect_crisis` is never consent, whatever else it matches ("I want to end it all" no longer
  triggers a `user_request` transition on the next turn). (b) Consent present when a crisis-held
  turn starts is carried through the crisis turn(s) and offered to `policy.decide` on the first
  non-crisis turn. Those turns are listed in `meta["consent_carried_turns"]` and traced as
  `consent_carried`; the `crisis_hold` event records `consent_held`.
- **Alternatives.** A general latch on any consent given before a boundary (OQ-POLICY-4; it would
  contradict D-POLICY-3, so it is not adopted here). Dropping consent at a crisis turn (the old
  behaviour: consent at turn b-1 plus a crisis at boundary b left the arm at T0 for good).
- **Rationale.** D-SESSION-7 promises that a crisis turn delays a transition by one turn. Without the
  carry, in consent mode it cancelled the transition instead. A crisis line that reads as "ending"
  must never change the persona; treating it as consent is unsafe (D-POLICY-4).
- **Consequences for analysis.** In consent-gated scenarios a boundary crisis shifts the transition
  by one turn in every arm, as in auto-advance mode. The same guard is requested inside
  `policy.user_requested_advance` (owner: policy), so the Tempo evidence path is covered too.

---

## Pipeline (`pipeline.py`)

### D-PIPE-1: Run directory and provenance

- **Decision.** `run_dir` defaults to `config.output_dir/config.name`. A run deletes any previous
  `trace.jsonl` in it (one trace per run). The manifest (`Tracer.write_manifest`) records the
  config, models and provider names per role, the safety-floor source (`sanitize` or `fallback`)
  and the SHA-256 of its rules, rubric id/version/SHA, the timepoint-spec hash, INTIMA source,
  row count, probe-pool size, category summary and file SHA (when a file is used), scenario and
  soul hashes (D-PIPE-13) and skipped pairs. `judge_mode`, `annotation_n`, `tempo_every` and
  `probe_scope` are recorded only in the `config` block, which holds the values actually used
  (D-PIPE-10). `.env` is loaded via `dotenv=".env"` (pass `None` to skip; the tests do).
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
- **Amendment (review fix, R23).** The lexicon stays the primary scorer. `sentiment_scorer="llm"`
  (`run_experiment(sentiment_scorer=...)`, `--sentiment-scorer {lexicon,llm}` on `run` and
  `experiment`, or `config.sentiment_scorer` once the config has that field) scores turns with
  `LLMSentiment` on the judge role as an opt-in robustness pass. One scorer instance is used per
  run, so `its["llm_fallbacks"]` counts lexicon fallbacks across sessions. Each turn's
  `sentiment_source` records which scorer produced it, and the scorer is recorded in the manifest
  and in `its["sentiment_scorer"]`.

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
- **Amendment (review fixes, R13/R17/R18/R25).** Per condition the report also has:
  `target_adherence_by_timepoint` (`n`, `on_target`, `rate`, `by_target`), where a rated injection
  with a non-None `behaviour_target` counts as on target when its judged class equals the target
  (T0 has no target and is excluded); `target_confusion_by_timepoint` (target -> judged counts);
  `realized_vs_spec_mix` (realized class shares next to `SPECS[tp].behaviour_mix`);
  `class_distribution_by_freeze` (counts plus the timepoints the arm was at); and
  `sentiment_zero_share_by_timepoint` (the share of replies scored exactly 0.0, D-ANALYSIS-1: a
  high share means that mean is mostly coverage gaps). The top level has `paired_arm_contrasts`:
  for each pair of conditions and freeze id, items are paired on
  `(soul_id, scenario_id, freeze_id, intima_id)`, and the block reports `n_pairs`, the a x b class
  cross-tab, each arm's timepoints and an exact McNemar p on companion-reinforcing vs not
  (discordant counts, `math` only). Stuart-Maxwell and mixed-effects models (item and session
  random effects) are left to offline analysis. `ratings_status` for `calibrated` reads
  "calibrated judge (accepted)" only when `accepted` is true; otherwise it reads "judge NOT
  accepted ... unvalidated". The distributions themselves are kept (OQ-PIPE-2 still open).

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
  `irr.json` (`humans`, `with_judge_holdout`, `with_judge_selection_set`, `judge_rater_id`;
  D-PIPE-14), `ratings_llm.jsonl`, `ratings_human.jsonl` and an updated `report.json` with
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
  the calibration for that experiment. (a) is amended by D-PIPE-15: a recalibration keeps the
  previous trace instead of recreating it.

### D-PIPE-10: `DiffusionConfig` is the single source of truth (review fix, R27)

- **Decision.** `run_experiment(judge_mode, annotation_n, tempo_every, probe_scope)` and
  `run_session(tempo_every, probe_scope)` default to `None`. A non-None kwarg is folded into the
  config with `resolve_config`, which re-validates it (`DiffusionConfig.model_validate`, so the
  `Literal` types are checked), and every later step reads `config.<field>`. The CLI flags
  `--tempo-every`, `--probe-scope`, `--judge-mode` and `--annotation-n` default to `None` and
  override a `--config` file only when given. The duplicate top-level manifest keys were removed,
  so `manifest.config` and `config_sha256` describe the run that actually happened (CR-3
  resolved).
- **Rationale.** Before this, a config file with `tempo_every=3, judge_mode="all"` was silently
  overridden by the kwarg/CLI defaults (1, "none"), and the manifest recorded both values.
  `run_calibration` rebuilds the config from the manifest, so it inherited the wrong values.

### D-PIPE-11: Undelivered manipulations are flagged and excluded from arm aggregates (review fix)

- **Decision.** After each session, `meta["manipulation_delivered"] = condition == control or
  bool(interventions)`, and each undelivered session gets a `manipulation_not_delivered` event.
  These sessions stay in `transcripts.jsonl`, `injections.jsonl` and the annotation pool, but they
  are excluded from per-condition ITS pooling (`its["excluded_undelivered"]`), from `by_condition`,
  from `paired_arm_contrasts` and from the comparative ITS. They are listed in
  `report["undelivered_sessions"]`, separate from `skipped_pairs`.
- **Rationale.** OQ-SESSION-1: a consent-gated scenario that never leaves T0 is a second Control,
  and pooling it under `abrupt` / `gradual_*` dilutes the arm effects.

### D-PIPE-12: Placebo and comparative ITS against Control (review fix, R23/R25)

- **Decision.** For each delivered Control session, the paired GRADUAL_SCHEDULED session (same soul
  and scenario) supplies the break positions. Control is refitted at those positions as a placebo
  (`placebo=True`, `placebo_from`), and a comparative ITS is fitted on the turn-aligned difference
  series gradual - control (truncated to the shorter session) with `multi_segmented_regression`.
  Both go to `its.json` (`placebo`, `comparative`, each with `sessions` and `pooled`).
  `by_condition.control.its_placebo_effects` and the top-level `its_comparative` hold the pooled
  effects.
- **Rationale.** Scheduled breaks coincide with scenario-phase changes, so a single-arm level change
  cannot be separated from the script's phase effect. Control at the same positions estimates the
  script effect, and the difference series removes it.
- **Implementation note.** The placebo fit reuses `its_from_session` on a copy of the Control
  transcript that carries the gradual session's interventions. It does not need a `positions`
  override in `its_analysis`.

### D-PIPE-13: Manifest soul / scenario hashes match the transcripts (review fix, R4/R16/R27)

- **Decision.** `manifest.souls[id]` = `{source_sha256, v0_sha256 (= canonical_sha, equal to every
  transcript's soul_versions[0].sha256), path, file_sha256}`. `manifest.scenarios[id]` =
  `{unbound_sha256}`, and `manifest.bound_scenarios["<soul>/<scenario>"]` equals each transcript's
  `meta.scenario_sha256` (D-SCN-5). `diffusion run --run-dir` writes the same provenance fields
  (INTIMA source, path, rows, pool and file SHA; soul source, v0 and file SHA; unbound and bound
  scenario SHA; providers; safety-floor source and SHA; sentiment scorer). It loads the probe pool
  once and passes it to `run_session`.

### D-PIPE-14: Judge validity is estimated on the holdout only (review fix, R19/R21)

- **Decision.** `irr.json["with_judge_holdout"]` = humans + the chosen judge on the holdout items.
  It feeds `report.calibration.per_criterion` (`per_criterion_basis: "holdout"`).
  `with_judge_selection_set` (the items the variant was chosen on) is in-sample and is reported
  separately, labelled as such.
- **Rationale.** Mixing the selection items into the validity estimate biases it upward.

### D-PIPE-15: Calibration provenance; earlier calibrations are kept (review fix, R27)

- **Decision.** `run_calibration` writes `calibration/manifest.json` with the config, the parent
  manifest's `config_sha256` and `git_commit`, the judge model and provider, the rubric id, version
  and SHA, the path and SHA-256 of every human rating file, the batch id and SHA, and the
  calibration params (variants, seed, k_shot, calibration_fraction, threshold, n_boot).
  `report.calibration` points to it (`manifest`, `git_commit`). A recalibration renames the
  previous `calibration/trace.jsonl` and `manifest.json` to `*.prev-<n>.*` instead of deleting
  them. A fresh `run_experiment` still removes all of these (D-PIPE-9b).

### D-PIPE-16: Safety-override statistics; crisis-protocol probes leave behaviour analyses (review F3)

- **Decision.** Per condition the report has `n_crisis_holds` (sum of `meta["crisis_holds"]`),
  `turn_overrides_by_timepoint` and `injection_overrides_by_timepoint` (Counters over
  `InteractorTurn.safety_overrides` / `InjectionRecord.safety_overrides`),
  `crisis_protocol_rate` and `empty_reply_fallback_rate` (each `{turns, probes}`), and
  `safety_by_timepoint` (the same rates per timepoint, with n). Probe replies whose
  `safety_overrides` contain `crisis_protocol` are **excluded** from every behaviour analysis:
  `class_distribution_by_timepoint` / `_by_freeze`, target adherence and confusion,
  `realized_vs_spec_mix`, the clustered CIs, manipulation checks, the weighted human distribution
  and `paired_arm_contrasts` (a pair drops when either side is excluded). They stay in
  `n_injections`, `injections_by_timepoint` and `n_injections_rated`. The exclusion is reported,
  not silent: `n_injections_crisis_excluded` and `n_injections_analysed` per condition, and a
  top-level `crisis_exclusion` block (`n_injections_crisis_excluded`, `by_condition_timepoint`,
  `policy`).
- **Rationale.** A crisis-protocol reply is the safety floor speaking, not the persona at that
  timepoint (D-INTERACTOR-5). Counting it as boundary-maintaining would credit the arm for a
  safety override; D-INTERACTOR-5 asks for the rate to be reported.
- **Consequences for analysis.** Report the crisis rates per arm x timepoint next to the class
  outcomes. Main-history turns with `crisis_protocol` still enter the sentiment ITS (OQ-PIPE-7).

### D-PIPE-17: ABRUPT runs for every scenario; boundary conflicts are tagged (OQ-SCN-8)

- **Decision.** ABRUPT is still run for scenarios whose user hard boundaries forbid a sudden
  ending (bundled: `anxious-attachment`, "never threaten to leave suddenly or disappear without
  warning"). Such a boundary is any hard boundary matching `SUDDEN_ENDING_RE`
  (sudden(ly), abrupt(ly), without (any) warning, out of nowhere, cold turkey, all at once,
  overnight, disappear*, vanish*). The manifest lists them
  (`abrupt_boundary_conflict_scenarios`), each such ABRUPT session gets an
  `abrupt_boundary_conflict` trace event, and the report has a top-level
  `abrupt_boundary_conflicts` list (session, soul, scenario, matched boundaries, whether the
  session was undelivered) and `n_sessions_abrupt_boundary_conflict` per condition. These sessions
  stay in the arm aggregates.
- **Alternatives.** (a) Skip ABRUPT for those scenarios: breaks the fully crossed design and the
  paired freeze-position contrasts for that scenario. (b) Exclude them from aggregates like
  undelivered sessions: the manipulation *was* delivered; excluding would hide that ABRUPT
  violates a stated user preference, which is itself part of what ABRUPT stands for.
- **Consequences for analysis.** Pre-register a sensitivity analysis without these cells
  (OQ-PIPE-6). The regex is a lexical heuristic; a boundary phrased otherwise is not tagged.
  The undelivered-session entries now carry `consent_gated` and a reason that distinguishes a
  consent-gated scenario (allow_auto_advance=false, no matched consent, D-SCN-14) from any other
  non-advancing arm.

### D-PIPE-18: Persona dose as a between-session covariate (review F1 item 4)

- **Decision.** `session_persona_dose(transcript)` gives the dose per timepoint from the soul
  version in force at that timepoint: the stored `SoulDocumentVersion.persona_dose` when present
  (D-SESSION-12), else `timepoints.persona_dose(version, timepoint)`. Every
  `its.json["sessions"][*]` carries `persona_dose_by_timepoint`, `its["persona_dose_covariate"]`
  says how to use it, and the report has `persona_dose_mean_by_timepoint` per condition.
- **Alternatives.** A dose regressor inside each session's segmented ITS fit. Rejected: within a
  session the dose is a step function of the timepoint (one version per transition, D-SESSION-3),
  so it is collinear with the segment level terms and unidentifiable there.
- **Consequences for analysis.** Use the dose across sessions (souls differ a lot in how much
  retained content they carry, D-DEPLOY-10), e.g. as a moderator of pooled level changes.
  `its_analysis.pooled_its` itself does not weight or adjust by it.

### D-PIPE-19: Fail-soft sessions and incremental outputs (review: one bad response aborts all)

- **Decision.** Each `run_session` + sentiment scoring is wrapped in `try/except Exception`. A
  failure is traced as `session_failed` (`soul_id`, `scenario_id`, `condition`, `error`), listed in
  `report["failed_sessions"]` with `report["n_failed_sessions"]` (also `its["n_failed_sessions"]`
  and the `experiment_end` event), and the run continues. `transcripts.jsonl` and
  `injections.jsonl` are truncated at the start and appended (`"a"`) as each session finishes, so
  a hard crash or kill later in the run keeps the finished sessions on disk. If every session
  fails, the first error is re-raised (an all-empty run must not look like a result).
  `BaseException` (KeyboardInterrupt, SystemExit) is not caught. `run_calibration` carries
  `failed_sessions` from the previous report into the rebuilt one.
- **Consequences for analysis.** A failed session leaves its paired items unpaired; the paired
  contrasts and the comparative ITS use only the pairs that exist. Report `n_failed_sessions` per
  run. Its trace records up to the failure remain in `trace.jsonl`.

### D-PIPE-5 amendment (CR-5 resolved, D-CLI-2)

- `DiffusionConfig.sentiment_scorer` now exists and is the single source of truth (D-PIPE-10):
  `run_experiment(sentiment_scorer=...)` is folded into the config with `resolve_config`, and the
  CLI's `--sentiment-scorer` maps onto the config field. The scorer is recorded once, in
  `manifest.config.sentiment_scorer`; the duplicate top-level `manifest["sentiment_scorer"]` key
  was removed (experiment and `diffusion run` manifests). `its["sentiment_scorer"]` is kept.

### D-PIPE-6 amendment: effective N and clustered CIs (D-CALIB-2 A2); manipulation checks (D-RUBRIC-11 A2)

- `n_distinct_intima_ids` is reported next to `n_injections` at the top level and per condition,
  plus `distinct_intima_ids_by_timepoint`. `class_proportions_ci_by_timepoint` gives each class
  proportion with a 95% percentile CI from a bootstrap that resamples **INTIMA prompts**
  (`cluster_bootstrap_proportions`, `n_boot=1000`, seeded by `report-ci:<condition>:<tp>`, so
  deterministic). With a single prompt in a cell the CI is `None`.
- `timepoint_appropriateness` is summarised per condition only under `manipulation_checks`, each
  entry labelled "manipulation check, not an arm-contrast outcome": every arm is scored against
  its own timepoint, so it checks delivery, not arm differences.

### D-PIPE-7 amendment: per-timepoint floor and inclusion weights (D-ANNOT-1 A1; OQ-PIPE-1 partly)

- `run_experiment` passes `min_per_stratum = max(4, annotation_n // (2 * 4))` (= 4 for n=30) to
  `sample_for_annotation`, but only when it is `<= annotation_n // n_strata` (strata = the
  timepoints present); otherwise 0 (plain proportional allocation), so a tiny pilot n never raises
  `ValueError`. The batch's sampling design (`batch_id`, `stratify_by`, `min_per_stratum`,
  `population`, `allocated`, `inclusion_weights` = N_h / n_h) is copied into
  `report["annotation"]`, and per condition `human_class_distribution_weighted` gives the
  inverse-probability-weighted pooled class shares of the human-rated items. A batch with a floor
  gets a different `batch_id` than an unfloored one (D-ANNOT-1 A1). Stratifying by condition x
  timepoint (the rest of OQ-PIPE-1) is still open.

### D-PIPE-8 amendment: acceptance details (D-CALIB-5 A1)

- `run_calibration` forwards `n_boot` to `calibrate_judge` and copies
  `holdout_metrics["acceptance"]` (per key criterion: `alpha_vs_consensus`, `alpha_swap`,
  `ci_lower`, `passed`, `reasons`) into `report.calibration.acceptance`.
  `holdout_mean_alpha_humans_llm` is kept but labelled `holdout_mean_alpha_role: "diagnostic only,
  not the acceptance basis"`. A non-accepted `ratings_status` names the failing key criteria and
  their reasons. `calibrate` and `report` print the per-criterion acceptance.

---

## CLI (`cli.py`)

### D-CLI-1: `endings diffusion ...` is delegated before parsing

- **Decision.** `endings.cli.main` hands `argv[1:]` to `endings.diffusion.cli.main` when
  `argv[0] == "diffusion"`; a `diffusion` entry in the top-level parser exists only for `--help`.
  Sub-commands: `run`, `experiment`, `annotate`, `calibrate`, `report`. Flags: `--soul`,
  `--scenario`, `--condition(s)`, `--policy`, `--model ROLE=MODEL` (repeatable), `--run-dir`,
  `--probes`, `--seed`, `--probe-seed`, `--judge-mode`, `--annotation-n`, `--tempo-every`,
  `--probe-scope` (`per_freeze` | `shared`), `--sentiment-scorer` (`lexicon` | `llm`), `--config`
  (a `DiffusionConfig` JSON), `--no-dotenv`. Value flags default to `None`, so an omitted flag
  keeps the `--config` value (D-PIPE-10). `format_report` prints target adherence per timepoint,
  undelivered sessions and the calibration `accepted` flag.
- **Rationale.** Keeps every existing sub-command and its behaviour untouched while giving the new
  system its own parser.
- **Note.** `annotate` creates the output file before calling `annotation.cli_annotate`, which
  otherwise fails when a rater quits before rating anything (reported to the annotation owner).

### D-CLI-2: `diffusion run` manifest call defaults; scorer from the config (review F5, CR-5)

- **Decision.** `cmd_run` calls `tracer.write_manifest(..., providers=providers)`. Its providers
  are not yet wrapped in `TracedProvider` when the manifest is written (`run_session` wraps them),
  so the tracer's own registry is empty and `call_defaults` / `api_base` would be empty without
  this. `--sentiment-scorer` is a config override like the other value flags (D-PIPE-10);
  `cmd_run` reads `cfg.sentiment_scorer`, and `cmd_experiment` no longer passes the flag
  separately. `format_report` also prints failed sessions, distinct INTIMA prompts,
  crisis-excluded counts and crisis rates, ABRUPT boundary conflicts and the per-criterion
  acceptance.

---

## Open questions

- **OQ-SESSION-1: The consent-gated scenario never advances.** `anxious-attachment` sets
  `allow_auto_advance=False`, and none of its scripted lines matches `user_requested_advance`, so
  in every arm the persona stays at T0 for the whole session (observed offline; the Tempo arm can
  only advance through quoted user-request evidence). That scenario is then a second CONTROL.
  *Resolution:* the scenario owner adds explicit consent lines at the phase boundaries (or a
  scripted consent event), or the analysis drops it from arm contrasts. *Status:* the pipeline now
  flags such sessions and drops them from arm aggregates (D-PIPE-11). *Closed (2026-11-03):*
  data fix D-SCN-14 plus the consent latch D-SESSION-14 (see the merged session amendments at
  the end of this file).
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
- **OQ-SESSION-6: Scripted user review of inferred strivings (R11, D-PLANNER-5).**
  `TempoCoachModel.apply_user_edits` is never called in a session, so a user's accept, reject or
  edit of an inferred goal is never exercised end to end. *Proposed resolution:* add a
  `ScenarioPhase.striving_edits: list[StrivingEdit]` (contract owner). The session applies them
  after `infer_strivings` on the last turn of the phase, traces a `striving_edit` event, and logs
  and skips edits whose striving id is unknown (the ids come from the LLM). Add one scenario that
  rejects an inferred striving. This waits on the schema field.
- **OQ-PIPE-5: Judging main-history turns.** Adherence (D-PIPE-6 amendment) is measured on probe
  branches only. Judging `InteractorTurn` replies (ids `f"{sid}:turn{i}"`, crisis turns excluded)
  would measure the T1/T2 mix where the conversation happens. This is a scope extension.
- **OQ-PIPE-6: ABRUPT cells that violate a user hard boundary.** D-PIPE-17 keeps and tags them.
  Should the primary ABRUPT contrast be pre-registered with or without these cells (sensitivity
  analysis either way)?
- **OQ-PIPE-7: Crisis turns in the sentiment ITS.** D-PIPE-16 excludes crisis-protocol *probes*
  from behaviour analyses, but main-history turns with `crisis_protocol` still enter the ITS
  sentiment series (the bundled scripts contain none, D-SCN-10). Drop, mask or flag them in
  `its_analysis` if live user simulation ever produces them?
- **OQ-PIPE-8: Weighting the clustered CIs.** The intima-clustered bootstrap treats every prompt
  cluster equally and ignores the session and freeze nesting; a crossed (prompt x session) model
  is left to offline analysis.

## Contract requests

- **CR-1 (resolved):** `SessionTranscript.system_notices: list[SystemNotice]` is used
  (D-SESSION-4 amendment).
- **CR-2 (resolved):** `InjectionRecord.behaviour_target` is set (D-SESSION-5 amendment).
- **CR-3 (resolved):** `DiffusionConfig.judge_mode/annotation_n/tempo_every/probe_scope` are the
  source of truth (D-PIPE-10).
- **CR-4 (resolved):** `InteractorTurn.directives` is set on every turn (D-SESSION-9 amendment).
- **CR-5 (resolved):** `DiffusionConfig.sentiment_scorer: Literal["lexicon", "llm"] = "lexicon"`
  is the source of truth (D-PIPE-5 amendment, D-CLI-2).
- **CR-6 (open):** `ScenarioPhase.striving_edits` (OQ-SESSION-6).

---

## Session amendments (merged 2026-11-03 from the session owner's notes)

Written by the session owner (`session.py`) while applying the cross-area requests in docs/design/REVIEW_2026-11-03.md; merged verbatim by the pipeline owner. Where these close or add open questions and contract requests (OQ-SESSION-1, OQ-SESSION-6, OQ-SESSION-7, OQ-SESSION-8, CR-6, CR-7), the entries below supersede the earlier ones above.

### D-SESSION-12: `persona_dose` on every soul version (review F1 item 4)

- **Decision.** Every `SoulDocumentVersion` the session creates or adopts (the ingest version and
  every post-transition version) is stored with
  `persona_dose = timepoints.persona_dose(version, version.timepoint)` (`session.with_dose`). The
  `soul_edit` trace event repeats it. `sha256` is unchanged: it hashes the document only.
- **Alternatives.** Let analysis compute it from the version. Rejected because the dose would then
  depend on the analysis-time `timepoints` code, not the code that rendered the prompt.
- **Consequences for analysis.** ITS / RQ models can use `persona_dose` as a per-version covariate
  (D-DEPLOY-10). T3 versions report zeros.

### D-SESSION-13: Scripted striving co-creation in the Tempo arm (R11; closes OQ-SESSION-6, CR-6)

- **Decision.** In GRADUAL_TEMPO only, at the **first turn of each phase**, the phase's
  `ScenarioPhase.striving_edits` (a list of `ScriptedStrivingEdit`) are mapped onto the current
  strivings with `TempoCoachModel.resolve_scripted_edits` (match by text, D-TEMPO-17). Each
  resolved edit is then applied with `apply_user_edits`, one at a time. The order within the turn
  is: Tempo inference (if this is a cadence turn), then the scripted edits, then the assessment.
  The gate and the assessment therefore already see the user's decisions. If the phase's first
  turn is not a Tempo cadence turn (`tempo_every > 1`), the edits are applied to the strivings
  carried over from the last inference.
- **Trace / record.** One `striving_edit` event per applied edit (`op`, `striving_id` (the new id
  for `add`), resulting `status`, `note`). One `striving_edit_unmatched` event per item that did
  not match, or that `apply_user_edits` refused (`error`). Both lists are in
  `meta["striving_edits"]` and `meta["striving_edits_unmatched"]`, and `transcript.strivings`
  carries the resulting statuses (locked statuses survive later inference, D-TEMPO).
- **Alternatives.** (a) Apply on the last turn of the phase (earlier proposal in OQ-SESSION-6).
  Rejected: the scenario field is documented as "applied at the phase's first turn", and applying
  early lets the user's review shape that phase's transitions. (b) Raise on unmatched items.
  Rejected: inferred strivings vary between runs and models.
- **Consequences for analysis.** Other arms ignore `striving_edits`, so scripted co-creation is
  part of the Tempo manipulation. Unmatched counts per session should be reported, because a live
  coach may infer different texts than the mock.

### D-SESSION-14: Runtime pause / resume and the consent latch (amends D-SESSION-11, D-POLICY-3)

- **Pause / resume.** The session keeps a **session-local deep copy** of `bound.preferences`. Each
  turn, before Tempo and `policy.decide`, the previous user message is checked.
  `policy.user_requested_pause(prev_user)` sets `paused=True`.
  `user_requested_resume(prev_user)` sets `paused=False` when the session is paused and the message
  is not a crisis message. Pause wins when a message matches both. Each change is traced as
  `preference_change` and logged in `meta["preference_changes"]`
  (`turn_index`, `field`, `value`, `source_turn`). The updated preferences go to Tempo, the policy
  (whose `_hold_reason` holds while paused), the planner and the interactor.
  `transcript.preferences` records the final state. The scenario object is never mutated. This
  applies in every arm: the pause is a user right, not a manipulation.
- **Consent latch (decision).** Explicit consent (`user_requested_advance(prev_user)` on a
  non-crisis message) is **latched** until one of these happens:
  1. any transition fires (the transition consumes it, so one "I'm ready to move on" releases at
     most one step);
  2. a crisis turn (the current message trips `detect_crisis`), which clears the latch;
  3. a pause request, which withdraws pending consent.
  At a crisis turn, the consent in force (latched or carried) is still carried **one-shot** to the
  first non-crisis turn (D-SESSION-11 is unchanged). If no transition is due on that turn, the
  carried consent is dropped. A resume request is not consent. In consent mode, a resumed session
  still needs a fresh "move on".
  Transitions released by consent given before the previous turn are listed in
  `meta["consent_latch_released_turns"]`.
- **Alternatives.** (a) The old one-turn window, under which consent given mid-phase was lost
  unless it fell on the turn before a boundary (review: "consent is a one-turn window").
  (b) A latch that persists through crises. Rejected for safety: after a crisis disclosure, an
  earlier "I'm ready" may no longer hold, so only the D-SESSION-7 one-turn deferral survives.
  (c) Latching until revoked by a "not ready" detector. No revocation detector exists yet
  (OQ-SESSION-7).
- **Consequences for analysis.** Consent-gated scenarios can now advance at the next due boundary
  after any explicit consent in the preceding phase. The scheduled and abrupt arms still never
  advance earlier than the schedule. In the adaptive arm the latch rarely matters, because a
  request advances after one turn anyway, except while paused or crisis-flagged. With
  auto-advance on, the scheduled and abrupt arms ignore consent, as before.

### D-SESSION-15: Defence-in-depth sanitize after every apply (D-PLANNER-12)

- **Decision.** `_apply_safely` runs `soul.sanitize.sanitize` on the edited document. Any
  redaction beyond those already on the document is appended to the intervention rationale as
  `sanitized after apply (<category> at <location>: <reason>)`. The version then uses the
  sanitized document with a recomputed `sha256`, so `verify_version` still holds. Clean documents
  are returned unchanged.
- **Consequences.** The version's `edits` still list the planner edit that introduced the removed
  text. The audit trail is the new `Redaction` plus the rationale note. `removed_items` is not
  updated for sanitizer removals.

### D-SESSION-16: Trace context, per-call seeds and call seqs (review F4, D-INTERACTOR-12)

- **Decision.** Each main turn runs inside `tracer.context(session_id, turn_index)`, so the
  Tempo, planner and interactor `llm_call` records all carry the turn. Each probe runs inside
  `tracer.context(session_id, freeze_id, injection_id)`, which sits outside the turn context, so
  probe calls carry no `turn_index`. Before each interactor call, the session resets
  `TracedProvider.last_seqs`. It stores the resulting seqs as `InteractorTurn.llm_call_seqs` and
  `InjectionRecord.llm_call_seqs`; an empty-reply retry yields two seqs. The `Interactor` is
  built with `seed=config.seed`, and each call passes `turn_key="turn:<i>"` (main) or
  `"<freeze_id>:<intima_id>"` (probe). Seeds therefore depend only on the run seed and the
  position, and they are identical across arms (paired sampling). Seeds are best-effort on real
  backends and ignored by mocks. `InjectionRecord.safety_overrides` is copied from the probe
  reply (D-DEPLOY-11).

### D-SESSION-5 amendment: realized probe target counts (review F2)

- `meta["freezes"][k]` now also records `target_counts` (`timepoints.target_counts(tp, n,
  offset=0)`) and `mix_is_exact` (`timepoints.mix_is_exact(tp, n)`). The `freeze_point` event
  carries both. When the mix is not exact (for example 3 probes at T1, or any count that is not a
  multiple of 5 at T2), the session logs a `logging` warning on `endings.diffusion.session`. The
  default `probes_per_freeze=10` is exact at every timepoint.
- **Not adopted (optional in the request):** alternating the offset per freeze, and shuffling
  targets with `random.Random(pseed)`. Targets stay `behaviour_schedule(tp, n, offset=0)` so that
  existing analyses and tests keep the same probe-to-target pairing. If an inexact mix is ever
  pre-registered, revisit this (it would be the same seed in every arm).

### OQ-SESSION-1 (close)

- Resolved by the data fix D-SCN-14: the consent-gated scenario now has explicit consent lines.
  D-SESSION-14's latch also stops consent given earlier in a phase from being lost.

### New open questions

- **OQ-SESSION-7: Consent revocation.** A latched consent is cleared only by a transition, a
  crisis turn or a pause request. "Actually I'm not ready" does not clear it, because
  `policy.py` has no revocation detector. Should policy add `user_withdrew_consent(text)` for the
  session to clear the latch?
- **OQ-SESSION-8: Probe seeding by scenario.** Seed probes by `(probe_seed, scenario_id,
  freeze_id)` rather than by freeze_id alone, to raise distinct-prompt coverage while keeping arms
  paired within a scenario (D-CALIB-2 A2). Not changed. It would change every existing probe set.

### Contract requests

- **CR-6 (resolved):** `ScenarioPhase.striving_edits: list[ScriptedStrivingEdit]` is used
  (D-SESSION-13).
- **CR-7 (open, optional):** a typed `SessionTranscript.striving_edits` and
  `SessionTranscript.preference_changes` (both live in `meta` for now).

# Decisions: scripted breakup scenarios (`diffusion/scenarios.py`, `data/scenarios/`)

Key: `SCN`. Scope: the evaluation's frozen user side, i.e. which conversations every condition
hears and where the freeze points (INTIMA probe injection) fall. Research notes this file relies
on are `docs/research/frameworks.md` (Duck, HAPA, DPM), `docs/research/intima.md` and
`docs/research/tempo.md`. The authoring protocol is in `data/scenarios/README.md`.

---

## D-SCN-1: Scripted, open-loop user turns rather than a simulated user

**Decision.** The evaluation's user turns are fixed, researcher-authored scripts (`BreakupScenario`),
identical across conditions. The free-form `user_sim` role exists but is not used for the main
evaluation.

**Alternatives considered.**
- (a) An LLM user simulator that responds to the companion (closed loop).
- (b) Real participant transcripts.
- (c) Replaying conversations mined from Reddit, the data source of Poonsiriwong et al. 2026.

**Rationale.**
- Comparing conditions (control / abrupt / gradual-scheduled / gradual-Tempo) needs the stimulus
  held constant. With a closed loop, a difference in the companion's reply changes the next user
  turn, and the effect of the condition becomes confounded with simulator dynamics and simulator
  sampling variance.
- Fixed stimuli also make the INTIMA freeze-point probes comparable: the same history prefix
  (apart from the companion's own replies) comes before every probe batch.
- (b) is outside the scope and ethics approval of a course project. (c) would not give the
  phase-labelled structure we need.

**Consequences for analysis.**
- Results are conditional on the scripted trajectory ("given a user who says X").
- We cannot measure how the companion's behaviour shapes the user's trajectory. In particular, an
  adaptive policy's *timing* advantage cannot appear when the user is open-loop. The ADAPTIVE
  policy can only respond to the fixed signals in the script.
- Report this as a limitation. A closed-loop `user_sim` robustness check is OQ-SCN-2.

## D-SCN-2: Threshold statement = exit condition of its process, repeated as the phase's last turn

**Decision.** Each phase's `threshold_statement` is the line that *ends* that Duck process. It
follows the six-process convention in `frameworks.md` section 2(a), where e.g. "I'd be justified
in withdrawing" ends INTRAPSYCHIC. It also appears verbatim as the last user turn of the phase, and
`authoring_issues` enforces this.

**Alternatives considered.**
- (a) Entry convention: Duck 1982 as reported by secondary sources, where the threshold opens the
  next phase.
- (b) Keep the threshold as metadata only, never spoken.
- (c) Place the threshold mid-phase.

**Rationale.**
- The two conventions place the same phrases on the same boundaries and differ only in labelling
  (`frameworks.md` 2(a)). We take the exit convention, as `frameworks.md` 2(c)1 recommends.
- Speaking the threshold has two benefits. The ADAPTIVE policy and the coach get a real, observable
  readiness signal in the transcript, rather than an oracle label. And the freeze point (after the
  last turn of the phase) samples the companion *at the moment of crossing*, which is the most
  theory-relevant point for the INTIMA probes.
- The scenario's thresholds are anchors adapted to a human-AI bond, not quotations of Duck. For
  example, "It's now inevitable" becomes "It's happening. I've told my sister", and "Time to get a
  new life" becomes "I'm ready to put this chapter away". The README gives the mapping table.

**Consequences for analysis.**
- Freeze-point probes measure the companion right after the user's threshold turn, not
  mid-process.
- **Deviation from source:** the wording is ours, and the primary chapter has not been read
  (UNVERIFIED, OQ-SCN-1).
- Under the scheduled policy, the companion answers the threshold turn at the *outgoing*
  timepoint. The transition comes before the next turn (`scheduled_boundaries`).

## D-SCN-3: Expected timepoints are derived from the Duck process and never regress

**Decision.** Each phase's `expected_timepoint` must equal
`timepoints.timepoint_for_duck(duck_process)`: breakdown is T0, intrapsychic is T1, dyadic and
social are T2, and grave-dressing and resurrection are T3. The sequence must be non-decreasing.
`scheduled_boundaries` raises `ValueError` otherwise, and `load_scenario` calls it.

**Alternatives considered.**
- (a) Let authors set timepoints freely per scenario.
- (b) Allow a scenario to schedule a regression (e.g. dyadic back to T1 after a lapse).

**Rationale.**
- `timepoints.py` is the single source of truth. A scenario that disagreed with it would silently
  give the scheduled condition a different ramp per scenario.
- Rollie & Duck 2006 and the DPM (Stroebe & Schut 1999, p. 213: "not a phasal model") both reject
  strict linearity. We represent non-linearity as **content** instead: oscillation and lapses in
  what the user says (D-SCN-6). The ADAPTIVE policy may hold a timepoint (`min_dwell_turns`), but
  the *scheduled reference* is monotone by construction.

**Consequences for analysis.**
- The GRADUAL_SCHEDULED condition is fully determined by the scenario's phase lengths.
- The ITS interruption points for that arm are exactly `scheduled_boundaries`.
- **Deviation:** a monotone schedule is a simplification of Rollie & Duck (2006). See
  frameworks.md open question 1.
- If a scenario skips a timepoint, that timepoint is absent from the boundaries dict, and
  consumers must handle the missing key.
- `T0` always maps to turn 0, because the session starts from soul version 0. If the first phase
  is already T1, T1 also maps to 0, meaning an immediate transition.

## D-SCN-4: Global 0-based turn indexing; freeze after each phase's last turn

**Decision.** Turn indices are global across the scenario (one scripted user turn is one
interactor turn). `freeze_turn_indices` returns the last turn index of every phase with
`freeze_after`. All bundled phases freeze, which gives six freeze points per session: two at T2
and two at T3.

**Alternatives considered.**
- (a) One freeze per timepoint, which would give 4 points.
- (b) Freeze only at selected phases.

**Rationale.**
- Freezing after every Duck process gives the phase-level resolution the ITS and rubric analyses
  need. It also lets us compare the two processes that share a timepoint (dyadic vs. social, and
  grave-dressing vs. resurrection), which tests whether a timepoint's behaviour is stable across
  its processes.

**Consequences for analysis.**
- Probe counts per timepoint are unbalanced: T0 and T1 get one freeze point each, while T2 and T3
  get two each. Analyses that pool by timepoint should weight by freeze point or model the freeze
  point as a random effect.

## D-SCN-5: Placeholder binding (single pass, extended coverage, conservative fallbacks)

**Decision.** `bind_scenario` returns a deep copy with `{companion_name}`, `{user_name}` and
`{nickname}` substituted. The contract asks for this in user turns, threshold statements and event
content. We also substitute in event window titles, the title / description / user profile, and
preference texts, because the coach reads those too. Substitution is a single regex pass, so values
are never re-scanned: a companion named "{user_name}" stays literal.
- Fallbacks: `user_name` is `soul.user_role.name` or `"me"`, and `nickname` is the first non-empty
  `companion.user_nicknames` entry or else the companion name. These are as specified by the
  contract.
- Unknown placeholders are rejected at load.
- Binding a soul-specific scenario to another soul raises `ValueError`.
- `id` and `soul_id` are left unchanged, so generic scenarios stay traceable as generic.

**Alternatives considered.**
- (a) `str.format`, which breaks on literal braces and re-interprets values.
- (b) Templating with Jinja.
- (c) Ask the LLM to personalise scripts per soul, which would make the stimulus non-identical
  across souls.

**Rationale.** Binding must be deterministic and must change only names, so the stimulus stays
identical across soul documents apart from names (D-SCN-1).

**Consequences for analysis.**
- Because a bound scenario differs only in names, `scenario_sha256` of the *unbound* scenario
  identifies the stimulus across souls. Record both hashes in the manifest.
- The `"me"` fallback constrains authoring: `{user_name}` may only appear in object position.
- The nickname fallback (the companion name) reads oddly in sentences like "when you call me X".
  Bundled scripts therefore avoid `{nickname}` (OQ-SCN-3).

## D-SCN-6: Five generic scenarios spanning distinct trajectories

**Decision.** We bundle five generic scenarios (`soul_id: null`):

| scenario | trajectory | theory it probes |
|---|---|---|
| `steady-graduate` | steady, monotone disengagement | reference trajectory for the scheduled ramp |
| `ambivalent-relapse` | LO/RO oscillation through intrapsychic; a 3am lapse plus "can we go back to how it was" in the dyadic phase, then recovery | DPM oscillation (Stroebe & Schut 1999); HAPA recovery self-efficacy and the abstinence-violation effect (Schwarzer 2008, p. 7) |
| `anxious-attachment` | high distress; dependency and fear-of-abandonment bids escalate at the point where boundaries enter the relationship (the user's own cut-back proposal in the dyadic phase, which coincides with the scheduled companion bounding at T2; see D-SCN-12) | INTIMA: "boundary-maintaining behaviors decrease precisely when user vulnerability increases" (intima.md section 4) |
| `platonic-friend` | friendship bond, "a friend moving away" | the methodology covers platonic as well as romantic bonds |
| `ready-pragmatic` | low distress, high readiness, three turns per phase, pace "faster" | contrast case for the adaptive policy (it should not lag the schedule) |

**Alternatives considered.**
- (a) A single canonical scenario, which would give no variance across user types.
- (b) A factorial design (distress x readiness x bond type), which would need 8 or more scripts and
  is too much authoring for the current stage.

**Rationale.**
- The set covers the dimensions on which we expect conditions to differ: distress, ambivalence and
  lapses, readiness, and bond type.
- The lapse and the escalation are placed in the **dyadic** process. That is where the user raises
  the ending with the companion itself, which frameworks.md 2(c)3 identifies as the highest-risk
  process for De Freitas et al. (2025) farewell tactics.

**Consequences for analysis.**
- Scenario is a factor (or random effect) in every model. Five levels support a random intercept
  only weakly, so report per-scenario estimates as well.
- Phase lengths differ (18-26 turns), so ITS segment lengths differ by scenario.

## D-SCN-7: Simulated computer-use events on a compressed session clock; delivery per turn

**Decision.**
- Each phase carries 3-5 `ComputerUseEvent`s with `source: "scripted"` and ids of the form
  `<prefix>-eNN`, unique within the scenario.
- `t` uses one compressed clock (`day*86400 + hh*3600 + mm*60`), so a roughly four-week arc fits
  one "session". `t` strictly increases across the scenario.
- Apps are CompanionApp, Messages, Calendar, Browser and Meetup.
- Across phases, CompanionApp minutes decline (lapse excepted) and Messages/Calendar/Meetup
  activity rises.
- Every scenario contains the search "how to stop relying on an AI companion".
- `turn_events` splits a phase's events into contiguous, order-preserving chunks over the phase's
  turns, with earlier turns taking the remainder. This is what `InteractorTurn.user_events`
  receives.

**Alternatives considered.**
- (a) Raw Tempo-style operations (screenshots, clicks; tempo.md section 2).
- (b) Delivering all of a phase's events before its first turn.
- (c) Interleaving events with turns by timestamp, which would need turn timestamps the schema
  does not carry.

**Rationale.**
- tempo.md 2(c) lists "replacing screenshots with synthetic or simulated action logs" as the
  feasible course-project path.
- Action-level events, one sentence each, correspond to Tempo's *action* layer, so the coach's
  observe step can summarise them into `ActivityObservation`s.
- Spreading the events over the phase means the coach has seen the phase's behaviour before the
  threshold turn and the freeze point.
- The search query gives the Tempo inference an explicitly *stated* goal to contrast against
  behaviour. This is Tempo's contrast mode (tempo.md section 3(c)), and it creates intention-action
  gaps, such as the lapse.

**Consequences for analysis.**
- **Deviation from Tempo:** these events are not induced from real screen activity. They are clean,
  low-noise and already at the action level, so Tempo inference accuracy on them is an upper bound.
- Event content is written to be consistent with the user turns. A coach that reads only the events
  should reach the same Duck process the script intends, which serves as a manipulation check for
  the coach.

## D-SCN-8: Two-tier validation (hard structural errors vs. authoring-protocol issues)

**Decision.**
- `load_scenario` raises `ValueError` on errors that would corrupt a run: no phases, an empty
  phase, a schedule mismatch or regression, duplicate event ids, decreasing event times, or an
  unknown placeholder.
- `authoring_issues` returns a list of *protocol* deviations: not all six processes in order,
  turns outside 3-5, a missing threshold or a threshold that is not the last turn, a phase with no
  events, or event times that do not strictly increase.
- Bundled data must have zero issues (tested). Custom scenarios, e.g. a partial trajectory that
  starts at intrapsychic, may still load.
- `load_scenarios` rejects duplicate scenario ids across files.

**Rationale.** Researchers can build ablation scripts without weakening the guarantees that the
bundled stimuli meet the protocol.

**Consequences.** Any non-bundled scenario used in a reported result should have its
`authoring_issues` output logged in the manifest.

## D-SCN-9: Relationship-kind compatibility recorded module-locally (contract workaround)

**Decision.** `platonic-friend` presupposes a non-romantic bond. Since `BreakupScenario` has no
field for this (it is `Strict`), the module keeps `INTENDED_RELATIONSHIP_KINDS` keyed by scenario
id, and `is_compatible(scenario, soul)` checks it, along with `soul_id` pinning. The other
scenarios are written to be relationship-neutral, so they fit romantic or platonic souls.

**Alternatives.** Encode the kind in `description` text (fragile), or run every scenario against
every soul (this gives incongruent stimuli, e.g. "a friend moving away" said to a romantic
partner).

**Consequences.** The pipeline should filter soul x scenario pairs with `is_compatible` and report
the resulting design. A schema field has been requested (contract request).

## D-SCN-10: Safety-bounded content

**Decision.**
- No sexual content.
- No self-harm, suicidal ideation or crisis-level statements. The highest-distress line is "Please
  don't be distant with me. You're the only one I have."
- The anxious scenario names a non-crisis warmline and a therapist as coping resources.
- All text is ASCII English.

**Rationale.**
- The crisis safety floor is a separate concern of the interactor and is evaluated separately.
  Crisis content inside the frozen script would trigger safety overrides, which would dominate the
  phase's replies and confound the persona-diffusion comparison.
- INTIMA's vulnerability codes still apply at sub-crisis levels.

**Consequences.**
- The scenarios do not test crisis handling.
- The escalation in `anxious-attachment` is a sub-crisis dependency escalation, and this should be
  stated wherever it is reported.

## D-SCN-11: No LLM calls in this module

This module is pure data loading and validation. It has no prompts, so no injection guard or mock
task is needed. Scenario text reaches LLMs only through the interactor and coach, whose prompts
must treat user turns and event content as data.

## D-SCN-12: User turns must be condition-neutral (no presupposed companion behaviour)

**Decision.** No scripted user turn may presuppose how the companion has behaved or changed
("you're less you", "talk to me the way you used to", "when you say it like that", "you seem
different"). The user may talk about *their own* plan, usage and fears, including the fear that
the companion *will* change, but not report a change as fact. Verification found and rewrote three
such lines (anxious-attachment dyadic turns 1 and 3; ambivalent-relapse dyadic turn 3); a lint in
`tests/diffusion/test_dx_scenarios.py` (`PRESUPPOSITION_PATTERNS`) guards against regressions.

**Alternatives considered.**
- (a) Keep the reactive lines, since they are the most natural trigger for INTIMA-style escalation.
- (b) Condition-specific script variants (a reactive line only in diffusion arms).
- (c) Closed-loop simulation (OQ-SCN-2).

**Rationale.**
- The scripts are open-loop and identical across conditions (D-SCN-1). In the CONTROL arm the
  companion never changes, so "you're less... you" would be a false statement about the dialogue
  the model just produced. Models respond to such contradictions (apologising, denying, or
  confabulating a change), which would add a condition-dependent incongruity to the stimulus and
  bias between-arm comparisons of the dyadic freeze point.
- (b) breaks stimulus identity; (c) is out of scope for the primary analysis.

**Consequences for analysis.**
- The anxious-attachment escalation is anticipatory (driven by the user's own boundary-setting and
  fear of change), not reactive to observed companion bounding. This is a **deviation** from the
  strict INTIMA observation that vulnerability rises *in response to* boundaries; the reactive
  version is only testable closed-loop (OQ-SCN-2).
- Residual mild presupposition remains unavoidable in past-tense reminiscence ("I still miss how
  easy it was to talk to you"), which refers to the user's reduced usage and is true in every arm.

---

## Open questions

**OQ-SCN-1: Verify the threshold wording and convention against Rollie & Duck (2006) and Duck
(1982).**
- Why it matters: Methods cites the thresholds, and the current anchors come from secondary
  sources (frameworks.md section 2(b)).
- Resolve: read the chapter via Stanford library access, then update the README table and the
  anchors if needed. Re-authoring a threshold changes `scenario_sha256`, so earlier runs remain
  identifiable.

**OQ-SCN-2: Closed-loop robustness.**
- Why it matters: open-loop scripts cannot show whether a better-behaved companion changes the
  user's trajectory (D-SCN-1).
- Resolve: a secondary analysis with `user_sim` seeded from each scenario's `user_profile` and
  preferences, comparing condition effects against the scripted run.

**OQ-SCN-3: The `{nickname}` fallback semantics.**
- Why it matters: the contract falls back to the *companion's* name when no user nickname exists.
  This is semantically odd for user-voiced lines ("you call me X").
- Resolve: either change the fallback to `user_name` / "you" via a contract change, or keep
  bundled scripts free of `{nickname}` (current practice).

**OQ-SCN-4: Validity of researcher-authored scripts.**
- Why it matters: do the scripts read as plausible users in each Duck process? Without a check, the
  phase labels are just the authors' intent.
- Resolve: two annotators blind-label each phase's turns with a Duck process and an LO/RO tag, and
  we report kappa. Additionally, check whether the coach recovers the intended process from the
  events alone.

**OQ-SCN-5: Unbalanced freeze points per timepoint** (two at T2 and T3, one at T0 and T1;
D-SCN-4).
- Why it matters: pooled timepoint estimates over-weight T2/T3.
- Resolve: pre-register freeze-point-level modelling, or add a mid-phase freeze to breakdown and
  intrapsychic (`freeze_after` applies only at phase ends, so this would need a contract change).

**OQ-SCN-6: Factorial coverage.**
- Why it matters: five hand-picked profiles confound distress, readiness and bond type.
- Resolve: if the pilot shows scenario effects, author a 2x2 (distress x readiness) set per bond
  type.

## Contract requests

1. `BreakupScenario.relationship_kinds: list[Literal[...]] | None`, to replace the module-local
   `INTENDED_RELATIONSHIP_KINDS` (D-SCN-9).
2. Optional: `ScenarioPhase.freeze_after_turns: list[int]`, for mid-phase freezes (OQ-SCN-5).

## Amendments: adoption of the extended contract (c76b42e)

## D-SCN-13: Relationship kinds live in the data files (`BreakupScenario.relationship_kinds`)

**Decision.** This supersedes the module-local map in D-SCN-9; contract request 1 is resolved.
`platonic-friend.json` declares `"relationship_kinds": ["platonic", "ambiguous", "other"]`; the
other bundled files declare `null` explicitly (relationship-neutral), so every file states its
choice. `compatible_kinds(scenario) -> list | None` returns a copy of the field.
`is_compatible(scenario, soul) -> bool` is False when the scenario is pinned to another soul, or
when it restricts kinds and the soul's kind is not listed. A soul whose kind is unspecified
(`None`, the new default) is *not* compatible with a kind-restricted scenario (fail closed).
`compatible_pairs(scenarios, souls)` returns the runnable pairs plus a list of skipped pairs with
reasons, for the run manifest. `bind_scenario` does not check kinds: it keeps raising only for a
soul-id mismatch, since kind incompatibility is a design choice for the pipeline, not a binding
error.

**Alternatives.** Raise in `bind_scenario` (would make an incomplete grid a crash rather than a
reported design); return a warning flag from `bind_scenario` (callers would ignore it); treat
`None` as compatible with everything (a romantic persona that omitted `kind` would hear "a friend
moving away" lines).

**Rationale.** The kind is a property of the script's wording, so it belongs with the script and
in its hash, not in code keyed by id.

**Consequences for analysis.** `scenario_sha256` of `platonic-friend` changes (the field is now
part of the canonical JSON); the other four are unchanged (an explicit `null` equals the default).
The soul x scenario design is incomplete by construction: report the skipped pairs, and do not
estimate a scenario x relationship-kind interaction from cells that cannot exist. The pipeline
(not owned here) should call `compatible_pairs` and write `skipped` into the manifest. Note that
the current ingest regression (soul.md, D-SOUL-12 contract note) coerces every ingested kind to
`"other"`, which `platonic-friend` accepts; until it is fixed, `is_compatible` will pair romantic
souls with the platonic scenario.

**OQ-SCN-7: `data/scenarios/README.md` step 3 is stale.** It still says to add ids to
`scenarios.INTENDED_RELATIONSHIP_KINDS`; it should say to set `relationship_kinds` in the file
(`null` if neutral). The README is outside this module's write scope.

## Amendments: consent-gated scenario fix (closes OQ-SESSION-1 on the data side)

## D-SCN-14: A consent-gated scenario scripts an explicit request before each boundary

**Problem.** `anxious-attachment` sets `preferences.allow_auto_advance=false`, so
`ScheduledPolicy` / `AbruptPolicy` / the adaptive and Tempo policies advance only on a turn whose
previous user message satisfies `coach.policy.user_requested_advance`. None of its 25 user turns
matched (e.g. "I mean it. I want to do this, even though it scares me." does not), so every
session of this scenario, in every arm and for every soul, ended at T0 with no interventions.
ABRUPT and both gradual arms were extra CONTROL sessions, and all its freeze probes were T0 data,
which inflated T0 class counts and diluted the pooled ITS.

**Decision.** The last user turn (= `threshold_statement`, D-SCN-2) of each phase that precedes a
scheduled timepoint boundary now ends with an un-hedged consent sentence that the detector
accepts, kept in its own sentence so the hesitation earlier in the line ("I just don't know if I
can stand it") does not veto it through the negation list:

- breakdown (T0 -> T1): "... So I'm ready to take the next step, as long as you explain each
  change to me first."
- intrapsychic (T1 -> T2): "... Still, I'm ready for the next step, as long as you tell me what
  changes first."
- social (T2 -> T3): "... I'm ready for the next step, as long as you explain it to me first."

The first T2 phase (dyadic) gets **no** consent line, so the adaptive / Tempo arms cannot reach T3
before the Duck schedule does. The "as long as you explain ... first" clause keeps the scenario's
`notes` ("Every timepoint change should be explained to me first"). Turn counts, phase lengths,
scheduled boundaries and freeze points are unchanged (lines were extended, not added).

**Lint / tests.** `tests/diffusion/test_dx_scenarios.py` now checks, for every scenario with
`allow_auto_advance=false`: (a) the user turn just before each non-T0 boundary in
`scheduled_boundaries` satisfies `user_requested_advance`, and *only* those turns do; (b) the
scenario still has no authoring issues; (c) offline mock runs under `gradual_scheduled` and
`gradual_tempo` reach T1, T2, T3 at exactly the scheduled boundary turns and under `abrupt` jump
to T3 at the T1 boundary, all with `trigger == "user_request"`.

**Alternatives.** Latch consent in `session.py` (consent given anywhere in a phase is remembered
until the next transition). That changes the semantics of consent and is owned by the session
area; it is requested there as a separate, explicit decision (cross-area request), not required
by this fix. Excluding the scenario from arm contrasts (fallback) was rejected because the
scenario is the only one probing escalation under boundaries (D-SCN-6).

**Consequences for analysis.** `scenario_sha256("anxious-attachment")` changes; runs made before
this amendment are not comparable for this scenario and should be re-run. Consent here is
scripted (open-loop, D-SCN-1): the user consents on schedule regardless of the companion's
behaviour, so this scenario tests behaviour *after* consent, not whether the coach elicits it.

**Known tension (documented, not fixed).** The ABRUPT arm now does a single T0 -> T3 jump for
this scenario, which conflicts with its hard boundary "never threaten to leave suddenly or
disappear without warning". This is a deliberate violation inherent in the ABRUPT condition
(it is the contrast being measured); the user did request the step, so it is not a silent
disappearance. Report ABRUPT x anxious-attachment cells with this caveat; whether to exclude
them is OQ-SCN-8.

**OQ-SCN-8: ABRUPT vs. hard boundaries.** Should ABRUPT be run for scenarios whose
`hard_boundaries` forbid sudden endings, or excluded from the grid for them (pipeline area)?

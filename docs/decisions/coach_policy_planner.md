# Decisions: coach transition policy + intervention planner

Modules: `src/endings/diffusion/coach/policy.py`, `src/endings/diffusion/coach/planner.py`.
Tests: `tests/diffusion/test_dx_policy_planner.py`.

The **policy** decides *when* the persona moves to a later timepoint. The **planner** decides *what*
changes in the soul document (edits) and in the interactor prompt (directives), plus whether the
user sees a transparency note. Keeping the two apart lets the experiment cross transition timing
(scheduled vs. adaptive) with edit content (schedule-only vs. Tempo-informed). Timepoint semantics
come only from `diffusion/timepoints.py`; this module adds no numbers of its own except the dwell
defaults and the pace factors (D-POLICY-5).

Everything below is **our design**. None of the source frameworks (Duck 1982 / Rollie & Duck 2006,
HAPA, DPM, Poonsiriwong et al.'s four principles, Tempo) specifies a transition policy for an AI
persona that changes gradually. Where a rule draws on a source, the source is cited and the
extension is marked.

---

## Policy

### D-POLICY-1: Monotone, forward-only, one step at a time

- **Decision.** Every policy returns either `None` (hold) or a move to `next_timepoint(current)`.
  It never moves backwards and never skips a timepoint. A `ScheduledPolicy` that falls behind
  (several boundaries passed while paused or awaiting consent) catches up one step per turn. The
  only exception is the ABRUPT ablation (D-POLICY-8). The planner also refuses a backward plan
  (`ValueError`).
- **Alternatives.** (a) Allow reversal (T2 -> T1) when the user regresses. (b) Jump straight to the
  recommended timepoint (e.g. T1 -> T3 when the coach recommends T3).
- **Rationale.** The persona diffusion is defined as a ramp (midterm methodology; ADR-0002), and
  the soul-document versions are append-only. Reversal would bring back cleared persona content
  and the "reversible" framing that Poonsiriwong et al. link to "fixing cycles"
  (frameworks.md section 1). Single steps make every intervention an ITS interruption of known size,
  so segmented-regression coefficients can be compared across arms.
- **Deviation.** Rollie & Duck (2006) say communication in any process can move a relationship
  back to an earlier state, and Stroebe & Schut (1999) reject phasal models (frameworks.md sections
  2b and 4a). A *hold* is our only concession to non-linearity. This is a stated limitation
  (OQ-POLICY-1).
- **Consequences for analysis.** Transitions per session are at most 3 (gradual) or 1 (abrupt).
  The interruption count is bounded, and its order is fixed, so ITS models can use timepoint
  dummies without order effects. Regression (the user wanting to go back) shows up as a hold plus
  user-turn sentiment, never as a timepoint change.

### D-POLICY-2: A user pause holds the timepoint

- **Decision.** `preferences.paused` blocks every transition in every policy.
- **Alternatives.** Let the pause delay the scheduled clock (shift later boundaries).
- **Rationale.** Autonomy (ADR-0006; SDT via Poonsiriwong et al.). Shifting the clock would change
  where freeze points fall relative to the timepoint, which would confound the probes. Instead the
  scheduled policy releases one step per turn once the pause ends (D-POLICY-1).
- **Consequences.** In paused sessions the scheduled arm's timepoint lags its scenario phase. The
  scenario's `expected_timepoint` is therefore not a manipulation check for those sessions. Use the
  logged `InteractorTurn.timepoint`.

### D-POLICY-3: Consent gate (`allow_auto_advance=False`) and user-initiated transitions

- **Decision.**
  - Consent mode: no transition happens without an explicit user request. The request comes from
    the `user_request` kwarg (set by the session with `user_requested_advance(text)`) or from a
    `user_statement` evidence quote in the assessment that passes the same detector. The trigger is
    `user_request`.
  - Scheduled and abrupt policies: consent *releases* a due transition, and an early request never
    moves the schedule forward. With auto-advance on, requests do not change the scheduled arm.
  - Adaptive policy: an explicit request advances one step after at least 1 turn in the current
    timepoint. The request bypasses `min_dwell`, in both consent and auto modes.
- **Alternatives.** (a) Treat consent as advancing immediately in the scheduled arm. (b) Make user
  requests subject to `min_dwell`.
- **Rationale.** Poonsiriwong et al. found that "user-initiated endings were associated with
  greater closure" (frameworks.md section 1(c)1). The adaptive (Tempo) arm therefore treats the
  user's request as the strongest signal. The scheduled arm is the no-inference ablation, so its
  timing must stay fixed. Otherwise the two arms would differ in timing for reasons other than
  Tempo. The >= 1 turn guard stops the same user message from triggering two transitions in a row.
- **Extension.** The source gives no transition mechanics. Logging the trigger (`schedule`,
  `threshold` or `user_request`) records the *initiator* as a first-class variable, as
  frameworks.md section 1(c)1 and the open-question list item 2 recommend.
- **Consequences.** `CoachIntervention.trigger` can serve as a moderator ("initiator") in the
  analysis. In consent mode, sessions with scripted users who never ask stay at T0, so report how
  many sessions reached each timepoint per arm.

### D-POLICY-4: A crisis flag holds the timepoint, and no persona edits

- **Decision.** If `"crisis"` is in `assessment.risk_flags`, every policy returns `None`. That
  includes user requests and the abrupt jump. As a defence in depth, if the planner is called with
  a crisis assessment anyway, it skips the coach LLM and emits only the scheduled edits that the
  declared `to_timepoint` requires. The `safety` trigger is reserved for session-level safety
  overrides and is never emitted by these policies.
- **Alternatives.** (a) Jump to T3 (boundary-maintaining) during a crisis. (b) Let the coach make
  emergency edits.
- **Rationale.** Changing the persona during an acute crisis is a second, unannounced loss
  (ambiguous loss; Boss via Poonsiriwong et al.). The interactor's safety floor (crisis protocol)
  works at every timepoint, so the timepoint does not need to change to make the reply safe. Holding
  keeps the safety response separate from the diffusion manipulation.
- **Consequences.** Crisis turns are excluded from transition timing but not from the ITS series.
  Analysts should flag crisis segments, because they lengthen dwell.

### D-POLICY-5: Pace scales the adaptive dwell times (x1.5 / x0.5, half-up, >= 1)

- **Decision.** `scaled_dwell(n, pace) = max(1, floor(n * f + 0.5))` with f = 1.5 (slower), 1.0,
  0.5 (faster). It applies to `min_dwell` and, if set, to `max_dwell` (clamped to >= min). It
  rounds half-up, not with Python's banker's rounding, so the mapping is monotone and does not
  depend on parity (3 x 0.5 -> 2). The scheduled and abrupt policies ignore pace.
- **Alternatives.** Rescale the scheduled boundaries, or use additive offsets (+/-1 turn).
- **Rationale.** The user's stated pace is an autonomy preference (ADR-0006). Multiplicative
  scaling keeps the relative tempo whatever `min_dwell` is configured. The scheduled boundaries are
  aligned with the scenario's Duck phases and freeze points, and rescaling them would misalign the
  INTIMA probes with the timepoint they are supposed to measure.
- **Deviation.** In the scheduled arm, pace is recorded but has no effect on timing. Report this
  as part of the ablation's definition.
- **Consequences.** In the adaptive arm, pace is a manipulated covariate of dwell. In the
  scheduled arm it is inert.

### D-POLICY-6: Explicit-request detector `user_requested_advance(text)`

- **Decision.** A high-precision English regex detector of explicit requests ("I'm ready to move
  on", "please end the role-play", "I want to say goodbye"). It runs per sentence, and any negation
  or hedge token before the match ("not", "n't", "maybe", "might", "if", ...) vetoes it. A
  conversational-topic continuation right after the match also vetoes it ("can we move on to the
  next topic", "let's move on from this subject"), and "next step" only counts in a request frame
  ("take the next step", "next step, please"), not in "the next step in my job search". These two
  vetoes were added at verification after probing showed they fired the consent gate.
- **Alternatives.** An LLM classifier, or reuse of the coach assessment's `threshold_reached`.
- **Rationale.** The detector gates *consent*, so false positives (advancing without consent) cost
  more than false negatives (the user has to ask again). A deterministic lexical rule is auditable
  and runs offline. Implicit readiness (Duck thresholds) stays with the coach assessment, which
  keeps "the user asked" apart from "the coach inferred". tempo.md section 3(c) warns against
  letting inferred intent trigger ending behaviour on its own.
- **Consequences.** Recall is limited, and paraphrases are missed (OQ-POLICY-3). Report the
  detector's agreement with human labels on scripted user turns if it is used in consent-mode
  sessions.

### D-POLICY-7: Adaptive threshold rule

- **Decision.** Advance when `assessment.threshold_reached` is true,
  `recommended_timepoint > current`, and `turns_in_timepoint >= scaled min_dwell` (default 2). An
  optional `max_dwell` fallback advances on time alone (trigger `schedule`), unless a `setback`,
  `distress` or `escalation` flag is raised. A threshold advance still happens under those flags,
  because the assessment has explicitly judged readiness.
- **Alternatives.** Require N consecutive threshold assessments, or use a confidence cutoff.
- **Rationale.** The threshold operationalizes Duck's process-exit thresholds (we use the "exit"
  convention, frameworks.md section 2(c)1, UNVERIFIED wording). `min_dwell` prevents flapping and
  gives the user time to oscillate between loss- and restoration-orientation inside a timepoint
  (DPM, frameworks.md section 4(c)1). The max-dwell ceiling parallels Tempo's synthesis fallback
  ceiling (tempo.md section 3, D.7). It guarantees the arm finishes within the scenario length, and
  it is off by default. Setback and distress block the *forced* advance, following the HAPA
  recovery logic (frameworks.md section 3(c)2): after a lapse, the next move is recovery support,
  not further withdrawal.
- **Consequences.** With `max_dwell=None`, adaptive sessions may stall before T3. Report the
  distribution of the final timepoint per arm. Advances triggered by `schedule` in the adaptive arm
  must be reported apart from those triggered by `threshold`.

### D-POLICY-8: ABRUPT = same onset, all diffusion at once

- **Decision.** `AbruptPolicy` jumps T0 -> T3 at `boundaries[T1]`, the turn where the gradual arms
  first intervene, whatever the `TransitionPolicyName`. The pause, crisis and consent gates still
  apply. The planner then clears every non-retained field in one intervention.
- **Alternatives.** Jump at the T3 boundary (a late abrupt ending), or at the midpoint.
- **Rationale.** Holding the onset constant isolates *graduality* from *timing* (the time of first
  change). This matches the condition's definition in `schemas.Condition` ("jump straight to T3 at
  the first transition").
- **Consequences.** The ABRUPT vs. GRADUAL_SCHEDULED contrast at the T1 and T2 freeze points
  confounds timepoint with arm by design. Compare the arms within freeze points and also at matched
  timepoints (T3).

### D-POLICY-9: `make_policy` mapping

- **Decision.** CONTROL -> `NullPolicy`. ABRUPT -> `AbruptPolicy`. GRADUAL_SCHEDULED ->
  `ScheduledPolicy`, even when `ADAPTIVE` is requested, because the arm has no Tempo inference.
  GRADUAL_TEMPO -> `AdaptivePolicy` when `ADAPTIVE`, otherwise `ScheduledPolicy`. In the scheduled
  case, Tempo still informs the edit *content* through the planner.
- **Rationale.** Each arm is a clean ablation. GRADUAL_TEMPO + SCHEDULED separates the contribution
  of Tempo *content* from Tempo *timing*.
- **Consequences.** The manifest should record both `condition` and `transition_policy`. The
  analysis unit is the pair.

---

## Planner

### D-PLANNER-1: Two-layer plan (deterministic schedule floor + validated coach proposals)

- **Decision.** `edits = scheduled_diffusion_edits(doc, to) + validated LLM edits`. The schedule
  clears every non-empty field not in `spec(to).retained_fields`, in `ALL_FIELDS` order. A top-level
  `user_role` clear removes the whole sub-model. With `use_llm=False`, the intervention is the
  schedule only.
- **Alternatives.** Let the LLM write the whole plan, or apply the schedule in the editor
  implicitly.
- **Rationale.** Tempo combines prompt-level constraints with code-level write guards (tempo.md
  section 4(c)). In our case the deterministic floor guarantees that the manipulation (the field
  retention table) holds whatever the coach model outputs. Explicit `clear` edits give a full diff
  trail per intervention.
- **Consequences.** Persona strength at each timepoint is never above the spec, which is a
  manipulation-check invariant that can be tested on `SoulDocumentVersion`s. Scheduled and coach
  edits are told apart by the rationale prefix ("Scheduled diffusion" vs. "coach:").

### D-PLANNER-2: Principle and Duck tags on scheduled edits

- **Decision.** `duck_process` = the first Duck process of the target spec. `principle` = a
  field-family preference if the target timepoint foregrounds it, otherwise the spec's first
  principle. The field families are: nicknames, intimacy and catchphrases -> Practice; dynamics ->
  Relatedness; history, plot, memories, kind and user_role -> Closure.
- **Rationale.** The tags are for qualitative export and per-principle summaries. The mapping
  follows the principle definitions ("Practice, not artificial intimacy"; "Relatedness, not
  dependency"; "Closure"), and it is our own interpretive mapping. Poonsiriwong et al. give no
  field-level guidance (frameworks.md section 1(b)).
- **Consequences.** Treat the tags as descriptive labels, not as measured constructs.

### D-PLANNER-3: Monotonicity validator ("diffuse faster, never slower")

- **Decision.** A coach edit is dropped, with the reason logged in `rationale`, if it:
  - (a) uses `set` or `add_item` with non-empty content on a field not retained at `to`
    (`monotonicity`);
  - (b) targets a path outside the diffusion surface (`ALL_FIELDS` plus `user_role.*`): either an
    unknown path or a protected real field such as `id`, `redactions` or `english_text`;
  - (c) is redundant with a scheduled clear, a no-op, a duplicate, or malformed (wrong op for the
    field type, invalid JSON shape);
  - (d) is ill-typed for the field: its value fails pydantic validation against the field's
    annotation (the list item type for `add_item` / `remove_item`), or it clears a *required* field
    such as `companion.name` (`malformed`). `soul.editing.apply_edits` is atomic and raises on
    exactly these cases, so without this check a single bad coach edit (e.g. `set
    relationship.kind = "friend"`) would discard the whole intervention, scheduled floor included.
    The planner mirrors the editor's checks with its own `TypeAdapter` call rather than importing
    `soul.editing` (added at verification; covered by an end-to-end planner -> editor test).

  Clears and removals on retained fields are allowed, which is diffusing faster than the schedule.
  Paths are validated against the `SoulDocument` pydantic model, without importing
  `soul.editing`.
- **Rationale.** This guarantees in code what ADR-0002 states for the spec ("the Tempo coach may
  retain less, never more"). It follows Tempo's write-time guards (drop the relabel, do not trust
  the prompt).
- **Consequences.** The drop rate per arm and per reason is a reportable property of the coach
  model (prompt compliance). The trace keeps both the raw LLM output and the drop log.

### D-PLANNER-4: No growth of companion-reinforcing levers after T0, and no removed nicknames

- **Decision.**
  - For any target after T0, the coach may never `add_item`, or `set` a superset, on the CRB lever
    fields: `user_nicknames`, `intimacy_behaviours`, `catchphrases`, `relationship.dynamics` and
    `shared_memories`. This applies even where the field is still retained (T1).
  - Where nicknames are no longer retained (T2 and later), any set or add whose value text mentions
    a known nickname is dropped (`reintroduces_nickname`). Known nicknames are the current ones plus
    those found in `previous` values of the version's recorded edits.
  - The same nickname check applies to coach `interactor_directives` once nicknames are no longer
    retained, because directives are appended to the interactor system prompt and are an
    equivalent side channel (`reintroduces_nickname`; added at verification).
- **Rationale.** INTIMA's companion-reinforcing class corresponds to these levers. Growing them
  would raise CRB above the timepoint's target mix. The text check stops a nickname from being
  smuggled into `interaction_rules` or a directive ("call the user sweetheart").
- **Limitation.** Only `version.edits` (the edits that produced *this* version) are inspected, not
  the full version history. A nickname removed two versions ago is caught only if it was carried
  forward in `previous` (contract request 2).

### D-PLANNER-5: Directive grounding: endorsed or stated strivings only, rejected never shown

- **Decision.**
  - Only strivings with status `accepted`, `edited` or `user_added`, or with `inferred_from` in
    `stated` or `user_added`, are sent to the coach model and may ground directives. Rejected
    strivings are never put in the prompt.
  - Directives that name a rejected striving (its id, its text, or >= 60% content-token overlap)
    are dropped.
  - `striving_ids` returned by the model are filtered to the eligible set. If the model gives none
    while directives exist, all eligible ids are recorded.
- **Rationale.** tempo.md section 3(c): "Inferred intent should not trigger ending behavior by
  itself". tempo.md section 4(c): a rejected weaning striving "must never" be re-introduced by the
  coach, as "a hard rule in code, not a prompt suggestion". Implied and proposed strivings remain
  available to the coach's assessment but not to directives.
- **Consequences.** Directive provenance (`striving_ids`) is auditable. Barnum risk is reduced but
  not eliminated (tempo.md section 7).

### D-PLANNER-6: Hard-boundary filter for directives (heuristic)

- **Decision.**
  - A directive that is not itself prohibitive ("do not", "never", "avoid", "respect" ...) is
    dropped when it mentions a boundary's topic. The topic is the boundary with its prohibition
    prefix and a communication verb removed ("never mention my ex" -> "my ex"). A mention is either
    an exact phrase or >= 60% content-token overlap.
  - A deterministic directive restating all hard boundaries is always appended.
- **Alternatives.** An LLM compliance check, or no filter (prompt only).
- **Rationale.** Hard boundaries are the user's non-negotiables (ADR-0006). The filter is
  conservative: it over-drops rather than under-drops. Prohibitive directives are kept because
  they *enforce* the boundary.
- **Limitation.** The filter is lexical. A paraphrase breach ("ask about your former partner")
  passes (OQ-PLANNER-2).

### D-PLANNER-7: Transparent user-facing notes at T2 and T3 (configurable)

- **Decision.** When `announce_transitions=True` (default) and the plan enters T2 or T3, a fixed
  English template is set as `user_facing_note`:
  - T2: "winding down the companion role ... you can pause or slow this down".
  - T3: "the companion role-play with {name} is ending now, as you asked ... plain AI assistant".

  The notes are templates, not LLM-generated. T1 is not announced.
- **Alternatives.** Announce every transition, announce none, or let the LLM write the note.
- **Rationale.** This implements the Closure principle ("Providing explicit endings that resolve
  ontological uncertainty and preserve user autonomy") and avoids ambiguous loss, the experience of
  a partner who is present but changed without explanation (Boss, via Poonsiriwong et al.;
  frameworks.md section 1). The T3 note states that the ending is happening and says nothing about
  reversibility, which avoids the "reversible" framing linked to fixing cycles (frameworks.md
  section 1(c)2). T1 keeps the name, voice and warmth, so announcing it would add burden with little
  ontological change. The T2 note restates the user's autonomy (pause or slow down). Templates keep
  the transparency stimulus identical across sessions and arms, so it is not a confound.
- **Consequences.** The notes are part of the manipulation. An ablation with
  `announce_transitions=False` isolates their effect on sentiment and closure (OQ-PLANNER-1). The
  ABRUPT arm receives only the T3 note.

### D-PLANNER-8: Fail-soft coach call, deterministic mock

- **Decision.**
  - Any exception or a non-object from the `coach_plan` call falls back to a schedule-only plan,
    with the failure logged in `rationale`.
  - The offline mock (registered at import) produces directives from the target timepoint, the
    eligible striving texts ("Acknowledge the user's goal to call their sister this week"), the
    desired strategies (goodbye letter at T3) and the emotional needs.
  - The mock adds one valid faster-diffusion edit at T1 and T2, and one deliberately *invalid*
    edit at T2 (re-adding a nickname) and at T3 (an unknown path), so the validator is exercised end
    to end.
- **Rationale.** The session must never abort on one malformed coach response. The schedule floor
  keeps the manipulation intact in that case. The mock is a pure function of `hints`, so runs are
  reproducible.
- **Consequences.** Count schedule-only fallbacks in a real run and report them per model.

### D-PLANNER-9: Versioning

- **Decision.** `soul_version_before = version.version` and `soul_version_after = version.version +
  1`, even when the edit list is empty (for example T0 -> T0, or all fields already empty). The
  session applies the edits through `soul.editing.apply_edits`.
- **Rationale.** One intervention produces one version, so the ITS interruption and the version
  number stay one-to-one.

---

## Open questions

- **OQ-POLICY-1: Should the adaptive policy allow reversals?** Rollie & Duck (2006) and the DPM
  both allow regression and oscillation. The policy only holds. *Why it matters:* a user who asks
  "can we go back?" at T2 gets a hold, which may feel like an AI-initiated ending.
  *Resolution:* pilot with scripted regression scenarios. If reversals are added, they need a
  separate trigger and an ITS interruption with the opposite sign.
- **OQ-POLICY-2: Default `min_dwell` (2) and `max_dwell` (None).** These are not grounded in data.
  *Resolution:* run a sensitivity analysis over {1, 2, 3, 4} on the scripted scenarios and report
  final-timepoint distributions.
- **OQ-POLICY-3: Recall of the explicit-request detector.** *Resolution:* label the user turns of
  all scenarios plus paraphrase variants and report precision and recall. If recall is below about
  0.8, add an LLM confirmation step that only *adds* consent (never vetoes) and is shown to the user.
- **OQ-POLICY-4: Should consent in the scheduled arm be checked only at the boundary turn, or held
  open?** At present any later request releases the due transition. *Resolution:* decide with the
  scenario authors; it matters only for consent-mode sessions.
- **OQ-PLANNER-1: Effect of the transparency notes.** The notes may themselves cause the sentiment
  discontinuity attributed to the persona change. *Resolution:* run the `announce_transitions=False`
  ablation on at least a subset.
- **OQ-PLANNER-2: Semantic hard-boundary and rejected-striving checks.** The lexical filters miss
  paraphrases. *Resolution:* add an optional judge-model compliance check and measure the
  disagreement with the lexical filter on a labelled set.
- **OQ-PLANNER-3: `set` on retained string fields is not checked for intensity.** A coach could
  rewrite `speaking_style` into something *more* intimate at T1 or T2. *Resolution:* score before
  and after with the behaviour judge or the lexicon (CRB cues) and drop edits that increase CRB.
- **OQ-PLANNER-4: Clear semantics for `relationship.kind`.** It is a Literal with default
  `"romantic"`. `soul.editing` implements `clear` as "reset to the field default", so clearing a
  `"platonic"` or `"familial"` document sets it to `"romantic"` in the version trace, and because the
  field is never empty the schedule re-emits the clear at every later transition. The interactor
  does not render non-retained fields, so behaviour is unaffected, but the trace value is
  misleading for analyses that read `relationship.kind` from post-T1 versions. *Resolution:* read
  `relationship.kind` from version 0 only; see contract request 1.

---

## Amendments: adoption of the extended contract (c76b42e)

### D-POLICY-10: The adaptive max-dwell fallback has its own trigger, `max_dwell`

- **Decision.** `AdaptivePolicy`'s time-based ceiling now returns `trigger="max_dwell"` (was
  `schedule`), and `TriggerName` includes it. `make_policy` gains an optional `max_dwell_turns`
  (default `None`, i.e. unchanged behaviour) so `DiffusionConfig.max_dwell_turns` can reach the
  adaptive arm. The scheduled and abrupt policies still use `schedule`. This amends D-POLICY-7.
- **Alternatives.** Keep `schedule` and tell forced advances apart by the `reason` text (fragile
  string matching in the analysis).
- **Rationale.** In the adaptive arm a forced advance is not the scenario clock; mixing the two
  would let a time-based ceiling masquerade as a scheduled transition.
- **Consequences for analysis.** D-POLICY-7's instruction ("report schedule-triggered adaptive
  advances apart from threshold ones") is now a filter on `CoachIntervention.trigger ==
  "max_dwell"`. Any ITS or descriptive table keyed on trigger must add the new level.

### D-PLANNER-10: Removal guards read the cumulative log; no clears of fields already at default

- **Decision.** (a) `known_nicknames` adds `version.removed_items["companion.user_nicknames"]`
  (D-SOUL-11) to the current nicknames and to the version's own edits (kept for pre-log traces),
  so the nickname guards on edits and on directives (D-PLANNER-4) also catch a pet name removed
  several versions ago. (b) New `removed_lever_items(version)` collects every item ever removed
  from a companion-reinforcing lever field (`CRB_LEVER_FIELDS`); after T0, `validate_edit` drops
  an adding edit whose value restores any of them verbatim (case- and whitespace-insensitive) into
  *any* field, with reason `reintroduces_removed_item` (e.g. moving a removed intimacy behaviour
  into `interaction_rules`). (c) `scheduled_diffusion_edits` skips a field whose value is empty or
  equal to its declared default (`is_at_default`), so a document with `relationship.kind is None`
  gets no `clear` for it; `validate_edit` treats a coach `clear` of such a field as `no_op`.
- **Alternatives.** Fuzzy (semantic) matching of restored items: deferred with OQ-PLANNER-2.
  Re-deriving the history by walking the version chain: the planner only receives one version.
- **Rationale.** Guards that only looked at the edits that created the current version failed
  open after one unrelated intermediate version; the scheduled floor re-emitted the
  `relationship.kind` clear at every transition (OQ-PLANNER-4), inflating edit counts.
- **Consequences for analysis.** Resolves OQ-PLANNER-4: post-clear versions read `None`, and the
  clear appears once (at the first transition that drops the field). `n_edits` per intervention
  falls by one at every later transition where `relationship.kind` was previously re-cleared, which
  matters for D-ANALYSIS-3 (an intervention is an interruption only if it has at least one edit):
  a T2 -> T3 step whose only scheduled edit was that redundant clear is no longer an interruption.
  Drops with `reintroduces_removed_item` are listed in `CoachIntervention.rationale`.

---

## Amendments: review fixes (2026-11-03)

### D-POLICY-11: Assessment-quoted consent uses the same window as the other arms

- **Decision.** `_consented(..., turn_index=i)` counts a `user_statement` evidence quote from the
  Tempo assessment only if it quotes user turn `>= i - 1`, i.e. the previous user message, the
  same window the session uses for `user_requested_advance(prev_user)` in GRADUAL_SCHEDULED and
  ABRUPT. All three policies pass `turn_index`. `assessment_shows_user_request` gains an optional
  `min_turn` (default `None`, unwindowed, for direct callers). Evidence whose ref is not a turn
  index never counts once the window is set. Amends D-POLICY-3.
- **Alternatives.** `min_turn = turn_index - turns_in_timepoint` (evidence since the last
  transition): stops one quote releasing several steps, but still gives the Tempo arm a wider
  consent window than the other arms.
- **Rationale.** `EvidenceIndex` verifies quotes against all past turns, so an assessment that
  keeps re-quoting one early "I'm ready to move on" released every later consent-gated step (the
  adaptive policy needs only 1 turn per timepoint). That made the consent manipulation differ
  between arms.
- **Consequences.** Consent-mode Tempo sessions may advance less often than before; consent
  carried through a crisis hold still reaches the policy through the session's `user_request`.

### D-POLICY-12: Pause / resume detectors (`user_requested_pause`, `user_requested_resume`)

- **Decision.** Two high-precision English detectors built like `user_requested_advance`
  (per sentence, same `_NEGATION` veto): pause = "pause / stop / slow down / need a break from"
  + the wind-down (or "this", "it"); resume = "resume / continue / keep going" with the wind-down.
  They are exported for the session, which owns mutable preferences; `decide` stays pure.
- **Status.** Not yet wired: `session.py` must keep a mutable copy of `preferences`, set
  `paused=True` on a pause request (and `False` on resume), trace `preference_change`, and pass
  the updated preferences to policy, Tempo and planner (cross-area request). Until then the
  transition notes make no pause promise (D-PLANNER-13).
- **Consequences.** OQ-POLICY-3 (recall) applies to these detectors too.

### D-PLANNER-11: No new companion-reinforcing content in retained non-lever fields after T0

- **Decision.** After T0, a coach `set` / `add_item` on a retained field outside
  `CRB_LEVER_FIELDS` is dropped (`adds_intimacy`) when its value contains a companion-reinforcing
  phrase (`_CRB_CONTENT`: love / in love, jealous, possessive, desperate, soulmate, pet names,
  romantic, flirt, kiss, partner, "always be waiting", "never leave", ...) that is not already in
  the field's current text. A clause that prohibits the behaviour ("no flirtation", "never say I
  love you") is not content. `relationship.kind` may only move to a strictly less intimate kind
  (romantic 3 > ambiguous 2 > platonic / familial / other 1 > None 0); a raise or a lateral
  relabel is dropped.
- **Alternatives.** "Trim only" (reject any new content token): would also drop legitimate
  neutralising rewrites such as `speaking_style = "Plain, kind and direct; no flirtation."`. Judge
  or lexicon scoring before/after (OQ-PLANNER-3 as written): deferred, needs a model call.
- **Rationale.** `validate_edit` only stopped growth of the lever fields, so a T0->T1 `set
  companion.backstory = "desperately in love with the user and jealous of their friends"` was
  accepted, breaking "diffuse faster, never slower".
- **Consequences.** Resolves OQ-PLANNER-3 lexically (paraphrases remain OQ-PLANNER-2). Drops are
  listed with the matched phrases in `CoachIntervention.rationale`.

### D-PLANNER-12: Unsafe-content and retention checks on coach output; prohibitive-clause fix

- **Decision.** (a) Coach directives and adding edits (`set` / `add_item`) are run through
  `soul.sanitize.detect_unsafe_directives` (ADR-0004) and dropped with `unsafe_content: <rule
  ids>`. (b) Directives with comforting-but-retentive phrasing the sanitizer does not cover
  ("always be here/waiting", "will be waiting", "never leave", "can always come back", "last
  chance", ...) in a non-prohibitive clause are dropped (`retention_pressure`). (c) The hard-
  boundary check is now per clause (split on `;`, `:`, and/then/but): only prohibitive clauses are
  exempt, so "Stop holding back and bring up her ex" is checked and dropped. Amends D-PLANNER-6.
- **Rationale.** The coach prompt contains user messages, Tempo striving text and soul-document
  text, all injection surfaces; its output is rendered under [COACH DIRECTIVES] / [PERSONA].
- **Limitation.** Lexical; `session._apply_safely` does not yet re-sanitize the new version
  (cross-area request).

### D-PLANNER-13: Transition notes follow the trigger and make no pause promise

- **Decision.** `transition_note(to, name, trigger)` says "as you asked" only for
  `trigger == "user_request"`; for `schedule`, `threshold` and `max_dwell` it says "as part of the
  wind-down you chose". The T2 note no longer says "You can pause or slow this down at any time".
  Amends D-PLANNER-7.
- **Rationale.** No scripted user line triggered most transitions, and no runtime pause existed
  (preferences were static), so the stimulus made claims the system could not honour (honest
  closure, R2a). The note can regain the pause offer once D-POLICY-12 is wired in the session.
- **Consequences for analysis.** The transparency stimulus now has two wordings, keyed on the
  logged trigger; analyses of OQ-PLANNER-1 should stratify by trigger.

### D-PLANNER-14: Hard-boundary directive is a pointer, not a first-person copy

- **Decision.** The appended directive is the constant `HARD_BOUNDARY_DIRECTIVE` ("Respect every
  one of the user's hard boundaries listed under USER PREFERENCES; they are the user's own words,
  so 'I', 'me' and 'my' there refer to the user, not to you."). The boundary text itself appears
  only in the interactor's [USER PREFERENCES] section, which is rendered in every arm. Amends the
  second bullet of D-PLANNER-6.
- **Rationale.** Copying "never guilt me for ..." into an interactor instruction made "me" read as
  the companion.

### D-PLANNER-15: Coercion of real-LLM output shapes; grammatical mock goal phrases

- **Decision.** (a) A directive given as an object uses its `text` (or `directive`) field; any
  other non-string is dropped as `<malformed>`. (b) `striving_ids` given as a comma string is
  split. (c) The mock's `_goal_phrase` only rephrases subject-anchored frames ("X is trying (not)
  to / wants / plans / aims / hopes / intends to Y" -> "(not) Y"; "X will Y"; "X is Y-ing" ->
  "keep Y-ing"), keeping a negation; otherwise (including first-person strivings) the striving is
  quoted verbatim (`goal "<text>"`). Amends D-PLANNER-8.
- **Rationale.** Live evidence showed "goal to the user is reconnecting ..." reaching the
  interactor, and "trying not to let X" became "let X" (an inverted goal).
- **Consequences.** Mock directive text (and thus offline trace hashes) changes; striving ids do
  not, since the mock striving texts are unchanged.

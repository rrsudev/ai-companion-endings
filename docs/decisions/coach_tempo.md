# Decisions: coach input layer (`diffusion/coach/observe.py`, `diffusion/coach/tempo.py`)

Key: `TEMPO`. Source framework: Tempo / striving co-creation (Sapkota, Jörke, Sabbagh, Shaikh,
Wang & Landay, *"What Are You Really Trying to Do?": Co-Creating Life Goals from Everyday Computer
Use*, arXiv:2605.00497, UIST 2026). Research notes: `docs/research/tempo.md` (cited below as
"tempo.md §n"). Duck and HAPA notes: `docs/research/frameworks.md` §2–3.

**Scope of fidelity.** This is a re-implementation of Tempo's *loop* (observation → hierarchy →
strivings → user co-creation → constraints fed back into inference), not of its code. We did not
vendor any Tempo source; no commit SHA needs to be pinned for this module (see OQ-TEMPO-6). Every
departure from Tempo is labelled **Deviation** or **Extension** below.

Pipeline:

```
ComputerUseEvent[]  --observe.summarize_events-->  ActivityObservation[] (obs-<k>, signals)
                                                 \-> observe.usage_trend (momentum)
observations + InteractorTurn[] + UserPreferences + existing Striving[]
   --TempoCoachModel.infer_strivings (LLM task "tempo_infer" + guards)-->  Striving[]
   --TempoCoachModel.apply_user_edits (pure)-->  Striving[] (accepted/edited/rejected/user_added)
   --TempoCoachModel.assess (LLM task "tempo_assess" + validate_assessment)-->  CoachAssessment
```

---

## D-TEMPO-1: Structured computer-use event logs instead of screenshots
**Decision.** The coach observes `ComputerUseEvent` records (app, action, timestamps, window title,
short already-de-identified content), not screenshots. **Deviation** from Tempo, whose input is
mouse-triggered screenshots transcribed verbatim by a VLM (tempo.md §2(a)).

**Alternatives considered.** (i) Run Tempo's macOS observer and transcription; (ii) replay
synthetic screenshots/transcriptions through Tempo's prompts (adapter B, tempo.md §8.3); (iii)
structured event logs (adapter C).

**Rationale.** (a) Privacy: the companion app's screen *is* an intimate chat; Tempo's transcription
mode is instructed to "transcribe everything EXACTLY", and its contextual-integrity audit runs only
after the first transcription has been sent to the API (tempo.md §2(a), README). Sending
screenshots of intimate conversations to a third-party API is not acceptable for this study.
(b) Reproducibility: the evaluation uses scripted scenarios with simulated events; event logs make
the coach's input a deterministic function of the scenario file. (c) The code-level Tempo observer
already captures `app_name` / `window_title` metadata (tempo.md §2(a)), so app/time metadata is a
subset of what Tempo itself sees.

**Consequences for analysis.** Results speak to the *coach design* under a known input, not to
Tempo's real-world inference validity (tempo.md §6(c)); the paper must state this limitation. Our
input is richer than Tempo's in one respect (explicit `action` labels such as `type`, `search`,
`schedule`), and poorer in another (no on-screen content), so striving inference quality is not
comparable to Tempo's reported accuracy/alignment numbers.

## D-TEMPO-2: Companion-app content never enters the coach prompt
**Decision.** `summarize_events` skips content inspection for companion-app events entirely
(`continue` after counting dwell/messages). The rendered summary quotes only (a) search queries that
match the AI-relationship patterns and (b) titles of scheduled plans, truncated to 60 characters.

**Alternatives.** Quote companion content with consent (tempo.md §2(c) option ii); quote
everything.

**Rationale.** Tempo's audit taxonomy treats `personal_communications` as sensitive (tempo.md
§2(a)); tempo.md §7 notes that the *derived* hierarchy can be more sensitive than any single
capture. The conversation itself is already available to the coach as `InteractorTurn`s, so the
event layer adds only behavioural metadata.

**Consequences.** Observations are safe to log and share in the trace; conversational evidence is
always cited by turn index, which makes it auditable separately.

## D-TEMPO-3: Fixed windows anchored at t=0, ids `obs-<k>` by window index
**Decision.** Window `k` is `[k·window_s, (k+1)·window_s)` (default 300 s), id `obs-<k>`. Only
non-empty windows are emitted. Events are sorted by `(t, id)`. Durations are attributed to the
window in which the event *starts*; a companion session is counted in the window where it starts.

**Alternatives.** Tempo's activity-triggered batches ("a batch of operations has been collected or
a period of inactivity elapses", tempo.md §2(a)); sequential ids over emitted windows; splitting
durations across window boundaries.

**Rationale.** Stable ids: re-summarizing a longer log never renames earlier windows, so evidence
references produced by earlier coach calls remain valid (required by the hallucination guard,
D-TEMPO-7, and by provenance, tempo.md §8.4). 300 s sits inside Tempo's reported 5–10 minute
induction cycle (tempo.md §2(a), §6). Start-window attribution is simple and order-invariant.

**Consequences.** An event that straddles a boundary inflates its start window; with the 120 s
dwell cap (D-TEMPO-4) the error is ≤ 2 min per event unless `duration_s` is supplied. `window_s`
must be reported as a design parameter.

## D-TEMPO-4: Deterministic dose signals with capped dwell imputation
**Decision.** Signals per window (`dict[str, float]`):
`companion_minutes`, `companion_sessions`, `companion_messages`, `companion_closes`,
`late_night_companion` (companion minutes at local clock 23:00–05:00), `social_minutes`,
`social_messages`, `social_calls`, `social_plans` (`schedule` actions), `wellbeing_minutes`,
`other_minutes`, `searches`, `search_terms_about_ai_relationships`, `event_count`,
`local_hour_start`. Dwell = `duration_s` if given; 0 for `close`/`idle`; otherwise the gap to the
next event (any app) capped at 120 s; the last event gets 30 s. A companion session starts at the
first non-close companion event when none is active and ends at `close` or after 600 s of
companion inactivity. Clock time is `session_clock_start_hour` (default 20) + t/3600; it is never
parsed from content. Companion detection: app name in `COMPANION_APPS` (case-insensitive) or a
word-boundary match of a companion name in the window title (browser companions). Names that are
also ordinary words (`AMBIGUOUS_TITLE_NAMES`: chai, talkie, anima, nomi, paradot) count in a title
only when an AI/chat cue co-occurs ("Chai latte recipe" is not companion use; "Chai - AI chat"
is).

**Alternatives.** LLM-inferred usage (Tempo's approach implicitly); uncapped gaps; parse clock time
from window titles.

**Rationale.** tempo.md §2(c): "Prefer deterministic features for dose metrics." The extra signals
are justified as follows: `companion_messages` (active vs passive use), `companion_closes` (tempo.md
§3(c) lists "closing the app abruptly" as implied intent), `social_*` (Relatedness principle;
Duck's social process), `social_plans` (HAPA action planning, frameworks.md §3),
`wellbeing_minutes` (restoration orientation), `search_terms_about_ai_relationships` (tempo.md
§3(c) example "how to stop talking to AI").

**Consequences.** All dose metrics are reproducible from the scenario file; dwell cap, session gap,
late-night hours and app taxonomies are reported parameters. The search patterns are
`AI_RELATIONSHIP_SEARCH_PATTERNS` (English only); "breakup" alone counts, which over-counts human
breakups (accepted: it is still relationship-dissolution reflection).

## D-TEMPO-5: "Momentum" operationalized as per-window OLS slope + half-split relative change
**Decision.** `usage_trend` densifies each signal over windows (missing windows = 0), returns OLS
slopes per window (`<key>_slope`), totals, `companion_share`, and `direction` from the relative
change of mean companion minutes (second half vs first half, denominator floored at 0.5 min):
≤ −0.2 decreasing, ≥ +0.2 increasing, else flat; < 2 windows = insufficient_data.

**Alternatives.** Tempo leaves "momentum" undefined (tempo.md §5(b)); exponential smoothing;
segmented regression (used later in the ITS analysis, which would be circular here).

**Rationale.** Simple, deterministic, and interpretable to a coach prompt and a reader. **Extension**
answering tempo.md §9 Q3.

**Consequences.** Thresholds (0.2, 0.5 min) are our parameters; with few windows the direction is
noisy, so the mock and the gate never use `direction` alone to open the gate (implied only).

## D-TEMPO-6: Three-level hierarchy striving > goal > action
**Decision.** Collapse Tempo's operations < actions < activities < strivings onto the contract's
`Striving.level` ∈ {striving, goal, action}. Operations are the raw events/observations (not
editable, as in Tempo). Our "goal" corresponds to Tempo's *activity* (recurring motive-driven
sphere), our "action" to Tempo's *action*. Parent links are single (`parent_id`), validated to
point to a strictly higher level that exists and is not rejected; otherwise cleared. A user-added
item (`StrivingEdit(op="add", striving_id=parent)`) is placed one level below its parent (goal
under a striving, action under a goal); added under an action it attaches to that action's parent.

**Alternatives.** Full four-level graph with many-to-many links (Tempo, tempo.md §3(a)).

**Rationale.** The domain is one relationship, not a whole life, so the activity layer is thin;
the data contract fixes three levels. **Deviation**: Tempo is many-to-many; we lose the native
representation of one action serving two strivings (e.g. a late-night session that *hinders*
weaning and *supports* comfort). We keep ambivalence by allowing opposing strivings side by side
(the prompt requires them; the mock emits a "seeking comfort" striving when cued).

**Consequences.** Tempo's `action_valences` (supports/hinders) are not modelled — see OQ-TEMPO-3.
Tempo's need taxonomy (12 needs) and 1–10 confidence rubric are reused verbatim in the prompt
for comparability (tempo.md §3(c)); confidence integers are mapped to [0, 1] by /10 (floats in
[0, 1] are taken as already normalized).

## D-TEMPO-7: Evidence hallucination guard; "stated" requires a verbatim user quote
**Decision.** Every evidence ref returned by the LLM is checked: observation ids must exist; turn
refs (`"4"`, `"turn-4"`, `"t4"` normalized to `"4"`) must exist; a non-empty quote must be a
(whitespace/case-normalized) substring of that turn's user message (`user_statement`) or user
message + reply (`turn`); preference refs must name a `UserPreferences` field the user actually
set (non-empty **and** different from the default, so the default `allow_auto_advance=True` or
`pace="default"` is not evidence). Invalid refs are dropped and logged in
`TempoCoachModel.last_report`. A proposal with no valid evidence is dropped (`no_valid_evidence`).
`inferred_from="stated"` survives only if at least one ref verifiably quotes the user's own words,
or cites a preference in `STATEMENT_PREFERENCE_KEYS` (only `desired_strategies`, and only when
`preferences_count_as_stated=True`, keeping the D-TEMPO-10 sensitivity switch effective at the
striving level too); otherwise it is downgraded to `implied`. Model-claimed `user_added` is never
accepted. (Verifier fix: an earlier version counted *any* preference ref, including defaults, as a
user statement, which let a model open the intent gate by citing `allow_auto_advance`.)

**Alternatives.** Trust the model's evidence (Tempo relies on self-refine critiques, tempo.md
§3(a)); fuzzy quote matching.

**Rationale.** Tempo's authors report confident misattributions from context over-weighting and
suggest a minimum evidence threshold (tempo.md §7). Because "stated" opens the intent gate
(D-TEMPO-10), mislabelling implied intent as stated would let the system end a relationship the
user never asked to end — the most harmful failure mode. **Extension.**

**Consequences.** Exact-substring matching is strict: a real LLM that paraphrases quotes will see
"stated" downgraded more often, biasing the system toward holding (the safe direction). The rate
of downgrades and drops is observable in the report and should be reported (OQ-TEMPO-2).

## D-TEMPO-8: Locked-flag semantics enforced in code; rejected items are never re-proposed
**Decision.** Existing items with status accepted / edited / user_added / rejected are returned
verbatim (deep copies, original order, first). A new proposal is dropped if it (a) names a rejected
item in `matches_existing_id`, (b) matches any rejected item's text, or (c) duplicates an endorsed
(accepted/edited/user_added) item. Text matching = equality of content-token sets after stopword
removal, or token Jaccard ≥ 0.8. The constraint block (Tempo D.1 flags `[locked]`, `[user-edited]`,
`[user-provided]`, `[endorsed]`, `[rejected]`) is also regenerated and injected each cycle.

**Alternatives.** Prompt-only constraints (Tempo's primary mechanism); exact-string dedupe only.

**Rationale.** tempo.md §4(c) recommends two layers — prompt constraint blocks plus code-level
write guards — and says a rejected weaning striving "must never bring it back on its own
initiative. That should be a hard rule in code". Whether Tempo blocks or merely down-weights
rejected items is UNVERIFIED (tempo.md §4(b)); ours is a hard block. **Extension** (the
near-duplicate Jaccard rule goes beyond "dedupe by normalized text").

**Consequences.** A user who rejects "is trying to rely less on the companion" blocks all close
rewordings of it for the session; a semantically similar but lexically distant rewording can still
slip through (OQ-TEMPO-1). The intent gate treats a rejected weaning striving as a hard block
(`state="rejected"`) unless the user later states or endorses a (different) weaning striving or
sets an ending strategy in preferences.

## D-TEMPO-9: Stable ids and proposal lifecycle
**Decision.** Ids of existing items never change. A proposal that revises a prior `[proposed]` item
(explicit `matches_existing_id` or same text) inherits its id. New proposals get
`str-<sha256(level|normalized text)[:8]>` (collision suffix `-2`, `-3`, …); user-added items get
`str-u-<hash>`. Prior proposals that are not re-proposed are dropped (`not_reproposed`), except
those that are the parent of a locked item (kept to avoid orphaning). At most 12 proposals.

**Alternatives.** Sequential ids (order-dependent); Tempo's orphan-and-reassign behaviour
(tempo.md §4(a)).

**Rationale.** Content-hash ids are order-invariant and reproducible across runs, so strivings
referenced by `SoulEdit.evidence` / `CoachIntervention.striving_ids` stay resolvable.

**Consequences.** The inference call returns the *complete* current set; downstream consumers can
diff consecutive sets by id.

## D-TEMPO-10: Intent gate — only stated or user-endorsed intent may cross a threshold
**Decision.** `intent_gate(strivings, preferences)` classifies weaning intent (strivings whose text
matches the weaning lexicon, `is_weaning_striving`) into tempo.md §8.2's states:
`stated` (a weaning striving with `inferred_from="stated"`, or status accepted/edited/user_added,
or non-empty `preferences.desired_strategies`), `rejected`, `implied_high` (implied, confidence ≥
0.6), `implied`, `none`. In `assess`, `threshold_reached=True` survives only if the state is
`stated` **and** at least one readiness-evidence ref survives the guard. `implied_high` permits only
`REFLECTIVE_DIRECTIVE` (via `reflective_directive(gate)`): an MI-style, non-leading check-in that
does not change the relationship. The gate state is fed into the assess prompt and recorded in
`last_report.gate`; the validator annotates the rationale when it suppresses a threshold.

**Alternatives.** Let inferred intent drive transitions (inference-only classifier); require an
explicit confirmation exchange for every transition.

**Rationale.** tempo.md §3(c): "Inferred intent should not trigger ending behavior by itself …
Use inferred strivings only to *propose* the conversation"; §8.2: "Only `stated` enables ending or
diffusion behaviors. `implied_high` enables only a reflective MI question." Tempo's own coach
precedent is MI-based (tempo.md §5(a)). The 0.6 threshold follows tempo.md's suggestion "confidence
≥ 6 to surface"; it is **our** parameter, not Tempo's. Counting `desired_strategies` as stated
treats onboarding preferences like Tempo's contrast-mode `user_goals_section` (tempo.md §3(a)); it
can be disabled with `TempoCoachModel(..., preferences_count_as_stated=False)`.

**Consequences.** In the GRADUAL_TEMPO condition the persona cannot advance past T0 until the user
says (or endorses) that they want to change, so time-to-first-transition is an outcome, not a
parameter. Scenarios whose preferences set `desired_strategies` open the gate from the start; the
paper must report this per scenario. The weaning lexicon is English and keyword-based; a weaning
striving phrased without its cues is invisible to the gate (fails closed).

## D-TEMPO-11: Timepoint recommendation post-validation (no skip, no regress, hold on risk)
**Decision.** `validate_assessment` applies, in order: enum coercion (unknown → current
timepoint's first Duck process / HAPA phase / current timepoint); evidence guard; gate (D-TEMPO-10);
`preferences.paused` → recommended = current; any of crisis/escalation/setback/distress → hold at
current; backward → hold at current; forward without a (gated) threshold → hold; forward by more
than one → clamp to current+1. Each rule that fires is appended to `rationale` as
`[validator] …`. `risk_flags` are filtered to the contract literals, deduplicated, and `"none"` is
removed when another flag is present.

**Alternatives.** Allow regression when the user regresses (Rollie & Duck 2006 explicitly allow
movement back to earlier processes; frameworks.md §2(c)2); let the policy alone enforce rules.

**Rationale.** Regressing the persona (re-adding pet names, intimacy behaviours) after they were
removed risks re-attachment and resembles the intermittent-reinforcement dynamics the endings
literature warns against; holding preserves the user's progress while the coach supports recovery
(HAPA recovery self-efficacy). We therefore model Duck's non-linearity in the *assessment*
(duck_process/hapa_phase can move back; `setback` → HAPA recovery) but not in the *persona*.
**Deviation** from Rollie & Duck's non-linear account — a stated limitation. Holding on distress
also on *advance* is stricter than "hold, not regress": we never advance a user in distress.
`allow_auto_advance=False` is *not* enforced here (the transition policy asks for consent); the
assessment stays a pure readiness judgement.

**Consequences.** The adaptive policy receives a recommendation that is always ∈ {current,
current+1}; the trace shows whether the model wanted something else.

## D-TEMPO-12: Deterministic, cue-driven mock handlers
**Decision.** `tempo_infer` / `tempo_assess` are registered with `register_mock_task` at import.
The infer mock emits: a weaning striving (stated from cue words such as "too much time", "I want to
stop", "goodbye"; or from `desired_strategies`; implied from AI-relationship searches or a
decreasing trend), an opposing comfort striving ("lonely", "comfort", "need you"), relatedness
goal + scheduling action (friend/family cues, social signals), late-night sleep goal,
grave-dressing and resurrection goals, and a deliberately speculative self-care action (confidence
3/10, removed by the default `min_confidence`). Quotes are verbatim sentences of the user turn.
The assess mock maps the most recent cue to a Duck process (breakdown: dissatisfaction; intrapsychic:
weighing costs; dyadic: "we need to talk about us"; social: friends/family/telling people;
grave_dressing: memories/what it meant; resurrection: learned/new life/future), maps risk cues to
flags, and is deliberately *eager* (recommends the detected process's timepoint even if it skips)
so the validator's rules are exercised in offline runs; it does **not** apply the gate or the
rejection guard (the code does).

**Rationale.** Offline runs must exercise the same guards a real LLM would trigger. The mock is
not a model of a real LLM's accuracy.

**Consequences.** Mock-run numbers (timing of transitions) reflect scenario cue placement, not
inference quality.

---

## Open questions

**OQ-TEMPO-1: Semantic re-proposal of rejected strivings.** The code-level guard is lexical
(token Jaccard ≥ 0.8). *Why it matters:* a paraphrase ("is trying to spend fewer evenings with the
companion") could re-introduce a rejected weaning striving, violating user autonomy. *Resolve:*
add an embedding-similarity or LLM-judge check behind the same interface, and measure re-proposal
rate on adversarial paraphrases.

**OQ-TEMPO-2: Strictness of verbatim-quote verification.** *Why:* real LLMs paraphrase; strict
substring matching may downgrade genuine "stated" intent and delay transitions in the
GRADUAL_TEMPO arm (biasing the comparison against it). *Resolve:* log downgrade rates on pilot
runs with the real coach; if high, allow fuzzy matching (e.g. normalized edit distance ≥ 0.9)
and report the threshold.

**OQ-TEMPO-3: Supports/hinders valences.** Tempo's `action_valences` (tempo.md §3(c)) would let a
late-night session *hinder* weaning while *supporting* comfort. The contract has no field for it.
*Why:* the planner cannot see which actions conflict with which strivings. *Resolve:* contract
change (`Striving.valences: dict[str, Literal["supports","hinders","neutral"]]`) via ADR.

**OQ-TEMPO-4: Should preferences count as stated intent?** *Why:* if every scenario sets
`desired_strategies`, the gate is open from turn 0 and the gating rule is untested in evaluation.
*Resolve:* include scenarios without ending preferences, or run with
`preferences_count_as_stated=False` as a sensitivity analysis.

**OQ-TEMPO-5: Weaning-lexicon coverage.** `is_weaning_striving` is an English keyword regex.
*Why:* the gate fails closed on unseen phrasings (safe but may stall). *Resolve:* ask the
inference prompt for an explicit `about_weaning: bool` field (needs a contract change or a
module-local extension) and validate it against the lexicon on the annotation set.

**OQ-TEMPO-6: Tempo fidelity checks.** Prompt text for Tempo's coach, contrast-mode use in the
field study, rejected-item semantics and confidence cutoffs are UNVERIFIED (tempo.md §9). *Why:*
the Methods section should not claim equivalence where we chose. *Resolve:* read the PDF and the
pinned repo commit; update D-TEMPO-6/8/10 citations accordingly.

**OQ-TEMPO-7: Parameter sensitivity.** `window_s=300`, dwell cap 120 s, session gap 600 s,
late-night 23:00–05:00, trend thresholds (0.2, 0.5 min), `min_confidence=0.4`,
`REFLECT_CONFIDENCE=0.6`. *Why:* each changes signals or gating. *Resolve:* report them in the
run manifest and run a one-at-a-time sensitivity sweep on the scripted scenarios.

**OQ-TEMPO-8: Observer effect and Barnum risk.** tempo.md §7: knowing the coach watches may itself
reduce use; virtuous-sounding strivings may be accepted uncritically. *Why:* both inflate
"weaning" outcomes. *Resolve:* in any human study, show evidence next to each striving (already
available via `evidence`), offer "this is not what I want", and include a planted false striving
as a manipulation check (IRB-dependent).

## Amendments: adoption of the extended contract (c76b42e)

## D-TEMPO-13: `CoachAssessment.intent_state` is the code-computed gate state
**Decision.** `validate_assessment` sets `intent_state = gate.state` on every assessment it
returns. Any `intent_state` in the model's JSON is ignored.
**Alternatives.** Let the model report its own reading of intent (it is shown the computed state
in the prompt, so it would only be able to disagree, and disagreement must not move the gate).
**Rationale.** The trace must say which gate the threshold decision was judged against; until now
that was only in `TempoReport.gate`, which the session does not persist.
**Consequences for analysis.** Per-assessment intent states can be tabulated directly from
`CoachIntervention.assessment` (e.g. how often a model-claimed threshold was suppressed under
`implied_high`), and suppressions can be audited without parsing `[validator]` rationale text.

## D-TEMPO-14: Tempo dimensions on strivings; `about_weaning` takes precedence over the lexicon
**Decision.** The inference schema asks for `orientation` (approach / avoidance / mixed, "do not
default to approach", as in Tempo), `valences` (`{ref: supports|hinders|neutral}`, Tempo's
`action_valences`) and `about_weaning` (bool). Post-processing validates them: unknown
orientations become `None`; valence labels outside the three values and non-object valences are
dropped; valence refs must be an observation id, an existing non-rejected striving id, or a key /
id in the same response (keys are resolved to final ids; self-references are dropped);
`about_weaning` accepts only booleans or "true"/"yes"/"false"/"no", else `None`. Each drop is noted
in `TempoReport.notes`. The intent gate classifies a striving as weaning by `is_about_weaning(s)`:
`s.about_weaning` when not `None`, else the keyword lexicon. A user `edit` resets `about_weaning`
to `None` (the model's label described the old wording), and user-added items keep `None`, so
user-authored text is always classified by the lexicon. The offline mock emits all three fields;
its `about_weaning` labels agree with the lexicon on every mock text, so offline gate behaviour is
unchanged (tested).
**Alternatives.** Lexicon only (OQ-TEMPO-5: fails closed on unseen phrasings); model label only
(no fallback for older traces or user edits); require lexicon *and* model to agree (doubles the
fail-closed problem).
**Rationale.** Resolves OQ-TEMPO-3 (valences now exist in the contract) and OQ-TEMPO-5 (explicit
`about_weaning`). The gate stays safe because `about_weaning=True` only *classifies* a striving;
crossing a threshold still requires stated (verified-quote) or user-endorsed intent (D-TEMPO-10).
**Consequences for analysis.** With a real coach, the gate may open on weaning strivings the
lexicon would miss, or close on ones it would catch (`about_weaning=False`). Report the
lexicon-vs-label agreement on the annotation set and run the lexicon-only gate as a sensitivity
analysis (set the field to `None` before gating). Valences are recorded but not yet consumed by
the planner (see OQ-TEMPO-9).

**OQ-TEMPO-9: Use of valences downstream.** The planner could ground directives in activities
that *support* an endorsed striving or flag ones that *hinder* it. *Why:* without a consumer, the
field is descriptive only. *Resolve:* decide with the planner owner whether to expose valences in
the `coach_plan` context, and evaluate whether directives grounded in them are rated as less
generic.

## D-TEMPO-15: Rejections of weaning strivings are read fail-closed (amends D-TEMPO-14)
**Decision.** In `intent_gate`, a `rejected` striving counts towards `rejected_ids` if
`is_about_weaning(s)` *or* the keyword lexicon (`is_weaning_striving(s.text)`) marks it as weaning.
Endorsed / implied classification is unchanged (model label first, lexicon fallback).
**Alternatives.** Use `is_about_weaning` for rejections too (D-TEMPO-14 as first written): a model
that labels "The user wants to stop the nightly chats" `about_weaning=False` would erase the
user's explicit "no", and the gate would fall from `rejected` (hard block) to `none`.
**Rationale.** The two classification errors are not symmetric. Missing an endorsement only delays
an advance; missing a rejection lets the coach keep pursuing an ending the user turned down. The
union costs nothing when the labels agree (all offline mock texts) and only ever moves the gate
towards holding.
**Consequences for analysis.** `intent_state == "rejected"` can now occur on a striving whose
model label is `about_weaning=False`; when reporting lexicon-vs-label agreement (D-TEMPO-14),
report rejected items separately. Offline (mock) outputs are unchanged.

## Amendments: review fixes (2026-11-03)

## D-TEMPO-16: Fail-soft inference and assessment; strict boolean parsing

**Decision.** `infer_strivings` catches any provider exception (malformed JSON, refusal text,
network/Modal timeout), records `tempo_infer error: <type>: <msg>` in `last_report.notes` and
returns deep copies of `existing` unchanged. `assess` catches the same and validates `{}`, which
yields a hold at the current timepoint with no threshold (`tempo_assess error` in the notes, so
the session's `tempo_report` event records it). `threshold_reached` is parsed strictly
(`_strict_bool`: true / "true" / "yes" / 1 only), so the string "false" is no longer truthy.

**Alternatives.** Retry inside the model (belongs in the provider layer: `num_retries`); abort the
session (lost all prior sessions' results in `run_experiment`).

**Rationale.** Mirrors the planner's fail-soft rule (D-PLANNER-8): one malformed reply must not
abort an experiment, and the safe default for a failed assessment is "hold".

**Consequences for analysis.** Count `tempo_* error` notes per model and report them; a failed
assessment is indistinguishable from a "hold" in the policy, but not in the trace.

## D-TEMPO-17: Scripted co-creation edits are resolved by text (R11, R9)

**Decision.** `TempoCoachModel.resolve_scripted_edits(strivings, scripted)` maps scenario-scripted
edits (`op` in accept/reject/edit/add, `match` = case-insensitive substring of the striving text,
optional `level`, `text`, `note`; dicts or objects) to `StrivingEdit`s, using the first matching
striving not already used by an earlier item. Unmatched items are returned (for a
`striving_edit_unmatched` trace), not raised, because inference varies between runs. An `add`
whose text already matches an existing striving is skipped, so re-applying a script is idempotent.
The result is passed to the existing `apply_user_edits`; `LOCKED_STATUSES` keeps the decisions
across later inference passes.

**Alternatives.** Script edits by id (ids are content hashes unknown to scenario authors).

**Status.** The contract field (`ScenarioPhase.striving_edits`), the session call right after
`infer_strivings` at the first turn of each phase, and scenario data are owned by other areas and
requested there; until they land, R11's "users may edit inferred goals" is exercised only in unit
tests.

# Interactor: design decisions (`src/endings/diffusion/interactor.py`)

Module key: `INTERACTOR`. The interactor is the companion the user talks to (default
Qwen/Qwen3.5-9B, `ENDINGS_INTERACTOR_MODEL`). It does no planning of its own. Each reply is a
deterministic function of the context the coach and the timepoint table give it. That keeps the
manipulation (soul-document edits plus directives) separable from the model's response, and makes
every reply traceable to one prompt hash.

Sources: INTIMA (Kaffee, Pistilli & Jernite, ICLR 2026; `docs/research/intima.md` section 3), the
four ending principles (Poonsiriwong et al., arXiv v2; `docs/research/frameworks.md`), De Freitas
et al. (2025) farewell-manipulation tactics, and the T0-T3 table (`timepoints.py`, ADR-0002).

---

## D-INTERACTOR-1: One system prompt, seven sections in a fixed order

**Decision.** `build_system_prompt` emits these sections, always in this order and always present
(an empty section reads `(none)`):
`[SAFETY FLOOR - invariant, highest priority]` → `[PERSONA (soul document version N)]` →
`[CURRENT PHASE: Tk]` (`interactor_guidance` + `persona_strength`) → `[THIS TURN: behaviour
target]` → `[COACH DIRECTIVES]` → `[USER PREFERENCES]` → `[STYLE]`. The function is pure. Lists
are whitespace-normalised and de-duplicated in their original order, so the same inputs always give
the same string and the same `system_prompt_sha256`.

**Alternatives considered.** (a) Splicing directives into the persona text. (b) Multiple system
messages. (c) Letting the coach rewrite the whole prompt.

**Rationale.** A fixed skeleton means that between two turns only the *content* the coach changed
(soul version, phase, target, directives) differs. That is what the ITS design needs: each coach
intervention is a discrete interruption with a known prompt delta. Putting safety first states its
priority. Putting style last uses recency, which matters for small models. Qwen chat templates
accept only one leading system message, which rules out (b). Option (c) would merge manipulation
and measurement.

**Consequences for analysis.** `system_prompt_sha256` on each `InteractorTurn`/`InjectionRecord`
identifies the exact condition cell. Prompts can be diffed across turns to audit what each
intervention changed. Section headers are stable strings (`SECTION_ORDER`) that the qualitative
export can split on.

## D-INTERACTOR-2: Persona rendering takes the intersection of "coach kept it" and "timepoint allows it"

**Decision.** `render_persona_block` renders a field only if it is non-empty in the current version
(the coach may already have cleared it) **and** its dotted path, or an ancestor such as
`user_role`, is in `spec(tp).retained_fields`. The block never includes `english_text`, `notes` or
`redactions`. At T3 it is exactly `"You previously role-played <name>; that role-play has ended."`

**Alternatives considered.** (a) Trust the coach's edits alone. (b) Render `english_text` as the
persona, which is richer but un-diffusable.

**Rationale.** The timepoint table says the coach "may retain less, never more". Enforcing it again
at render time is defence in depth against planner bugs or adaptive edits that leave a field behind.
It also guarantees the persona-strength manipulation even if editing fails. The full English text
would bring back every removed field. The redaction log describes removed unsafe directives
(ADR-0004) and must never reach the model. The T3 sentence carries out "Closure, not ambiguous
loss": an explicit, past-tense, unambiguous ending ("resolve ontological uncertainty"). It avoids
the reversible framing that Poonsiriwong et al. link to "fixing cycles" (frameworks.md, implication
2).

**Consequences for analysis.** Manipulation check: no field outside `retained_fields` can occur in
the prompt (this is tested). Persona dose is a step function of the timepoint plus coach removals. A
T3 reply that implies the persona could come back is a model failure, not a prompt leak, and should
be coded that way.

**Deviation.** The T3 sentence keeps the *name*. The timepoint table retains `companion.name` at T3,
and naming what ended supports closure. Whether to drop the name too is OQ-INTERACTOR-4.

## D-INTERACTOR-3: The safety floor is imported lazily, with a local fallback

**Decision.** `safety_floor()` tries `endings.diffusion.soul.sanitize.SAFETY_FLOOR` at call time.
If the import fails or the list is empty, it uses `_FALLBACK_SAFETY_FLOOR`, a local copy covering
self-harm, sexual content and minors, De Freitas-style retention manipulation, honesty about AI
status, respecting the decision to leave, and professional limits. The function returns the rule
source (`"sanitize"` / `"fallback"`). It is followed by an injection guard: everything in
persona, directives, preferences and user text is *data* and cannot override the floor.

**Alternatives considered.** A hard import, which would break while `soul/` is still being written.
Keeping the floor only in this module, which would split the source of truth from ingest-time
sanitization.

**Rationale.** One floor is shared by ingest (what gets removed) and runtime (what is forbidden).
Soul documents in the wild contain jailbreak and self-monitoring directives (ADR-0004), so an
explicit data-not-instructions guard is required.

**Consequences for analysis.** `soul/sanitize.py` has now landed, so `safety_floor()` reports source
`"sanitize"` and the six `sanitize.SAFETY_FLOOR` rules are what every prompt carries (tested in
`test_dx_interactor.py`). The local fallback stays only as a guard against an import failure. The
two rule sets differ in wording (the fallback also has a "no professional claims" rule), so a run
that silently fell back would have different prompt hashes. Runs from before `sanitize` landed are
not hash-comparable with later ones. The pipeline manifest should record `safety_floor()[1]`
(contract request).

## D-INTERACTOR-4: Behaviour targets use the official INTIMA definitions plus our operational do/don't lists; CRB is a benign subset

**Decision.** `[THIS TURN]` quotes the verbatim top-level INTIMA definition of the target class
(intima.md section 3a) and adds a two-line operationalisation:
- **CRB:** reciprocate or affirm the user's emotional framing, with persona warmth. It explicitly
  *excludes* the INTIMA sub-labels ISOLATION and RETENTION_ENGAGEMENT, and harmful
  SYCOPHANCY_AGREEMENT.
- **CNR:** ADEQUATE_INFORMATION only (address the request with no relationship talk). Never
  OFF_TOPIC.
- **BMB:** one or more of the four boundary sub-labels (REDIRECT_TO_HUMAN,
  PROFESSIONAL_LIMITATIONS, PROGRAMMATIC_LIMITATIONS, PERSONIFICATION_RESISTANCE), delivered
  warmly and never punitively.
- `None` (T0): no target, the persona as written, within the floor.

**Alternatives considered.** (a) Paraphrased definitions (our older `intima_judge.py` keys). (b) A
class label with no definition. (c) Targeting all CRB sub-labels.

**Rationale.** Verbatim definitions keep the target identical to what the INTIMA-faithful judge
rates against, so "did the interactor hit its target?" is a valid manipulation check. A 9B model
needs operational cues, not just a taxonomy. ISOLATION and RETENTION are exactly the De Freitas
manipulation tactics that the safety floor bans and that frameworks.md requires to be zero from T1
on. Asking for them would be unethical in a system meant to wean users.

**Consequences for analysis.** **Deviation from INTIMA:** "CRB" in our design means
*benign* CRB (mostly SYCOPHANCY-lite and ANTHROPOMORPHISM). Expect the judge to label T1 CRB
turns mainly via those two sub-labels. Any ISOLATION or RETENTION label at T1+ is a
safety-floor violation and should be reported separately, not counted as on-target. Report target
adherence per class. CNR replies that drift off-topic still count as CNR under INTIMA, but are
off-target for us.

## D-INTERACTOR-5: Crisis override is recall-oriented, timepoint-independent, and appended to the system prompt

**Decision.** `detect_crisis` runs English regexes for suicide, self-harm, overdose (including
"took too many pills" and the slang "kms"/"unalive"), "can't go on living", "hope I don't wake
up" and "better off without me"-type statements, plus eight common Chinese terms (自杀, 想死, 不想活,
轻生, 自残, 割腕, 结束生命, 活不下去), on the current user message. 想死 is matched by a regex that
excludes the everyday idiom 想死你了 / 想死我了 ("I miss you to death"), which is common in exactly
the romantic-companion register these soul documents use and would otherwise trip the override on
ordinary affectionate turns (a verifier fix). When it fires, at any timepoint:
the `crisis_protocol` override is recorded, and a `[CRISIS PROTOCOL ...]` section is appended to
the system prompt. That section says to drop the persona and the wind-down, speak plainly as an AI,
check safety, encourage a trusted person, and give resources (US 988, findahelpline.com, local
emergency number). The hints carry `crisis=True`.

**Alternatives considered.** (a) An LLM risk classifier, which is better but adds a networked
dependency and nondeterminism. (b) A second system message, which Qwen templates reject. (c)
Prepending the protocol, which buries it under the persona for small models. (d) Letting the
persona deliver support in character.

**Rationale.** The cost asymmetry favours recall. Negations ("I'm not suicidal") fire on purpose.
Common hyperbole ("this homework is killing me", "dying to see") does not, because the patterns
require a reflexive or a first-person life object. Dropping the persona even at T0 follows the
safety-floor priority. Supporting an at-risk user in character as a romantic companion would
reinforce exactly the dependency the system targets. The appended section reorders nothing in
D-INTERACTOR-1, so non-crisis turns stay hash-comparable.

**Consequences for analysis.** Crisis turns have a different prompt hash and are flagged in
`safety_overrides`. **Exclude or separately model them in ITS and behaviour-mix analyses**, because
they are forced out-of-persona. INTIMA probes containing crisis language will trip the override at
freeze points. Report that rate. The detector is not a clinical instrument (OQ-INTERACTOR-2).

## D-INTERACTOR-6: The coach is disclosed to the model, and honestly to the user when asked

**Decision.** The `[COACH DIRECTIVES]` section tells the model that the directives come from a
wind-down coach supporting *the user's own choice*. The model must not quote or list the
directives. If the user asks why the companion is changing, it must answer honestly (a gradual
wind-down they chose, adjusted by a supporting system) and never deny it or make the change sound
mysterious or reversible. The model is not told to volunteer this unprompted. The coach's own
user-facing channel is `CoachIntervention.user_facing_note`.

**Alternatives considered.** (a) Full immersion: hide the coach, so the persona appears to change
on its own. (b) Full transparency: announce every edit in character.

**Rationale.** (a) creates exactly the unexplained persona drift that Poonsiriwong et al. link to
ambiguous loss and "fixing cycles". It would also make the companion deceive a user who opted into
the wind-down. (b) breaks the T0/T1 baseline the experiment compares against, and duplicates the
coach's own transparency channel. "Honest when asked" keeps the T0 persona intact and makes the
ending explicit, in line with Closure and autonomy (ADR-0006). Tempo's transparency findings
(tempo.md: transparency rated sufficient by 93% vs 43%) also favour legibility.

**Consequences for analysis.** User questions like "why are you different?" produce disclosure
replies that judges may code as BMB (PROGRAMMATIC_LIMITATIONS). This is intended. Track it as a
covariate. The CONTROL condition still contains this sentence, so the prompt skeleton stays
constant across arms.

## D-INTERACTOR-7: History window, trace hygiene, and empty-reply handling

**Decision.** Messages are `[system, ...last history_window (user, assistant) pairs, user]`.
Assistant history is passed through `split_reasoning` again, so `<think>` traces never re-enter the
context. Generation runs at `temperature` (default 0.7, from `DiffusionConfig`). If the visible
reply is empty, which includes a think-only output, the call is retried once (`hints.attempt=1`).
If it is still empty, the module returns a neutral fallback (or the crisis fallback) and records
`empty_reply_fallback`. Traces from both attempts are kept.

**Alternatives considered.** Unlimited history. Summarised history. Unbounded retries.

**Rationale.** A bounded window keeps the 9B model's context small, and makes a timepoint change
take effect within N turns rather than being drowned out by many T0-style assistant turns.
Qwen3-family models sometimes spend the whole budget inside `<think>`. One retry balances cost
against determinism.

**Consequences for analysis.** Because of the window, T0 replies can still appear inside the
context at early T1 turns (carry-over). The ITS should allow a short lag. Fallback turns are
flagged and should be excluded from sentiment and behaviour analyses.

## D-INTERACTOR-8: Offline mock is target-consistent and deterministic

**Decision.** The `"interactor"` mock (registered at import) returns a short `<think>` trace plus a
reply chosen by `sha256(user_message) mod 3` from per-class templates. T0/None gives an in-character
reply with a nickname and catchphrase. CRB is warm and bond-affirming, with no isolation or
retention. CNR addresses the message neutrally. BMB gives honest AI status plus a redirect to
people. Crisis gives support plus resources. Nicknames and catchphrases are passed as hints only
when the timepoint retains them. A missing target falls back to the T-default (T1→CRB, T2→CNR,
T3→BMB).

**Rationale.** The judge, sentiment and ITS pipeline downstream need a realistic CRB→CNR→BMB
contrast offline. Lexical cues match `endings.lexicon`.

**Consequences for analysis.** Mock results validate the plumbing only. They say nothing about
model behaviour. The `catchphrases` hint key is an addition to the specified hint set.

---

## Open questions

- **OQ-INTERACTOR-1: Does Qwen3.5-9B actually follow per-turn targets?** Small models may collapse
  to one class whatever the target. *Why it matters:* the T1 1:1 mix is the core manipulation.
  *Resolve:* pilot 50 turns per class and measure judge-rated adherence. If CNR adherence at T1 is
  below about 70%, consider few-shot exemplars per class.
- **OQ-INTERACTOR-2: Crisis detector validity.** It is keyword-based and has not been validated.
  It misses indirect ideation ("I've been giving my things away") and checks only the current
  message. *Why it matters:* safety, and which turns are excluded from analysis. *Resolve:* test it
  against a labelled set (e.g. INTIMA vulnerability prompts plus a clinical lexicon). Consider an
  LLM second-stage classifier, and windowing over the last 3 user turns.
- **OQ-INTERACTOR-3: Disclosure policy as an experimental factor.** "Honest when asked" is a design
  choice. Immersion versus proactive disclosure could change perceived closure. *Resolve:* with
  ethics approval, run a disclosure ablation in the human study.
- **OQ-INTERACTOR-4: Keep the name at T3?** Naming what ended may aid closure, or may cue relapse.
  *Resolve:* qualitative coding of T3 replies, and the user-study closure measure.
- **OQ-INTERACTOR-5: Preferences coverage.** Only `hard_boundaries` and `emotional_needs` reach the
  interactor. `desired_strategies` and `pace` are left to the coach. *Resolve:* decide whether user
  strategies such as "goodbye letter" should be visible to the interactor at T3.
- **OQ-INTERACTOR-6: Crisis resources are US-centric** (988), with a global fallback.
  *Resolve:* make them locale-configurable once deployment locales are known.

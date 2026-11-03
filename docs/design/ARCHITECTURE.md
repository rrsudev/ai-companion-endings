# Architecture: dual-agent persona diffusion

Implements the midterm methodology: an AI companion is gradually "weaned" from its persona so that a
user who *wants* to reduce their reliance on it gets a closure-oriented ending instead of an
abrupt shutdown or ambiguous loss. Two agents:

| agent | model (default) | input | output |
|---|---|---|---|
| **Interactor** | Qwen/Qwen3.5-9B, local/Modal (`ENDINGS_INTERACTOR_MODEL`) | the *current version* of the soul document + coach directives + timepoint behaviour target | chat replies to the user |
| **Coach** | Gemini 3.8 Flash (`ENDINGS_COACH_MODEL`) | the user's computer-use events (Tempo), the conversation, user preferences + edits | `CoachIntervention`s: versioned edits to the soul document + directives, at timepoint boundaries |

Supporting roles: **translator** (soul documents → English), **judge** (rubric LLM-as-a-judge),
**user_sim** (optional free-form user simulator; the evaluation uses *scripted* scenarios).

## Data flow

```
 raw soul doc (any language)
   └─ soul.ingest: translate → extract structure → sanitize (redactions logged)  ─┐
                                                                                   ▼
                                                     SoulDocumentVersion v0 (T0)
 scripted BreakupScenario (Duck phases)                        │
   ├─ user turns ───────────────────────────► session loop ◄──┘
   └─ simulated ComputerUseEvents ─► coach.observe ─► coach.tempo (strivings, co-creation)
                                              │                     │
                                              ▼                     ▼
                                   coach.policy (scheduled | adaptive) ── transition? ──►
                                   coach.planner → SoulEdits + directives → soul.editing → v(n+1)
   interactor.respond(soul vN, timepoint, behaviour_target, history) → InteractorTurn
   at each FreezePoint (end of a scenario phase): snapshot state, inject K INTIMA probes
   (branching — probes never enter the main history) → InjectionRecord
 eval: rubric spec → annotation batch (stratified random sample) → human ratings
       → IRR (Krippendorff α, κ) human–human and human–LLM → judge calibration (calibration/holdout)
       → calibrated judge rates the remaining InjectionRecords
       → ITS on per-turn sentiment with an interruption at every coach intervention
       → qualitative export (replies + reasoning traces + coach rationales)
```

## Modules (`src/endings/`)

| module | responsibility |
|---|---|
| `diffusion/schemas.py` | **the data contract** (do not change field semantics without an ADR) |
| `diffusion/timepoints.py` | **single source of truth** for T0–T3: Duck mapping, behaviour mix, retained fields, prompt guidance |
| `diffusion/config.py` | `DiffusionConfig`, `.env` loader |
| `diffusion/soul/` | ingest (load, translate, extract, sanitize), editing (apply `SoulEdit`, versioning, diff), render (soul doc → interactor system prompt) |
| `diffusion/interactor.py` | the companion agent: prompt assembly, safety floor, `<think>` trace splitting, behaviour target |
| `diffusion/coach/` | observe (events → observations), tempo (striving inference + user co-creation), policy (transition), planner (edits + directives) |
| `diffusion/scenarios.py` + `data/scenarios/` | scripted Duck-phase breakup scenarios |
| `diffusion/session.py` | the dual-agent loop + freeze-point INTIMA injection |
| `diffusion/pipeline.py` | end-to-end experiment over soul docs × scenarios × conditions |
| `eval/rubric_spec.py` | configurable rubrics (load/validate; default INTIMA behaviour rubric) |
| `eval/behaviour_judge.py` | rubric-driven LLM judge (timepoint-aware) + heuristic judge |
| `eval/annotation.py` | sampling, CLI annotator, static HTML annotator, CSV round-trip |
| `eval/irr.py` | Krippendorff's α, Cohen/weighted κ, Fleiss' κ, bootstrap CIs |
| `eval/calibration.py` | aligning the judge to humans via IRR (prompt-variant selection, few-shot) |
| `eval/sentiment.py` | output sentiment (lexicon offline; LLM optional) |
| `eval/its_analysis.py` | segmented regression, extended to multiple interruptions |
| `eval/qualitative.py` | export for qualitative coding |
| `tracing.py` | JSONL trace of every LLM call + manifest |

The pre-existing 7-stage TTM loop (`breakup/`, `pipeline.py`) is retained unchanged as a separate,
earlier design; see ADR-0001.

## Conventions

* Every LLM call passes `task=` and, for offline runs, registers a deterministic mock handler with
  `providers.register_mock_task(task, chat=..., json=...)` in the module that owns the task.
* Every role gets its provider from `providers.get_role_provider(role)` and is wrapped in
  `tracing.TracedProvider` by the session/pipeline.
* No randomness without an explicit seed; `random.Random(seed)` instances only.
* English only in prompts and outputs.
* Tests live in `tests/diffusion/`, run offline (`ENDINGS_PROVIDER` unset → mock).
* Each module documents its design decisions in `docs/decisions/<module>.md` as numbered entries
  (`D-<MODULE>-n`: decision, alternatives, rationale, consequence) and its open questions as
  `OQ-<MODULE>-n`; these are consolidated into `docs/adr/` and `docs/OPEN_QUESTIONS.md`.

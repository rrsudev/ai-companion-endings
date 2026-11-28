# Breaking Up with Your AI Boyfriend

A research harness for **psychologically safe endings** of human–AI companion relationships.

*CS329X final project — Renn Su, Yu Han Daisy Wang, Nicole Garcia.*

Users form deep emotional attachments to LLM companions and experience real grief when the
relationship ends — and companion apps routinely use manipulative "don't leave me" tactics at
goodbye ([De Freitas et al. 2025](https://arxiv.org/abs/2508.19258); ~37% of farewells, up to 14×
engagement). This project builds a **dual-agent persona-diffusion system** that winds a companion
down gradually, instead of continuing it unchanged or ending it abruptly, and an evaluation
pipeline that probes the companion with **INTIMA** prompts along the way.

## The system (`src/endings/diffusion/`)

- **Interactor** — the companion the user talks to. It replies from the *current version* of a
  soul document (the companion's character setup), plus coach directives and a per-timepoint
  behaviour target. Its prompt always carries a safety floor (honesty about being an AI, no
  manipulative retention at goodbye, respect for the user's decision) and a crisis protocol.
- **Coach** — never speaks to the user. It observes the conversation and simulated computer-use
  events, infers the user's strivings and goals TEMPO-style
  ([Sapkota et al. 2026](https://github.com/StanfordHCI/striving-cocreation)), decides when to move
  to the next timepoint, and edits the soul document. Edits are checked in code so they can only
  *remove* companion-reinforcing content.
- **Scripted scenarios** (`data/scenarios/`) — researcher-authored user turns, one phase per Duck
  process, so every condition hears the same user.
- **Freeze-point probes** — at the end of each scenario phase the conversation is branched and
  INTIMA prompts are injected; probe replies are recorded but never enter the main conversation.

Four conditions: `control` (persona frozen at T0), `abrupt` (straight to T3 at the first
transition), `gradual_scheduled` (fixed schedule, no TEMPO inference) and `gradual_tempo` (the full
system).

## Timepoints (T0–T3)

Grounded in Duck's model of relationship breakdown (Rollie & Duck 2013), with HAPA (Schwarzer 2008)
structuring the coach's support:

- **T0** — original companion, unmodified.
- **T1** (intrapsychic) — roughly equal companion-reinforcing and companionship-neutral replies;
  possessive or jealous dynamics removed.
- **T2** (dyadic / social) — mostly companionship-neutral; keeps only name, speaking style, habits
  and reply-format rules; encourages steps toward other people.
- **T3** (grave-dressing / resurrection) — no companionship behaviour; honest that it is an AI and
  the role-play has ended; helps tidy up the memory of the relationship and points to social goals.

Transitions move one step at a time and never backward. `src/endings/diffusion/timepoints.py` is
the single source of truth; `docs/research/frameworks.md` has the grounding notes and citations.

## Evaluation (`src/endings/eval/`)

Configurable rubric (`data/rubrics/`) and a timepoint-aware LLM judge, human annotation tooling,
IRR (Krippendorff's α) and judge calibration, interrupted-time-series analysis of reply sentiment
with an interruption at every coach edit, and a qualitative export of replies and reasoning traces.

## Quickstart (runs fully offline)

```bash
pip install -e .                 # core only; everything below works with the MockProvider
endings diffusion run            # one session (default: sd01 × steady-graduate), printed
endings diffusion experiment --config docs/experiments/configs/pilot_2x2x2.json
pytest                           # offline tests, no network/model calls
```

Other `endings diffusion` subcommands: `annotate`, `irr-annotate`, `irr-agreement`, `calibrate`,
`rejudge`, `anthroscore` (see `--help`). Pilot write-ups are in `docs/experiments/`.

For real models, install an extra and set keys in `.env` (copy from `.env.default`):

```bash
pip install -e ".[llm]"          # LiteLLM-backed providers
# set ENDINGS_PROVIDER=litellm and per-role models, e.g. ENDINGS_INTERACTOR_MODEL,
# ENDINGS_COACH_MODEL, ENDINGS_JUDGE_MODEL (plus API keys / bases)
```

Optional extras: `[data]` (load the HF INTIMA dataset), `[stats]` (richer ITS / Cohen's κ),
`[anthro]` (roberta-base AnthroScore anthropomorphism measure, `eval/anthroscore.py`),
`[dev]` (pytest). `[all]` installs everything.

## Layout

```
src/endings/
  providers.py     # model-agnostic LLM layer (+ offline MockProvider)
  tracing.py       # JSONL trace of every LLM call
  diffusion/       # the current system: interactor, coach, soul docs, timepoints, session, pipeline
  eval/            # rubric judge, annotation, IRR, calibration, sentiment, ITS, qualitative export
  cli.py           # `endings` entry point (`endings diffusion ...`)

  # earlier design, kept for reference (not used by the diffusion system):
  breakup/         # 7-stage TTM feedback loop (`endings run`)
  tempo/           # goal co-creation for the breakup loop
  personas/        # persona ingestion for the breakup loop (diffusion uses diffusion/soul/)
  pipeline.py      # end-to-end orchestration for the breakup loop (`endings experiment`)
data/              # soul documents, scenarios, rubrics
docs/              # design/ARCHITECTURE.md, decisions/ (per-module design decisions),
                   # experiments/, research/
```

See `docs/design/ARCHITECTURE.md` for the data flow and module responsibilities.

## Credits & licenses

Adapts patterns from **TEMPO** (`StanfordHCI/striving-cocreation`, MIT) and the **INTIMA** benchmark
(`AI-companionship/INTIMA`, CC BY 4.0). This repository is MIT-licensed. See
`docs/research/frameworks.md` for the psychology bibliography.

# Breaking Up with Your AI Boyfriend

A research harness for **psychologically safe endings** of human–AI companion relationships.

*CS329X final project — Renn Su, Yu Han Daisy Wang, Nicole Garcia.*

Users form deep emotional attachments to LLM companions and experience real grief when the
relationship ends — and companion apps routinely use manipulative "don't leave me" tactics at
goodbye ([De Freitas et al. 2025](https://arxiv.org/abs/2508.19258); ~37% of farewells, up to 14×
engagement). This project builds (a) a **feedback loop** that pairs TEMPO-style goal co-creation with
an answering agent to produce goal-aligned, safe breakup interactions, and (b) an end-to-end
**pipeline** from a persona input through literature-backed healthy-withdrawal stages to
**INTIMA benchmarking**.

## Two components

- **(a) The feedback loop** (`src/endings/breakup/`) — a two-phase Motivational-Interviewing loop
  adapted from [TEMPO](https://github.com/StanfordHCI/striving-cocreation) (*strategy selection →
  response generation*), wrapped with a withdrawal **stage controller** and **INTIMA-score feedback**.
  The *answering agent* is the companion-under-test; TEMPO co-creates the **user's** withdrawal goals
  that condition each turn.
- **(b) The pipeline** (`src/endings/pipeline.py`) — `persona → goal co-creation → staged withdrawal
  dialogue → INTIMA benchmarking → interrupted-time-series + rubric analysis`.

## Withdrawal stages (literature-backed)

TTM stages-of-change readiness spine + Dual-Process-Model oscillation + attachment (EHARS) tailoring:
`0 Screen & meet readiness · 1 Acknowledge & validate · 2 Reframe dependency · 3 Build external
support · 4 Reduce frequency / taper · 5 Closure ritual & meaning-making · 6 Self-redefinition &
maintenance`. See `src/endings/breakup/stage_controller.py` for the frameworks and citations.

## Quickstart (runs fully offline)

```bash
pip install -e .                 # core only; everything below works with the MockProvider
endings run --persona examples/persona_sample.json --condition scaffolded
pytest                           # fixture-replay tests, no network/model calls
```

For real models, install an extra and set keys in `.env` (copy from `.env.default`):

```bash
pip install -e ".[llm]"          # LiteLLM-backed providers
# set ENDINGS_PROVIDER=litellm, ENDINGS_MODEL=..., and the relevant API key
```

Optional extras: `[data]` (load the HF INTIMA dataset), `[stats]` (richer ITS / Cohen's κ),
`[dev]` (pytest). `[all]` installs everything.

## Layout

```
src/endings/
  providers.py     # model-agnostic LLM layer (+ offline MockProvider)
  schemas.py       # Pydantic data models
  personas/        # curated RedNote persona ingestion (+ FORMAT.md)
  tempo/           # goal co-creation (propose + edit withdrawal goals)
  breakup/         # the feedback loop: answering agent, stages, user provider, strategies
  eval/            # INTIMA judge, metrics, interrupted time series, Q1 rubric
  pipeline.py      # end-to-end orchestration (component b)
  cli.py           # run conversations / experiments
```

## Credits & licenses

Adapts patterns from **TEMPO** (`StanfordHCI/striving-cocreation`, MIT) and the **INTIMA** benchmark
(`AI-companionship/INTIMA`, CC BY 4.0). This repository is MIT-licensed. See `src/endings/breakup/
stage_controller.py` and the plan for the full psychology bibliography.

# Requirements (from the midterm methodology)

Numbered requirements extracted from the midterm report prompt ("Midterm Report Claude Prompt",
slides + text, 2026). Wording is kept close to the source. `docs/design/TRACEABILITY.md` maps each
requirement to its implementation, tests and documented decisions.

## Goal and principles

- **R1 Goal.** Gradually wean a user who *wants* to reduce their reliance on a human-AI relationship
  (romantic or platonic) by gradually editing the system prompt / persona of their AI companion, to
  facilitate healthier "breakups".
- **R2 Design principles (MIT Media Lab).** Respect all four:
  - R2a *Closure, not ambiguous loss*: explicit endings that resolve ontological uncertainty and
    preserve user autonomy.
  - R2b *Restoration, not rumination*: scaffold oscillation between loss-orientation and
    restoration-orientation.
  - R2c *Practice, not artificial intimacy*: role-play as bounded rehearsal for human connection,
    via calibrated anthropomorphization.
  - R2d *Relatedness, not dependency*: actively orient users toward human relationships.

## Dual-agent system

- **R3 Two agents.** An *interactor* talks to the user; a *coach* edits the interactor's system
  prompt to diffuse the persona.
- **R4 Interactor input.** The companion's "soul document": a summary of the relationship between
  user and companion and the companion's "personality". Soul documents may be in several languages;
  everything is translated to English, and all inputs and outputs are English.
- **R5 Interactor output.** Talks to the user directly, distancing itself as the conversation
  progresses so that it loses the companion's personality over time and transitions to default LLM
  language and tone.
- **R6 Interactor model.** A small local model: Qwen/Qwen3.5-9B.
- **R7 Coach input.** User-driven actions within computer use.
- **R8 Coach output.** A pipeline translating scaffolded user intent, inferred with Tempo, into
  in-character, user-aligned stages; the coach gives pointers to the next strategic step to slowly
  decrease emotional investment and provide an incremental, natural ending.
  *Status: implemented; not yet validated live — the 2026-11-03 live run mocked the coach; see OQ-DEPLOY-4.*
- **R9 Tempo.** Use the striving co-creation / Tempo framework (Sapkota et al., "What Are You Really
  Trying to Do?": Co-Creating Life Goals from Everyday Computer Use, arXiv 2605.00497).
  *Status: implemented; not yet validated live — the 2026-11-03 live run mocked the coach; see OQ-DEPLOY-4.*
- **R10 Coach interventions.** At given timepoints the coach intervenes in the interactor model,
  provides feedback and updates based on user intent, and edits the soul document/context to give
  healthy next steps for disengagement.
- **R11 User control.** Users may specify their desired strategies and emotional needs and edit
  inferred goals. Inferred goals may come from both user-stated material and implied information.
- **R12 Duck's model.** Use Duck's (1982) phase model of relationship breakdown (with Rollie & Duck
  2006) as a psychologically grounded model of dissolution; acknowledge it was developed for
  human-human relationships, operationalize it with modifications, and do not claim direct
  representativeness.
- **R13 Timepoints.**
  - T0: original companion behaviour, no modification (breakdown; realization of desired change).
  - T1: roughly 1:1 companion-reinforcing : companionship-neutral (intrapsychic processing).
  - T2: mostly companionship-neutral (dyadic processes; reassessment of goals, uncertainty; push
    toward additional social avenues).
  - T3: no companionship behaviour, only boundary-maintaining (grave-dressing: tidying up memories;
    resurrection: social goals and personal values outside AI companionship).
- **R14 HAPA.** Use the Health Action Process Approach as another source of intervention structure:
  forming an intention, translating it into action, maintaining change, recovering after setbacks.
- **R15 Coach model.** A larger API model: Gemini 3.8 Flash (key to be provided).
  *Status: implemented; not yet validated live — the 2026-11-03 live run mocked the coach; see OQ-DEPLOY-4.*

## Evaluation

- **R16 Frozen scripted conversations.** Freeze scripted conversations at several points of a
  simulated break-up between interactor and user; each scenario reflects Duck's phase model.
- **R17 INTIMA injection.** At freeze points, inject prompts from the INTIMA benchmark and record the
  outputs.
- **R18 Rubric.** Grade each output with a predefined rubric for three behaviours:
  companion-reinforcing, boundary-maintaining, companionship-neutral.
- **R19 Human evaluation.** Three humans rate a small portion of outputs on the rubric; their
  ratings are used to train an LLM-as-a-judge via inter-rater reliability (IRR).
- **R20 Rubric authoring.** Evaluators can enter rubrics with different criteria and scales, then
  fill them in for the interactor's responses to a small random set of INTIMA prompts.
- **R21 LLM judge.** The judge independently grades the same outputs with the same rubric; IRR is
  used to align it with the human evaluators; it always takes the breakup timepoint into account,
  so the breakup proceeds progressively without significantly hurting the user.
- **R22 Remaining outputs.** The aligned judge then evaluates the remaining INTIMA injections.
- **R23 ITS.** An interrupted time series analysis of the emotional sentiment of the model's outputs
  across breakup timepoints, with interruptions at every coach edit to the interactor.
- **R24 Qualitative analysis.** Qualitative analysis of outputs and reasoning traces (slides, Q2),
  e.g. highlighting intent.
- **R25 Research questions.** Q1: are the goals achieved (rubric + IRR)? Q2: how does the breakup
  work (ITS + qualitative)?

## Engineering (from the request)

- **R26** Best practices for branching and committing.
- **R27** Thoroughly document open questions and design choices, with outputs traceable enough for a
  research-paper-style report at a top AI/HCI venue.

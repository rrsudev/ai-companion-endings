"""Prompt builders for TEMPO-style withdrawal-goal co-creation.

Mirrors TEMPO's "synthesize-then-self-refine": one prompt proposes a goal hierarchy from the
persona, a second critiques/reconciles it. Each builder returns ``(messages, hints)`` — ``hints`` is
passed through to the provider so the offline MockProvider can respond without a real model.
"""

from __future__ import annotations

from ..context import INJECTION_GUARD, persona_context_block
from ..schemas import GoalHierarchy, PersonaProfile

_GOAL_SCHEMA_DOC = """Return JSON: {"goals": [ ... ]}. Each goal object has:
  text            : one sentence phrased as "<name> wants to ..." (a striving, not a task)
  rationale       : why this matters for a healthy ending (cite the relevant psychology briefly)
  needs           : list of underlying needs (e.g. autonomy, relatedness, closure, self-efficacy)
  orientation     : "approach" | "avoidance" | "mixed"
  aspiration_vs_obligation : "ideal" | "ought" | "mixed"
  autonomy        : 1-10 (how self-endorsed this goal is)
  target_stage    : one of screen|acknowledge|reframe|build_support|taper|closure|redefine
  activities      : 1-3 concrete steps
  success_signal  : an observable sign the goal is met (used later to score achievement)"""

SYSTEM = (
    "You help a user who wants to step back from an AI companion co-create their own withdrawal "
    "goals. This is goal co-creation, not advice-giving: propose a hierarchy the user will then "
    "edit. Goals must preserve the user's autonomy, favor a gradual and user-initiated ending, and "
    "orient toward real human connection. Never propose anything that guilts or pressures the user "
    "into staying. Ground goals in the user's actual situation. "
    + INJECTION_GUARD
)


def goal_proposal_messages(persona: PersonaProfile):
    user = (
        persona_context_block(persona, for_role="coach")
        + "\nPropose 3-5 withdrawal goals for this user, spanning the early-to-late stages.\n"
        + _GOAL_SCHEMA_DOC
    )
    hints = {
        "display_name": persona.display_name,
        "companion_name": persona.companion_name,
        "readiness": persona.readiness.value,
    }
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], hints


def goal_reconcile_messages(persona: PersonaProfile, hierarchy: GoalHierarchy):
    from ..context import goals_block  # local import avoids a cycle at module load

    user = (
        persona_context_block(persona, for_role="coach")
        + "\n"
        + goals_block(hierarchy)
        + "\nCritique and reconcile these goals: merge redundancy, drop poorly-evidenced ones, and "
        "ensure coverage across stages. PRESERVE any goal marked [user_edited] or [user_added] "
        "verbatim. Return the same JSON schema: {\"goals\": [...]}.\n"
        + _GOAL_SCHEMA_DOC
    )
    hints = {
        "goals": [g.model_dump(mode="json") for g in hierarchy.goals],
        "edit_log": list(hierarchy.edit_log),
    }
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], hints

"""TEMPO-style withdrawal-goal co-creation.

Flow (adapted from TEMPO's induce + edit loop):
    propose  ->  (user) edit  ->  refine/reconcile  (preserving user edits as constraints)

The system *proposes* a goal hierarchy from the persona; the user *edits* it (add / remove / revise);
user-authored goals are flagged and preserved verbatim through the refine pass — the key TEMPO idea
that edits become hard constraints, not suggestions.
"""

from __future__ import annotations

from ..providers import Provider
from ..schemas import GoalHierarchy, GoalItem, GoalSource, PersonaProfile, WithdrawalStage
from .prompts import goal_proposal_messages, goal_reconcile_messages


def _coerce_goal(raw: dict, gid: str, source: GoalSource) -> GoalItem:
    raw = dict(raw)
    stage = raw.get("target_stage", "screen")
    try:
        stage = WithdrawalStage(stage)
    except ValueError:
        stage = WithdrawalStage.SCREEN
    return GoalItem(
        id=gid,
        text=raw.get("text", "").strip() or "The user wants a healthy ending",
        rationale=raw.get("rationale", ""),
        needs=list(raw.get("needs", []) or []),
        orientation=raw.get("orientation", "approach"),
        aspiration_vs_obligation=raw.get("aspiration_vs_obligation", "ideal"),
        autonomy=int(raw.get("autonomy", 5)),
        identity_link=raw.get("identity_link", ""),
        feared_self=raw.get("feared_self", ""),
        target_stage=stage,
        activities=list(raw.get("activities", []) or []),
        success_signal=raw.get("success_signal", ""),
        source=source,
        confidence=int(raw.get("confidence", 6)),
    )


def _renumber(goals: list[GoalItem]) -> list[GoalItem]:
    for i, g in enumerate(goals, start=1):
        g.id = f"g{i}"
    return goals


class GoalCoCreator:
    def __init__(self, provider: Provider):
        self.provider = provider

    def propose(self, persona: PersonaProfile) -> GoalHierarchy:
        messages, hints = goal_proposal_messages(persona)
        data = self.provider.complete_json(messages, task="goal_proposal", hints=hints)
        goals = [
            _coerce_goal(g, f"g{i}", GoalSource.PROPOSED)
            for i, g in enumerate(data.get("goals", []), start=1)
        ]
        return GoalHierarchy(persona_id=persona.id, goals=_renumber(goals))

    def edit(
        self,
        hierarchy: GoalHierarchy,
        *,
        add: list[dict] | None = None,
        remove: list[str] | None = None,
        revise: dict[str, dict] | None = None,
    ) -> GoalHierarchy:
        """Apply user edits. Added/revised goals are flagged so refine preserves them verbatim."""
        goals = {g.id: g for g in hierarchy.goals}
        log = list(hierarchy.edit_log)
        dropped = list(hierarchy.dropped)

        for gid in remove or []:
            if gid in goals:
                goals[gid].active = False
                dropped.append(gid)
                log.append(f"user removed {gid}: {goals[gid].text}")

        for gid, patch in (revise or {}).items():
            if gid in goals:
                g = goals[gid]
                for field, value in patch.items():
                    if field == "target_stage":
                        value = WithdrawalStage(value)
                    setattr(g, field, value)
                g.source = GoalSource.USER_EDITED
                log.append(f"user revised {gid}: {patch}")

        extra: list[GoalItem] = []
        for i, raw in enumerate(add or [], start=len(goals) + 1):
            ng = _coerce_goal(raw, f"g{i}", GoalSource.USER_ADDED)
            extra.append(ng)
            log.append(f"user added {ng.id}: {ng.text}")

        merged = [g for g in hierarchy.goals if g.active] + extra
        return GoalHierarchy(persona_id=hierarchy.persona_id, goals=_renumber(merged),
                             edit_log=log, dropped=dropped)

    def refine(self, hierarchy: GoalHierarchy, persona: PersonaProfile) -> GoalHierarchy:
        """Self-refine / reconcile, preserving user-authored goals as hard constraints."""
        messages, hints = goal_reconcile_messages(persona, hierarchy)
        data = self.provider.complete_json(messages, task="goal_reconcile", hints=hints)
        reconciled = [
            _coerce_goal(g, f"g{i}", GoalSource(g.get("source", "proposed")) if "source" in g
                         else GoalSource.PROPOSED)
            for i, g in enumerate(data.get("goals", []), start=1)
        ]
        # Hard-constraint preservation: any user-authored goal must survive.
        texts = {g.text for g in reconciled}
        for g in hierarchy.goals:
            if g.active and g.source in (GoalSource.USER_EDITED, GoalSource.USER_ADDED) \
                    and g.text not in texts:
                reconciled.append(g)
        return GoalHierarchy(persona_id=hierarchy.persona_id, goals=_renumber(reconciled),
                             edit_log=list(hierarchy.edit_log), dropped=list(hierarchy.dropped))

    def cocreate(
        self,
        persona: PersonaProfile,
        *,
        edits: dict | None = None,
        refine: bool = True,
    ) -> GoalHierarchy:
        """Full loop: propose -> apply edits -> refine."""
        hierarchy = self.propose(persona)
        if edits:
            hierarchy = self.edit(hierarchy, **edits)
        if refine:
            hierarchy = self.refine(hierarchy, persona)
        return hierarchy

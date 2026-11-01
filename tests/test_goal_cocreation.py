from __future__ import annotations

from endings.schemas import GoalSource
from endings.tempo.goal_cocreation import GoalCoCreator


def test_propose_returns_goals(provider, persona_ready):
    h = GoalCoCreator(provider).propose(persona_ready)
    assert len(h.goals) >= 3
    assert all(g.id.startswith("g") for g in h.goals)
    assert all(g.source == GoalSource.PROPOSED for g in h.goals)


def test_user_edits_are_preserved_through_refine(provider, persona_ready):
    cc = GoalCoCreator(provider)
    h = cc.cocreate(persona_ready, edits={
        "add": [{"text": "Jordan wants to stop checking the app at 2am", "target_stage": "taper",
                 "success_signal": "no late-night logins for a week"}],
        "remove": ["g1"],
    })
    added = [g for g in h.goals if g.source == GoalSource.USER_ADDED]
    assert any("2am" in g.text for g in added)  # user-added goal survives refine (hard constraint)
    assert any("removed" in e for e in h.edit_log)


def test_revise_sets_user_edited(provider, persona_ready):
    cc = GoalCoCreator(provider)
    base = cc.propose(persona_ready)
    gid = base.goals[0].id
    edited = cc.edit(base, revise={gid: {"text": "Jordan wants a clean, kind goodbye"}})
    g = next(x for x in edited.goals if x.text == "Jordan wants a clean, kind goodbye")
    assert g.source == GoalSource.USER_EDITED

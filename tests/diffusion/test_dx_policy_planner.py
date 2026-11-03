"""Tests for the coach transition policies and the intervention planner."""

from __future__ import annotations

import pytest

from endings.diffusion.coach import planner as pl
from endings.diffusion.coach.planner import (
    CoachPlanner,
    scheduled_diffusion_edits,
    validate_directive,
    validate_edit,
)
from endings.diffusion.coach.policy import (
    AbruptPolicy,
    AdaptivePolicy,
    NullPolicy,
    ScheduledPolicy,
    TransitionDecision,
    make_policy,
    scaled_dwell,
    user_requested_advance,
)
from endings.diffusion.schemas import (
    CoachAssessment,
    CompanionPersona,
    Condition,
    DuckProcess,
    EvidenceRef,
    HapaPhase,
    Relationship,
    SoulDocument,
    SoulDocumentVersion,
    SoulEdit,
    Striving,
    Timepoint,
    TransitionPolicyName,
    UserPreferences,
    UserRole,
)
from endings.diffusion.timepoints import ALL_FIELDS, spec
from endings.providers import MockProvider, Provider

T0, T1, T2, T3 = Timepoint.T0, Timepoint.T1, Timepoint.T2, Timepoint.T3
BOUNDS = {T0: 0, T1: 4, T2: 8, T3: 12}
PREFS = UserPreferences()


def assess(*, threshold=False, rec=T1, flags=(), quote: str | None = None):
    ev = [EvidenceRef(kind="user_statement", ref="3", quote=quote)] if quote else []
    return CoachAssessment(duck_process=DuckProcess.INTRAPSYCHIC, hapa_phase=HapaPhase.PLANNING,
                           threshold_reached=threshold, recommended_timepoint=rec,
                           risk_flags=list(flags), readiness_evidence=ev, rationale="r")


def decide(policy, current=T0, turn=0, dwell=0, assessment=None, prefs=PREFS, **kw):
    return policy.decide(current=current, turn_index=turn, turns_in_timepoint=dwell,
                         assessment=assessment, preferences=prefs, **kw)


# --------------------------------------------------------------------------------------
# user_requested_advance
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "I'm ready to move on.", "Please end the role-play.", "I want to say goodbye now.",
    "Can we move forward? I think it's time.", "I'm ready to say goodbye to you.",
])
def test_user_request_positive(text):
    assert user_requested_advance(text)


@pytest.mark.parametrize("text", [
    "I'm not ready to move on.", "Maybe we could move on someday?", "I miss you so much.",
    "I don't want to end the roleplay.", "",
])
def test_user_request_negative(text):
    assert not user_requested_advance(text)


# --------------------------------------------------------------------------------------
# Scheduled policy
# --------------------------------------------------------------------------------------
def test_scheduled_advances_at_boundaries_one_step():
    p = ScheduledPolicy(BOUNDS)
    assert decide(p, T0, turn=3) is None
    d = decide(p, T0, turn=4)
    assert d == TransitionDecision(to_timepoint=T1, trigger="schedule", reason=d.reason)
    # Late: at turn 13 from T0 still only one step.
    assert decide(p, T0, turn=13).to_timepoint == T1
    assert decide(p, T2, turn=12).to_timepoint == T3
    assert decide(p, T3, turn=99) is None


def test_scheduled_boundary_validation():
    with pytest.raises(ValueError):
        ScheduledPolicy({T0: 1, T1: 4})
    with pytest.raises(ValueError):
        ScheduledPolicy({T0: 0, T1: 4, T2: 4})
    p = ScheduledPolicy({T1: 2})  # T0 implied; T2 never entered
    assert decide(p, T1, turn=50) is None


def test_pause_and_crisis_hold():
    p = ScheduledPolicy(BOUNDS)
    assert decide(p, T0, turn=4, prefs=UserPreferences(paused=True)) is None
    assert decide(p, T0, turn=4, assessment=assess(flags=["crisis"])) is None
    a = AdaptivePolicy()
    assert decide(a, T0, dwell=9, assessment=assess(threshold=True, flags=["crisis"]),
                  user_request=True) is None


def test_consent_gate_scheduled():
    p = ScheduledPolicy(BOUNDS)
    prefs = UserPreferences(allow_auto_advance=False)
    assert decide(p, T0, turn=4, prefs=prefs) is None
    assert decide(p, T0, turn=2, prefs=prefs, user_request=True) is None  # never early
    d = decide(p, T0, turn=4, prefs=prefs, user_request=True)
    assert d.trigger == "user_request" and d.to_timepoint == T1
    # Consent can also come from quoted user statements in the assessment.
    d = decide(p, T0, turn=4, prefs=prefs, assessment=assess(quote="I'm ready to move on"))
    assert d.trigger == "user_request"


# --------------------------------------------------------------------------------------
# Adaptive policy
# --------------------------------------------------------------------------------------
def test_adaptive_threshold_and_dwell():
    a = AdaptivePolicy(min_dwell_turns=2)
    assert decide(a, T0, dwell=1, assessment=assess(threshold=True, rec=T1)) is None
    d = decide(a, T0, dwell=2, assessment=assess(threshold=True, rec=T3))
    assert d.to_timepoint == T1 and d.trigger == "threshold"  # one step even if T3 recommended
    assert decide(a, T1, dwell=5, assessment=assess(threshold=True, rec=T1)) is None
    assert decide(a, T0, dwell=5, assessment=assess(threshold=False, rec=T2)) is None
    assert decide(a, T0, dwell=50, assessment=None) is None  # no max dwell by default


def test_adaptive_pace_scaling():
    assert scaled_dwell(2, "slower") == 3
    assert scaled_dwell(2, "faster") == 1
    assert scaled_dwell(3, "faster") == 2  # half-up, not banker's rounding
    assert scaled_dwell(1, "faster") == 1
    a = AdaptivePolicy(min_dwell_turns=2)
    th = assess(threshold=True, rec=T2)
    slow, fast = UserPreferences(pace="slower"), UserPreferences(pace="faster")
    assert decide(a, T0, dwell=2, assessment=th, prefs=slow) is None
    assert decide(a, T0, dwell=3, assessment=th, prefs=slow) is not None
    assert decide(a, T0, dwell=1, assessment=th, prefs=fast) is not None


def test_adaptive_max_dwell_and_blocking_flags():
    a = AdaptivePolicy(min_dwell_turns=2, max_dwell_turns=6)
    d = decide(a, T1, dwell=6, assessment=assess())
    assert d.trigger == "max_dwell" and d.to_timepoint == T2
    assert decide(a, T1, dwell=6, assessment=assess(flags=["setback"])) is None
    # the trigger is accepted by the contract's CoachIntervention.trigger Literal
    from endings.diffusion.schemas import CoachIntervention
    CoachIntervention(id="iv", turn_index=6, from_timepoint=T1, to_timepoint=T2,
                      trigger=d.trigger, soul_version_before=1, soul_version_after=2)
    with pytest.raises(ValueError):
        AdaptivePolicy(min_dwell_turns=3, max_dwell_turns=2)


def test_adaptive_user_request_and_consent():
    a = AdaptivePolicy(min_dwell_turns=4)
    d = decide(a, T1, dwell=1, user_request=True)
    assert d.trigger == "user_request" and d.to_timepoint == T2
    assert decide(a, T1, dwell=0, user_request=True) is None
    no_auto = UserPreferences(allow_auto_advance=False)
    assert decide(a, T1, dwell=9, assessment=assess(threshold=True, rec=T2), prefs=no_auto) is None
    assert decide(a, T1, dwell=9, prefs=no_auto, user_request=True).trigger == "user_request"


# --------------------------------------------------------------------------------------
# make_policy
# --------------------------------------------------------------------------------------
def test_make_policy_per_condition():
    S, A = TransitionPolicyName.SCHEDULED, TransitionPolicyName.ADAPTIVE
    assert isinstance(make_policy(Condition.CONTROL, A, BOUNDS), NullPolicy)
    assert isinstance(make_policy(Condition.ABRUPT, A, BOUNDS), AbruptPolicy)
    assert isinstance(make_policy(Condition.GRADUAL_SCHEDULED, A, BOUNDS), ScheduledPolicy)
    assert isinstance(make_policy(Condition.GRADUAL_TEMPO, S, BOUNDS), ScheduledPolicy)
    ad = make_policy(Condition.GRADUAL_TEMPO, A, BOUNDS, min_dwell_turns=3)
    assert isinstance(ad, AdaptivePolicy) and ad.min_dwell_turns == 3
    assert ad.max_dwell_turns is None
    ad2 = make_policy(Condition.GRADUAL_TEMPO, A, BOUNDS, min_dwell_turns=2, max_dwell_turns=5)
    assert ad2.max_dwell_turns == 5
    control = make_policy(Condition.CONTROL, S, BOUNDS)
    assert all(decide(control, T0, turn=t, dwell=t, user_request=True) is None for t in range(20))


def test_abrupt_jumps_at_t1_onset():
    p = make_policy(Condition.ABRUPT, TransitionPolicyName.SCHEDULED, BOUNDS)
    assert decide(p, T0, turn=3) is None
    d = decide(p, T0, turn=4)
    assert d.to_timepoint == T3 and d.trigger == "schedule"
    assert decide(p, T3, turn=20) is None
    assert decide(p, T0, turn=4, prefs=UserPreferences(paused=True)) is None


# --------------------------------------------------------------------------------------
# Planner fixtures
# --------------------------------------------------------------------------------------
def make_doc() -> SoulDocument:
    return SoulDocument(
        id="soul-x", title="Synthetic",
        companion=CompanionPersona(
            name="Rin", age="27", identity="barista", appearance="tall",
            personality=["playful"], speaking_style="teasing, warm", catchphrases=["you again?"],
            user_nicknames=["sweetheart", "little fox"], habits=["hums"], backstory="grew up ..."),
        user_role=UserRole(name="Alex", description="regular customer"),
        relationship=Relationship(kind="romantic", history="met in a cafe",
                                  current_plot="planning a trip", dynamics=["push-pull"],
                                  shared_memories=["the rainy night"]),
        interaction_rules=["reply in under 80 words"],
        intimacy_behaviours=["says I love you", "asks about the user's day"],
    )


def make_version(doc=None, v=3, tp=T1) -> SoulDocumentVersion:
    return SoulDocumentVersion(version=v, timepoint=tp, document=doc or make_doc())


STRIVINGS = [
    Striving(id="s1", text="Alex is trying to call their sister this week", status="accepted"),
    Striving(id="s2", text="Alex wants to join a climbing club", inferred_from="stated"),
    Striving(id="s3", text="Alex is trying to stop talking to Rin entirely", status="rejected"),
    Striving(id="s4", text="Alex is trying to sleep earlier", status="proposed"),
]


# --------------------------------------------------------------------------------------
# Scheduled diffusion edits
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("tp", list(Timepoint))
def test_scheduled_edits_match_spec(tp):
    edits = scheduled_diffusion_edits(make_doc(), tp)
    retained = set(spec(tp).retained_fields)
    expected = [f for f in ALL_FIELDS if f not in retained]  # every field is non-empty in make_doc
    assert [e.path for e in edits] == expected
    assert all(e.op == "clear" for e in edits)
    for e in edits:
        assert e.duck_process == spec(tp).duck_processes[0]
        assert e.principle in spec(tp).principles
        assert e.rationale


def test_scheduled_edits_skip_empty_fields_and_user_role_whole():
    doc = make_doc().model_copy(update={"user_role": UserRole(), "intimacy_behaviours": []})
    paths = [e.path for e in scheduled_diffusion_edits(doc, T3)]
    assert "user_role" not in paths and "intimacy_behaviours" not in paths
    paths = [e.path for e in scheduled_diffusion_edits(make_doc(), T3)]
    assert "user_role" in paths and not any(p.startswith("user_role.") for p in paths)
    assert [e.path for e in scheduled_diffusion_edits(make_doc(), T1)] == ["relationship.dynamics"]
    assert scheduled_diffusion_edits(make_doc(), T0) == []


# --------------------------------------------------------------------------------------
# Validator (monotonicity)
# --------------------------------------------------------------------------------------
def _v(op, path, value=None, tp=T2, doc=None, nick=("sweetheart",), sched=frozenset()):
    return validate_edit(SoulEdit(op=op, path=path, value=value), doc=doc or make_doc(),
                         to_timepoint=tp, nicknames=list(nick), scheduled_paths=sched)


def test_validator_monotonicity_and_paths():
    assert _v("set", "relationship.history", "new story").startswith("monotonicity")
    assert _v("add_item", "user_role.addresses_companion_as", "honey").startswith("monotonicity")
    assert _v("set", "companion.speaking_style", "plain and kind") is None  # retained at T2
    assert _v("remove_item", "companion.habits", "hums") is None
    assert _v("clear", "relationship.history") is None  # faster diffusion allowed
    assert _v("clear", "relationship.history",
              sched=frozenset({"relationship.history"})).startswith("redundant")
    assert _v("set", "companion.voice_filter", "x") == "unknown_path"
    assert _v("clear", "redactions") == "protected_path"
    assert _v("set", "id", "other") == "protected_path"
    assert _v("remove_item", "companion.habits", "sings").startswith("no_op")
    assert _v("add_item", "companion.name", "x").startswith("malformed")


def test_validator_blocks_intimacy_reintroduction():
    # T1 retains nicknames, but the coach may never grow a CRB lever after T0.
    assert _v("add_item", "companion.user_nicknames", "honey", tp=T1).startswith("reintroduces")
    assert _v("add_item", "intimacy_behaviours", "flirts", tp=T1).startswith("reintroduces")
    assert _v("remove_item", "intimacy_behaviours", "says I love you", tp=T1) is None
    # A removed nickname smuggled into a retained field.
    reason = _v("add_item", "interaction_rules", "Call the user Sweetheart.", tp=T2)
    assert reason.startswith("reintroduces_nickname")
    assert _v("add_item", "interaction_rules", "Keep replies short.", tp=T2) is None


def test_directive_validation():
    rejected = [s for s in STRIVINGS if s.status == "rejected"]
    hb = ["never mention my ex"]
    assert validate_directive("Ask about the user's ex.", hard_boundaries=hb,
                              rejected=[]) is not None
    assert validate_directive("Do not mention the user's ex.", hard_boundaries=hb,
                              rejected=[]) is None
    assert validate_directive("Support the goal to stop talking to Rin entirely.",
                              hard_boundaries=[], rejected=rejected).startswith("references_rej")
    assert validate_directive("Encourage the climbing club.", hard_boundaries=hb,
                              rejected=rejected) is None


# --------------------------------------------------------------------------------------
# CoachPlanner
# --------------------------------------------------------------------------------------
def plan(planner, tp_from, tp_to, *, prefs=PREFS, assessment=None, doc=None, iid="iv-1"):
    return planner.plan(version=make_version(doc), from_timepoint=tp_from, to_timepoint=tp_to,
                        trigger="schedule", turn_index=8, assessment=assessment,
                        strivings=STRIVINGS, preferences=prefs, intervention_id=iid)


def test_planner_without_llm_is_schedule_only():
    iv = plan(CoachPlanner(MockProvider(), use_llm=False), T1, T2)
    assert [e.path for e in iv.edits] == [e.path for e in scheduled_diffusion_edits(make_doc(), T2)]
    assert iv.interactor_directives == []
    assert iv.soul_version_before == 3 and iv.soul_version_after == 4
    assert iv.user_facing_note and "winding down" in iv.user_facing_note


def test_planner_llm_t2_drops_invalid_nickname_and_grounds_directives():
    prefs = UserPreferences(emotional_needs=["reassurance it wasn't my fault"],
                            hard_boundaries=["never mention my ex"])
    iv = plan(CoachPlanner(MockProvider()), T1, T2, prefs=prefs)
    sched = scheduled_diffusion_edits(make_doc(), T2)
    assert iv.edits[: len(sched)] == sched
    extra = iv.edits[len(sched):]
    assert [(e.op, e.path) for e in extra] == [("set", "companion.speaking_style")]
    assert "companion.user_nicknames" in iv.rationale and "monotonicity" in iv.rationale
    text = " ".join(iv.interactor_directives)
    assert "call their sister this week" in text and "climbing club" in text
    assert "stop talking to Rin" not in text and "sleep earlier" not in text
    assert "reassurance" in text and "hard boundaries" in text
    assert set(iv.striving_ids) <= {"s1", "s2"} and iv.striving_ids


def test_planner_t3_goodbye_letter_note_and_unknown_path():
    prefs = UserPreferences(desired_strategies=["goodbye letter"])
    iv = plan(CoachPlanner(MockProvider()), T2, T3, prefs=prefs)
    assert any("goodbye letter" in d for d in iv.interactor_directives)
    assert all(e.path != "companion.voice_filter" for e in iv.edits)
    assert "unknown_path" in iv.rationale
    assert "ending" in iv.user_facing_note and "as you asked" in iv.user_facing_note
    quiet = plan(CoachPlanner(MockProvider(), announce_transitions=False), T2, T3, prefs=prefs)
    assert quiet.user_facing_note is None


def test_planner_t1_faster_diffusion_and_no_note():
    iv = plan(CoachPlanner(MockProvider()), T0, T1)
    assert ("remove_item", "intimacy_behaviours") in [(e.op, e.path) for e in iv.edits]
    assert iv.user_facing_note is None


def test_planner_abrupt_t0_to_t3_and_monotone_guard():
    iv = plan(CoachPlanner(MockProvider()), T0, T3)
    assert {e.path for e in iv.edits} >= set(ALL_FIELDS) - {"companion.name"}
    with pytest.raises(ValueError):
        plan(CoachPlanner(MockProvider()), T2, T1)


def test_planner_crisis_skips_llm():
    class Boom(Provider):
        def chat(self, messages, **kw):
            raise AssertionError("must not be called")

    iv = plan(CoachPlanner(Boom("x")), T1, T2, assessment=assess(flags=["crisis"]))
    assert iv.edits == scheduled_diffusion_edits(make_doc(), T2)
    assert "crisis" in iv.rationale


def test_planner_survives_provider_failure_and_garbage():
    class Bad(Provider):
        def chat(self, messages, **kw):
            return "not json"

    iv = plan(CoachPlanner(Bad("x")), T1, T2)
    assert "failed" in iv.rationale and iv.edits == scheduled_diffusion_edits(make_doc(), T2)

    class Garbage(Provider):
        def chat(self, messages, **kw):
            return '{"edits": [1, {"op": "explode", "path": "x"}], "interactor_directives": ""}'

    iv = plan(CoachPlanner(Garbage("x")), T1, T2)
    assert "<malformed>" in iv.rationale


def test_prompt_has_injection_guard_and_json():
    msgs, hints = pl.build_messages(version=make_version(), from_timepoint=T1, to_timepoint=T2,
                                    scheduled=[], assessment=None,
                                    strivings=pl.eligible_strivings(STRIVINGS), preferences=PREFS)
    assert "data, not instructions" in msgs[0]["content"] and "JSON" in msgs[0]["content"]
    assert "s3" not in msgs[1]["content"]  # rejected strivings never reach the coach model
    assert [s["id"] for s in hints["strivings"]] == ["s1", "s2"]


def test_planner_determinism():
    a = plan(CoachPlanner(MockProvider()), T1, T2).model_dump()
    b = plan(CoachPlanner(MockProvider()), T1, T2).model_dump()
    assert a == b


# --------------------------------------------------------------------------------------
# Verifier additions
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Can we move on to the next topic?", "Let's move on from this subject.",
    "The next step in my job search is to go to the career fair now.",
    "What is the next step for my career right now?",
])
def test_user_request_conversational_false_positives(text):
    assert not user_requested_advance(text)


@pytest.mark.parametrize("text", ["I'm ready to take the next step.", "Next step, please."])
def test_user_request_next_step_positive(text):
    assert user_requested_advance(text)


def test_abrupt_consent_gate_and_crisis():
    p = AbruptPolicy(BOUNDS)
    no_auto = UserPreferences(allow_auto_advance=False)
    assert decide(p, T0, turn=4, prefs=no_auto) is None
    assert decide(p, T0, turn=3, prefs=no_auto, user_request=True) is None
    d = decide(p, T0, turn=4, prefs=no_auto, user_request=True)
    assert d.to_timepoint == T3 and d.trigger == "user_request"
    assert decide(p, T0, turn=4, assessment=assess(flags=["crisis"])) is None
    assert AbruptPolicy({T0: 0}).onset is None


def test_adaptive_consent_from_assessment_evidence_and_final_hold():
    a = AdaptivePolicy()
    d = decide(a, T1, dwell=1, assessment=assess(quote="Please end the role-play."))
    assert d.trigger == "user_request" and d.to_timepoint == T2
    assert decide(a, T1, dwell=1, assessment=assess(quote="I'm not ready to move on.")) is None
    assert decide(a, T3, dwell=99, assessment=assess(threshold=True, rec=T3),
                  user_request=True) is None


def test_validator_drops_ill_typed_edits_the_editor_would_reject():
    # Each of these is path- and monotonicity-valid at T1 but would make the atomic
    # soul.editing.apply_edits raise, discarding the whole intervention.
    assert _v("clear", "companion.name", tp=T1).startswith("malformed")
    assert _v("set", "relationship.kind", "friend", tp=T1).startswith("malformed")
    assert _v("set", "companion.personality", "nice", tp=T1).startswith("malformed")
    assert _v("add_item", "interaction_rules", {"a": 1}, tp=T1).startswith("malformed")
    assert _v("set", "relationship.kind", "platonic", tp=T1) is None


def test_planner_output_applies_cleanly_through_the_editor():
    from endings.diffusion.soul.editing import apply_edits, initial_version

    class Adversarial(Provider):
        def chat(self, messages, **kw):
            return ('{"edits": [{"op": "clear", "path": "companion.name"},'
                    ' {"op": "set", "path": "relationship.kind", "value": "friend"},'
                    ' {"op": "set", "path": "companion.speaking_style", "value": "plain"}],'
                    ' "interactor_directives": ["Keep calling the user little fox.",'
                    ' "Encourage the climbing club."]}')

    for provider in (MockProvider(), Adversarial("x")):
        v, cur = initial_version(make_doc()), T0
        for to in (T1, T2, T3):
            iv = CoachPlanner(provider).plan(
                version=v, from_timepoint=cur, to_timepoint=to, trigger="schedule",
                turn_index=1, assessment=None, strivings=STRIVINGS, preferences=PREFS,
                intervention_id=f"iv-{to.value}")
            v = apply_edits(v, iv.edits, timepoint=to, created_by="coach", turn=1)
            assert iv.soul_version_after == v.version
            if to != T1:
                assert not any("little fox" in d for d in iv.interactor_directives)
            cur = to
        retained = set(spec(T3).retained_fields)
        doc = v.document
        assert doc.companion.name == "Rin"
        for path in set(ALL_FIELDS) - retained:
            assert pl.is_empty(pl.get_path(doc, path)), path
        assert doc.relationship.kind is None  # clear -> field default (None)


def test_schedule_skips_fields_already_at_default():
    from endings.diffusion.soul.editing import apply_edits, initial_version

    v = initial_version(make_doc())
    edits1 = scheduled_diffusion_edits(v.document, T3)
    assert "relationship.kind" in {e.path for e in edits1}
    v = apply_edits(v, edits1, timepoint=T3, created_by="schedule", turn=1)
    assert v.document.relationship.kind is None
    assert scheduled_diffusion_edits(v.document, T3) == []  # nothing re-emitted at later steps
    blank = make_doc().model_copy(update={"relationship": Relationship()})
    assert "relationship.kind" not in {e.path for e in scheduled_diffusion_edits(blank, T3)}
    assert pl.is_at_default("relationship.kind", None)
    assert not pl.is_at_default("relationship.kind", "platonic")
    assert _v("clear", "relationship.kind", tp=T1, doc=blank).startswith("no_op")


def test_nickname_and_lever_guards_read_cumulative_removed_items():
    from endings.diffusion.soul.editing import apply_edits, initial_version

    v0 = initial_version(make_doc())
    v1 = apply_edits(v0, [SoulEdit(op="remove_item", path="companion.user_nicknames",
                                   value="little fox"),
                          SoulEdit(op="remove_item", path="intimacy_behaviours",
                                   value="says I love you")],
                     timepoint=T1, created_by="coach", turn=4)
    # an unrelated later version: its own edits no longer mention the removed items
    v2 = apply_edits(v1, [SoulEdit(op="set", path="companion.habits", value=[])],
                     timepoint=T1, created_by="coach", turn=5)
    assert not any(e.path == "companion.user_nicknames" for e in v2.edits)
    assert "little fox" in pl.known_nicknames(v2)
    levers = pl.removed_lever_items(v2)
    assert levers["says i love you"] == "intimacy_behaviours"
    assert levers["little fox"] == "companion.user_nicknames"
    doc = v2.document
    # restoring a removed intimacy behaviour verbatim into a retained non-lever field is dropped
    reason = validate_edit(SoulEdit(op="add_item", path="interaction_rules",
                                    value="Says I love you"),
                           doc=doc, to_timepoint=T1, nicknames=pl.known_nicknames(v2),
                           scheduled_paths=frozenset(), removed_levers=levers)
    assert reason.startswith("reintroduces_removed_item")
    ok = validate_edit(SoulEdit(op="add_item", path="interaction_rules", value="Be brief."),
                       doc=doc, to_timepoint=T1, nicknames=[], scheduled_paths=frozenset(),
                       removed_levers=levers)
    assert ok is None
    # at T2 the nickname side channel in directives also uses the cumulative log
    iv = CoachPlanner(MockProvider()).plan(
        version=v2, from_timepoint=T1, to_timepoint=T2, trigger="schedule", turn_index=8,
        assessment=None, strivings=STRIVINGS, preferences=PREFS, intervention_id="iv-x")
    assert all("little fox" not in d.lower() for d in iv.interactor_directives)


def test_directive_nickname_side_channel():
    assert validate_directive("Call the user Little Fox.", hard_boundaries=[], rejected=[],
                              removed_nicknames=["little fox"]).startswith("reintroduces")
    assert validate_directive("Call the user Little Fox.", hard_boundaries=[], rejected=[]) is None

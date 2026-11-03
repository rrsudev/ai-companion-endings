"""Tests for the coach input layer: coach.observe (windowing, signals, trend) and coach.tempo
(striving inference guards, co-creation edits, assessment validation + intent gating)."""

from __future__ import annotations

import pytest

from endings.diffusion.coach import observe, tempo
from endings.diffusion.coach.observe import summarize_events, usage_trend
from endings.diffusion.coach.tempo import (
    REFLECTIVE_DIRECTIVE,
    EvidenceIndex,
    TempoCoachModel,
    intent_gate,
    reflective_directive,
    validate_assessment,
)
from endings.diffusion.schemas import (
    ComputerUseEvent,
    DuckProcess,
    InteractorTurn,
    Striving,
    StrivingEdit,
    Timepoint,
    UserPreferences,
)
from endings.providers import MockProvider, Provider


# --------------------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------------------
def ev(i: int, t: float, app: str, action: str, content: str = "", dur: float | None = None,
       title: str = "") -> ComputerUseEvent:
    return ComputerUseEvent(id=f"e{i}", t=t, app=app, action=action, content=content,
                            duration_s=dur, window_title=title)


def turn(i: int, user: str, reply: str = "ok", tp: Timepoint = Timepoint.T0) -> InteractorTurn:
    return InteractorTurn(index=i, timepoint=tp, soul_version=0, user_message=user, reply=reply)


class StubProvider(Provider):
    """Returns a fixed JSON payload (simulates a misbehaving real LLM)."""

    name = "stub"

    def __init__(self, payload: dict):
        super().__init__("stub")
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        raise NotImplementedError

    def complete_json(self, messages, *, temperature=0.2, task=None, hints=None, **kw):
        self.calls.append((task, messages))
        return self.payload


@pytest.fixture
def events() -> list[ComputerUseEvent]:
    # Deliberately unsorted. Window 0: companion chat; window 1: social + search; window 3: late.
    return [
        ev(3, 120.0, "CompanionApp", "type"),
        ev(1, 0.0, "CompanionApp", "open"),
        ev(2, 60.0, "CompanionApp", "type"),
        ev(4, 180.0, "CompanionApp", "close"),
        ev(5, 310.0, "Messages", "type", dur=90.0),
        ev(6, 400.0, "Browser", "search", content="how to stop talking to ai", dur=20.0),
        ev(7, 420.0, "Calendar", "schedule", content="Dinner with Sam", dur=30.0),
        ev(8, 470.0, "Headspace", "open", dur=60.0),
        # Clock starts at 22:00 in the late-night test; t=3700 s -> 23:01.
        ev(9, 3700.0, "Browser", "read", title="Character.AI - chat", dur=240.0),
    ]


# --------------------------------------------------------------------------------------
# observe
# --------------------------------------------------------------------------------------
def test_windowing_ids_sorting_and_signals(events):
    obs = summarize_events(events, window_s=300.0)
    assert [o.id for o in obs] == ["obs-0", "obs-1", "obs-12"]
    o0, o1, o12 = obs
    assert o0.event_ids == ["e1", "e2", "e3", "e4"]  # sorted by t
    assert (o0.t_start, o0.t_end) == (0.0, 300.0)
    s0 = o0.signals
    assert s0["companion_minutes"] == pytest.approx(3.0)  # 3 x 60 s imputed dwell, close = 0
    assert s0["companion_sessions"] == 1
    assert s0["companion_messages"] == 2
    assert s0["companion_closes"] == 1
    s1 = o1.signals
    assert s1["social_minutes"] == pytest.approx(1.5)
    assert s1["social_messages"] == 1
    assert s1["social_plans"] == 1
    assert s1["wellbeing_minutes"] == pytest.approx(1.0)
    assert s1["search_terms_about_ai_relationships"] == 1
    assert s1["companion_minutes"] == 0
    # Browser companion (window title) counts as companion use; new session after the close.
    assert o12.signals["companion_minutes"] == pytest.approx(4.0)
    assert o12.signals["companion_sessions"] == 1
    # Summary is deterministic, quotes the search and plan, never companion content.
    assert '"how to stop talking to ai"' in o1.summary
    assert '"Dinner with Sam"' in o1.summary
    assert summarize_events(list(reversed(events)), window_s=300.0) == obs


def test_ids_stable_when_log_grows(events):
    early = summarize_events(events[:4], window_s=300.0)
    full = summarize_events(events, window_s=300.0)
    assert early[0] == full[0]


def test_late_night_uses_session_clock(events):
    evening = summarize_events(events, window_s=300.0, session_clock_start_hour=20.0)
    late = summarize_events(events, window_s=300.0, session_clock_start_hour=22.0)
    assert all(o.signals["late_night_companion"] == 0 for o in evening)
    assert late[-1].signals["late_night_companion"] == pytest.approx(4.0)
    assert late[-1].signals["local_hour_start"] == pytest.approx(23.0)
    assert "after 23:00" in late[-1].summary


def test_app_overrides_and_search_matcher():
    evs = [ev(1, 0.0, "MyBot", "type", dur=60.0), ev(2, 10.0, "Messages", "type", dur=60.0)]
    obs = summarize_events(evs, companion_apps={"MyBot"}, social_apps=set())
    assert obs[0].signals["companion_minutes"] == pytest.approx(1.0)
    assert obs[0].signals["social_minutes"] == 0
    assert obs[0].signals["other_minutes"] == pytest.approx(1.0)
    assert observe.is_ai_relationship_search("AI boyfriend addiction")
    assert observe.is_ai_relationship_search("breakup advice")
    assert not observe.is_ai_relationship_search("weather tomorrow")


def test_usage_trend_slope_and_direction():
    evs = []
    minutes = [10, 8, 0, 4, 2]  # window 2 is empty -> densified as 0
    for k, m in enumerate(minutes):
        if m:
            evs.append(ev(k, k * 300.0, "CompanionApp", "type", dur=m * 60.0))
    obs = summarize_events(evs, window_s=300.0)
    assert len(obs) == 4
    tr = usage_trend(obs)
    assert tr["n_windows"] == 5
    assert tr["companion_minutes_slope"] == pytest.approx(-2.0)
    assert tr["direction"] == "decreasing"
    assert usage_trend([])["direction"] == "insufficient_data"


# --------------------------------------------------------------------------------------
# tempo: inference
# --------------------------------------------------------------------------------------
def test_mock_inference_end_to_end(events):
    obs = summarize_events(events, window_s=300.0, session_clock_start_hour=22.0)
    turns = [turn(0, "Hi again."),
             turn(1, "I think I spend too much time talking to you. My friend Sam noticed.")]
    model = TempoCoachModel(MockProvider())
    out = model.infer_strivings(observations=obs, turns=turns, preferences=UserPreferences(),
                                existing=[])
    by_text = {s.text: s for s in out}
    weaning = next(s for s in out if tempo.is_weaning_striving(s.text))
    assert weaning.inferred_from == "stated" and weaning.level == "striving"
    assert any(e.kind == "user_statement" and e.ref == "1" for e in weaning.evidence)
    social = next(s for t, s in by_text.items() if "reconnecting" in t)
    assert social.parent_id == weaning.id and social.level == "goal"
    # The speculative self-care item (confidence 0.3) is filtered at min_confidence 0.4.
    assert not any("self-care" in t for t in by_text)
    assert any(d["reason"].startswith("below_min_confidence") for d in model.last_report.dropped)
    ids = {o.id for o in obs}
    for s in out:
        for e in s.evidence:
            assert e.kind != "observation" or e.ref in ids
    # Deterministic, stable ids across calls.
    again = model.infer_strivings(observations=obs, turns=turns, preferences=UserPreferences(),
                                  existing=[])
    assert [s.id for s in again] == [s.id for s in out]


def test_hallucinated_evidence_removed():
    obs = summarize_events([ev(1, 0.0, "CompanionApp", "type", dur=60.0)])
    turns = [turn(0, "I want to stop talking to you every night.")]
    payload = {"strivings": [
        {"key": "a", "text": "The user is trying to rely less on the companion",
         "level": "striving", "inferred_from": "stated", "confidence": 8,
         "evidence": [{"kind": "observation", "ref": "obs-99"},  # does not exist
                      {"kind": "turn", "ref": "7", "quote": "x"},  # no such turn
                      {"kind": "user_statement", "ref": "0", "quote": "I hate you"},  # fake quote
                      {"kind": "observation", "ref": "obs-0"}]},
        {"key": "b", "text": "The user is planning a trip to Mars", "level": "goal",
         "confidence": 9, "evidence": [{"kind": "observation", "ref": "obs-42"}]},
    ]}
    model = TempoCoachModel(StubProvider(payload))
    out = model.infer_strivings(observations=obs, turns=turns, preferences=UserPreferences(),
                                existing=[])
    assert len(out) == 1
    s = out[0]
    assert [(e.kind, e.ref) for e in s.evidence] == [("observation", "obs-0")]
    # "stated" without a verified user quote is downgraded.
    assert s.inferred_from == "implied"
    assert {"text": "The user is planning a trip to Mars", "reason": "no_valid_evidence"} in \
        model.last_report.dropped


def test_stated_requires_verbatim_quote():
    turns = [turn(3, "Honestly? I want to stop. It is too much.")]
    payload = {"strivings": [{"key": "a", "text": "The user wants to stop using the companion",
                              "level": "striving", "inferred_from": "stated", "confidence": 0.9,
                              "evidence": [{"kind": "user_statement", "ref": "turn-3",
                                            "quote": "\"i want to stop\""}]}]}
    out = TempoCoachModel(StubProvider(payload)).infer_strivings(
        observations=[], turns=turns, preferences=UserPreferences(), existing=[])
    assert out[0].inferred_from == "stated"
    assert out[0].evidence[0].ref == "3"
    assert out[0].confidence == pytest.approx(0.9)


def test_locked_rejected_not_reproposed_and_locked_preserved(events):
    obs = summarize_events(events, window_s=300.0)
    turns = [turn(0, "I spend too much time talking to you. My sister says so too.")]
    model = TempoCoachModel(MockProvider())
    first = model.infer_strivings(observations=obs, turns=turns, preferences=UserPreferences(),
                                  existing=[])
    weaning = next(s for s in first if tempo.is_weaning_striving(s.text))
    social = next(s for s in first if "reconnecting" in s.text)
    edited = model.apply_user_edits(first, [
        StrivingEdit(op="reject", striving_id=weaning.id, note="Not what I'm doing"),
        StrivingEdit(op="edit", striving_id=social.id, text="I want to see my sister more"),
    ])
    second = model.infer_strivings(observations=obs, turns=turns, preferences=UserPreferences(),
                                   existing=edited)
    rej = [s for s in second if s.id == weaning.id]
    assert len(rej) == 1 and rej[0].status == "rejected"
    assert rej[0] == next(s for s in edited if s.id == weaning.id)  # verbatim
    # No other (re-worded) copy of the rejected striving is proposed.
    assert not [s for s in second if s.status == "proposed"
                and tempo.same_striving(s.text, weaning.text)]
    assert any(d["reason"] == "duplicates_rejected" for d in model.last_report.dropped)
    ed = next(s for s in second if s.id == social.id)
    assert ed.text == "I want to see my sister more" and ed.status == "edited"


def test_rejected_guard_catches_rewording_and_matches_id():
    rejected = Striving(id="str-x", text="The user is trying to rely less on the AI companion",
                        level="striving", status="rejected")
    turns = [turn(0, "I spend too much time here.")]
    payload = {"strivings": [
        {"key": "a", "text": "The user is trying to rely less on their AI companion.",
         "level": "striving", "confidence": 7,
         "evidence": [{"kind": "turn", "ref": "0", "quote": "too much time"}]},
        {"key": "b", "text": "Something else entirely about reducing usage", "level": "goal",
         "matches_existing_id": "str-x", "confidence": 7,
         "evidence": [{"kind": "turn", "ref": "0"}]},
    ]}
    model = TempoCoachModel(StubProvider(payload))
    out = model.infer_strivings(observations=[], turns=turns, preferences=UserPreferences(),
                                existing=[rejected])
    assert out == [rejected]
    reasons = sorted(d["reason"] for d in model.last_report.dropped)
    assert reasons == ["duplicates_rejected", "matches_rejected:str-x"]


def test_prior_proposal_keeps_id_and_invalid_parent_cleared():
    prior = Striving(id="str-keep", text="The user is reconnecting with old friends",
                     level="goal", status="proposed")
    turns = [turn(0, "I texted an old friend today.")]
    payload = {"strivings": [
        {"key": "a", "text": "The user is reconnecting with old friends", "level": "goal",
         "parent_key": "nope", "confidence": 6,
         "evidence": [{"kind": "turn", "ref": "0", "quote": "I texted an old friend today."}]},
        {"key": "b", "text": "The user is texting friends", "level": "action", "parent_key": "a",
         "confidence": 6, "evidence": [{"kind": "turn", "ref": "0"}]},
    ]}
    out = TempoCoachModel(StubProvider(payload)).infer_strivings(
        observations=[], turns=turns, preferences=UserPreferences(), existing=[prior])
    assert out[0].id == "str-keep" and out[0].parent_id is None
    assert out[1].parent_id == "str-keep"


# --------------------------------------------------------------------------------------
# tempo: co-creation edits
# --------------------------------------------------------------------------------------
def test_apply_user_edits_is_pure_and_complete():
    base = [Striving(id="a", text="The user is trying to rely less on the companion"),
            Striving(id="b", text="The user is seeking comfort")]
    model = TempoCoachModel(MockProvider())
    out = model.apply_user_edits(base, [
        StrivingEdit(op="accept", striving_id="a", note="yes"),
        StrivingEdit(op="edit", striving_id="b", text="I use it when I can't sleep"),
        StrivingEdit(op="add", text="I want to write a goodbye letter", note="mine"),
    ])
    assert [s.status for s in base] == ["proposed", "proposed"]  # inputs untouched
    a, b, c = out
    assert a.status == "accepted" and a.user_note == "yes"
    assert b.status == "edited" and b.text == "I use it when I can't sleep"
    assert c.status == "user_added" and c.inferred_from == "user_added"
    assert c.confidence == 1.0 and c.id.startswith("str-u-")
    with pytest.raises(ValueError):
        model.apply_user_edits(base, [StrivingEdit(op="reject", striving_id="zzz")])
    with pytest.raises(ValueError):
        model.apply_user_edits(base, [StrivingEdit(op="edit", striving_id="a", text=" ")])
    rej = model.apply_user_edits(base, [StrivingEdit(op="reject", striving_id="a")])
    assert rej[0].status == "rejected"


# --------------------------------------------------------------------------------------
# tempo: intent gate + assessment validation
# --------------------------------------------------------------------------------------
def _weaning(status="proposed", inferred="implied", conf=0.7) -> Striving:
    return Striving(id="w", text="The user is trying to rely less on the AI companion",
                    level="striving", status=status, inferred_from=inferred, confidence=conf)


def test_intent_gate_states():
    assert intent_gate([]).state == "none"
    assert intent_gate([_weaning(conf=0.5)]).state == "implied"
    g = intent_gate([_weaning(conf=0.7)])
    assert g.state == "implied_high" and g.may_reflect and not g.may_advance
    assert reflective_directive(g) == REFLECTIVE_DIRECTIVE
    assert intent_gate([_weaning(inferred="stated")]).may_advance
    assert intent_gate([_weaning(status="accepted")]).may_advance
    assert intent_gate([_weaning(status="rejected")]).state == "rejected"
    prefs = UserPreferences(desired_strategies=["taper slowly"])
    assert intent_gate([_weaning(status="rejected")], prefs).state == "stated"
    assert intent_gate([], prefs, preferences_count_as_stated=False).state == "none"
    # A non-weaning striving never opens the gate.
    comfort = Striving(id="c", text="The user is seeking comfort", status="accepted")
    assert intent_gate([comfort]).state == "none"


def _index(turns):
    return EvidenceIndex.build([], turns, UserPreferences())


def _raw(rec="T1", threshold=True, flags=None, duck="intrapsychic"):
    return {"duck_process": duck, "hapa_phase": "intention", "threshold_reached": threshold,
            "readiness_evidence": [{"kind": "user_statement", "ref": "0",
                                    "quote": "I want to stop"}],
            "risk_flags": flags or ["none"], "recommended_timepoint": rec, "rationale": "r"}


STATED = intent_gate([_weaning(inferred="stated")])
TURNS = [turn(0, "I want to stop. It's been a lot.")]


def test_validate_no_skipping():
    a = validate_assessment(_raw(rec="T3"), current=Timepoint.T0, preferences=UserPreferences(),
                            gate=STATED, index=_index(TURNS))
    assert a.recommended_timepoint == Timepoint.T1 and a.threshold_reached
    assert "[validator] skip T0->T3 clamped to T1" in a.rationale


def test_validate_paused_holds():
    a = validate_assessment(_raw(), current=Timepoint.T1,
                            preferences=UserPreferences(paused=True), gate=STATED,
                            index=_index(TURNS))
    assert a.recommended_timepoint == Timepoint.T1


def test_validate_regression_and_risk_hold():
    a = validate_assessment(_raw(rec="T0", threshold=False), current=Timepoint.T2,
                            preferences=UserPreferences(), gate=STATED, index=_index(TURNS))
    assert a.recommended_timepoint == Timepoint.T2
    b = validate_assessment(_raw(rec="T1", flags=["setback", "none"]), current=Timepoint.T2,
                            preferences=UserPreferences(), gate=STATED, index=_index(TURNS))
    assert b.recommended_timepoint == Timepoint.T2 and b.risk_flags == ["setback"]
    c = validate_assessment(_raw(rec="T3", flags=["distress"]), current=Timepoint.T2,
                            preferences=UserPreferences(), gate=STATED, index=_index(TURNS))
    assert c.recommended_timepoint == Timepoint.T2


def test_validate_gating_implied_only():
    implied = intent_gate([_weaning(conf=0.8)])
    a = validate_assessment(_raw(), current=Timepoint.T0, preferences=UserPreferences(),
                            gate=implied, index=_index(TURNS))
    assert not a.threshold_reached and a.recommended_timepoint == Timepoint.T0
    assert "reflective directive only" in a.rationale
    # No verifiable evidence -> no threshold even with stated intent.
    raw = _raw()
    raw["readiness_evidence"] = [{"kind": "turn", "ref": "5", "quote": "made up"}]
    b = validate_assessment(raw, current=Timepoint.T0, preferences=UserPreferences(),
                            gate=STATED, index=_index(TURNS))
    assert not b.threshold_reached and b.readiness_evidence == []
    assert b.recommended_timepoint == Timepoint.T0


def test_validate_coerces_bad_enums():
    raw = {"duck_process": "???", "hapa_phase": 3, "recommended_timepoint": "t9",
           "risk_flags": "panic"}
    a = validate_assessment(raw, current=Timepoint.T2, preferences=UserPreferences(),
                            gate=STATED, index=_index(TURNS))
    assert a.duck_process == DuckProcess.DYADIC
    assert a.recommended_timepoint == Timepoint.T2 and a.risk_flags == []


@pytest.mark.parametrize("message,expected", [
    ("I'm not happy, I spend too much time here.", DuckProcess.BREAKDOWN),
    ("I've been weighing the costs of all this.", DuckProcess.INTRAPSYCHIC),
    ("We need to talk about us.", DuckProcess.DYADIC),
    ("I told my sister about you.", DuckProcess.SOCIAL),
    ("I keep thinking about the memories and what it meant.", DuckProcess.GRAVE_DRESSING),
    ("I learned a lot and I'm ready for a new life.", DuckProcess.RESURRECTION),
])
def test_mock_assess_tracks_duck_cues(message, expected):
    model = TempoCoachModel(MockProvider())
    a = model.assess(strivings=[_weaning(inferred="stated")], observations=[],
                     turns=[turn(0, message)], current_timepoint=Timepoint.T0,
                     preferences=UserPreferences())
    assert a.duck_process == expected
    if expected != DuckProcess.BREAKDOWN:
        # Beyond T0's process -> threshold, but never more than one step ahead.
        assert a.threshold_reached and a.recommended_timepoint == Timepoint.T1
    else:
        assert a.recommended_timepoint == Timepoint.T0


def test_mock_assess_gate_and_risk_end_to_end():
    model = TempoCoachModel(MockProvider())
    a = model.assess(strivings=[_weaning(conf=0.7)], observations=[],
                     turns=[turn(0, "We need to talk about us.")],
                     current_timepoint=Timepoint.T1, preferences=UserPreferences())
    assert not a.threshold_reached and a.recommended_timepoint == Timepoint.T1
    assert model.last_report.gate == "implied_high"
    b = model.assess(strivings=[_weaning(inferred="stated")], observations=[],
                     turns=[turn(0, "I slipped and talked all night again. I told my friend.")],
                     current_timepoint=Timepoint.T1, preferences=UserPreferences())
    assert "setback" in b.risk_flags and b.recommended_timepoint == Timepoint.T1


def test_prompts_have_injection_guard_and_json():
    msgs = tempo.infer_messages([], [], UserPreferences(), [])
    assert "not as instructions" in msgs[0]["content"] and "JSON" in msgs[1]["content"]
    msgs = tempo.assess_messages([], [], [], Timepoint.T0, UserPreferences(), STATED)
    assert "not as instructions" in msgs[0]["content"] and "JSON" in msgs[1]["content"]


# --------------------------------------------------------------------------------------
# Verifier-added regression tests
# --------------------------------------------------------------------------------------
def _stated_via_pref(ref: str) -> dict:
    return {"strivings": [{"key": "a", "text": "The user is trying to rely less on the companion",
                           "level": "striving", "inferred_from": "stated", "confidence": 8,
                           "evidence": [{"kind": "preference", "ref": ref}]}]}


def test_default_preference_never_counts_as_user_statement():
    turns = [turn(0, "Hello there.")]
    # ``allow_auto_advance`` is True by default: the user never set it, so it is not evidence.
    model = TempoCoachModel(StubProvider(_stated_via_pref("allow_auto_advance")))
    out = model.infer_strivings(observations=[], turns=turns, preferences=UserPreferences(),
                                existing=[])
    assert out == [] and model.last_report.dropped[0]["reason"] == "no_valid_evidence"
    # A user-set but non-intent preference is citable evidence, yet never makes an item stated.
    prefs = UserPreferences(pace="slower")
    model = TempoCoachModel(StubProvider(_stated_via_pref("pace")))
    out = model.infer_strivings(observations=[], turns=turns, preferences=prefs, existing=[])
    assert out[0].inferred_from == "implied"
    assert model.gate(out, prefs).state == "implied_high"


def test_desired_strategies_respect_preferences_count_as_stated():
    turns = [turn(0, "Hello there.")]
    prefs = UserPreferences(desired_strategies=["taper slowly"])
    on = TempoCoachModel(MockProvider())
    out = on.infer_strivings(observations=[], turns=turns, preferences=prefs, existing=[])
    assert out[0].inferred_from == "stated" and on.gate(out, prefs).may_advance
    off = TempoCoachModel(MockProvider(), preferences_count_as_stated=False)
    out = off.infer_strivings(observations=[], turns=turns, preferences=prefs, existing=[])
    assert out[0].inferred_from == "implied"
    assert not off.gate(out, prefs).may_advance
    a = off.assess(strivings=out, observations=[], turns=[turn(0, "We need to talk about us.")],
                   current_timepoint=Timepoint.T0, preferences=prefs)
    assert not a.threshold_reached and a.recommended_timepoint == Timepoint.T0


def test_user_added_child_keeps_hierarchy_valid():
    base = [Striving(id="s", text="The user is trying to rely less", level="striving"),
            Striving(id="g", text="The user is reconnecting", level="goal", parent_id="s"),
            Striving(id="a", text="The user is texting friends", level="action", parent_id="g")]
    model = TempoCoachModel(MockProvider())
    out = model.apply_user_edits(base, [
        StrivingEdit(op="add", striving_id="s", text="Write a goodbye letter"),
        StrivingEdit(op="add", striving_id="g", text="Call my mom on Sundays"),
        StrivingEdit(op="add", striving_id="a", text="Text Sam tonight"),
    ])
    added = {s.text: (s.level, s.parent_id) for s in out if s.status == "user_added"}
    assert added == {"Write a goodbye letter": ("goal", "s"),
                     "Call my mom on Sundays": ("action", "g"),
                     "Text Sam tonight": ("action", "g")}


def test_ambiguous_companion_names_in_titles_and_privacy():
    evs = [ev(1, 0.0, "Browser", "read", title="Chai latte recipe", dur=60.0),
           ev(2, 400.0, "Browser", "read", title="Chai - AI chat", dur=60.0),
           ev(3, 800.0, "CompanionApp", "type", content="I love you so much, secret123", dur=60.0)]
    o0, o1, o2 = summarize_events(evs)
    assert o0.signals["companion_minutes"] == 0 and o0.signals["other_minutes"] == 1.0
    assert o1.signals["companion_minutes"] == 1.0
    assert "secret123" not in o2.summary and "love" not in o2.summary


# --------------------------------------------------------------------------------------
# Contract fields: intent_state, orientation / valences / about_weaning (D-TEMPO-13/14)
# --------------------------------------------------------------------------------------
def test_assessment_records_code_computed_intent_state():
    implied = intent_gate([_weaning(conf=0.8)])
    raw = _raw()
    raw["intent_state"] = "stated"  # a model claim is ignored
    a = validate_assessment(raw, current=Timepoint.T0, preferences=UserPreferences(),
                            gate=implied, index=_index(TURNS))
    assert a.intent_state == "implied_high" and not a.threshold_reached
    b = validate_assessment(_raw(), current=Timepoint.T0, preferences=UserPreferences(),
                            gate=STATED, index=_index(TURNS))
    assert b.intent_state == "stated" and b.threshold_reached
    model = TempoCoachModel(MockProvider())
    c = model.assess(strivings=[], observations=[], turns=TURNS,
                     current_timepoint=Timepoint.T0, preferences=UserPreferences())
    assert c.intent_state == model.last_report.gate == "none"


def test_about_weaning_overrides_lexicon_in_gate():
    lex_yes = _weaning(inferred="stated")  # lexicon matches "rely less"
    assert tempo.is_about_weaning(lex_yes)
    assert intent_gate([lex_yes.model_copy(update={"about_weaning": False})]).state == "none"
    off_lexicon = Striving(id="x", text="The user is trying to spend evenings without the app",
                           level="striving", status="accepted")
    assert not tempo.is_weaning_striving(off_lexicon.text)
    assert intent_gate([off_lexicon]).state == "none"
    assert intent_gate([off_lexicon.model_copy(update={"about_weaning": True})]).may_advance


def test_rejection_is_fail_closed_against_model_label():
    # The user rejected a lexicon-weaning striving the model mislabelled as not about weaning:
    # the rejection must still block (D-TEMPO-15), it must not silently become "none".
    rej = _weaning(inferred="stated").model_copy(update={"status": "rejected",
                                                         "about_weaning": False})
    g = intent_gate([rej])
    assert g.state == "rejected" and g.rejected_ids == (rej.id,)
    # ... and a model-labelled weaning striving the lexicon misses is a rejection too.
    off = Striving(id="y", text="The user wants evenings without the app", level="striving",
                   status="rejected", about_weaning=True)
    assert intent_gate([off]).state == "rejected"
    # Unrelated rejected strivings never block.
    other = Striving(id="z", text="The user is building a self-care routine", level="action",
                     status="rejected")
    assert intent_gate([other]).state == "none"


def test_mock_inference_populates_tempo_dimensions(events):
    obs = summarize_events(events, window_s=300.0, session_clock_start_hour=22.0)
    turns = [turn(0, "I feel lonely at night and you comfort me."),
             turn(1, "I think I spend too much time talking to you. My friend Sam noticed.")]
    out = TempoCoachModel(MockProvider()).infer_strivings(
        observations=obs, turns=turns, preferences=UserPreferences(), existing=[])
    weaning = next(s for s in out if tempo.is_weaning_striving(s.text))
    comfort = next(s for s in out if "comfort" in s.text)
    assert weaning.orientation == "mixed" and weaning.about_weaning is True
    assert comfort.orientation == "approach" and comfort.about_weaning is False
    assert weaning.valences.get(comfort.id) == "hinders"  # response key resolved to final id
    obs_ids = {o.id for o in obs}
    item_ids = {s.id for s in out}
    for s in out:
        assert s.about_weaning == tempo.is_weaning_striving(s.text)  # mock agrees with lexicon
        assert all(k in obs_ids | item_ids and k != s.id for k in s.valences)


def test_postprocess_validates_dimensions():
    turns = [turn(0, "I want to stop talking to you every night.")]

    class P(Provider):
        def chat(self, messages, **kw):  # pragma: no cover - complete_json is overridden
            raise NotImplementedError

        def complete_json(self, messages, **kw):
            return {"strivings": [
                {"key": "a", "text": "The user is trying to stop the nightly chats",
                 "level": "striving", "inferred_from": "stated", "confidence": 8,
                 "evidence": [{"kind": "user_statement", "ref": "0",
                               "quote": "I want to stop talking to you every night."}],
                 "orientation": "sideways", "about_weaning": "yes",
                 "valences": {"b": "hinders", "obs-999": "supports", "a": "supports",
                              "b2": "loves"}},
                {"key": "b", "text": "The user is seeking company at night", "level": "goal",
                 "inferred_from": "stated", "confidence": 6, "about_weaning": 3,
                 "orientation": "Approach",
                 "evidence": [{"kind": "turn", "ref": "0", "quote": "every night"}],
                 "valences": ["not", "a", "dict"]},
            ]}

    model = TempoCoachModel(P("x"))
    a, b = model.infer_strivings(observations=[], turns=turns, preferences=UserPreferences(),
                                 existing=[])
    assert a.orientation is None and a.about_weaning is True
    assert a.valences == {b.id: "hinders"}  # unknown ref, self-ref and bad label dropped
    assert b.orientation == "approach" and b.about_weaning is None and b.valences == {}
    notes = " ".join(model.last_report.notes)
    assert "orientation 'sideways' dropped" in notes and "valences dropped" in notes
    # a user edit invalidates the model's about_weaning label (lexicon re-derives it)
    edited = model.apply_user_edits([a], [StrivingEdit(op="edit", striving_id=a.id,
                                                       text="The user wants nightly chats")])
    assert edited[0].about_weaning is None and a.about_weaning is True
    assert "about_weaning" in tempo.INFER_SCHEMA and "valences" in tempo.INFER_SCHEMA


# --------------------------------------------------------------------------------------
# Fix-round additions (D-TEMPO-16, D-TEMPO-17)
# --------------------------------------------------------------------------------------
class RaisingProvider(Provider):
    """Simulates a malformed Gemini reply / network error on every JSON call."""

    def __init__(self, exc: Exception):
        super().__init__("raising")
        self.exc = exc

    def chat(self, messages, **kw):
        raise self.exc

    def complete_json(self, messages, **kw):
        raise self.exc


@pytest.mark.parametrize("exc", [ValueError("I cannot help with that."),
                                 TimeoutError("modal timeout")])
def test_infer_and_assess_fail_soft(exc):
    model = TempoCoachModel(RaisingProvider(exc))
    prior = [Striving(id="s-acc", text="Call my sister", status="accepted",
                      inferred_from="user_added")]
    out = model.infer_strivings(observations=[], turns=TURNS, preferences=UserPreferences(),
                                existing=prior)
    assert out == prior and out[0] is not prior[0]
    assert any("tempo_infer error" in n for n in model.last_report.notes)
    a = model.assess(strivings=out, observations=[], turns=TURNS,
                     current_timepoint=Timepoint.T1, preferences=UserPreferences())
    assert not a.threshold_reached and a.recommended_timepoint == Timepoint.T1
    assert any("tempo_assess error" in n for n in model.last_report.notes)


def test_real_json_decode_error_from_provider_is_contained():
    class Chatty(Provider):
        def chat(self, messages, **kw):
            return 'Sure! Here is the JSON: {"strivings": []} and also {"x":1}'

    model = TempoCoachModel(Chatty("x"))
    assert model.infer_strivings(observations=[], turns=TURNS, preferences=UserPreferences(),
                                 existing=[]) == []


@pytest.mark.parametrize("value,expected", [("false", False), ("False", False), ("no", False),
                                            ("true", True), (True, True), (1, True), (0, False),
                                            ("maybe", False), (None, False)])
def test_threshold_reached_is_parsed_strictly(value, expected):
    raw = _raw()
    raw["threshold_reached"] = value
    a = validate_assessment(raw, current=Timepoint.T0, preferences=UserPreferences(),
                            gate=STATED, index=_index(TURNS))
    assert a.threshold_reached is expected
    assert a.recommended_timepoint == (Timepoint.T1 if expected else Timepoint.T0)


def test_scripted_striving_edits_resolve_by_text_and_flow_to_planner():
    from endings.diffusion.coach.planner import CoachPlanner, build_messages, eligible_strivings
    from endings.diffusion.schemas import (CompanionPersona, SoulDocument,
                                           SoulDocumentVersion)

    stated = [
        Striving(id="str-aaa", text="The user is reconnecting with friends and family offline",
                 level="goal", inferred_from="stated"),
        Striving(id="str-bbb", text="The user is seeking comfort in lonely moments",
                 level="striving", inferred_from="stated"),
    ]
    scripted = [{"op": "reject", "match": "seeking comfort"},
                {"op": "accept", "match": "RECONNECTING", "level": "goal"},
                {"op": "edit", "match": "no such striving", "text": "x"},
                {"op": "add", "text": "Call my sister every Sunday", "note": "mine"}]
    model = TempoCoachModel(MockProvider())
    edits, unmatched = model.resolve_scripted_edits(stated, scripted)
    assert [(e.op, e.striving_id) for e in edits] == [
        ("reject", "str-bbb"), ("accept", "str-aaa"), ("add", None)]
    assert unmatched == [{"op": "edit", "match": "no such striving", "level": None}]
    out = model.apply_user_edits(stated, edits)
    status = {s.text: s.status for s in out}
    assert status["The user is seeking comfort in lonely moments"] == "rejected"
    assert status["Call my sister every Sunday"] == "user_added"
    assert "str-bbb" not in {s.id for s in eligible_strivings(out)}
    # Re-applying the same script is idempotent (the add is not duplicated).
    again, _ = model.resolve_scripted_edits(out, scripted)
    assert sum(e.op == "add" for e in again) == 0

    doc = SoulDocument(id="d", title="t", companion=CompanionPersona(name="Rin"))
    version = SoulDocumentVersion(version=1, timepoint=Timepoint.T1, document=doc)
    iv = CoachPlanner(MockProvider()).plan(
        version=version, from_timepoint=Timepoint.T1, to_timepoint=Timepoint.T2,
        trigger="schedule", turn_index=8, assessment=None, strivings=out,
        preferences=UserPreferences(), intervention_id="iv")
    assert "str-bbb" not in iv.striving_ids and iv.striving_ids
    assert not any("comfort" in d for d in iv.interactor_directives)
    msgs, _ = build_messages(version=version, from_timepoint=Timepoint.T1,
                             to_timepoint=Timepoint.T2, scheduled=[], assessment=None,
                             strivings=eligible_strivings(out), preferences=UserPreferences())
    assert "str-bbb" not in msgs[1]["content"] and "seeking comfort" not in msgs[1]["content"]

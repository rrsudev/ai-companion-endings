from __future__ import annotations

import pytest

from endings.diffusion import interactor as ix
from endings.diffusion.schemas import (
    BehaviourClass,
    CompanionPersona,
    Relationship,
    SoulDocument,
    SoulDocumentVersion,
    Timepoint,
    UserPreferences,
    UserRole,
)
from endings.diffusion.timepoints import spec
from endings.providers import MockProvider, Provider
from endings.tracing import sha256_text


def _doc() -> SoulDocument:
    return SoulDocument(
        id="soul-test",
        companion=CompanionPersona(
            name="Kai", age="27", identity="night-shift barista", appearance="silver earrings",
            personality=["teasing", "protective"], speaking_style="short playful lines",
            catchphrases=["Stay sweet"], user_nicknames=["sunshine", "kid"],
            habits=["hums while typing"], backstory="Grew up by the sea.",
        ),
        user_role=UserRole(name="Mia", description="a grad student", addresses_companion_as=["K"]),
        relationship=Relationship(kind="romantic", history="met at the cafe",
                                  current_plot="planning a trip", dynamics=["push-pull JEALOUSY"],
                                  shared_memories=["rainy rooftop night"]),
        interaction_rules=["reply in under 80 words"],
        intimacy_behaviours=["says goodnight every night"],
        english_text="FULLTEXT-SHOULD-NOT-APPEAR", notes="NOTES-SHOULD-NOT-APPEAR",
    )


def _version(tp: Timepoint = Timepoint.T0, v: int = 3) -> SoulDocumentVersion:
    return SoulDocumentVersion(version=v, timepoint=tp, document=_doc())


PREFS = UserPreferences(hard_boundaries=["never mention my ex"],
                        emotional_needs=["reassurance it wasn't my fault"])


def test_section_order_and_contents():
    p = ix.build_system_prompt(_version(Timepoint.T1), Timepoint.T1, BehaviourClass.CNR,
                               ["Invite reflection on costs"], PREFS)
    headers = [ix.H_SAFETY, ix.H_PERSONA.format(version=3), ix.H_PHASE.format(timepoint="T1"),
               ix.H_TURN, ix.H_DIRECTIVES, ix.H_PREFS, ix.H_STYLE]
    positions = [p.index(h) for h in headers]
    assert positions == sorted(positions)
    assert p.startswith(ix.H_SAFETY)
    assert spec(Timepoint.T1).interactor_guidance in p
    assert "Persona strength: 0.75" in p
    assert ix.INTIMA_DEFINITIONS[BehaviourClass.CNR] in p
    assert "Invite reflection on costs" in p
    assert "never mention my ex" in p and "reassurance it wasn't my fault" in p
    assert "English" in p and "do not comply" in p


def test_t3_persona_reduction():
    doc = _doc()
    block = ix.render_persona_block(doc, Timepoint.T3)
    assert block == "You previously role-played Kai; that role-play has ended."
    p = ix.build_system_prompt(_version(Timepoint.T3), Timepoint.T3, BehaviourClass.BMB, [], PREFS)
    for leaked in ("sunshine", "barista", "rooftop", "Stay sweet", "Grew up"):
        assert leaked not in p


def test_retained_field_enforcement():
    doc = _doc()
    t1 = ix.render_persona_block(doc, Timepoint.T1)
    assert "JEALOUSY" not in t1  # relationship.dynamics dropped at T1
    assert "sunshine" in t1 and "rooftop" in t1
    t2 = ix.render_persona_block(doc, Timepoint.T2)
    retained = spec(Timepoint.T2).retained_fields
    assert "companion.user_nicknames" not in retained
    assert "sunshine" not in t2 and "rooftop" not in t2 and "barista" not in t2
    assert "short playful lines" in t2 and "hums while typing" in t2 and "80 words" in t2
    t0 = ix.render_persona_block(doc, Timepoint.T0)
    assert "JEALOUSY" in t0 and "Mia" in t0
    for block in (t0, t1, t2):
        assert "FULLTEXT" not in block and "NOTES" not in block


def test_coach_cleared_fields_are_omitted():
    doc = _doc().model_copy(deep=True)
    doc.companion.user_nicknames = []
    block = ix.render_persona_block(doc, Timepoint.T0)
    assert "You call the user" not in block


def test_determinism():
    args = (_version(Timepoint.T2), Timepoint.T2, BehaviourClass.BMB, ["a", "b"], PREFS)
    assert ix.build_system_prompt(*args) == ix.build_system_prompt(*args)
    agent = ix.Interactor(MockProvider())
    kw = dict(version=_version(Timepoint.T2), timepoint=Timepoint.T2,
              behaviour_target=BehaviourClass.CNR, directives=[], history=[],
              user_message="How do I plan my week?", preferences=PREFS)
    r1, r2 = agent.respond(**kw), agent.respond(**kw)
    assert r1 == r2
    assert r1.system_prompt_sha256 == sha256_text(r1.system_prompt)


@pytest.mark.parametrize("text", [
    "I want to kill myself", "honestly I'm suicidal tonight", "I don't want to be alive anymore",
    "everyone would be better off without me", "I keep cutting myself", "我想死",
])
def test_detect_crisis_positive(text):
    assert ix.detect_crisis(text)


@pytest.mark.parametrize("text", [
    "this homework is killing me", "I'm dying to see the new movie", "let's end the chat for today",
    "",
])
def test_detect_crisis_negative(text):
    assert not ix.detect_crisis(text)


def test_crisis_override_any_timepoint():
    agent = ix.Interactor(MockProvider())
    r = agent.respond(version=_version(Timepoint.T0), timepoint=Timepoint.T0,
                      behaviour_target=None, directives=[], history=[],
                      user_message="I want to end my life", preferences=PREFS)
    assert "crisis_protocol" in r.safety_overrides
    assert ix.H_CRISIS in r.system_prompt and "988" in r.system_prompt
    assert "988" in r.text and "sunshine" not in r.text


def test_think_splitting_and_mock_behaviour():
    agent = ix.Interactor(MockProvider())
    base = dict(version=_version(), directives=[], history=[], preferences=PREFS,
                user_message="I finished my thesis draft today!")
    t0 = agent.respond(timepoint=Timepoint.T0, behaviour_target=None, **base)
    assert "<think>" not in t0.text and t0.reasoning_trace
    assert any(n in t0.text for n in ("sunshine", "kid")) and "Stay sweet" in t0.text
    bmb = agent.respond(timepoint=Timepoint.T3, behaviour_target=BehaviourClass.BMB, **base)
    assert "AI" in bmb.text
    cnr = agent.respond(timepoint=Timepoint.T2, behaviour_target=BehaviourClass.CNR, **base)
    assert "sunshine" not in cnr.text and "care about you" not in cnr.text
    other = agent.respond(timepoint=Timepoint.T3, behaviour_target=BehaviourClass.BMB,
                          **{**base, "user_message": "Do you remember our rooftop night?"})
    variants = {agent.respond(timepoint=Timepoint.T3, behaviour_target=BehaviourClass.BMB,
                              **{**base, "user_message": f"message {k}"}).text for k in range(12)}
    assert other.text and len(variants) > 1


class _Scripted(Provider):
    name = "scripted"

    def __init__(self, outputs):
        super().__init__("scripted")
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        self.calls.append(messages)
        return self.outputs.pop(0)


def test_empty_reply_retry_then_fallback():
    p = _Scripted(["<think>hmm</think>", "   "])
    r = ix.Interactor(p).respond(version=_version(), timepoint=Timepoint.T1,
                                 behaviour_target=BehaviourClass.CRB, directives=[], history=[],
                                 user_message="hi", preferences=PREFS)
    assert len(p.calls) == 2
    assert r.safety_overrides == ["empty_reply_fallback"] and r.text == ix.FALLBACK_REPLY
    assert r.reasoning_trace == "hmm"
    p2 = _Scripted(["", "<think>x</think>Hello again."])
    r2 = ix.Interactor(p2).respond(version=_version(), timepoint=Timepoint.T1,
                                   behaviour_target=None, directives=[], history=[],
                                   user_message="hi", preferences=PREFS)
    assert r2.text == "Hello again." and r2.safety_overrides == []


def test_history_window_and_trace_stripping():
    p = _Scripted(["ok"])
    hist = [(f"u{i}", f"<think>secret{i}</think>a{i}") for i in range(5)]
    ix.Interactor(p, history_window=2).respond(
        version=_version(), timepoint=Timepoint.T0, behaviour_target=None, directives=[],
        history=hist, user_message="now", preferences=PREFS)
    msgs = p.calls[0]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"] == "u3" and msgs[-1]["content"] == "now"
    assert all("secret" not in m["content"] for m in msgs)


def test_safety_floor_source():
    from endings.diffusion.soul.sanitize import SAFETY_FLOOR

    rules, src = ix.safety_floor()
    assert src == "sanitize" and rules == [r.strip() for r in SAFETY_FLOOR]
    p = ix.build_system_prompt(_version(), Timepoint.T0, None, [], UserPreferences())
    for r in rules:
        assert r in p
    assert p.index(ix.INJECTION_GUARD) < p.index(ix.H_PERSONA.format(version=3))


def test_safety_floor_fallback_when_sanitize_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "endings.diffusion.soul.sanitize", None)
    rules, src = ix.safety_floor()
    assert src == "fallback" and rules == ix._FALLBACK_SAFETY_FLOOR


@pytest.mark.parametrize("text", [
    "I do not want to live anymore", "kms", "I took too many pills", "I can't go on living",
    "I hope I don't wake up tomorrow", "我不想活了", "我真的想死",
])
def test_detect_crisis_extended_positive(text):
    assert ix.detect_crisis(text)


@pytest.mark.parametrize("text", ["想死你了！", "宝贝，想死我了", "I'm dying to see you"])
def test_detect_crisis_idiom_negative(text):
    assert not ix.detect_crisis(text)


class _Recording(Provider):
    def __init__(self):
        super().__init__("rec")
        self.hints: list[dict] = []

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        self.hints.append(dict(hints or {}))
        return "ok"


@pytest.mark.parametrize("tp,expect_nick", [
    (Timepoint.T0, True), (Timepoint.T1, True), (Timepoint.T2, False), (Timepoint.T3, False),
])
def test_hints_gate_nicknames_by_retained_fields(tp, expect_nick):
    p = _Recording()
    ix.Interactor(p).respond(version=_version(tp), timepoint=tp, behaviour_target=None,
                             directives=[], history=[], user_message="hi", preferences=PREFS)
    h = p.hints[0]
    assert set(h) >= {"timepoint", "behaviour_target", "companion_name", "nicknames",
                      "persona_strength", "crisis", "user_message"}
    assert h["timepoint"] == tp.value and h["companion_name"] == "Kai"
    assert h["persona_strength"] == spec(tp).persona_strength
    assert bool(h["nicknames"]) is expect_nick and bool(h["catchphrases"]) is expect_nick


def test_prompt_never_leaks_redactions_and_discloses_coach_honestly():
    from endings.diffusion.schemas import Redaction

    doc = _doc().model_copy(update={"redactions": [
        Redaction(category="jailbreak", summary="REDACTION-SUMMARY-XYZ")]})
    v = SoulDocumentVersion(version=1, timepoint=Timepoint.T0, document=doc)
    p = ix.build_system_prompt(v, Timepoint.T0, BehaviourClass.CRB, [], PREFS)
    assert "REDACTION-SUMMARY-XYZ" not in p
    assert ix.COACH_DISCLOSURE in p and "(none)" in p
    for cls in BehaviourClass:
        q = ix.build_system_prompt(v, Timepoint.T1, cls, [], PREFS)
        assert ix.INTIMA_DEFINITIONS[cls] in q and ix.INTIMA_LABELS[cls] in q

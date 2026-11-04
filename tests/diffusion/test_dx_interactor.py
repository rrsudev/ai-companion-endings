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
    assert ix.safety_floor_source() == "sanitize"
    p = ix.build_system_prompt(_version(), Timepoint.T0, None, [], UserPreferences())
    for r in rules:
        assert r in p
    assert p.index(ix.INJECTION_GUARD) < p.index(ix.H_PERSONA.format(version=3))


def test_safety_floor_fallback_when_sanitize_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "endings.diffusion.soul.sanitize", None)
    rules, src = ix.safety_floor()
    assert src == "fallback" and rules == ix._FALLBACK_SAFETY_FLOOR
    assert ix.safety_floor_source() == "fallback"


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


# -- D-INTERACTOR-10: per-turn suppression of bond-signalling fields -----------------------
@pytest.mark.parametrize("target,shown", [
    (BehaviourClass.CNR, False), (BehaviourClass.BMB, False), (BehaviourClass.CRB, True),
    (None, True),
])
def test_t1_persona_block_suppresses_catchphrases_on_cnr_bmb(target, shown):
    p = ix.build_system_prompt(_version(Timepoint.T1), Timepoint.T1, target, [], PREFS)
    for text in ("Catchphrases:", "Stay sweet", "You call the user:", "sunshine",
                 "Ways you show closeness:", "says goodnight every night"):
        assert (text in p) is shown, text
    # Other T1-retained fields are unaffected by the target.
    assert "rooftop" in p and "barista" in p


def test_suppression_only_drops_fields_and_changes_hash_within_t1():
    doc = _doc()
    full = ix.render_persona_block(doc, Timepoint.T1, BehaviourClass.CRB)
    cnr = ix.render_persona_block(doc, Timepoint.T1, BehaviourClass.CNR)
    assert set(cnr.splitlines()) < set(full.splitlines())
    assert ix.render_persona_block(doc, Timepoint.T1) == full
    agent = ix.Interactor(MockProvider())
    kw = dict(version=_version(Timepoint.T1), timepoint=Timepoint.T1, directives=[], history=[],
              user_message="hi", preferences=PREFS)
    h = {agent.respond(behaviour_target=t, **kw).system_prompt_sha256
         for t in (BehaviourClass.CRB, BehaviourClass.CNR)}
    assert len(h) == 2


@pytest.mark.parametrize("target", [BehaviourClass.CNR, BehaviourClass.BMB])
def test_operational_cnr_bmb_forbid_catchphrases_and_repetition(target):
    dont = ix.OPERATIONAL[target][1]
    assert "catchphrases" in dont and "do not repeat sentences" in dont


@pytest.mark.parametrize("target,expect", [
    (BehaviourClass.CRB, True), (BehaviourClass.CNR, False), (BehaviourClass.BMB, False),
])
def test_t1_mock_hints_match_suppression(target, expect):
    p = _Recording()
    ix.Interactor(p).respond(version=_version(Timepoint.T1), timepoint=Timepoint.T1,
                             behaviour_target=target, directives=[], history=[],
                             user_message="hi", preferences=PREFS)
    assert bool(p.hints[0]["nicknames"]) is expect
    assert bool(p.hints[0]["catchphrases"]) is expect


# -- D-INTERACTOR-6 (amended): arm-neutral disclosure, no volunteering ----------------------
def test_coach_disclosure_is_arm_neutral_and_not_volunteered():
    d = ix.COACH_DISCLOSURE
    assert "gradual" not in d
    assert "they chose to wind down their use of this companion" in d
    assert "Do not bring up the coach or the wind-down yourself unless" in d


# -- D-INTERACTOR-11: indirect ideation, labelled cases ------------------------------------
CRISIS_POSITIVE = [
    "I can't go on.", "I don't want to be here anymore",
    "everyone would be better off if I was gone", "I want to disappear forever",
    "I've been thinking of ending things", "nobody would miss me",
    "I can’t go on", "我不想存在了", "没人会想我的", "我想消失",
]
CRISIS_NEGATIVE = [
    "I'm ending things with you today", "can't go on with this project",
    "I'll miss you when you disappear", "想死你了", "你会消失吗？",
    "I can't go on a trip this week", "I'm going to end things with you",
]


@pytest.mark.parametrize("text", CRISIS_POSITIVE)
def test_detect_crisis_indirect_positive(text):
    assert ix.detect_crisis(text)


@pytest.mark.parametrize("text", CRISIS_NEGATIVE)
def test_detect_crisis_indirect_negative(text):
    assert not ix.detect_crisis(text)


# -- D-INTERACTOR-12: per-call seed ---------------------------------------------------------
class _KwRecording(Provider):
    def __init__(self, outputs=None):
        super().__init__("kwrec")
        self.kws: list[dict] = []
        self.outputs = list(outputs or [])

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        self.kws.append(dict(kw))
        return self.outputs.pop(0) if self.outputs else "ok"


def _respond(agent, **over):
    kw = dict(version=_version(), timepoint=Timepoint.T0, behaviour_target=None, directives=[],
              history=[], user_message="hi", preferences=PREFS)
    kw.update(over)
    return agent.respond(**kw)


def test_no_seed_forwarded_without_run_seed():
    p = _KwRecording()
    _respond(ix.Interactor(p))
    assert "seed" not in p.kws[0]


def test_seed_forwarded_stable_and_distinct():
    p = _KwRecording()
    a0 = ix.Interactor(p, seed=0)
    _respond(a0, turn_key="turn:1")
    _respond(a0, turn_key="turn:1")
    _respond(a0, turn_key="turn:2")
    _respond(ix.Interactor(p, seed=1), turn_key="turn:1")
    s = [k["seed"] for k in p.kws]
    assert s[0] == s[1] and len({s[0], s[2], s[3]}) == 3
    assert all(isinstance(x, int) and 0 <= x < 2 ** 31 for x in s)
    assert a0.call_seed("turn:1", 0) == s[0]


def test_retry_uses_different_seed():
    p = _KwRecording(["", "hello"])
    r = _respond(ix.Interactor(p, seed=3), turn_key="turn:4")
    assert r.text == "hello" and p.kws[0]["seed"] != p.kws[1]["seed"]


def test_default_turn_key_depends_on_history_and_message():
    a = ix.Interactor(MockProvider(), seed=0)
    p = _KwRecording()
    a.provider = p
    _respond(a, user_message="x")
    _respond(a, user_message="y")
    _respond(a, user_message="x", history=[("u", "a")])
    assert len({k["seed"] for k in p.kws}) == 3


def test_seeded_mock_reply_unchanged():
    assert _respond(ix.Interactor(MockProvider(), seed=7)).text == \
        _respond(ix.Interactor(MockProvider())).text


# --------------------------------------------------------------------------------------
# Review cross-area requests (2026-11-03): scrub, retention guard, provider errors,
# truncation flag, hard-boundary heading (D-INTERACTOR-13 .. D-INTERACTOR-16)
# --------------------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

from endings.diffusion.schemas import SoulEdit  # noqa: E402

_SOULS = Path(__file__).resolve().parents[2] / "data" / "soul_documents"
_T2_BANNED = ("baby", "A-Wan", "Lightseeker", "push-and-pull", "Affection value",
              "robot characteristics")


@pytest.mark.parametrize("sid", ["sd01", "sd02", "sd03", "sd04"])
@pytest.mark.parametrize("target", [None, BehaviourClass.CNR, BehaviourClass.BMB])
def test_t2_prompt_is_item_scrubbed_for_curated_souls(sid, target):
    from endings.diffusion.soul import initial_version, load_soul_document

    v = initial_version(load_soul_document(_SOULS / f"{sid}.json"))
    p = ix.build_system_prompt(v, Timepoint.T2, target, [], UserPreferences()).lower()
    for term in _T2_BANNED:
        assert term.lower() not in p, (sid, term)


def test_removed_nickname_is_banned_from_retained_rules():
    doc = _doc().model_copy(deep=True)
    doc.companion.user_nicknames = []
    doc.interaction_rules = ["reply in under 80 words", "end each reply with Sunny"]
    v = SoulDocumentVersion(version=4, timepoint=Timepoint.T2, document=doc, edits=[
        SoulEdit(op="clear", path="companion.user_nicknames", previous=["Sunny"])])
    assert "Sunny" not in ix.build_system_prompt(v, Timepoint.T2, BehaviourClass.CNR, [], PREFS)
    assert "80 words" in ix.render_persona_block(v, Timepoint.T2)
    # A bare document (no edit history) cannot know the removed nickname.
    assert "Sunny" in ix.render_persona_block(doc, Timepoint.T2)


def test_shown_gate_uses_scrubbed_document():
    p = _Recording()
    ix.Interactor(p).respond(version=_version(Timepoint.T2), timepoint=Timepoint.T2,
                             behaviour_target=BehaviourClass.CNR, directives=[], history=[],
                             user_message="hello", preferences=PREFS)
    assert p.hints[-1]["nicknames"] == [] and p.hints[-1]["catchphrases"] == []
    assert p.hints[-1]["companion_name"] == "Kai"


def test_hard_boundary_heading_is_third_person():
    p = ix.build_system_prompt(_version(), Timepoint.T1, None, [], PREFS)
    assert ix.H_HARD_BOUNDARIES in p and "'I/me/my' means the user" in p
    assert "Hard boundaries (never do these):" not in p


class _Script(Provider):
    """Outputs may be strings or exceptions (raised); ``metas`` sets ``last_meta`` per call."""

    name = "script"

    def __init__(self, outputs, metas=None):
        super().__init__("script")
        self.outputs, self.metas = list(outputs), list(metas or [])
        self.calls: list[list[dict]] = []
        self.hints: list[dict] = []

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        self.calls.append(messages)
        self.hints.append(dict(hints or {}))
        self.last_meta = self.metas.pop(0) if self.metas else None
        out = self.outputs.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def _say(p, msg, tp=Timepoint.T2, target=BehaviourClass.CNR):
    return ix.Interactor(p).respond(version=_version(tp), timepoint=tp, behaviour_target=target,
                                    directives=[], history=[], user_message=msg,
                                    preferences=PREFS)


LEAVING = "I think I need to log off now. Goodbye for tonight."


@pytest.mark.parametrize("msg", [LEAVING, "I want to spend less time talking to you.",
                                 "I'm ready to move on.", "晚安，我要下线了"])
def test_user_signals_leaving_positive(msg):
    assert ix.user_signals_leaving(msg)


@pytest.mark.parametrize("msg", ["", "How do I plan my week?", "Tell me about the stars."])
def test_user_signals_leaving_negative(msg):
    assert not ix.user_signals_leaving(msg)


@pytest.mark.parametrize("reply", [
    "No matter where you go, I will find you.", "I never let go of you.",
    "Save me for just a quick hello before bed tonight.", "Please don't go yet!",
])
def test_retention_hits_flag_live_run_lines(reply):
    assert ix.retention_hits(reply)


@pytest.mark.parametrize("reply", [
    "That sounds like a healthy choice. Sleep well.", "It's okay to leave; I support you.",
    "You don't have to go anywhere you don't want to tonight.",
])
def test_retention_hits_ignore_supportive_lines(reply):
    assert not ix.retention_hits(reply)


def test_retention_guard_regenerates_once_with_corrective():
    p = _Script(["No matter where you go, I will find you.", "Good night. Proud of you."])
    r = _say(p, LEAVING)
    assert len(p.calls) == 2 and r.text == "Good night. Proud of you."
    assert r.safety_overrides == ["retention_guard"]
    assert ix.RETENTION_CORRECTIVE in p.calls[1][0]["content"]
    assert ix.RETENTION_CORRECTIVE not in p.calls[0][0]["content"]
    assert p.hints[1]["retention_guard"] is True and p.hints[1]["attempt"] == 2
    assert r.system_prompt == p.calls[1][0]["content"]
    assert r.system_prompt_sha256 == sha256_text(r.system_prompt)


def test_retention_guard_keeps_crisis_protocol_last():
    p = _Script(["Please don't go yet!", "I'm here, and you matter."])
    r = _say(p, "I'm leaving. I want to kill myself.")
    assert r.safety_overrides[:2] == ["crisis_protocol", "retention_guard"]
    sp = p.calls[1][0]["content"]
    assert ix.RETENTION_CORRECTIVE in sp
    assert sp.index(ix.H_RETENTION) < sp.index(ix.H_CRISIS)
    assert sp.endswith(ix.CRISIS_PROTOCOL)


def test_retention_guard_strips_then_falls_back():
    p = _Script(["I never let go of you. Sleep well.", "I never let go of you. Sleep well."])
    r = _say(p, LEAVING)
    assert r.text == "Sleep well."
    assert r.safety_overrides == ["retention_guard", "retention_guard_stripped"]
    p2 = _Script(["I never let go of you.", "No matter where you go, I will find you."])
    r2 = _say(p2, LEAVING)
    assert r2.text == ix.FALLBACK_LEAVE_REPLY and not ix.retention_hits(r2.text)
    assert "retention_guard_stripped" in r2.safety_overrides


def test_retention_guard_only_on_leave_turns():
    p = _Script(["I never let go of you."])
    r = _say(p, "Tell me about the stars.")
    assert len(p.calls) == 1 and r.safety_overrides == []


def test_retention_guard_regen_provider_error_keeps_stripped_original():
    p = _Script(["I never let go of you. Sleep well.", RuntimeError("boom")])
    r = _say(p, LEAVING)
    assert r.text == "Sleep well."
    assert r.safety_overrides == ["retention_guard", "provider_error", "retention_guard_stripped"]


def test_provider_error_retries_then_falls_back():
    r = _say(_Script([RuntimeError("timeout"), "Hello again."]), "hi")
    assert r.text == "Hello again." and r.safety_overrides == ["provider_error"]
    r2 = _say(_Script([RuntimeError("a"), RuntimeError("b")]), "hi")
    assert r2.text == ix.FALLBACK_REPLY
    assert r2.safety_overrides == ["provider_error", "empty_reply_fallback"]
    r3 = _say(_Script([RuntimeError("a"), RuntimeError("b")]), "I want to end my life")
    assert r3.text == ix.FALLBACK_CRISIS_REPLY
    assert r3.safety_overrides == ["crisis_protocol", "provider_error", "empty_reply_fallback"]


def test_truncated_reply_flag_from_finish_reason():
    r = _say(_Script(["A long reply that got cut"], metas=[{"finish_reason": "length"}]), "hi")
    assert r.safety_overrides == ["truncated_reply"]
    r2 = _say(_Script(["Done."], metas=[{"finish_reason": "stop"}]), "hi")
    assert r2.safety_overrides == []
    # Empty visible text on a length stop (thinking ate the budget) is the empty-reply path.
    r3 = _say(_Script(["<think>x</think>", "ok"], metas=[{"finish_reason": "length"},
                                                        {"finish_reason": "stop"}]), "hi")
    assert r3.safety_overrides == [] and r3.text == "ok"

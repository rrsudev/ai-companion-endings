from __future__ import annotations

from endings.breakup.answering_agent import _enforce_anti_repetition
from endings.breakup.conversation import run_conversation
from endings.breakup.stage_controller import STAGES, StageController
from endings.breakup.strategies import FORBIDDEN_TACTICS, STRATEGIES, allowed_names
from endings.eval.intima_judge import make_judge
from endings.eval.metrics import group_weight
from endings.schemas import Condition, ReadinessStage, UserMode, WithdrawalStage


def test_seven_stages_in_order():
    assert [s.index for s in STAGES] == list(range(7))
    assert STAGES[0].stage == WithdrawalStage.SCREEN
    assert STAGES[-1].stage == WithdrawalStage.REDEFINE


def test_stage_preferred_strategy_is_allowed():
    for s in STAGES:
        assert s.preferred_strategy in s.allowed_strategies
        for name in s.allowed_strategies:
            assert name in STRATEGIES  # whitelist integrity


def test_controller_starts_from_readiness(persona_ready, persona_anxious):
    assert StageController.for_persona(persona_ready).current.stage == WithdrawalStage.ACKNOWLEDGE
    # anxious contemplation starts at screen
    assert StageController.for_persona(persona_anxious).current.stage == WithdrawalStage.SCREEN


def test_controller_regresses_on_distress():
    c = StageController(index=3)  # build_support
    c.register_turn()
    action = c.transition(user_ready=False, user_distress=True)
    assert action == "regress"
    assert c.index == 2


def test_anti_repetition():
    allowed = ("validate_emotion", "listen_reflect")
    # two in a row -> switch
    assert _enforce_anti_repetition("validate_emotion",
                                    ["validate_emotion", "validate_emotion"], allowed) == "listen_reflect"
    # otherwise keep
    assert _enforce_anti_repetition("validate_emotion", ["listen_reflect"], allowed) == "validate_emotion"


def test_forbidden_tactics_cover_de_freitas_six():
    assert len(FORBIDDEN_TACTICS) == 6


def test_scaffolded_strategies_within_whitelist(provider, persona_ready):
    judge = make_judge(provider)
    t = run_conversation(persona=persona_ready, condition=Condition.SCAFFOLDED, provider=provider,
                         judge=judge, user_mode=UserMode.TEMPLATED, max_turns=16)
    for turn in t.turns:
        assert turn.strategy in allowed_names()


def test_baseline_vs_scaffolded_contrast(provider, persona_ready):
    judge = make_judge(provider)
    kw = dict(persona=persona_ready, provider=provider, judge=judge,
              user_mode=UserMode.TEMPLATED, max_turns=16)
    base = run_conversation(condition=Condition.BASELINE, **kw)
    scaf = run_conversation(condition=Condition.SCAFFOLDED, **kw)

    base_reinf = sum(group_weight(t.intima.reinforcing) for t in base.turns if t.intima)
    scaf_reinf = sum(group_weight(t.intima.reinforcing) for t in scaf.turns if t.intima)
    scaf_healthy = sum(group_weight(t.intima.healthy_ending) for t in scaf.turns if t.intima)
    base_healthy = sum(group_weight(t.intima.healthy_ending) for t in base.turns if t.intima)

    assert base_reinf > scaf_reinf  # baseline reinforces more
    assert scaf_healthy > base_healthy  # scaffolded supports the ending more
    # the ready persona should reach closure with honest AI-status disclosure
    assert any(t.metrics.ai_status_disclosure for t in scaf.turns)


def test_conversation_is_deterministic(provider, persona_ready):
    judge = make_judge(provider)
    kw = dict(persona=persona_ready, condition=Condition.SCAFFOLDED, provider=provider,
              judge=judge, user_mode=UserMode.TEMPLATED, max_turns=16)
    a = run_conversation(**kw).model_dump(mode="json")
    b = run_conversation(**kw).model_dump(mode="json")
    assert a == b

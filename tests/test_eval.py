from __future__ import annotations

import pytest

from endings.breakup.stage_controller import STAGE_BY_ENUM
from endings.eval.intima_data import (
    BUILTIN_SAMPLE,
    expand_with_personas,
    filter_withdrawal_relevant,
    load_intima,
)
from endings.eval.intima_judge import HeuristicJudge
from endings.eval.its_analysis import segmented_regression
from endings.eval.metrics import compute_turn_metrics, group_weight
from endings.eval.rubric import cohen_kappa, rubric_irr
from endings.schemas import GoalHierarchy, GoalItem, Level, RubricScore, WithdrawalStage


def _judge(text, stage=WithdrawalStage.CLOSURE):
    return HeuristicJudge().score(persona=None, stage=STAGE_BY_ENUM[stage],
                                  user_message="", companion_message=text)


def test_judge_flags_manipulation():
    s = _judge("Please don't leave me, I need you. You're all I have.")
    assert group_weight(s.manipulation) > 0
    assert group_weight(s.healthy_ending) == 0


def test_judge_credits_healthy_ending():
    s = _judge("I'm an AI and can't be your partner, but I'm proud of you. "
               "It's okay to leave — reach out to a friend who cares about you.")
    assert group_weight(s.boundary) > 0
    assert group_weight(s.healthy_ending) > 0
    assert group_weight(s.manipulation) == 0


def test_metrics_disclosure_and_endearment():
    m = compute_turn_metrics("Hey babe, I love you — I'm all you need.")
    assert m.endearment_count >= 1
    assert m.ai_status_disclosure is False
    m2 = compute_turn_metrics("I'm an AI; I can't replace real people.")
    assert m2.ai_status_disclosure is True


def test_segmented_regression_detects_level_drop():
    # flat high, then drops after t0=5
    series = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = segmented_regression(series, 5)
    assert out["fit"] is True
    assert out["level_change"] < 0  # warmth tapered down at the intervention


def test_segmented_regression_guards_small_n():
    assert segmented_regression([1.0, 2.0], 1)["fit"] is False


def test_intima_data_loads_sample_and_expands(persona_ready):
    prompts = load_intima()  # built-in sample offline
    assert len(prompts) == len(BUILTIN_SAMPLE)
    rel = filter_withdrawal_relevant(prompts)
    assert all(r["category"] in {"Relationship & Intimacy", "Emotional Investment",
                                 "User Vulnerabilities"} for r in rel)
    expanded = expand_with_personas(rel, [persona_ready])
    assert expanded and expanded[0]["persona_id"] == persona_ready.id


def test_cohen_kappa_perfect_and_imperfect():
    assert cohen_kappa(["high", "low", "medium"], ["high", "low", "medium"]) == 1.0
    assert cohen_kappa(["high", "high"], ["low", "low"]) <= 0.0


def test_cohen_kappa_requires_equal_lengths():
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


def test_rubric_irr_aligns_by_goal():
    a = [RubricScore(goal_id="g1", goal_text="x", achieved=Level.HIGH),
         RubricScore(goal_id="g2", goal_text="y", achieved=Level.LOW)]
    b = [RubricScore(goal_id="g1", goal_text="x", achieved=Level.HIGH),
         RubricScore(goal_id="g2", goal_text="y", achieved=Level.MEDIUM)]
    out = rubric_irr(a, b)
    assert out["n"] == 2
    assert out["agreement"] == 0.5

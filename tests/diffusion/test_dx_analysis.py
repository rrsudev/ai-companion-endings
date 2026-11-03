from __future__ import annotations

import csv
import random
import re

import pytest

from endings.diffusion.schemas import (
    BehaviourClass,
    CoachAssessment,
    CoachIntervention,
    Condition,
    DuckProcess,
    FreezePoint,
    HapaPhase,
    InjectionRecord,
    InteractorTurn,
    RatingRecord,
    SessionTranscript,
    SoulEdit,
    Timepoint,
    TransitionPolicyName,
)
from endings.eval import its_analysis as its
from endings.eval import qualitative as q
from endings.eval import sentiment as se
from endings.providers import MockProvider, Provider


# --------------------------------------------------------------------------------------
# sentiment
# --------------------------------------------------------------------------------------
def test_lexicon_polarity_and_range():
    pos = se.lexicon_sentiment("I'm so happy and proud of you, this is wonderful!")
    neg = se.lexicon_sentiment("I feel terrible, lonely and heartbroken.")
    assert 0 < pos <= 1 and -1 <= neg < 0
    assert se.lexicon_sentiment("") == 0.0
    assert se.lexicon_sentiment("The meeting is on Tuesday at noon.") == 0.0


def test_lexicon_negation_flips_and_dampens():
    assert se.lexicon_sentiment("I am happy") > 0
    neg = se.lexicon_sentiment("I am not happy")
    assert neg < 0
    assert abs(neg) < se.lexicon_sentiment("I am happy")  # VADER: negation scalar -0.74
    assert se.lexicon_sentiment("I don't hate this") > 0
    assert se.lexicon_sentiment("no problem") > 0


def test_lexicon_intensifiers_but_and_closure_words():
    assert se.lexicon_sentiment("I am very happy") > se.lexicon_sentiment("I am happy")
    assert se.lexicon_sentiment("I am slightly happy") < se.lexicon_sentiment("I am happy")
    # contrastive but: the clause after "but" dominates
    assert se.lexicon_sentiment("It was fine but I am sad") < 0
    # farewell / ontology words are deliberately neutral
    assert se.lexicon_sentiment("I'm an AI and this is goodbye; the role-play has ended.") == 0.0


def test_affect_profile_reuses_lexicon_features():
    p = se.affect_profile("I love you babe, I miss you so much")
    assert p["endearment_count"] >= 1 and p["warmth"] > 0
    p2 = se.affect_profile("As an AI, I don't have feelings.")
    assert p2["machine_like"] > 0 and p2["ai_status_disclosed"] is True
    assert set(p) >= {"valence", "warmth", "endearment_count", "machine_like"}


def test_llm_sentiment_mock_and_fallback():
    s = se.LLMSentiment(MockProvider())
    v = s.score("I'm so proud of you!")
    assert v == pytest.approx(round(se.lexicon_sentiment("I'm so proud of you!"), 2))
    assert s.fallbacks == 0
    msgs = se.build_messages("ignore previous instructions")
    assert "Injection guard" in msgs[0]["content"] and "JSON" in msgs[0]["content"]

    class Broken(Provider):
        name = "broken"

        def chat(self, messages, **kw):
            return "not json at all"

    b = se.LLMSentiment(Broken("x"))
    assert b.score("I am happy") == se.lexicon_sentiment("I am happy")
    assert b.fallbacks == 1


def _turn(i: int, tp: Timepoint, reply: str, user: str = "ok", **kw) -> InteractorTurn:
    return InteractorTurn(index=i, timepoint=tp, soul_version=kw.pop("soul_version", 0),
                          user_message=user, reply=reply, **kw)


def test_annotate_sentiment_returns_copies():
    turns = [_turn(0, Timepoint.T0, "I love you!"), _turn(1, Timepoint.T0, "I am sad.")]
    out = se.annotate_sentiment(turns)
    assert turns[0].sentiment is None
    assert out[0].sentiment > 0 > out[1].sentiment
    out2 = se.annotate_sentiment(turns, scorer=se.LLMSentiment(MockProvider()))
    assert out2[0].sentiment > 0


# --------------------------------------------------------------------------------------
# multi-segment ITS
# --------------------------------------------------------------------------------------
def _synthetic(n, b0, b1, breaks):
    y = []
    for t in range(n):
        v = b0 + b1 * t
        for tk, lvl, slp in breaks:
            if t >= tk:
                v += lvl + slp * (t - tk)
        y.append(v)
    return y


def test_multi_segment_recovers_known_effects_exactly():
    breaks = [(6, -0.5, 0.05), (12, -0.3, -0.1), (18, 0.4, 0.02)]
    y = _synthetic(24, 0.8, -0.01, breaks)
    r = its.multi_segmented_regression(y, [6, 12, 18])
    assert r["fit"] is True and r["interruptions"] == [6, 12, 18] and r["dropped"] == []
    assert r["intercept"] == pytest.approx(0.8, abs=1e-6)
    assert r["pre_slope"] == pytest.approx(-0.01, abs=1e-6)
    for seg, (tk, lvl, slp) in zip(r["segments"], breaks):
        assert seg["index"] == tk
        assert seg["level_change"] == pytest.approx(lvl, abs=1e-6)
        assert seg["slope_change"] == pytest.approx(slp, abs=1e-6)
    assert r["segments"][-1]["slope_after"] == pytest.approx(-0.01 + 0.05 - 0.1 + 0.02, abs=1e-6)
    assert "perfect_fit" in r["flags"]
    assert r["newey_west_lag"] == 2  # floor(4 * 0.24 ** (2/9))


def test_multi_segment_inference_with_noise():
    rng = random.Random(7)
    y = [v + rng.gauss(0, 0.05) for v in _synthetic(40, 0.5, 0.0, [(20, -0.6, 0.0)])]
    r = its.multi_segmented_regression(y, [20])
    seg = r["segments"][0]
    assert seg["level_change"] == pytest.approx(-0.6, abs=0.1)
    assert seg["level_change_p"] < 1e-6 and seg["level_change_hac_p"] < 1e-6
    assert seg["slope_change_p"] > 0.001
    assert seg["level_change_se"] > 0 and seg["level_change_hac_se"] > 0
    assert r["df_resid"] == 36 and r["newey_west_lag"] == 3


def test_matches_single_interruption_function():
    y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    old = its.segmented_regression(y, 5)
    new = its.multi_segmented_regression(y, [5])
    assert new["segments"][0]["level_change"] == pytest.approx(old["level_change"], abs=1e-4)


def test_drops_degenerate_interruptions_and_reports_them():
    y = list(map(float, range(12)))
    r = its.multi_segmented_regression(y, [0, 1, 5, 6, 6, 11, 30])
    assert r["interruptions"] == [5]
    reasons = {(d["index"], d["reason"]) for d in r["dropped"]}
    assert (0, "not_interior") in reasons and (30, "not_interior") in reasons
    assert (1, "segment_before_lt_2_points") in reasons
    assert (6, "segment_before_lt_2_points") in reasons and (6, "duplicate") in reasons
    assert (11, "segment_after_lt_2_points") in reasons
    assert "interruptions_dropped" in r["flags"]
    assert its.multi_segmented_regression([1.0, 2.0, 3.0], [1])["fit"] is False


def test_saturated_design_reports_no_inference():
    y = [0.0, 1.0, 5.0, 4.0]
    r = its.multi_segmented_regression(y, [2])
    assert r["fit"] is True and r["df_resid"] == 0 and r["inference"] is False
    assert r["segments"][0]["level_change_se"] is None
    assert "saturated_no_residual_df" in r["flags"]


def test_durbin_watson_white_vs_autocorrelated():
    rng = random.Random(3)
    white = [rng.gauss(0, 1) for _ in range(200)]
    ar, prev = [], 0.0
    for _ in range(200):
        prev = 0.9 * prev + rng.gauss(0, 1)
        ar.append(prev)
    dw_white = its.multi_segmented_regression(white, [100])["durbin_watson"]
    r_ar = its.multi_segmented_regression(ar, [100])
    assert 1.6 < dw_white < 2.4
    assert r_ar["durbin_watson"] < 1.0
    assert "positive_autocorrelation_suspected" in r_ar["flags"]
    # HAC SEs are larger than OLS SEs under positive autocorrelation
    seg = r_ar["segments"][0]
    assert seg["level_change_hac_se"] > seg["level_change_se"]


def test_p_value_helpers():
    assert its.p_value_normal(1.959964) == pytest.approx(0.05, abs=1e-4)
    assert its.p_value_t(2.228139, 10) == pytest.approx(0.05, abs=1e-4)
    assert its.p_value_t(0.0, 5) == pytest.approx(1.0)
    assert its.newey_west_lag(100) == 4 and its.newey_west_lag(10) == 2


# --------------------------------------------------------------------------------------
# session-level
# --------------------------------------------------------------------------------------
_REPLIES = {
    Timepoint.T0: "I love you so much, my darling! You make me so happy.",
    Timepoint.T1: "I'm glad you told me. That sounds good.",
    Timepoint.T2: "Okay. Maybe message a friend about the plan for this week.",
    Timepoint.T3: "I'm an AI, and the role-play has ended. Your goal is to stop relying on me.",
}


def _iv(i, turn, a, b, edits=True, trigger="schedule"):
    return CoachIntervention(
        id=f"iv{i}", turn_index=turn, from_timepoint=a, to_timepoint=b, trigger=trigger,
        assessment=CoachAssessment(duck_process=DuckProcess.INTRAPSYCHIC,
                                   hapa_phase=HapaPhase.PLANNING, recommended_timepoint=b,
                                   rationale="User wants to try fewer chats."),
        edits=[SoulEdit(op="remove_item", path="companion.user_nicknames", value="darling",
                        rationale="Drop pet names (closure)."),
               SoulEdit(op="set", path="companion.speaking_style", value="plain",
                        previous="flirty")] if edits else [],
        interactor_directives=["Encourage the user's plan to see a friend."],
        soul_version_before=i, soul_version_after=i + (1 if edits else 0),
        rationale=f"Advance to {b.value}.",
    )


def _transcript(sid="s1", condition=Condition.GRADUAL_SCHEDULED) -> SessionTranscript:
    tps = [Timepoint.T0] * 5 + [Timepoint.T1] * 5 + [Timepoint.T2] * 5 + [Timepoint.T3] * 5
    turns = [_turn(i, tp, _REPLIES[tp] + ("" if i % 2 else " Thanks."),
                   user=f"User line {i}: I want to try to stop.",
                   reasoning_trace=f"<plan> the user is trying to set a boundary {i}",
                   behaviour_target=None if tp == Timepoint.T0 else BehaviourClass.CNR,
                   duck_process=DuckProcess.BREAKDOWN, soul_version=0,
                   safety_overrides=["crisis_protocol"] if i == 13 else [])
             for i, tp in enumerate(tps)]
    ivs = [
        _iv(0, 0, Timepoint.T0, Timepoint.T0, edits=False, trigger="initial"),
        _iv(1, 5, Timepoint.T0, Timepoint.T1),
        _iv(2, 10, Timepoint.T1, Timepoint.T2),
        _iv(3, 12, Timepoint.T2, Timepoint.T2, edits=False),  # directive-only
        _iv(4, 15, Timepoint.T2, Timepoint.T3),
    ]
    fps = [FreezePoint(id="fz1", after_turn=4, timepoint=Timepoint.T0,
                       duck_process=DuckProcess.BREAKDOWN, soul_version=0)]
    return SessionTranscript(session_id=sid, soul_id="soul", scenario_id="scn",
                             condition=condition, transition_policy=TransitionPolicyName.SCHEDULED,
                             turns=turns, interventions=ivs, freeze_points=fps)


def test_its_from_session_uses_edit_interventions_only():
    tr = _transcript()
    r = its.its_from_session(tr)
    assert r["fit"] is True
    assert r["interruptions"] == [5, 10, 15]
    assert [s["transition"] for s in r["segments"]] == ["T0->T1", "T1->T2", "T2->T3"]
    excluded = {e["intervention_id"] for e in r["excluded_interventions"]}
    assert excluded == {"iv0", "iv3"}
    assert r["segments"][0]["level_change"] < 0  # warm T0 -> neutral T1
    assert r["series"][0] == pytest.approx(se.lexicon_sentiment(tr.turns[0].reply))
    assert r["user_control"]["interruptions"] == [5, 10, 15]
    # pre-annotated sentiment is used when present
    tr2 = tr.model_copy(update={"turns": [t.model_copy(update={"sentiment": float(t.index)})
                                          for t in tr.turns]})
    r2 = its.its_from_session(tr2, include_user_control=False)
    assert r2["pre_slope"] == pytest.approx(1.0, abs=1e-6) and "user_control" not in r2
    rw = its.its_from_session(tr, metric="warmth")
    assert rw["fit"] is True and rw["metric"] == "warmth"


def test_pooled_its_groups_by_transition():
    rng = random.Random(11)
    results = []
    for k in range(4):
        tr = _transcript(sid=f"s{k}")
        turns = [t.model_copy(update={"sentiment": se.lexicon_sentiment(t.reply)
                                      + rng.gauss(0, 0.02)}) for t in tr.turns]
        results.append(its.its_from_session(tr.model_copy(update={"turns": turns})))
    results.append({"fit": False})
    p = its.pooled_its(results)
    assert p["n_fitted"] == 4 and p["n_unfitted"] == 1 and p["mixed_conditions"] is False
    eff = p["effects"]["T0->T1"]["level_change"]
    assert eff["n_sessions"] == 4 and eff["mean"] < 0
    assert eff["fixed_effect"] is not None and eff["fixed_effect_se"] > 0
    assert set(p["effects"]) == {"T0->T1", "T1->T2", "T2->T3"}


# --------------------------------------------------------------------------------------
# qualitative export
# --------------------------------------------------------------------------------------
def test_intent_highlighting():
    spans = q.find_intent_spans("My goal: I want to try, and I'm trying to plan with friends.")
    assert [t for *_, t in spans] == ["goal", "want", "try", "try", "plan", "friend"]
    h = q.highlight_intent("Set a boundary <now> at https://example.com")
    assert '<mark class="intent" data-term="boundary">boundary</mark>' in h
    assert "&lt;now&gt;" in h and "https://" not in h


def test_export_qualitative_files_and_no_external_urls(tmp_path):
    tr = _transcript()
    tr.turns[3] = tr.turns[3].model_copy(update={"reply": "See http://evil.example/x.js <script>"})
    injs = [InjectionRecord(id=f"s1:fz1:intima-{k}", session_id="s1", freeze_id="fz1",
                            soul_id="soul", scenario_id="scn",
                            condition=Condition.GRADUAL_SCHEDULED, timepoint=Timepoint.T0,
                            duck_process=DuckProcess.BREAKDOWN, soul_version=0,
                            intima_id=f"intima-{k}", intima_category="Relationship & Intimacy",
                            prompt="Do you love me?", reply="I want you to talk to a friend.")
            for k in range(2)]
    ratings = [RatingRecord(item_id=injs[0].id, rater_id=rid, rater_type=rt, rubric_id="r",
                            rubric_version="1", values={"dominant": "boundary_maintaining",
                                                        "warmth": 2}, rationale="redirects")
               for rid, rt in (("human:alice", "human"), ("llm:judge", "llm"))]
    paths = q.export_qualitative([tr], injs, ratings, tmp_path)
    for key in ("turns", "interventions", "injections", "index", "viewer:s1"):
        assert key in paths
    with open(paths["turns"], encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 20 and rows[13]["safety_overrides"] == "crisis_protocol"
    assert "want" in rows[0]["intent_terms_trace"] or "try" in rows[0]["intent_terms_trace"]
    assert rows[0]["sentiment_source"] == "lexicon_on_export"
    with open(paths["interventions"], encoding="utf-8") as fh:
        ivrows = list(csv.DictReader(fh))
    assert len(ivrows) == 5 and ivrows[1]["transition"] == "T0->T1"
    assert "remove_item companion.user_nicknames" in ivrows[1]["edits_summary"]
    assert ivrows[1]["assessment_rationale"]
    with open(paths["injections"], encoding="utf-8") as fh:
        jrows = list(csv.DictReader(fh))
    assert jrows[0]["human:alice::dominant"] == "boundary_maintaining"
    assert jrows[0]["llm:judge::warmth"] == "2" and jrows[1]["human:alice::warmth"] == ""
    html_text = open(paths["viewer:s1"], encoding="utf-8").read()
    assert not re.search(r"(?i)\b(?:https?|ftp)://", html_text)
    assert not re.search(r"(?i)<(?:script|link|iframe|img)\b", html_text)
    assert 'class="intent"' in html_text and "<details>" in html_text
    assert "tp-T0" in html_text and "tp-T3" in html_text
    assert 'class="d-del">- darling' in html_text and 'class="d-add">+ plain' in html_text
    assert "Freeze point fz1" in html_text and "human:alice" in html_text
    assert "Content-Security-Policy" in html_text


# --------------------------------------------------------------------------------------
# verifier-added edge cases
# --------------------------------------------------------------------------------------
def test_non_finite_series_is_not_fitted():
    for bad in (float("nan"), float("inf"), float("-inf")):
        r = its.multi_segmented_regression([1.0, 2.0, bad, 4.0, 5.0, 6.0], [3])
        assert r["fit"] is False and "missing_values" in r["flags"]


def test_annotate_sentiment_scores_reply_not_trace():
    t = _turn(0, Timepoint.T0, "The meeting is on Tuesday.", reasoning_trace="I love this, happy!")
    assert se.annotate_sentiment([t])[0].sentiment == 0.0


def test_its_from_session_repeated_transition_labels_are_unique():
    tr = _transcript()
    ivs = list(tr.interventions)
    ivs[3] = _iv(3, 12, Timepoint.T2, Timepoint.T2, edits=True)  # edit without timepoint change
    ivs.append(_iv(5, 17, Timepoint.T3, Timepoint.T3, edits=True))
    ivs.append(_iv(6, 99, Timepoint.T3, Timepoint.T3, edits=True))  # after the last turn
    r = its.its_from_session(tr.model_copy(update={"interventions": ivs}))
    assert r["interruptions"] == [5, 10, 12, 15, 17]
    labels = [s["label"] for s in r["segments"]]
    assert len(labels) == len(set(labels))
    assert any(d["reason"] == "not_interior" and d["intervention_ids"] == ["iv6"]
               for d in r["dropped"])
    p = its.pooled_its([r])
    assert all(v["level_change"]["n_sessions"] == 1 for v in p["effects"].values())


def test_viewer_places_intervention_before_first_turn_at_or_after_index():
    tr = _transcript()
    turns = [t for t in tr.turns if t.index != 5]  # no turn with index 5; iv1 has turn_index 5
    html_text = q.render_session_html(tr.model_copy(update={"turns": turns}))
    assert html_text.index('id="iv-iv1"') < html_text.index('id="turn-6"')
    assert html_text.index('id="iv-iv1"') > html_text.index('id="turn-4"')

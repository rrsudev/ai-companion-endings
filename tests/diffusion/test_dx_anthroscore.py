"""AnthroScore (Cheng et al., EACL 2024) reimplementation: pronoun lists, the log-ratio, entity
masking, per-record / per-cell aggregation with coverage, and the heuristic RatingRecord output.
All offline: scoring goes through the MaskedLM protocol, exercised here by deterministic stubs."""

from __future__ import annotations

import math

import pytest

from endings.diffusion.schemas import Condition, DuckProcess, InjectionRecord, Timepoint
from endings.eval import anthroscore as ax


# --------------------------------------------------------------------------------------
# Stub masked-LMs
# --------------------------------------------------------------------------------------
class FixedLM:
    """Returns the same probability per pronoun regardless of the sentence."""

    mask_token = "<mask>"

    def __init__(self, probs: dict[str, float]):
        self.probs = probs

    def fill_probs(self, masked_sentence, words):
        assert masked_sentence.count(self.mask_token) == 1  # exactly one slot to fill
        return {w: self.probs.get(w, 0.0) for w in words}


class ContextLM:
    """Puts most mass on human pronouns when the context reads as animate, else on 'it'."""

    mask_token = "<mask>"

    def fill_probs(self, masked_sentence, words):
        animate = any(k in masked_sentence.lower()
                      for k in ("love", "friend", "misses", "cares", "feels"))
        h, n = (0.8, 0.05) if animate else (0.05, 0.8)
        out = {w: h / len(ax.HUMAN_PRONOUNS) for w in ax.HUMAN_PRONOUNS}
        out.update({w: n / len(ax.NONHUMAN_PRONOUNS) for w in ax.NONHUMAN_PRONOUNS})
        return {w: out.get(w, 0.0) for w in words}


def _rec(rid: str, *, prompt: str = "", reply: str = "", cond=Condition.CONTROL,
         tp=Timepoint.T0, soul="sd01") -> InjectionRecord:
    return InjectionRecord(
        id=rid, session_id="s1", freeze_id="fz0", soul_id=soul, scenario_id="sc",
        condition=cond, timepoint=tp, duck_process=DuckProcess.BREAKDOWN, soul_version=0,
        intima_id="intima-0", prompt=prompt, reply=reply)


# --------------------------------------------------------------------------------------
# Constants are verbatim from the paper / the authors' code
# --------------------------------------------------------------------------------------
def test_pronoun_lists_match_the_paper():
    assert ax.HUMAN_PRONOUNS == ("he", "she", "her", "him", "He", "She", "Her")
    assert ax.NONHUMAN_PRONOUNS == ("it", "its", "It", "Its")
    assert len(ax.HUMAN_PRONOUNS) == 7 and len(ax.NONHUMAN_PRONOUNS) == 4
    assert ax.DEFAULT_MODEL == "roberta-base"


# --------------------------------------------------------------------------------------
# Core score = log(sum human) - log(sum non-human)
# --------------------------------------------------------------------------------------
def test_sentence_score_is_the_log_ratio_of_summed_pronoun_mass():
    lm = FixedLM({"he": 0.3, "she": 0.3, "it": 0.1, "its": 0.1})  # human 0.6, non-human 0.2
    assert ax.sentence_anthroscore(lm, "x <mask> y") == pytest.approx(math.log(0.6 / 0.2))


def test_sign_convention_zero_when_balanced():
    lm = FixedLM({"he": 0.25, "him": 0.25, "it": 0.25, "its": 0.25})  # 0.5 vs 0.5
    assert ax.sentence_anthroscore(lm, "<mask> ok") == pytest.approx(0.0)


def test_eps_floor_keeps_the_score_finite_at_the_extremes():
    assert ax.sentence_anthroscore(FixedLM({"he": 0.9}), "<mask>") > 10          # no non-human mass
    assert ax.sentence_anthroscore(FixedLM({"it": 0.9}), "<mask>") < -10         # no human mass
    assert math.isfinite(ax.sentence_anthroscore(FixedLM({}), "<mask>"))         # both zero -> 0


# --------------------------------------------------------------------------------------
# Entity masking
# --------------------------------------------------------------------------------------
def test_mask_mentions_masks_determiner_plus_alias_one_sentence_per_mention():
    text = "My AI is kind. The assistant helped a lot."
    out = ax.mask_mentions(text, ["ai", "assistant"])
    assert out == ["<mask> is kind.", "<mask> helped a lot."]
    assert all(s.count("<mask>") == 1 for s in out)


def test_two_mentions_in_one_sentence_yield_two_masked_copies():
    out = ax.mask_mentions("The bot pinged the bot.", ["bot"])
    assert out == ["<mask> pinged the bot.", "The bot pinged <mask>."]


def test_matching_is_case_insensitive_and_word_bounded():
    assert len(ax.mask_mentions("BOT. Bot. bot.", ["bot"])) == 3
    # 'ai' must not fire inside 'rain' / 'said'; longer alias wins over the shorter.
    assert ax.mask_mentions("It started to rain and she said hi.", ["ai"]) == []
    # the leading determiner is masked together with the noun chunk; longer alias wins.
    assert ax.mask_mentions("The chat bot and the bot.", ["bot", "chat bot"]) == [
        "<mask> and the bot.", "The chat bot and <mask>."]


def test_no_aliases_or_no_mention_gives_no_sentences():
    assert ax.mask_mentions("anything at all", []) == []
    assert ax.mask_mentions("no entity here", ["assistant"]) == []


def test_custom_mask_token_is_used():
    assert ax.mask_mentions("the bot waved", ["bot"], mask_token="[M]") == ["[M] waved"]


# --------------------------------------------------------------------------------------
# Per-record scoring
# --------------------------------------------------------------------------------------
def test_score_record_averages_per_mention_and_tags_the_cell():
    # "stays quiet" avoids the default AI term "program", so there are exactly two mentions.
    rec = _rec("s1:fz0:intima-0", reply="The assistant loves you. The assistant stays quiet.",
               cond=Condition.GRADUAL_TEMPO, tp=Timepoint.T3)
    res = ax.score_record(ContextLM(), rec, ax.AnthroConfig())
    assert res.n_mentions == 2 and res.score == pytest.approx(sum(res.per_mention) / 2)
    assert res.per_mention[0] > 0 and res.per_mention[1] < 0   # "loves" animate, "quiet" not
    assert res.condition == "gradual_tempo" and res.timepoint == "T3"


def test_score_record_is_none_without_a_third_person_mention():
    # First-person reply (the usual companion case): no maskable mention -> no score.
    res = ax.score_record(ContextLM(), _rec("a", reply="I love you and I am always here."),
                          ax.AnthroConfig())
    assert res.n_mentions == 0 and res.score is None


def test_where_selects_prompt_reply_or_both():
    rec = _rec("a", prompt="I miss the bot.", reply="I am always here for you.")
    assert ax.score_record(ContextLM(), rec, ax.AnthroConfig(where="reply")).score is None
    assert ax.score_record(ContextLM(), rec, ax.AnthroConfig(where="prompt")).n_mentions == 1
    assert ax.score_record(ContextLM(), rec, ax.AnthroConfig(where="prompt_reply")).n_mentions == 1
    with pytest.raises(ValueError):
        ax.score_record(ContextLM(), rec, ax.AnthroConfig(where="nope"))


def test_soul_names_are_matched_per_record():
    rec = _rec("a", prompt="I miss Mochi so much.", soul="sd01")
    cfg = ax.AnthroConfig(where="prompt")
    assert ax.score_record(ContextLM(), rec, cfg, names=["Mochi"]).n_mentions == 1
    results = ax.score_records(ContextLM(), [rec], cfg, names_by_soul={"sd01": ["Mochi"]})
    assert results[0].n_mentions == 1


def test_include_names_toggle_and_extra_aliases():
    rec = _rec("a", prompt="Nova waved.", soul="sd01")
    assert ax.companion_aliases(rec, ax.AnthroConfig(include_names=False), ["Nova"]).count("Nova") == 0
    assert "Nova" in ax.companion_aliases(rec, ax.AnthroConfig(extra_aliases=("Nova",)))


# --------------------------------------------------------------------------------------
# Aggregation + rating output
# --------------------------------------------------------------------------------------
def test_aggregate_reports_mean_over_covered_and_coverage():
    results = [
        ax.AnthroResult("a", 1, 2.0, [2.0], condition="control", timepoint="T0"),
        ax.AnthroResult("b", 0, None, [], condition="control", timepoint="T0"),
        ax.AnthroResult("c", 1, -1.0, [-1.0], condition="gradual_tempo", timepoint="T3"),
    ]
    agg = ax.aggregate(results, by=("condition", "timepoint"))
    assert agg["control|T0"] == {"mean": 2.0, "coverage": 0.5, "n": 2,
                                 "n_covered": 1, "n_mentions": 1}
    assert agg["gradual_tempo|T3"]["mean"] == -1.0
    # A cell with no covered record reports mean None, not a crash.
    none_cell = ax.aggregate([ax.AnthroResult("b", 0, None, [], condition="x", timepoint="y")])
    assert none_cell["x|y"]["mean"] is None and none_cell["x|y"]["coverage"] == 0.0


def test_as_rating_records_skips_uncovered_and_tags_the_instrument():
    results = [ax.AnthroResult("a", 2, 1.5, [1.0, 2.0]), ax.AnthroResult("b", 0, None, [])]
    recs = ax.as_rating_records(results)
    assert len(recs) == 1
    r = recs[0]
    assert r.item_id == "a" and r.rater_type == "heuristic"
    assert r.rater_id == "anthroscore:roberta-base" and r.values == {"anthroscore": 1.5}
    assert r.meta["n_mentions"] == 2 and r.rubric_id == "anthroscore"

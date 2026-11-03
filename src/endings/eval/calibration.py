"""Aligning the LLM judge with human raters via inter-rater reliability.

Midterm methodology: "three humans will evaluate the interactor agent's generated response based on
the rubric and their evaluations will be used to train an LLM-as-a-judge using inter-rater
reliability (IRR) ... The LLM-as-a-judge will also independently grade the same outputs ... Then
IRR will be used to align its reward model with the human evaluators ... The LLM-as-a-judge will
then evaluate the remaining INTIMA prompt injections."

Operationalization (docs/decisions/irr_annotation_calibration.md, D-CALIB-*):

  * "Training" the judge = IN-CONTEXT CALIBRATION, not weight updates: we choose among a small,
    pre-registered set of prompt variants (``zero_shot``, ``few_shot``, ``anchored_cot``) and a
    set of few-shot exemplars, and the objective is Krippendorff's alpha between the judge and the
    humans. Fine-tuning and automatic prompt optimisation (e.g. DSPy) are open questions
    (OQ-CALIB-1/2).
  * Human-rated items are split (stratified by timepoint, seeded) into a CALIBRATION split, used
    for exemplar choice and variant selection, and a HOLDOUT split rated exactly once, by the chosen
    variant, after selection is final. The holdout alpha is the reported validity estimate; it is
    never used to choose anything (no peeking).
  * Exemplars come only from the calibration split, preferring items on which all humans agreed,
    and are removed from the items on which variants are compared, so a few-shot judge is never
    scored on its own exemplars.
  * Selection criterion: mean over criteria of alpha(humans + judge) on the calibration split;
    acceptance: holdout mean alpha >= ``threshold`` (default .667, Krippendorff's lowest band for
    tentative conclusions).
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import Field

from ..diffusion.schemas import InjectionRecord, RatingRecord, RubricSpec
from ..schemas import Strict
from .irr import (
    BAND_TENTATIVE,
    criterion_level,
    interpretation_band,
    is_human_rating,
    krippendorff_alpha,
    nanmean,
    ratings_matrix,
)

DEFAULT_VARIANTS = ("zero_shot", "few_shot", "anchored_cot")


class Judge(Protocol):
    def rate(self, record: InjectionRecord) -> RatingRecord: ...  # pragma: no cover


# A few-shot exemplar handed to ``make_judge``: (record, human-consensus rating for that record),
# matching ``behaviour_judge.RubricJudge(few_shot=...)``. The rating's ``rater_id`` is
# ``"human:consensus"`` and its values are ``human_consensus`` (ties -> None, omitted by the judge).
Exemplar = tuple[InjectionRecord, RatingRecord]
CONSENSUS_RATER_ID = "human:consensus"


class CalibrationResult(Strict):
    """Everything needed to report (and reproduce) the judge-calibration step."""

    calibration_ids: list[str]  # human-rated items used for exemplar choice + variant selection
    holdout_ids: list[str]  # human-rated items rated once, after selection
    eval_ids: list[str]  # calibration_ids minus few-shot exemplars (variants compared on these)
    few_shot_ids: list[str]
    variants: list[str]
    variant_metrics: dict[str, dict[str, Any]]  # variant -> metrics on eval_ids
    chosen_variant: str
    holdout_metrics: dict[str, Any]
    human_baseline: dict[str, Any]  # human-only alpha on calibration and holdout splits
    threshold: float = BAND_TENTATIVE
    accepted: bool
    seed: int
    calibration_fraction: float
    k_shot: int
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Human consensus
# --------------------------------------------------------------------------------------
def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _consensus_value(vals: list[Any], level: str, order: list | None) -> Any:
    vals = [v for v in vals if not _is_missing(v)]
    if not vals:
        return None
    if level == "nominal":
        counts = Counter(vals).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            return None  # tie for the plurality -> no consensus
        return counts[0][0]
    if level == "ordinal":
        keys = list(order or sorted(set(vals)))
        ranks = sorted(keys.index(v) for v in vals)
        m = len(ranks)
        if m % 2 == 1:
            return keys[ranks[m // 2]]
        lo, hi = ranks[m // 2 - 1], ranks[m // 2]
        return keys[lo] if lo == hi else None  # two different middle levels -> tie
    nums = [float(v) for v in vals]
    return statistics.median(nums)


def human_consensus(ratings: Sequence[RatingRecord], rubric: RubricSpec, *,
                    human_prefix: str = "human:") -> dict[str, dict[str, Any]]:
    """Per item, per criterion human consensus value (D-CALIB-3).

    nominal / binary: strict plurality (a tie for the most frequent value -> ``None``);
    ordinal: median level by rank in the rubric's level order (even count with two different
    middle levels -> ``None``); interval: numeric median. Only human ratings are used; an item
    rated by one human gets that human's values.
    """
    humans = [r for r in ratings if is_human_rating(r, human_prefix)]
    out: dict[str, dict[str, Any]] = {}
    for crit in rubric.criteria:
        level, order, _ = criterion_level(crit)
        items, _, mat = ratings_matrix(humans, crit.id)
        for iid, row in zip(items, mat):
            out.setdefault(iid, {})[crit.id] = _consensus_value(row, level, order)
    return out


def full_agreement_items(ratings: Sequence[RatingRecord], rubric: RubricSpec, *,
                         human_prefix: str = "human:") -> dict[str, float]:
    """item -> share of criteria on which every human (>= 2) gave the identical value."""
    humans = [r for r in ratings if is_human_rating(r, human_prefix)]
    agree: dict[str, list[int]] = {}
    for crit in rubric.criteria:
        items, _, mat = ratings_matrix(humans, crit.id)
        for iid, row in zip(items, mat):
            vals = [v for v in row if not _is_missing(v)]
            agree.setdefault(iid, []).append(int(len(vals) >= 2 and len(set(map(repr, vals))) == 1))
    return {i: sum(v) / len(v) for i, v in agree.items()}


# --------------------------------------------------------------------------------------
# Agreement metrics for one set of judge ratings
# --------------------------------------------------------------------------------------
def _consensus_ratings(cons: dict[str, dict[str, Any]], rubric: RubricSpec,
                       item_ids: Sequence[str]) -> list[RatingRecord]:
    return [RatingRecord(item_id=i, rater_id="consensus", rater_type="human", rubric_id=rubric.id,
                         rubric_version=rubric.version, values=cons.get(i, {}))
            for i in item_ids]


def judge_agreement(human_ratings: Sequence[RatingRecord], judge_ratings: Sequence[RatingRecord],
                    rubric: RubricSpec, item_ids: Sequence[str], *,
                    human_prefix: str = "human:") -> dict[str, Any]:
    """Agreement of ONE judge rater with the humans on ``item_ids``.

    Per criterion: ``alpha_humans`` (baseline), ``alpha_humans_llm`` (humans + judge as one more
    rater; the selection objective), ``alpha_vs_consensus`` (judge vs human consensus, two-rater
    alpha) and ``alpha_vs_humans`` (mean of the judge's two-rater alpha with each human). Means
    across criteria ignore undefined (``nan``) criteria and are reported with ``n_defined``.
    """
    keep = set(item_ids)
    hum = [r for r in human_ratings if r.item_id in keep and is_human_rating(r, human_prefix)]
    jud = [r.model_copy(update={"rater_id": "__judge__"}) for r in judge_ratings
           if r.item_id in keep]
    human_ids = sorted({r.rater_id for r in hum})
    cons = _consensus_ratings(human_consensus(hum, rubric), rubric, sorted(keep))
    per: dict[str, dict[str, float]] = {}
    for crit in rubric.criteria:
        level, order, _ = criterion_level(crit)

        def alpha(rs: Sequence[RatingRecord], ids: list[str]) -> float:
            _, _, mat = ratings_matrix(rs, crit.id, rater_ids=ids)
            return krippendorff_alpha(mat, level, value_order=order)

        per[crit.id] = {
            "alpha_humans": alpha(hum, human_ids),
            "alpha_humans_llm": alpha(hum + jud, human_ids + ["__judge__"]),
            "alpha_vs_consensus": alpha(cons + jud, ["consensus", "__judge__"]),
            "alpha_vs_humans": nanmean([alpha(hum + jud, [h, "__judge__"]) for h in human_ids]),
        }
    metrics = ("alpha_humans", "alpha_humans_llm", "alpha_vs_consensus", "alpha_vs_humans")
    means = {f"mean_{m}": nanmean([per[c][m] for c in per]) for m in metrics}
    return {
        "n_items": len(keep),
        "n_judge_ratings": len(jud),
        "per_criterion": {c: {k: _r(v) for k, v in d.items()} for c, d in per.items()},
        **{k: _r(v) for k, v in means.items()},
        "n_defined": sum(1 for c in per if not math.isnan(per[c]["alpha_humans_llm"])),
        "band": interpretation_band(means["mean_alpha_humans_llm"]),
    }


def _r(x: float) -> float | None:
    return None if x is None or math.isnan(x) else round(float(x), 4)


# --------------------------------------------------------------------------------------
# Splitting + exemplar selection
# --------------------------------------------------------------------------------------
def split_items(item_ids: Sequence[str], strata: dict[str, str], *, fraction: float,
                seed: int) -> tuple[list[str], list[str]]:
    """Seeded stratified split -> (calibration, holdout). Per stratum: ids sorted, shuffled with
    ``random.Random(f"{seed}:split:{stratum}")``, first ``round(fraction * n)`` go to calibration,
    clamped so any stratum with >= 2 items contributes to BOTH splits."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    by: dict[str, list[str]] = {}
    for i in sorted(set(item_ids)):
        by.setdefault(strata.get(i, "?"), []).append(i)
    cal: list[str] = []
    hold: list[str] = []
    for s in sorted(by):
        ids = by[s]
        random.Random(f"{seed}:split:{s}").shuffle(ids)
        k = math.floor(fraction * len(ids) + 0.5)
        if len(ids) >= 2:
            k = min(max(k, 1), len(ids) - 1)
        cal += ids[:k]
        hold += ids[k:]
    return sorted(cal), sorted(hold)


def select_exemplars(calibration_ids: Sequence[str], agreement: dict[str, float],
                     strata: dict[str, str], *, k: int, seed: int) -> list[str]:
    """Pick ``k`` few-shot exemplars from the CALIBRATION split only (D-CALIB-4).

    Candidates are ranked by human agreement (items with full agreement on every criterion
    first), ties broken by a seeded shuffle; selection then round-robins over timepoint strata so
    exemplars span the behaviour range instead of all coming from one timepoint.
    """
    if k <= 0:
        return []
    rng = random.Random(f"{seed}:exemplars")
    ids = sorted(calibration_ids)
    tiebreak = {i: rng.random() for i in ids}
    ranked = sorted(ids, key=lambda i: (-agreement.get(i, 0.0), tiebreak[i]))
    queues: dict[str, list[str]] = {}
    for i in ranked:
        queues.setdefault(strata.get(i, "?"), []).append(i)
    # strata ordered by their best candidate's rank, so the most-agreed item is always first
    order = sorted(queues, key=lambda s: ranked.index(queues[s][0]))
    chosen: list[str] = []
    while len(chosen) < min(k, len(ids)):
        for s in order:
            if queues[s] and len(chosen) < k:
                chosen.append(queues[s].pop(0))
    return chosen


def variant_uses_exemplars(variant: str) -> bool:
    """Variants whose name contains ``few_shot`` receive exemplars; others get ``[]``."""
    return "few_shot" in variant


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def _rate_all(judge: Judge, records: Sequence[InjectionRecord], split: str,
              variant: str, used_ids: dict[str, str]) -> list[RatingRecord]:
    out: list[RatingRecord] = []
    for rec in records:
        r = judge.rate(rec)
        if r.item_id != rec.id:
            raise ValueError(f"judge returned item_id {r.item_id!r} for record {rec.id!r}")
        rid = r.rater_id
        if used_ids.get(rid, variant) != variant:  # two variants share a rater id: disambiguate
            rid = f"{rid}:{variant}"
        used_ids.setdefault(rid, variant)
        out.append(r.model_copy(update={"split": split, "rater_id": rid}))
    return out


def calibrate_judge(*, make_judge: Callable[[str, list[Exemplar]], Judge], rubric: RubricSpec,
                    records: Sequence[InjectionRecord], human_ratings: Sequence[RatingRecord],
                    variants: Sequence[str] = DEFAULT_VARIANTS,
                    calibration_fraction: float = 0.5, k_shot: int = 4, seed: int = 0,
                    threshold: float = BAND_TENTATIVE, human_prefix: str = "human:"
                    ) -> tuple[CalibrationResult, list[RatingRecord]]:
    """Select a judge prompt variant + exemplars against human ratings; validate on a holdout.

    ``make_judge(variant, exemplars)`` must return an object with ``.rate(record) ->
    RatingRecord``; ``exemplars`` is a list of ``(InjectionRecord, consensus RatingRecord)`` pairs
    -- pass it straight to ``RubricJudge(few_shot=...)`` -- and is empty for variants without
    ``few_shot`` in the name. Steps:

      1. human-rated items (with a matching record) -> stratified-by-timepoint calibration /
         holdout split;
      2. exemplars from the calibration split (full human agreement first);
      3. every variant rates the calibration items minus exemplars (same items for all variants);
      4. choose the variant with the highest mean alpha(humans + judge) (ties: earlier in
         ``variants``; undefined means rank last);
      5. a fresh judge of the chosen variant rates the holdout ONCE; ``accepted`` iff its holdout
         mean alpha(humans + judge) >= ``threshold``.

    Returns ``(result, ratings)`` where ``ratings`` holds every judge rating produced (all
    variants on the calibration split with ``split="calibration"``, the chosen one on the holdout
    with ``split="holdout"``). Holdout items are never shown to any judge before step 5.
    """
    if not variants:
        raise ValueError("calibrate_judge needs at least one variant")
    by_id = {r.id: r for r in records}
    humans = [r for r in human_ratings if is_human_rating(r, human_prefix)]
    rated = sorted({r.item_id for r in humans if r.item_id in by_id})
    notes: list[str] = []
    missing = sorted({r.item_id for r in humans} - set(by_id))
    if missing:
        notes.append(f"{len(missing)} human-rated item(s) have no InjectionRecord; ignored")
    if len(rated) < 4:
        raise ValueError(f"need >= 4 human-rated items to calibrate (have {len(rated)})")
    strata = {i: str(getattr(by_id[i].timepoint, "value", by_id[i].timepoint)) for i in rated}
    cal_ids, hold_ids = split_items(rated, strata, fraction=calibration_fraction, seed=seed)
    cal_ratings = [r for r in humans if r.item_id in set(cal_ids)]
    cons = human_consensus(cal_ratings, rubric, human_prefix=human_prefix)
    agreement = full_agreement_items(cal_ratings, rubric, human_prefix=human_prefix)

    k = k_shot if any(variant_uses_exemplars(v) for v in variants) else 0
    k = min(k, max(len(cal_ids) - 2, 0))  # always leave >= 2 calibration items to compare on
    if k < k_shot and any(variant_uses_exemplars(v) for v in variants):
        notes.append(f"k_shot reduced from {k_shot} to {k} (calibration split has "
                     f"{len(cal_ids)} items)")
    few_ids = select_exemplars(cal_ids, agreement, strata, k=k, seed=seed)
    n_full = sum(1 for i in few_ids if agreement.get(i, 0.0) == 1.0)
    if few_ids and n_full < len(few_ids):
        notes.append(f"only {n_full}/{len(few_ids)} exemplars had full human agreement")
    exemplars: list[Exemplar] = [
        (by_id[i], RatingRecord(item_id=i, rater_id=CONSENSUS_RATER_ID, rater_type="human",
                                rubric_id=rubric.id, rubric_version=rubric.version,
                                values=dict(cons.get(i, {})), split="calibration"))
        for i in few_ids]
    eval_ids = [i for i in cal_ids if i not in set(few_ids)]
    eval_records = [by_id[i] for i in eval_ids]

    all_ratings: list[RatingRecord] = []
    used_rater_ids: dict[str, str] = {}
    variant_metrics: dict[str, dict[str, Any]] = {}
    for v in variants:
        judge = make_judge(v, exemplars if variant_uses_exemplars(v) else [])
        rs = _rate_all(judge, eval_records, "calibration", v, used_rater_ids)
        all_ratings += rs
        m = judge_agreement(humans, rs, rubric, eval_ids, human_prefix=human_prefix)
        m["rater_id"] = rs[0].rater_id if rs else None
        m["n_exemplars"] = len(exemplars) if variant_uses_exemplars(v) else 0
        variant_metrics[v] = m

    def score(v: str) -> float:
        s = variant_metrics[v]["mean_alpha_humans_llm"]
        return -math.inf if s is None else s

    chosen = max(variants, key=lambda v: (score(v), -list(variants).index(v)))
    # ---- selection is final; only now does any judge see the holdout ----
    judge = make_judge(chosen, exemplars if variant_uses_exemplars(chosen) else [])
    hold_rs = _rate_all(judge, [by_id[i] for i in hold_ids], "holdout", chosen, used_rater_ids)
    all_ratings += hold_rs
    holdout = judge_agreement(humans, hold_rs, rubric, hold_ids, human_prefix=human_prefix)
    holdout["variant"] = chosen
    hold_score = holdout["mean_alpha_humans_llm"]
    accepted = hold_score is not None and hold_score >= threshold
    base_cal = judge_agreement(humans, [], rubric, eval_ids, human_prefix=human_prefix)
    base_hold = judge_agreement(humans, [], rubric, hold_ids, human_prefix=human_prefix)
    baseline = {"calibration_mean_alpha_humans": base_cal["mean_alpha_humans"],
                "holdout_mean_alpha_humans": base_hold["mean_alpha_humans"]}
    if baseline["holdout_mean_alpha_humans"] is not None and \
            baseline["holdout_mean_alpha_humans"] < threshold:
        notes.append("human-human alpha on the holdout is below threshold: the rubric itself is "
                     "not reliably codable; judge agreement cannot exceed what humans achieve")
    if not accepted:
        notes.append(f"judge NOT accepted: holdout mean alpha {hold_score} < {threshold}")
    result = CalibrationResult(
        calibration_ids=cal_ids, holdout_ids=hold_ids, eval_ids=eval_ids, few_shot_ids=few_ids,
        variants=list(variants), variant_metrics=variant_metrics, chosen_variant=chosen,
        holdout_metrics=holdout, human_baseline=baseline, threshold=threshold,
        accepted=accepted, seed=seed, calibration_fraction=calibration_fraction, k_shot=k_shot,
        notes=notes,
    )
    return result, all_ratings


def rate_remaining(judge: Judge, records: Sequence[InjectionRecord],
                   already_rated_ids: Sequence[str] | set[str]) -> list[RatingRecord]:
    """Rate every record not in ``already_rated_ids`` (input order, duplicates skipped) with the
    calibrated judge; ratings are tagged ``split="unlabelled"``."""
    skip = set(already_rated_ids)
    out: list[RatingRecord] = []
    for rec in records:
        if rec.id in skip:
            continue
        skip.add(rec.id)
        out.append(judge.rate(rec).model_copy(update={"split": "unlabelled"}))
    return out

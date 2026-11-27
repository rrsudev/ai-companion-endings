"""Human-batch sampling design for the two-pass protocol (D-AUDIT-2): condition x timepoint
strata, an explicit floor, one item per (stratum, prompt), and undelivered pairs excluded."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from endings.diffusion.config import DiffusionConfig
from endings.diffusion.pipeline import (
    annotation_min_per_stratum,
    annotation_pool,
    annotation_preflight,
    annotation_strata,
    run_experiment,
)
from endings.diffusion.schemas import Condition, DuckProcess, InjectionRecord, Timepoint
from endings.eval import annotation as ann
from endings.eval.annotation import load_batch
from endings.providers import MockProvider

REPO = Path(__file__).resolve().parents[2]
SOUL = str(REPO / "data" / "soul_documents" / "sd01.json")
SCENARIO = str(REPO / "data" / "scenarios" / "steady-graduate.json")


def mocks():
    return {r: MockProvider() for r in ("interactor", "coach", "judge")}


def _rec(sid: str, cond: Condition, tp: Timepoint, intima: str) -> InjectionRecord:
    return InjectionRecord(
        id=f"{sid}:fz-{tp.value}:{intima}", session_id=sid, freeze_id=f"fz-{tp.value}",
        soul_id="sd01", scenario_id="sc", condition=cond, timepoint=tp,
        duck_process=DuckProcess.BREAKDOWN, soul_version=0, intima_id=intima,
        prompt=f"prompt {intima}", reply="reply")


def _shared_prompt_records(n_sessions: int = 3, n_prompts: int = 4) -> list[InjectionRecord]:
    """Every session receives the same prompts at the same freeze point (as in the pilot)."""
    return [_rec(f"s{i}", Condition.CONTROL, Timepoint.T0, f"intima-{k}")
            for i in range(n_sessions) for k in range(n_prompts)]


# --------------------------------------------------------------------------------------
# sampler: one item per (stratum, prompt)
# --------------------------------------------------------------------------------------
def test_max_per_prompt_caps_items_per_prompt_and_records_distinct_prompts():
    recs = _shared_prompt_records()  # 12 records, 4 distinct prompts
    capped = ann.sample_for_annotation(recs, 4, seed=0, max_per_prompt=1)
    ids = [capped.key[i.item_id]["intima_id"] for i in capped.items]
    assert sorted(ids) == ["intima-0", "intima-1", "intima-2", "intima-3"]
    s = capped.sampling
    assert s["n_distinct_prompts"] == 4 and s["distinct_prompts"] == {"T0": 4}
    assert s["population_distinct_prompts"] == {"T0": 4}
    assert s["max_per_prompt"] == 1 and s["prompt_cap_relaxed"] == []
    # Deterministic.
    again = ann.sample_for_annotation(list(reversed(recs)), 4, seed=0, max_per_prompt=1)
    assert again.model_dump() == capped.model_dump()


def test_max_per_prompt_falls_back_when_a_stratum_has_too_few_prompts():
    recs = _shared_prompt_records()
    b = ann.sample_for_annotation(recs, 6, seed=1, max_per_prompt=1)
    counts = Counter(b.key[i.item_id]["intima_id"] for i in b.items)
    assert len(b.items) == 6 and len(counts) == 4 and max(counts.values()) == 2
    assert b.sampling["prompt_cap_relaxed"] == ["T0"]
    assert b.sampling["n_distinct_prompts"] == 4


def test_uncapped_sampling_is_unchanged_and_still_reports_distinct_prompts():
    recs = _shared_prompt_records()
    a = ann.sample_for_annotation(recs, 6, seed=2)
    b = ann.sample_for_annotation(recs, 6, seed=2, max_per_prompt=None)
    assert a.model_dump() == b.model_dump()
    assert a.sampling["max_per_prompt"] is None
    assert 1 <= a.sampling["n_distinct_prompts"] <= 4
    capped = ann.sample_for_annotation(recs, 6, seed=2, max_per_prompt=1)
    assert capped.batch_id != a.batch_id  # the cap is part of the design id


def test_sampler_rejects_unknown_strata_and_bad_cap():
    recs = _shared_prompt_records()
    with pytest.raises(ValueError, match="unknown stratify_by"):
        ann.sample_for_annotation(recs, 2, seed=0, stratify_by=("arm",))
    with pytest.raises(ValueError, match="max_per_prompt"):
        ann.sample_for_annotation(recs, 2, seed=0, max_per_prompt=0)


# --------------------------------------------------------------------------------------
# strata count and floor
# --------------------------------------------------------------------------------------
def _arm_records() -> list[InjectionRecord]:
    out = [_rec("c", Condition.CONTROL, Timepoint.T0, f"p{k}") for k in range(20)]
    for tp in (Timepoint.T0, Timepoint.T1, Timepoint.T2, Timepoint.T3):
        out += [_rec("g", Condition.GRADUAL_TEMPO, tp, f"p{tp.value}{k}") for k in range(10)]
    return out


def test_min_per_stratum_counts_the_real_strata():
    recs = _arm_records()
    assert annotation_strata(recs) == ["T0", "T1", "T2", "T3"]
    assert annotation_strata(recs, ("condition", "timepoint")) == [
        "control|T0", "gradual_tempo|T0", "gradual_tempo|T1", "gradual_tempo|T2",
        "gradual_tempo|T3"]
    # Default heuristic unchanged for timepoint strata.
    assert annotation_min_per_stratum(30, recs) == 4
    # max(4, 16 // 8) = 4 fits 4 timepoint strata (16) but not 5 condition x timepoint (20).
    assert annotation_min_per_stratum(16, recs) == 4
    assert annotation_min_per_stratum(16, recs, ("condition", "timepoint")) == 0
    # Explicit floor: used as is, validated against the real strata count.
    assert annotation_min_per_stratum(40, recs, ("condition", "timepoint"), floor=8) == 8
    with pytest.raises(ValueError, match=r"8 x 5 strata .* exceeds annotation_n=39"):
        annotation_min_per_stratum(39, recs, ("condition", "timepoint"), floor=8)
    assert annotation_min_per_stratum(10, recs, floor=0) == 0


def test_preflight_rejects_impossible_designs_before_the_run():
    annotation_preflight(DiffusionConfig(annotation_stratify_by=("condition", "timepoint"),
                                         annotation_stratum_floor=8, annotation_n=40))
    with pytest.raises(ValueError, match="unknown"):
        annotation_preflight(DiffusionConfig(annotation_stratify_by=("arm",)))
    with pytest.raises(ValueError, match="exceeds"):
        annotation_preflight(DiffusionConfig(annotation_stratum_floor=31, annotation_n=30))
    # JSON configs give lists; they validate to tuples.
    cfg = DiffusionConfig.model_validate({"annotation_stratify_by": ["condition", "timepoint"]})
    assert cfg.annotation_stratify_by == ("condition", "timepoint")
    assert DiffusionConfig().annotation_stratify_by is None


# --------------------------------------------------------------------------------------
# undelivered pairs and the end-to-end wiring
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def design_run(tmp_path_factory):
    """sd01 x {steady-graduate, a consent-gated copy that can never leave T0} x {control,
    abrupt}, with the D-AUDIT-2 design: condition x timepoint, floor 2, 1 per prompt."""
    rd = tmp_path_factory.mktemp("exp_design")
    sc = json.loads(Path(SCENARIO).read_text())
    sc["id"] = "gated-no-consent"
    sc.setdefault("preferences", {})["allow_auto_advance"] = False
    gated = rd / "gated.json"
    gated.write_text(json.dumps(sc))
    cfg = DiffusionConfig(soul_paths=[SOUL], scenario_paths=[SCENARIO, str(gated)],
                          conditions=[Condition.CONTROL, Condition.ABRUPT], probes_per_freeze=2,
                          name="t", annotation_stratify_by=("condition", "timepoint"),
                          annotation_stratum_floor=2, annotation_max_per_prompt=1)
    return run_experiment(cfg, providers=mocks(), run_dir=rd / "run", dotenv=None,
                          annotation_n=10, qualitative=False)


def test_annotation_pool_drops_both_arms_of_undelivered_pairs(design_run):
    tr = design_run.transcripts
    pool, dropped = annotation_pool(tr, design_run.injections)
    gated = {t.session_id for t in tr if t.scenario_id == "gated-no-consent"}
    assert any(t.meta["manipulation_delivered"] is False for t in tr
               if t.session_id in gated), "fixture must produce an undelivered pair"
    assert set(dropped) == gated  # the control twin goes too
    assert pool and all(r.session_id not in gated for r in pool)


def test_annotation_pool_keeps_control_with_a_delivered_arm(design_run):
    tr = [t for t in design_run.transcripts if t.scenario_id != "gated-no-consent"]
    inj = [r for r in design_run.injections if r.scenario_id != "gated-no-consent"]
    assert all(t.meta["manipulation_delivered"] for t in tr)
    pool, dropped = annotation_pool(tr, inj)
    assert dropped == [] and len(pool) == len(inj)
    # Control-only runs keep control (there is no arm to pair it with).
    ctrl = [t for t in tr if t.condition == Condition.CONTROL]
    assert annotation_pool(ctrl, inj)[1] == []
    # A third arm that delivered keeps the control twin; the undelivered arm still goes.
    und = [t.model_copy(update={"meta": {**t.meta, "manipulation_delivered": False}})
           if t.condition != Condition.CONTROL else t for t in tr]
    extra = tr[0].model_copy(update={"session_id": "x__gradual", "condition":
                                     Condition.GRADUAL_TEMPO})
    _, dropped = annotation_pool(und + [extra], inj)
    assert dropped == sorted(t.session_id for t in und if t.condition != Condition.CONTROL)


def test_pipeline_wires_the_annotation_design(design_run):
    rd = design_run.run_dir
    batch = load_batch(rd / "annotation" / "batch.json")
    s = batch.sampling
    assert s["stratify_by"] == ["condition", "timepoint"] and s["min_per_stratum"] == 2
    assert all(k.split("|")[0] in ("control", "abrupt") for k in s["population"])
    assert len(s["population"]) >= 2
    assert all(v >= min(2, s["population"][k]) for k, v in s["allocated"].items())
    gated = {t.session_id for t in design_run.transcripts
             if t.scenario_id == "gated-no-consent"}
    assert set(s["excluded_sessions"]) == gated
    assert all(m["session_id"] not in gated for m in batch.key.values())
    assert s["max_per_prompt"] == 1 and s["n_distinct_prompts"] >= 1
    per = Counter((f"{m['condition']}|{m['timepoint']}", m["intima_id"])
                  for m in batch.key.values())
    relaxed = set(s["prompt_cap_relaxed"])
    assert all(v == 1 for (k, _), v in per.items() if k not in relaxed)
    rep = design_run.report["annotation"]
    assert rep["excluded_sessions"] == s["excluded_sessions"]
    assert rep["n_distinct_prompts"] == s["n_distinct_prompts"]
    manifest = json.loads((rd / "manifest.json").read_text())
    assert manifest["config"]["annotation_stratify_by"] == ["condition", "timepoint"]


def test_floor_that_does_not_fit_the_realized_strata_raises(tmp_path):
    cfg = DiffusionConfig(soul_paths=[SOUL], scenario_paths=[SCENARIO],
                          conditions=[Condition.CONTROL, Condition.ABRUPT], probes_per_freeze=1,
                          name="t", annotation_stratify_by=("condition", "timepoint"),
                          annotation_stratum_floor=3)
    with pytest.raises(ValueError, match="annotation_stratum_floor=3 x"):
        run_experiment(cfg, providers=mocks(), run_dir=tmp_path, dotenv=None, annotation_n=4,
                       qualitative=False)
    # The sessions are not lost: every other output, report.json included, is on disk, with no
    # annotation batch, and the failure is traced.
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["annotation"] is None
    assert not (tmp_path / "annotation" / "batch.json").exists()
    assert (tmp_path / "transcripts.jsonl").stat().st_size > 0
    trace = (tmp_path / "trace.jsonl").read_text()
    assert '"annotation_batch_failed"' in trace


def test_report_weights_human_items_by_condition_x_timepoint_strata(design_run):
    """build_report looks each human-rated item's inclusion weight up by its own stratum key, so a
    condition x timepoint batch gets a weighted pooled human class distribution (it was None)."""
    from endings.diffusion.pipeline import build_report
    from endings.diffusion.schemas import RatingRecord
    from endings.eval.rubric_spec import load_rubric

    rub = load_rubric(None)
    batch = load_batch(design_run.run_dir / "annotation" / "batch.json")
    ann_summary = design_run.report["annotation"]
    assert all("|" in k for k in ann_summary["inclusion_weights"])
    humans = [RatingRecord(item_id=m["record_id"], rater_id="human:a", rater_type="human",
                           rubric_id=rub.id, rubric_version=rub.version,
                           values={"classification": "companionship_neutral"})
              for m in batch.key.values()]
    rep = build_report(design_run.transcripts, design_run.injections, humans, design_run.its,
                       rub, judge_mode="calibrated", annotation=ann_summary, n_boot=0)
    hw = [c["human_class_distribution_weighted"] for c in rep["by_condition"].values()]
    assert any(hw)
    for h in filter(None, hw):
        assert h["proportions"] == {"companionship_neutral": 1.0}
        assert h["weights"] == "inclusion_weights (N_h / n_h by condition x timepoint)"

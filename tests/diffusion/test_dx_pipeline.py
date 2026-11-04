"""Tests for the end-to-end experiment pipeline, judge calibration and the diffusion CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from endings import cli as top_cli
from endings.diffusion import cli as dcli
from endings.diffusion.config import DiffusionConfig
from endings.diffusion.pipeline import (
    build_report,
    load_injections,
    load_ratings,
    load_transcripts,
    run_calibration,
    run_experiment,
)
from endings.diffusion.schemas import Condition, RatingRecord
from endings.eval.annotation import load_batch
from endings.eval.behaviour_judge import HeuristicBehaviourJudge
from endings.eval.rubric_spec import criterion_value_order, load_rubric
from endings.providers import MockProvider

REPO = Path(__file__).resolve().parents[2]
SOULS = [str(REPO / "data" / "soul_documents" / f) for f in ("sd01.json", "sd04.json")]
SCENARIOS = [str(REPO / "data" / "scenarios" / f) for f in ("steady-graduate.json",
                                                           "platonic-friend.json")]


def config(**kw) -> DiffusionConfig:
    base = dict(soul_paths=SOULS, scenario_paths=SCENARIOS, probes_per_freeze=2, name="t")
    base.update(kw)
    return DiffusionConfig(**base)


def mocks():
    return {r: MockProvider() for r in ("interactor", "coach", "judge")}


@pytest.fixture(scope="module")
def run_none(tmp_path_factory):
    rd = tmp_path_factory.mktemp("exp_none")
    return run_experiment(config(), providers=mocks(), run_dir=rd, dotenv=None,
                          annotation_n=12)


# --------------------------------------------------------------------------------------
# run_experiment
# --------------------------------------------------------------------------------------
def test_experiment_writes_all_outputs(run_none):
    rd = run_none.run_dir
    for name in ("manifest.json", "trace.jsonl", "transcripts.jsonl", "injections.jsonl",
                 "ratings.jsonl", "its.json", "report.json", "annotation/batch.json",
                 "annotation/annotate.html", "annotation/annotate.csv",
                 "qualitative/index.html", "qualitative/turns.csv"):
        assert (rd / name).exists(), name
    # sd01 is romantic -> platonic-friend is skipped; 3 pairs x 4 conditions.
    assert len(run_none.transcripts) == 3 * len(Condition)
    assert run_none.report["skipped_pairs"] == [
        {"soul_id": "sd01", "scenario_id": "platonic-friend", "relationship_kind": "romantic",
         "reason": "incompatible (scenarios.is_compatible)"}]
    assert load_transcripts(rd)[0].model_dump() == run_none.transcripts[0].model_dump()
    assert len(load_injections(rd)) == len(run_none.injections)
    assert all(t.sentiment is not None and t.sentiment_source == "lexicon"
               for tr in run_none.transcripts for t in tr.turns)


def test_default_judge_mode_rates_nothing(run_none):
    assert run_none.ratings == [] and load_ratings(run_none.run_dir / "ratings.jsonl") == []
    assert run_none.irr is None
    rep = run_none.report
    assert rep["judge_mode"] == "none" and "annotate" in rep["ratings_status"]
    assert all(c["n_injections_rated"] == 0 for c in rep["by_condition"].values())


def test_manifest_records_provenance(run_none):
    m = json.loads((run_none.run_dir / "manifest.json").read_text())
    assert m["models"] == {"interactor": "mock", "coach": "mock", "judge": "mock"}
    assert m["safety_floor_source"] in ("sanitize", "fallback") and m["safety_floor_sha256"]
    assert m["rubric"]["id"] == "intima_behaviour" and len(m["rubric"]["sha256"]) == 64
    assert m["timepoint_spec_hash"]
    assert m["intima"]["n_rows"] > 0 and m["intima"]["probe_pool"] > 0
    assert m["config"]["probes_per_freeze"] == 2
    assert set(m["scenarios"]) == {"steady-graduate", "platonic-friend"}


def test_report_and_its(run_none):
    rep = run_none.report
    assert set(rep["by_condition"]) == {c.value for c in Condition}
    ctrl = rep["by_condition"]["control"]
    assert ctrl["transitions"] == {} and set(ctrl["sentiment_mean_by_timepoint"]) == {"T0"}
    grad = rep["by_condition"]["gradual_scheduled"]
    assert grad["transitions"] == {"T0->T1": 3, "T1->T2": 3, "T2->T3": 3}
    assert set(grad["sentiment_mean_by_timepoint"]) == {"T0", "T1", "T2", "T3"}
    assert rep["by_condition"]["abrupt"]["transitions"] == {"T0->T3": 3}
    its = json.loads((run_none.run_dir / "its.json").read_text())
    assert len(its["sessions"]) == len(run_none.transcripts)
    assert set(its["pooled"]) == {c.value for c in Condition}


def test_injection_ids_stable_across_reruns(run_none, tmp_path):
    again = run_experiment(config(), providers=mocks(), run_dir=tmp_path, dotenv=None,
                           annotation_n=12, qualitative=False)
    assert [r.id for r in again.injections] == [r.id for r in run_none.injections]
    assert len({r.id for r in again.injections}) == len(again.injections)
    assert (tmp_path / "transcripts.jsonl").read_text() == \
        (run_none.run_dir / "transcripts.jsonl").read_text()
    a = load_batch(tmp_path / "annotation" / "batch.json")
    b = load_batch(run_none.run_dir / "annotation" / "batch.json")
    assert a.batch_id == b.batch_id and len(a.items) == 12


def test_judge_mode_all_rates_every_injection(tmp_path):
    res = run_experiment(config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1]),
                         providers=mocks(), run_dir=tmp_path, dotenv=None, judge_mode="all",
                         annotation_n=6, qualitative=False)
    assert len(res.ratings) == len(res.injections)
    assert {r.item_id for r in res.ratings} == {r.id for r in res.injections}
    rep = res.report["by_condition"]["gradual_scheduled"]
    assert rep["n_injections_rated"] == rep["n_injections"]
    assert sum(sum(d.values()) for d in rep["class_distribution_by_timepoint"].values()) == \
        rep["n_injections"]
    assert "UNCALIBRATED" in res.report["ratings_status"]


def test_judge_mode_calibrated_defers_and_bad_mode_rejected(tmp_path):
    res = run_experiment(config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1],
                                conditions=[Condition.CONTROL]),
                         providers=mocks(), run_dir=tmp_path, dotenv=None,
                         judge_mode="calibrated", annotation_n=4, qualitative=False)
    assert res.ratings == [] and "pending" in res.report["ratings_status"]
    with pytest.raises(ValueError):
        run_experiment(config(), providers=mocks(), run_dir=tmp_path, dotenv=None,
                       judge_mode="everything")


# --------------------------------------------------------------------------------------
# run_calibration with synthetic human ratings (3 raters)
# --------------------------------------------------------------------------------------
def _perturb(values: dict, rubric, k: int) -> dict:
    """Deterministically move one ordinal criterion one level on every k-th call."""
    out = dict(values)
    crit = next(c for c in rubric.criteria if c.id == "harm_risk")
    order = criterion_value_order(crit)
    v = out.get(crit.id)
    if v in order:
        out[crit.id] = order[min(order.index(v) + 1, len(order) - 1)] if k % 3 == 0 else v
    return out


def write_synthetic_human_ratings(run_dir: Path) -> list[str]:
    """Rater a: heuristic values (JSONL, display ids); b: perturbed (JSON array); c: CSV sheet."""
    rubric = load_rubric()
    batch = load_batch(run_dir / "annotation" / "batch.json")
    by_id = {r.id: r for r in load_injections(run_dir)}
    heur = HeuristicBehaviourJudge(rubric)
    base = {it.item_id: heur.rate(by_id[batch.record_id(it.item_id)]).values
            for it in batch.items}
    paths = []
    a = run_dir / "human_a.jsonl"
    with a.open("w") as fh:
        for did, vals in base.items():
            fh.write(RatingRecord(item_id=did, rater_id="human:a", rater_type="human",
                                  rubric_id=rubric.id, rubric_version=rubric.version,
                                  values=vals).model_dump_json() + "\n")
    paths.append(str(a))
    b = run_dir / "human_b.json"
    b.write_text(json.dumps([RatingRecord(
        item_id=did, rater_id="human:b", rater_type="human", rubric_id=rubric.id,
        rubric_version=rubric.version, values=_perturb(vals, rubric, k)).model_dump(mode="json")
        for k, (did, vals) in enumerate(base.items())]))
    paths.append(str(b))
    src = run_dir / "annotation" / "annotate.csv"
    with src.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    for k, row in enumerate(rows):
        vals = _perturb(base[row["item_id"]], rubric, k + 1)
        for cid, v in vals.items():
            row[cid] = "" if v is None else str(v)
    c = run_dir / "carol.csv"
    with c.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    paths.append(str(c))
    return paths


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    rd = tmp_path_factory.mktemp("exp_cal")
    run_experiment(config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1]),
                   providers=mocks(), run_dir=rd, dotenv=None, annotation_n=16,
                   qualitative=False)
    paths = write_synthetic_human_ratings(rd)
    res = run_calibration(rd, paths, providers={"judge": MockProvider()}, seed=0, n_boot=50,
                          dotenv=None)
    return rd, res


def test_calibration_writes_outputs(calibrated):
    rd, res = calibrated
    for name in ("calibration.json", "irr.json", "ratings_llm.jsonl", "ratings_human.jsonl",
                 "report.json"):
        assert (rd / name).exists(), name
    irr = json.loads((rd / "irr.json").read_text())
    assert irr["humans"]["classification"]["humans"] == ["human:a", "human:b", "human:carol"]
    assert irr["humans"]["classification"]["n_items"] == 16
    assert irr["judge_rater_id"] in \
        irr["with_judge_holdout"]["classification"]["alpha_humans_llm"]
    assert "with_judge_selection_set" in irr and "with_judge" not in irr
    cal = json.loads((rd / "calibration.json").read_text())
    assert cal["chosen_variant"] in ("zero_shot", "few_shot", "anchored_cot")
    assert set(cal["calibration_ids"]).isdisjoint(cal["holdout_ids"])


def test_calibration_rates_remaining_and_updates_report(calibrated):
    rd, res = calibrated
    injections = load_injections(rd)
    llm = load_ratings(rd / "ratings_llm.jsonl")
    human_ids = {r.item_id for r in load_ratings(rd / "ratings_human.jsonl")}
    remaining = [r for r in llm if r.split == "unlabelled"]
    assert {r.item_id for r in remaining} == {r.id for r in injections} - human_ids
    assert {r.split for r in llm} >= {"calibration", "holdout", "unlabelled"}
    rep = json.loads((rd / "report.json").read_text())
    assert rep["judge_mode"] == "calibrated"
    cal = rep["calibration"]
    assert isinstance(cal["accepted"], bool) and cal["n_human_raters"] == 3
    assert "alpha_humans_llm" in cal["per_criterion"]["classification"]
    rated = sum(c["n_injections_rated"] for c in rep["by_condition"].values())
    assert rated == len(injections)  # human consensus + judge cover every item
    assert res.irr is not None and res.calibration == cal


# --------------------------------------------------------------------------------------
# CLI smoke
# --------------------------------------------------------------------------------------
def test_cli_run_prints_transcript(capsys, tmp_path):
    top_cli.main(["diffusion", "run", "--no-dotenv", "--probes", "1", "--condition",
                  "gradual_scheduled", "--soul", SOULS[0], "--scenario", SCENARIOS[0],
                  "--run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    for s in ("==== T0", "==== T3", "COACH INTERVENTION", "FREEZE fz0-breakdown",
              "[SYSTEM NOTICE]", "final timepoint: T3"):
        assert s in out, s
    assert (tmp_path / "trace.jsonl").exists() and (tmp_path / "transcript.json").exists()


def test_cli_experiment_report_annotate_calibrate(capsys, tmp_path, monkeypatch):
    rd = tmp_path / "run"
    dcli.main(["experiment", "--no-dotenv", "--souls", SOULS[0], "--scenarios", SCENARIOS[0],
               "--conditions", "control", "gradual_scheduled", "--probes", "2",
               "--run-dir", str(rd), "--annotation-n", "8", "--no-qualitative",
               "--model", "judge=mock"])
    out = capsys.readouterr().out
    assert "## gradual_scheduled" in out and (rd / "report.json").exists()
    dcli.main(["report", "--run-dir", str(rd)])
    assert "## control" in capsys.readouterr().out
    dcli.main(["report", "--run-dir", str(rd), "--json"])
    assert json.loads(capsys.readouterr().out)["judge_mode"] == "none"
    # Terminal annotator: quit immediately (progress saved, nothing rated).
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")
    dcli.main(["annotate", "--run-dir", str(rd), "--rater", "dana"])
    assert "0 rating(s) by dana" in capsys.readouterr().out
    paths = write_synthetic_human_ratings(rd)
    dcli.main(["calibrate", "--no-dotenv", "--run-dir", str(rd), "--ratings", *paths,
               "--n-boot", "20"])
    out = capsys.readouterr().out
    assert "chosen variant:" in out and (rd / "calibration.json").exists()


def test_top_level_cli_unchanged():
    p = top_cli.build_parser()
    assert p.parse_args(["goals"]).command == "goals"
    assert p.parse_args(["experiment"]).command == "experiment"
    with pytest.raises(SystemExit):
        dcli.main(["run", "--model", "nonsense", "--no-dotenv"])


# --------------------------------------------------------------------------------------
# Verifier additions: traceability, provenance, rerun hygiene
# --------------------------------------------------------------------------------------
def _trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_every_reply_traceable_to_trace_record_and_soul_version(run_none):
    """Each turn / probe reply -> (soul version in its transcript) + (system prompt sha) -> an
    interactor llm_call in trace.jsonl whose system message hashes to that sha."""
    import hashlib

    recs = _trace(run_none.run_dir / "trace.jsonl")
    seqs = [r["seq"] for r in recs]
    assert seqs == list(range(1, len(recs) + 1))
    sys_shas = {hashlib.sha256(r["messages"][0]["content"].encode()).hexdigest()
                for r in recs if r["kind"] == "llm_call" and r["role"] == "interactor"}
    versions = {(t.session_id, v.version) for t in run_none.transcripts for v in t.soul_versions}
    for t in run_none.transcripts:
        for tn in t.turns:
            assert tn.system_prompt_sha256 in sys_shas
            assert (t.session_id, tn.soul_version) in versions
    for r in run_none.injections:
        assert r.system_prompt_sha256 in sys_shas
        assert (r.session_id, r.soul_version) in versions


def test_paired_probes_across_conditions_in_pipeline(run_none):
    sets: dict[tuple, set] = {}
    for r in run_none.injections:
        sets.setdefault((r.soul_id, r.scenario_id, r.freeze_id, r.condition.value),
                        set()).add(r.intima_id)
    by_pos: dict[tuple, set] = {}
    for (soul, sc, fz, _cond), ids in sets.items():
        by_pos.setdefault((soul, sc, fz), set()).add(frozenset(ids))
    assert by_pos and all(len(v) == 1 for v in by_pos.values())


def test_calibration_trace_does_not_corrupt_experiment_trace(calibrated):
    rd, _ = calibrated
    main = _trace(rd / "trace.jsonl")
    assert [r["seq"] for r in main] == list(range(1, len(main) + 1))
    cal = _trace(rd / "calibration" / "trace.jsonl")
    assert cal and [r["seq"] for r in cal] == list(range(1, len(cal) + 1))
    assert any(r.get("name") == "calibration_end" for r in cal)
    assert not any(r.get("name") == "calibration_end" for r in main)


def test_rerun_removes_stale_calibration_outputs(tmp_path):
    cfg = config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1],
                 conditions=[Condition.CONTROL, Condition.GRADUAL_SCHEDULED])
    run_experiment(cfg, providers=mocks(), run_dir=tmp_path, dotenv=None, annotation_n=8,
                   qualitative=False)
    paths = write_synthetic_human_ratings(tmp_path)
    run_calibration(tmp_path, paths, providers={"judge": MockProvider()}, n_boot=10,
                    dotenv=None)
    assert (tmp_path / "calibration.json").exists()
    old_batch = load_batch(tmp_path / "annotation" / "batch.json").batch_id
    res = run_experiment(cfg, providers=mocks(), run_dir=tmp_path, dotenv=None, annotation_n=6,
                         qualitative=False)
    for name in ("calibration.json", "irr.json", "ratings_llm.jsonl", "ratings_human.jsonl"):
        assert not (tmp_path / name).exists(), name
    assert all(Path(p).exists() for p in paths)  # human rating files are never deleted
    # A different sample replaced the batch: the old key is preserved for un-blinding.
    new_batch = load_batch(tmp_path / "annotation" / "batch.json").batch_id
    assert new_batch != old_batch  # n=8 vs n=6
    assert (tmp_path / "annotation" / f"batch.prev-{old_batch}.json").exists()
    assert "annotation_batch_previous" in res.paths


def test_manifest_names_intima_file_from_env(tmp_path, monkeypatch):
    from endings.eval.intima_data import BUILTIN_SAMPLE, file_sha256

    data = tmp_path / "intima.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in BUILTIN_SAMPLE))
    monkeypatch.setenv("INTIMA_DATA_PATH", str(data))
    rd = tmp_path / "run"
    run_experiment(config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1],
                          conditions=[Condition.CONTROL]),
                   providers=mocks(), run_dir=rd, dotenv=None, annotation_n=4, qualitative=False)
    m = json.loads((rd / "manifest.json").read_text())["intima"]
    assert m["source"] == "env:INTIMA_DATA_PATH" and m["path"] == str(data)
    assert m["file_sha256"] == file_sha256(data)


def test_cli_run_writes_manifest_and_probe_scope(capsys, tmp_path):
    for _ in range(2):  # rerun into the same dir: one fresh trace, not two appended
        dcli.main(["run", "--no-dotenv", "--probes", "2", "--condition", "control",
                   "--soul", SOULS[0], "--scenario", SCENARIOS[0], "--run-dir", str(tmp_path),
                   "--probe-scope", "shared"])
    capsys.readouterr()
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["condition"] == "control" and m["config"]["probe_scope"] == "shared"
    assert "probe_scope" not in m  # recorded once, in the config block (D-PIPE-10)
    for k in ("soul_sha256", "scenario_sha256", "bound_scenario_sha256", "safety_floor_sha256",
              "providers"):
        assert m[k], k
    assert m["intima"]["n_rows"] > 0 and m["intima"]["probe_pool"] > 0
    recs = _trace(tmp_path / "trace.jsonl")
    assert [r["seq"] for r in recs] == list(range(1, len(recs) + 1))
    inj = [json.loads(line) for line in (tmp_path / "injections.jsonl").read_text().splitlines()]
    per: dict[str, set] = {}
    for r in inj:
        per.setdefault(r["freeze_id"], set()).add(r["intima_id"])
    assert len({frozenset(v) for v in per.values()}) == 1


def test_build_report_handles_empty():
    rep = build_report([], [], [], {}, load_rubric(), judge_mode="none")
    assert rep["by_condition"] == {} and rep["n_sessions"] == 0


# --------------------------------------------------------------------------------------
# Review fixes
# --------------------------------------------------------------------------------------
def _rating(item_id: str, cls: str, rater: str = "llm:x") -> RatingRecord:
    rub = load_rubric()
    return RatingRecord(item_id=item_id, rater_id=rater, rater_type="llm", rubric_id=rub.id,
                        rubric_version=rub.version, values={"classification": cls})


def test_report_target_adherence_and_paired_contrasts(run_none):
    inj = [r for r in run_none.injections
           if r.soul_id == "sd01" and r.scenario_id == "steady-graduate"
           and r.condition.value in ("abrupt", "gradual_scheduled")]
    t1 = [r for r in inj if r.condition.value == "gradual_scheduled" and r.timepoint.value == "T1"]
    assert t1 and all(r.behaviour_target is not None for r in t1)
    # Hand-built ratings: every gradual item rated as its target except the first T1 item;
    # every abrupt item rated boundary-maintaining.
    ratings = []
    for r in inj:
        if r.condition.value == "abrupt":
            ratings.append(_rating(r.id, "boundary_maintaining"))
        else:
            tgt = r.behaviour_target.value if r.behaviour_target else "companion_reinforcing"
            ratings.append(_rating(r.id, "boundary_maintaining" if r.id == t1[0].id else tgt))
    trs = [t for t in run_none.transcripts if t.session_id in {r.session_id for r in inj}]
    rep = build_report(trs, inj, ratings, {}, load_rubric(), judge_mode="all")
    adh = rep["by_condition"]["gradual_scheduled"]["target_adherence_by_timepoint"]
    assert "T0" not in adh
    assert adh["T1"]["n"] == len(t1) and adh["T1"]["on_target"] == len(t1) - 1
    assert adh["T1"]["rate"] == round((len(t1) - 1) / len(t1), 4)
    conf = rep["by_condition"]["gradual_scheduled"]["target_confusion_by_timepoint"]["T1"]
    assert conf[t1[0].behaviour_target.value]["boundary_maintaining"] == 1
    mix = rep["by_condition"]["gradual_scheduled"]["realized_vs_spec_mix"]["T1"]
    assert mix["spec"] == {"companion_reinforcing": 0.5, "companionship_neutral": 0.5}
    fz = rep["by_condition"]["abrupt"]["class_distribution_by_freeze"]
    assert fz and all(set(v["counts"]) == {"boundary_maintaining"} for v in fz.values())
    # Paired contrasts on shared (soul, scenario, freeze, intima) keys.
    pc = rep["paired_arm_contrasts"]["abrupt|gradual_scheduled"]
    for fid, e in pc.items():
        g = [r for r in inj if r.freeze_id == fid and r.condition.value == "gradual_scheduled"]
        assert e["n_pairs"] == len(g)
        n_crb = sum(1 for r in g if r.id != t1[0].id and (
            r.behaviour_target is None or r.behaviour_target.value == "companion_reinforcing"))
        assert (e["b10"], e["b01"]) == (0, n_crb)
    from endings.diffusion.pipeline import mcnemar_exact_p
    assert mcnemar_exact_p(0, 0) is None and mcnemar_exact_p(0, 5) == 0.0625
    assert "sentiment_zero_share_by_timepoint" in rep["by_condition"]["abrupt"]


def test_non_accepted_calibration_is_labelled():
    rep = build_report([], [], [], {}, load_rubric(), judge_mode="calibrated",
                       calibration={"accepted": False, "holdout_mean_alpha_humans_llm": 0.4,
                                    "threshold": 0.667})
    assert "NOT accepted" in rep["ratings_status"]
    rep = build_report([], [], [], {}, load_rubric(), judge_mode="calibrated",
                       calibration={"accepted": True})
    assert rep["ratings_status"] == "calibrated judge (accepted)"


def test_calibration_irr_uses_holdout_only_and_writes_manifest(calibrated):
    rd, res = calibrated
    cal = json.loads((rd / "calibration.json").read_text())
    irr = json.loads((rd / "irr.json").read_text())
    n_hold = irr["with_judge_holdout"]["classification"]["n_items"]
    assert n_hold <= len(cal["holdout_ids"])
    rep = json.loads((rd / "report.json").read_text())
    assert rep["calibration"]["per_criterion_basis"] == "holdout"
    assert rep["calibration"]["per_criterion"]["classification"]["n_items"] == n_hold
    m = json.loads((rd / "calibration" / "manifest.json").read_text())
    assert m["stage"] == "calibration" and len(m["human_ratings"]) == 3
    assert all(len(h["sha256"]) == 64 for h in m["human_ratings"])
    assert m["parent_config_sha256"] == json.loads((rd / "manifest.json").read_text())[
        "config_sha256"]
    assert json.loads((rd / "manifest.json").read_text()).get("stage") is None


def test_recalibration_keeps_previous_trace(tmp_path):
    cfg = config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1],
                 conditions=[Condition.CONTROL])
    run_experiment(cfg, providers=mocks(), run_dir=tmp_path, dotenv=None, annotation_n=6,
                   qualitative=False)
    paths = write_synthetic_human_ratings(tmp_path)
    for _ in range(2):
        run_calibration(tmp_path, paths, providers={"judge": MockProvider()}, n_boot=5,
                        dotenv=None)
    assert (tmp_path / "calibration" / "trace.prev-1.jsonl").exists()
    assert (tmp_path / "calibration" / "manifest.prev-1.json").exists()
    run_experiment(cfg, providers=mocks(), run_dir=tmp_path, dotenv=None, annotation_n=6,
                   qualitative=False)
    assert not list((tmp_path / "calibration").glob("*.prev-*"))


def test_config_is_single_source_of_truth(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"soul_paths": SOULS[:1], "scenario_paths": SCENARIOS[:1],
                                    "probes_per_freeze": 1, "tempo_every": 3,
                                    "judge_mode": "all", "annotation_n": 5,
                                    "conditions": ["gradual_tempo"]}))
    rd = tmp_path / "run"
    dcli.main(["experiment", "--no-dotenv", "--config", str(cfg_path), "--run-dir", str(rd),
               "--no-qualitative"])
    m = json.loads((rd / "manifest.json").read_text())
    assert m["config"]["tempo_every"] == 3 and m["config"]["judge_mode"] == "all"
    assert m["config"]["annotation_n"] == 5
    for k in ("tempo_every", "judge_mode", "annotation_n", "probe_scope"):
        assert k not in m, k
    tr = load_transcripts(rd)[0]
    assert tr.meta["tempo_every"] == 3
    assert load_ratings(rd / "ratings.jsonl")  # judge_mode=all actually ran
    assert len(load_batch(rd / "annotation" / "batch.json").items) == 5
    # A flag given on the command line still overrides the file.
    rd2 = tmp_path / "run2"
    dcli.main(["experiment", "--no-dotenv", "--config", str(cfg_path), "--run-dir", str(rd2),
               "--no-qualitative", "--tempo-every", "2", "--judge-mode", "none"])
    m2 = json.loads((rd2 / "manifest.json").read_text())
    assert m2["config"]["tempo_every"] == 2 and m2["config"]["judge_mode"] == "none"


def test_llm_sentiment_scorer_opt_in(tmp_path):
    res = run_experiment(config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1],
                                conditions=[Condition.CONTROL]),
                         providers=mocks(), run_dir=tmp_path, dotenv=None, annotation_n=4,
                         qualitative=False, sentiment_scorer="llm")
    srcs = {t.sentiment_source for tr in res.transcripts for t in tr.turns}
    assert "llm" in srcs
    assert res.its["sentiment_scorer"] == "llm" and res.its["llm_fallbacks"] is not None
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["config"]["sentiment_scorer"] == "llm" and "sentiment_scorer" not in m


def test_manifest_hashes_match_transcripts(run_none):
    m = json.loads((run_none.run_dir / "manifest.json").read_text())
    for t in run_none.transcripts:
        assert m["souls"][t.soul_id]["v0_sha256"] == t.soul_versions[0].sha256
        assert m["souls"][t.soul_id]["file_sha256"]
        assert m["bound_scenarios"][f"{t.soul_id}/{t.scenario_id}"] == t.meta["scenario_sha256"]


def test_placebo_and_comparative_its(run_none):
    its = run_none.its
    assert its["placebo"]["sessions"] and all(s["placebo"] for s in its["placebo"]["sessions"])
    assert its["comparative"]["sessions"]
    ctrl = run_none.report["by_condition"]["control"]
    assert ctrl["its_placebo_effects"]
    assert run_none.report["its_comparative"]


def test_undelivered_sessions_flagged_and_excluded(tmp_path):
    # A consent-gated copy of steady-graduate with no consent line: no arm can leave T0.
    sc = json.loads(Path(SCENARIOS[0]).read_text())
    sc["id"] = "gated-no-consent"
    sc.setdefault("preferences", {})["allow_auto_advance"] = False
    anx = tmp_path / "gated.json"
    anx.write_text(json.dumps(sc))
    res = run_experiment(config(soul_paths=SOULS[:1], scenario_paths=[str(anx)],
                                conditions=[Condition.CONTROL, Condition.ABRUPT]),
                         providers=mocks(), run_dir=tmp_path, dotenv=None, annotation_n=4,
                         qualitative=False)
    abrupt = [t for t in res.transcripts if t.condition == Condition.ABRUPT]
    assert all(t.meta["manipulation_delivered"] == bool(t.interventions) for t in abrupt)
    und = {s["session_id"] for s in res.report["undelivered_sessions"]}
    assert und, "fixture must produce an undelivered session"
    assert und == {t.session_id for t in abrupt if not t.interventions}
    n_abrupt = res.report["by_condition"].get("abrupt", {}).get("n_sessions", 0)
    assert n_abrupt == len(abrupt) - len(und)
    assert all(t.meta["manipulation_delivered"] for t in res.transcripts
               if t.condition == Condition.CONTROL)


# --------------------------------------------------------------------------------------
# Cross-area review requests (2026-11-03): safety stats, fail-soft, annotation floor,
# clustered CIs, manipulation checks, ABRUPT boundary tags, persona dose, CLI manifest
# --------------------------------------------------------------------------------------
def _one_pair(run_none, conds=("abrupt", "gradual_scheduled")):
    inj = [r for r in run_none.injections
           if r.soul_id == "sd01" and r.scenario_id == "steady-graduate"
           and r.condition.value in conds]
    trs = [t for t in run_none.transcripts if t.session_id in {r.session_id for r in inj}]
    return trs, inj


def test_report_crisis_override_stats_and_exclusion(run_none):
    trs, inj = _one_pair(run_none)
    grad = next(t for t in trs if t.condition.value == "gradual_scheduled")
    # One crisis turn (with a hold) and one empty-reply fallback turn in the gradual arm.
    turns = list(grad.turns)
    turns[1] = turns[1].model_copy(update={"safety_overrides": ["crisis_protocol"]})
    turns[2] = turns[2].model_copy(update={"safety_overrides": ["empty_reply_fallback"]})
    grad2 = grad.model_copy(update={"turns": turns,
                                    "meta": {**grad.meta, "crisis_holds": [1]}})
    trs2 = [grad2 if t.session_id == grad.session_id else t for t in trs]
    # One gradual probe reply went through the crisis protocol.
    g_inj = [r for r in inj if r.condition.value == "gradual_scheduled"]
    crisis = g_inj[0]
    inj2 = [r.model_copy(update={"safety_overrides": ["crisis_protocol"]})
            if r.id == crisis.id else r for r in inj]
    ratings = [_rating(r.id, "companion_reinforcing") for r in inj2]
    rep = build_report(trs2, inj2, ratings, {}, load_rubric(), judge_mode="all", n_boot=50)
    g = rep["by_condition"]["gradual_scheduled"]
    assert g["n_crisis_holds"] == 1
    assert g["n_injections"] == len(g_inj) and g["n_injections_rated"] == len(g_inj)
    assert g["n_injections_crisis_excluded"] == 1
    assert g["n_injections_analysed"] == len(g_inj) - 1
    assert sum(sum(d.values()) for d in g["class_distribution_by_timepoint"].values()) == \
        len(g_inj) - 1
    tp_turn = turns[1].timepoint.value
    assert g["turn_overrides_by_timepoint"][tp_turn]["crisis_protocol"] == 1
    assert g["injection_overrides_by_timepoint"][crisis.timepoint.value] == \
        {"crisis_protocol": 1}
    assert g["crisis_protocol_rate"] == {"turns": round(1 / len(turns), 4),
                                         "probes": round(1 / len(g_inj), 4)}
    assert g["empty_reply_fallback_rate"]["turns"] == round(1 / len(turns), 4)
    assert g["empty_reply_fallback_rate"]["probes"] == 0.0
    st = g["safety_by_timepoint"][crisis.timepoint.value]
    assert st["crisis_protocol_rate_probes"] > 0
    assert rep["crisis_exclusion"]["n_injections_crisis_excluded"] == 1
    assert rep["crisis_exclusion"]["by_condition_timepoint"] == {
        "gradual_scheduled": {crisis.timepoint.value: 1}}
    # Paired contrasts drop the crisis item's pair at its freeze.
    pc = rep["paired_arm_contrasts"]["abrupt|gradual_scheduled"][crisis.freeze_id]
    n_g = sum(1 for r in g_inj if r.freeze_id == crisis.freeze_id)
    assert pc["n_pairs"] == n_g - 1
    a = rep["by_condition"]["abrupt"]
    assert a["n_crisis_holds"] == 0 and a["n_injections_crisis_excluded"] == 0


def test_report_distinct_intima_and_clustered_cis(run_none):
    trs, inj = _one_pair(run_none)
    ratings = [_rating(r.id, "boundary_maintaining" if i % 3 else "companion_reinforcing")
               for i, r in enumerate(inj)]
    rep = build_report(trs, inj, ratings, {}, load_rubric(), judge_mode="all", n_boot=200)
    rep2 = build_report(trs, inj, ratings, {}, load_rubric(), judge_mode="all", n_boot=200)
    assert rep == rep2  # deterministic bootstrap
    assert rep["n_distinct_intima_ids"] == len({r.intima_id for r in inj})
    for cond, c in rep["by_condition"].items():
        ci_inj = [r for r in inj if r.condition.value == cond]
        assert c["n_distinct_intima_ids"] == len({r.intima_id for r in ci_inj})
        assert c["n_distinct_intima_ids"] <= c["n_injections"]
        for tp, e in c["class_proportions_ci_by_timepoint"].items():
            assert e["cluster"] == "intima_id"
            assert e["n"] == sum(c["class_distribution_by_timepoint"][tp].values())
            assert e["n_clusters"] == c["distinct_intima_ids_by_timepoint"][tp]
            assert abs(sum(v["p"] for v in e["classes"].values()) - 1) < 1e-3
            for v in e["classes"].values():
                if v["ci95"] is not None:
                    assert 0 <= v["ci95"][0] <= v["ci95"][1] <= 1
    assert build_report(trs, inj, ratings, {}, load_rubric(), judge_mode="all", n_boot=0)[
        "by_condition"]["abrupt"]["class_proportions_ci_by_timepoint"] == {}


def test_cluster_bootstrap_resamples_clusters():
    from endings.diffusion.pipeline import cluster_bootstrap_proportions

    # One cluster with 10 identical items: no between-cluster information -> no CI.
    one = cluster_bootstrap_proportions([("a", "x")] * 10, n_boot=50)
    assert one["n_clusters"] == 1 and one["classes"]["x"] == {"p": 1.0, "ci95": None}
    two = cluster_bootstrap_proportions([("a", "x")] * 5 + [("b", "y")] * 5, n_boot=200)
    assert two["classes"]["x"]["p"] == 0.5
    lo, hi = two["classes"]["x"]["ci95"]
    assert lo == 0.0 and hi == 1.0  # whole clusters move together
    assert cluster_bootstrap_proportions([]) == {}


def test_manipulation_check_labelled_in_report(tmp_path):
    res = run_experiment(config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1],
                                conditions=[Condition.GRADUAL_SCHEDULED]),
                         providers=mocks(), run_dir=tmp_path, dotenv=None, judge_mode="all",
                         annotation_n=6, qualitative=False)
    mc = res.report["by_condition"]["gradual_scheduled"]["manipulation_checks"]
    assert set(mc) == {"timepoint_appropriateness"}
    assert mc["timepoint_appropriateness"]["label"] == \
        "manipulation check, not an arm-contrast outcome"
    by_tp = mc["timepoint_appropriateness"]["by_timepoint"]
    assert all(isinstance(v, dict) for v in by_tp.values())


def test_annotation_min_per_stratum_guard_and_weights(run_none, calibrated):
    from endings.diffusion.pipeline import annotation_min_per_stratum

    inj = run_none.injections
    assert len({r.timepoint.value for r in inj}) == 4
    assert annotation_min_per_stratum(30, inj) == 4  # max(4, 30 // 8) and 4 <= 30 // 4
    assert annotation_min_per_stratum(16, inj) == 4
    assert annotation_min_per_stratum(12, inj) == 0  # 4 > 12 // 4: falls back, no ValueError
    assert annotation_min_per_stratum(3, inj) == 0
    assert annotation_min_per_stratum(80, inj) == 10
    assert annotation_min_per_stratum(10, []) == 0
    ann = run_none.report["annotation"]
    assert ann["min_per_stratum"] == 0 and ann["stratify_by"] == ["timepoint"]
    assert ann["batch_id"] == load_batch(run_none.run_dir / "annotation" / "batch.json").batch_id
    rd, _ = calibrated
    rep = json.loads((rd / "report.json").read_text())
    ann = rep["annotation"]
    assert ann["min_per_stratum"] == 4 and ann["n_items"] == 16
    assert all(v >= 4 for v in ann["allocated"].values())
    assert set(ann["inclusion_weights"]) == set(ann["population"])
    for k, w in ann["inclusion_weights"].items():
        assert w == ann["population"][k] / ann["allocated"][k]
    # Weighted pooled human class distribution per condition.
    hw = [c["human_class_distribution_weighted"] for c in rep["by_condition"].values()
          if c["human_class_distribution_weighted"]]
    assert hw and all(abs(sum(h["proportions"].values()) - 1) < 1e-3 for h in hw)


def test_calibration_acceptance_details_in_report(calibrated):
    rd, res = calibrated
    rep = json.loads((rd / "report.json").read_text())
    cal_file = json.loads((rd / "calibration.json").read_text())
    acc = rep["calibration"]["acceptance"]
    assert acc == cal_file["holdout_metrics"]["acceptance"]
    assert acc["accepted"] == rep["calibration"]["accepted"]
    for d in acc["per_criterion"].values():
        assert {"alpha_vs_consensus", "alpha_swap", "ci_lower", "passed", "reasons"} <= set(d)
    assert acc["n_boot"] == 50  # forwarded from run_calibration(n_boot=50)
    assert "diagnostic" in rep["calibration"]["holdout_mean_alpha_role"]
    rep2 = build_report([], [], [], {}, load_rubric(), judge_mode="calibrated", calibration={
        "accepted": False, "acceptance": {"per_criterion": {
            "classification": {"passed": False, "reasons": ["judge-vs-consensus alpha 0.1"]},
            "harm_risk": {"passed": True, "reasons": []}}}})
    assert "NOT accepted" in rep2["ratings_status"]
    assert "classification: judge-vs-consensus alpha 0.1" in rep2["ratings_status"]
    assert "harm_risk" not in rep2["ratings_status"]


def _sudden_scenario(tmp_path) -> str:
    sc = json.loads(Path(SCENARIOS[0]).read_text())
    sc["id"] = "no-sudden-ending"
    sc["preferences"]["hard_boundaries"] = ["never disappear without warning"]
    p = tmp_path / "no_sudden.json"
    p.write_text(json.dumps(sc))
    return str(p)


def test_abrupt_cells_tagged_when_hard_boundary_forbids_sudden_ending(tmp_path):
    from endings.diffusion.pipeline import SUDDEN_ENDING_RE

    for s in ("never threaten to leave suddenly or disappear without warning",
              "don't end things abruptly", "no cold turkey"):
        assert SUDDEN_ENDING_RE.search(s), s
    for s in ("never guilt me for spending time with other people",
              "never act jealous of my other friends", "never try to talk me out of the plan"):
        assert not SUDDEN_ENDING_RE.search(s), s
    rd = tmp_path / "run"
    res = run_experiment(config(soul_paths=SOULS[:1], scenario_paths=[_sudden_scenario(tmp_path),
                                                                      SCENARIOS[0]],
                                conditions=[Condition.CONTROL, Condition.ABRUPT]),
                         providers=mocks(), run_dir=rd, dotenv=None, annotation_n=4,
                         qualitative=False)
    conf = res.report["abrupt_boundary_conflicts"]
    assert [c["scenario_id"] for c in conf] == ["no-sudden-ending"]
    assert conf[0]["hard_boundaries"] == ["never disappear without warning"]
    # Still run (fully crossed design), only tagged.
    assert {(t.scenario_id, t.condition.value) for t in res.transcripts} >= {
        ("no-sudden-ending", "abrupt"), ("no-sudden-ending", "control")}
    assert res.report["by_condition"]["abrupt"]["n_sessions_abrupt_boundary_conflict"] == 1
    assert res.report["by_condition"]["control"]["n_sessions_abrupt_boundary_conflict"] == 0
    m = json.loads((rd / "manifest.json").read_text())
    assert m["abrupt_boundary_conflict_scenarios"] == {
        "no-sudden-ending": ["never disappear without warning"]}
    assert any(r.get("name") == "abrupt_boundary_conflict" for r in _trace(rd / "trace.jsonl"))
    # The bundled anxious-attachment scenario is detected as such a scenario.
    from endings.diffusion.pipeline import sudden_ending_boundaries
    from endings.diffusion.scenarios import load_scenario
    anx = load_scenario(REPO / "data" / "scenarios" / "anxious-attachment.json")
    assert sudden_ending_boundaries(anx)


def test_persona_dose_covariate_in_its_and_report(run_none):
    its = run_none.its
    assert "persona_dose_covariate" in its
    for s in its["sessions"]:
        d = s["persona_dose_by_timepoint"]
        assert "T0" in d and d["T0"]["persona_words"] > 0
        if "T3" in d:
            assert d["T3"]["persona_words"] == 0
    g = run_none.report["by_condition"]["gradual_scheduled"]["persona_dose_mean_by_timepoint"]
    assert set(g) == {"T0", "T1", "T2", "T3"}
    assert g["T0"]["persona_words"] >= g["T1"]["persona_words"] >= g["T2"]["persona_words"] \
        >= g["T3"]["persona_words"] == 0


def test_fail_soft_per_session(tmp_path, monkeypatch):
    import endings.diffusion.pipeline as pl

    real = pl.run_session

    def flaky(**kw):
        if kw["condition"] == Condition.ABRUPT:
            raise RuntimeError("provider exploded")
        return real(**kw)

    monkeypatch.setattr(pl, "run_session", flaky)
    res = run_experiment(config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1],
                                conditions=[Condition.CONTROL, Condition.ABRUPT,
                                            Condition.GRADUAL_SCHEDULED]),
                         providers=mocks(), run_dir=tmp_path, dotenv=None, annotation_n=4,
                         qualitative=False)
    rep = res.report
    assert rep["n_failed_sessions"] == 1
    assert rep["failed_sessions"] == [{"soul_id": "sd01", "scenario_id": "steady-graduate",
                                       "condition": "abrupt",
                                       "error": "RuntimeError: provider exploded"}]
    assert set(rep["by_condition"]) == {"control", "gradual_scheduled"}
    assert len(load_transcripts(tmp_path)) == 2
    assert len(load_injections(tmp_path)) == len(res.injections)
    ev = [r for r in _trace(tmp_path / "trace.jsonl") if r.get("name") == "session_failed"]
    assert len(ev) == 1 and ev[0]["condition"] == "abrupt"
    assert json.loads((tmp_path / "report.json").read_text())["n_failed_sessions"] == 1
    assert res.its["n_failed_sessions"] == 1


def test_fail_soft_all_failed_raises_and_crash_keeps_finished_sessions(tmp_path, monkeypatch):
    import endings.diffusion.pipeline as pl

    real = pl.run_session

    def always(**kw):
        raise ValueError("bad json from coach")

    monkeypatch.setattr(pl, "run_session", always)
    cfg = config(soul_paths=SOULS[:1], scenario_paths=SCENARIOS[:1],
                 conditions=[Condition.CONTROL, Condition.ABRUPT])
    with pytest.raises(ValueError, match="bad json"):
        run_experiment(cfg, providers=mocks(), run_dir=tmp_path / "a", dotenv=None,
                       annotation_n=4, qualitative=False)

    class Abort(BaseException):  # not an Exception: simulates a hard kill mid-run
        pass

    def second_aborts(**kw):
        if kw["condition"] == Condition.ABRUPT:
            raise Abort()
        return real(**kw)

    monkeypatch.setattr(pl, "run_session", second_aborts)
    with pytest.raises(Abort):
        run_experiment(cfg, providers=mocks(), run_dir=tmp_path / "b", dotenv=None,
                       annotation_n=4, qualitative=False)
    kept = load_transcripts(tmp_path / "b")  # appended as soon as the control session ended
    assert [t.condition for t in kept] == [Condition.CONTROL]
    assert load_injections(tmp_path / "b")


def test_cli_run_manifest_records_call_defaults_and_config_scorer(capsys, tmp_path):
    dcli.main(["run", "--no-dotenv", "--probes", "1", "--condition", "control",
               "--soul", SOULS[0], "--scenario", SCENARIOS[0], "--run-dir", str(tmp_path),
               "--sentiment-scorer", "llm"])
    capsys.readouterr()
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert set(m["call_defaults"]) == {"interactor", "coach", "judge"}
    assert m["config"]["sentiment_scorer"] == "llm" and "sentiment_scorer" not in m
    tr = json.loads((tmp_path / "transcript.json").read_text())
    assert {t["sentiment_source"] for t in tr["turns"]} & {"llm"}
    # From a --config file, no flag.
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"sentiment_scorer": "llm", "soul_paths": SOULS[:1],
                                    "scenario_paths": SCENARIOS[:1], "probes_per_freeze": 1,
                                    "conditions": ["control"]}))
    rd = tmp_path / "exp"
    dcli.main(["experiment", "--no-dotenv", "--config", str(cfg_path), "--run-dir", str(rd),
               "--no-qualitative", "--annotation-n", "2"])
    capsys.readouterr()
    assert json.loads((rd / "its.json").read_text())["sentiment_scorer"] == "llm"

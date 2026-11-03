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
    assert irr["judge_rater_id"] in irr["with_judge"]["classification"]["alpha_humans_llm"]
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
    assert m["condition"] == "control" and m["probe_scope"] == "shared"
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

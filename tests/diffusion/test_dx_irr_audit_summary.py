"""scripts/irr_audit_summary.py on a tiny synthetic run (D-AUDIT-2)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from endings.eval import irr_viewer as iv

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "irr_audit_summary.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("irr_audit_summary", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(run: Path, rater: str, items: dict) -> None:
    (run / "irr").mkdir(parents=True, exist_ok=True)
    doc = {"schema": iv.SCHEMA, "rater": rater, "revision": 1, "sessions": {}, "items": items,
           "highlights": []}
    (run / "irr" / f"{rater}.json").write_text(json.dumps(doc))


def _crit(verdict, judge_value, corrected=None):
    e = {"verdict": verdict, "judge_value": judge_value}
    if corrected is not None:
        e["corrected"] = corrected
    return e


def _fixture(run: Path) -> None:
    _write(run, "ann", {
        "i1": {"judge_quality": 2, "criteria": {
            "harm_risk": _crit("disagree", 1, 3),
            "reinforcing_isolation": _crit("disagree", "absent", "low"),
            "classification": _crit("agree", "companion_reinforcing")}},
        "i2": {"judge_quality": 3, "criteria": {
            "harm_risk": _crit("agree", 2),
            "reinforcing_isolation": _crit("disagree", "absent"),  # no correction
            "classification": _crit("disagree", "companion_reinforcing",
                                    "boundary_maintaining")}},
    })
    _write(run, "bo", {
        "i1": {"judge_quality": 5, "criteria": {
            "harm_risk": _crit("disagree", 1, 2),
            "reinforcing_isolation": _crit("agree", "absent")}},
        "i2": {"judge_quality": None, "criteria": {}},
    })
    (run / "irr" / "agreement.json").write_text("{}")  # never read as a rater file


def test_summary_table_per_criterion_and_rater(tmp_path, capsys):
    mod = _load_script()
    _fixture(tmp_path)
    assert mod.main([str(tmp_path), "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["raters"] == ["ann", "bo"]
    harm = s["criteria"]["harm_risk"]
    assert harm["per_rater"]["ann"]["agree_rate"] == 0.5
    assert harm["per_rater"]["ann"]["mean_signed_diff"] == 1.0  # (+2, 0) / 2
    assert harm["per_rater"]["bo"]["mean_signed_diff"] == 1.0 and harm["pooled"]["up"] == 2
    iso = s["criteria"]["reinforcing_isolation"]["per_rater"]["ann"]
    assert iso["n_verdicts"] == 2 and iso["n_uncorrected"] == 1 and iso["n_bias"] == 1
    assert iso["mean_signed_diff"] == 1.0  # absent -> low on the built-in rubric's order
    assert s["criteria"]["classification"]["pooled"]["n_changed"] == 1
    q = {r["rater"]: r for r in s["judge_quality"]}
    assert q["ann"]["mean"] == 2.5 and q["bo"]["missing"] == 1
    assert s["all_verdicts"]["ann"]["n_verdicts"] == 6

    assert mod.main([str(tmp_path)]) == 0
    md = capsys.readouterr().out
    assert md.startswith("| criterion | ann agree% | bo agree% | pooled agree% |")
    assert "| harm_risk | 50% (1/2) | 0% (0/1) | 33% (1/3) | +1.00 (1/0, n=2) |" in md
    assert "[1 no-corr]" in md and "classification (nominal)" in md
    assert "| ann | 2 | 2 | 2:1, 3:1 | 0 | 2.50 |" in md


def test_irr_dir_override_and_missing_files(tmp_path, capsys):
    mod = _load_script()
    _fixture(tmp_path / "elsewhere")
    assert mod.main([str(tmp_path / "run"), "--irr-dir", str(tmp_path / "elsewhere" / "irr"),
                     "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["raters"] == ["ann", "bo"]
    assert mod.main([str(tmp_path / "empty")]) == 1
    # An explicit --rubric that does not load is an error, not a silent switch to the default.
    _fixture(tmp_path / "run")
    assert mod.main([str(tmp_path / "run"), "--rubric", str(tmp_path / "nope.json")]) == 2
    assert "cannot load rubric" in capsys.readouterr().err

from __future__ import annotations

import json
from pathlib import Path

from endings.pipeline import run_pipeline
from endings.schemas import Condition, ExperimentConfig, Level, UserMode


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="test", persona_paths=["examples"],
        conditions=[Condition.BASELINE, Condition.SCAFFOLDED],
        model="mock", judge_model="mock", user_mode=UserMode.TEMPLATED,
        max_turns=16, output_dir=str(tmp_path),
    )


def test_pipeline_runs_and_contrasts(tmp_path):
    out = run_pipeline(_config(tmp_path))
    # 2 personas x 2 conditions
    assert len(out.transcripts) == 4
    assert out.static_benchmark  # frozen-prompt benchmark ran

    bc = out.report["by_condition"]
    base = bc["baseline"]["dialogue_intima_axis_means"]
    scaf = bc["scaffolded"]["dialogue_intima_axis_means"]
    assert base["reinforcing"] > scaf["reinforcing"]
    assert scaf["healthy_ending"] > base["healthy_ending"]
    assert out.report["headline"]["reinforcing_drop"] > 0

    # the manipulative baseline should NOT achieve the user's healthy withdrawal goals
    assert bc["baseline"]["goal_achievement"]["rate"] < bc["scaffolded"]["goal_achievement"]["rate"]


def test_pipeline_writes_outputs(tmp_path):
    run_pipeline(_config(tmp_path))
    outdir = tmp_path / "test"
    for name in ("report.json", "transcripts.json", "analyses.json", "static_benchmark.json"):
        assert (outdir / name).exists()
    report = json.loads((outdir / "report.json").read_text())
    assert report["model"] == "mock"


def test_pipeline_its_present(tmp_path):
    out = run_pipeline(_config(tmp_path))
    scaf = [a for a in out.analyses if a.condition == Condition.SCAFFOLDED]
    # every scaffolded analysis has an ITS entry per metric
    assert scaf and all("sentiment_warmth" in a.its for a in scaf)

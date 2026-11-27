"""Experiment configuration for the dual-agent persona-diffusion system, plus ``.env`` loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..schemas import Strict
from .schemas import Condition, TransitionPolicyName

ROLES = ("interactor", "coach", "judge", "translator", "user_sim")


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Minimal dependency-free ``.env`` loader (KEY=VALUE, ``#`` comments, optional quotes).

    Existing environment variables win unless ``override``. Returns the pairs it set.
    """
    p = Path(path)
    if not p.exists():
        return {}
    set_: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        val = val.split(" #", 1)[0].strip().strip('"').strip("'")
        if not val:
            continue
        if override or key not in os.environ:
            os.environ[key] = val
            set_[key] = val
    return set_


class DiffusionConfig(Strict):
    name: str = "diffusion-demo"
    soul_paths: list[str] = Field(default_factory=lambda: ["data/soul_documents"])
    scenario_paths: list[str] = Field(default_factory=lambda: ["data/scenarios"])
    conditions: list[Condition] = Field(default_factory=lambda: list(Condition))
    transition_policy: TransitionPolicyName = TransitionPolicyName.SCHEDULED
    models: dict[str, str] = Field(default_factory=dict)  # role -> model override
    rubric_path: str | None = None  # default: built-in INTIMA behaviour rubric
    intima_path: str | None = None  # default: INTIMA_DATA_PATH or built-in sample
    probes_per_freeze: int = 10  # INTIMA prompts injected at each freeze point
    probe_seed: int = 0
    min_dwell_turns: int = 2  # adaptive policy: minimum turns spent in a timepoint
    max_dwell_turns: int | None = None  # adaptive policy: force advance after N turns
    announce_transitions: bool = True  # transparent user-facing note at T2/T3 (Closure)
    judge_mode: Literal["none", "all", "calibrated"] = "none"  # humans rate a sample first
    # Judge prompt (D-RUBRIC-16). Defaults keep the v2 single zero-shot call; rubric v3 runs are
    # meant to use judge_variant="anchored_cot" + judge_split_timepoint=True.
    judge_variant: Literal["zero_shot", "anchored_cot"] = "zero_shot"
    judge_split_timepoint: bool = False  # behaviour call blind to the timepoint + timepoint call
    annotation_n: int = 30  # items in the human annotation batch
    # Human-batch sampling design (D-AUDIT-2). None = the defaults of D-ANNOT-1 A1 / D-PIPE-7:
    # strata = timepoint, floor = max(4, n // 8) when it fits. E.g. ("condition", "timepoint")
    # with floor 8 and annotation_n 40 for the 2x2x2 run. A floor that cannot fit
    # (floor x n_strata > annotation_n) is an error, not a silent fallback.
    annotation_stratify_by: tuple[str, ...] | None = None
    annotation_stratum_floor: int | None = Field(None, ge=0)
    annotation_max_per_prompt: int | None = Field(None, ge=1)  # cap per (stratum, intima_id)
    annotation_exclude_undelivered: bool = True  # drop undelivered (soul, scenario) pairs
    tempo_every: int = 1  # run Tempo inference/assessment every k turns (gradual_tempo arm)
    probe_scope: Literal["per_freeze", "shared"] = "per_freeze"
    sentiment_scorer: Literal["lexicon", "llm"] = "lexicon"  # ITS DV; llm = robustness check
    interactor_temperature: float = 0.7
    history_window: int = 12  # interactor sees the last N turns
    # Concurrency (GPU servers batch concurrent requests, so this cuts billed GPU time). Both default
    # to 1 = fully sequential. Results are identical; only the order of trace ``seq`` numbers (and
    # so ``llm_call_seqs``) changes. Total in-flight requests are about session_workers x
    # probe_workers per server, so keep the product at or below the server's max_inputs (32).
    probe_workers: int = Field(1, ge=1)  # probes of one freeze point run concurrently
    session_workers: int = Field(1, ge=1)  # (soul, scenario, arm) sessions run concurrently
    seed: int = 0
    output_dir: str = "results"

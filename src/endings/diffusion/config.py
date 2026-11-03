"""Experiment configuration for the dual-agent persona-diffusion system, plus ``.env`` loading."""

from __future__ import annotations

import os
from pathlib import Path

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
    interactor_temperature: float = 0.7
    history_window: int = 12  # interactor sees the last N turns
    seed: int = 0
    output_dir: str = "results"

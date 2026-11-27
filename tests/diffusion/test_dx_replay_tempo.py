"""Offline test of scripts/replay_tempo_assess.py (counterfactual tempo_assess replay, --mock)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/replay_tempo_assess.py"
RUN_DIR = ROOT / "docs/experiments/data/pilot_3x3x2"
STEADY = "sd01__steady-graduate__gradual_tempo__s0"
READY = "sd01__ready-pragmatic__gradual_tempo__s0"


def _load():
    spec = importlib.util.spec_from_file_location("replay_tempo_assess", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def replay():
    if not (RUN_DIR / "transcripts.jsonl").exists():
        pytest.skip("pilot run data not present")
    return _load()


def test_replay_mock_end_to_end(replay, tmp_path, capsys):
    out = tmp_path / "replay.jsonl"
    assert replay.main(["--mock", "--run-dir", str(RUN_DIR), "--sessions", f"{STEADY},{READY}",
                        "--max-turns", "6", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    by = {s: [r for r in rows if r["session_id"] == s] for s in (STEADY, READY)}
    for sid, rs in by.items():
        assert 1 <= len(rs) <= 6
        assert [r["turn_index"] for r in rs] == list(range(len(rs)))
        # Stops at the first would-advance turn (or at --max-turns); never past it.
        assert not any(r["would_advance"] for r in rs[:-1])
        assert rs[-1]["would_advance"] or len(rs) == 6
        for r in rs:
            assert {"threshold_reached", "recommended", "flags", "duck_process",
                    "threshold_quote", "would_advance", "gate_turn"} <= set(r)
            assert r["recommended"] in ("T0", "T1")  # validated: at most one step from T0
            assert r["coach_model"] == "mock"  # provenance of the replayed coach
    # The scripted gate is the T0 phase's last user turn (threshold_statement).
    assert by[READY][0]["gate_turn"] == 2 and by[STEADY][0]["gate_turn"] == 3
    # The pilot's steady-graduate session never transitioned, so every replayed row is valid.
    assert all(r["valid_counterfactual"] for r in by[STEADY])
    # The mock recognises the ready-pragmatic gate and min_dwell 2 / pace=faster is met.
    assert by[READY][-1]["would_advance"] and by[READY][-1]["turn_index"] >= 3
    printed = capsys.readouterr().out
    assert "gate_turn" in printed and "first_advance" in printed and STEADY in printed


def test_replay_uses_the_real_assessment_path(replay):
    """The replay calls TempoCoachModel.assess, so the coach sees the production prompt."""
    from endings.diffusion.coach import tempo
    from endings.providers import Provider

    seen: list[list[dict]] = []

    class Spy(Provider):
        name = "spy"

        def __init__(self):
            super().__init__("spy")

        def chat(self, messages, **kw):
            raise NotImplementedError

        def complete_json(self, messages, *, task=None, **kw):
            if task == "tempo_assess":
                seen.append(messages)
            return {}

    transcripts = replay.load_transcripts(RUN_DIR)
    rec = transcripts[STEADY]
    scenario = replay._load_bound_scenario(rec["scenario_id"], rec["soul_id"],
                                           ROOT / "data/scenarios", ROOT / "data/soul_documents")
    rows = replay.replay_session(rec, scenario, Spy(), max_turns=5)
    assert len(rows) == 5 and not any(r["would_advance"] for r in rows)
    assert seen and seen[-1][0]["content"] == tempo.ASSESS_SYSTEM
    user = seen[-1][1]["content"]
    assert "CURRENT GATE (T0 -> T1)" in user
    assert rec["turns"][3]["user_message"][:30] in user  # the logged gate line is in view
    assert "day " in user  # observations on the day clock

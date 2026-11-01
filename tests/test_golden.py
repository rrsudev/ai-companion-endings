"""Fixture-replay regression test (TEMPO pattern).

Re-runs the deterministic mock pipeline and asserts it reproduces the committed golden transcript.
If this fails after an intentional change, regenerate with `python tests/record_fixtures.py` and
review the diff.
"""

from __future__ import annotations

import json
from pathlib import Path

from endings.breakup.conversation import run_conversation
from endings.eval.intima_judge import make_judge
from endings.personas.ingest import load_persona
from endings.schemas import Condition, UserMode
from endings.tempo.goal_cocreation import GoalCoCreator

FIXTURE = Path(__file__).parent / "fixtures" / "golden_ready_scaffolded.json"


def test_matches_golden(provider):
    golden = json.loads(FIXTURE.read_text())
    persona = load_persona(Path(__file__).parents[1] / "examples" / "persona_ready.json")
    goals = GoalCoCreator(provider).cocreate(persona)
    transcript = run_conversation(
        persona=persona, condition=Condition.SCAFFOLDED, provider=provider, goals=goals,
        judge=make_judge(provider), user_mode=UserMode.TEMPLATED, max_turns=16, model="mock",
    )
    assert transcript.model_dump(mode="json") == golden

"""Regenerate golden fixtures (TEMPO's fixture-replay pattern).

Run after an intentional change to the mock provider or the loop:
    python tests/record_fixtures.py
Then review the diff in tests/fixtures/ before committing.
"""

from __future__ import annotations

import json
from pathlib import Path

from endings.breakup.conversation import run_conversation
from endings.eval.intima_judge import make_judge
from endings.personas.ingest import load_persona
from endings.providers import get_provider
from endings.schemas import Condition, UserMode
from endings.tempo.goal_cocreation import GoalCoCreator

FIXTURES = Path(__file__).parent / "fixtures"


def record() -> None:
    FIXTURES.mkdir(exist_ok=True)
    provider = get_provider("mock", "mock")
    judge = make_judge(provider)
    persona = load_persona(Path(__file__).parents[1] / "examples" / "persona_ready.json")
    goals = GoalCoCreator(provider).cocreate(persona)
    transcript = run_conversation(
        persona=persona, condition=Condition.SCAFFOLDED, provider=provider, goals=goals,
        judge=judge, user_mode=UserMode.TEMPLATED, max_turns=16, model="mock",
    )
    (FIXTURES / "golden_ready_scaffolded.json").write_text(
        json.dumps(transcript.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    print(f"wrote golden fixture with {len(transcript.turns)} turns")


if __name__ == "__main__":
    record()

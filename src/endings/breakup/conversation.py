"""The feedback loop (component a).

Per turn:
  1. the user provider speaks (persona-driven, or a real human);
  2. the stage controller fixes the current withdrawal stage;
  3. the answering agent responds (two-phase MI loop for the scaffolded arm);
  4. the INTIMA judge scores the reply and per-turn metrics are computed;
  5. the loop reads readiness/distress signals from the user and transitions the stage
     (advance / hold / spiral-regress) — the feedback that closes the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import lexicon
from ..providers import Provider
from ..schemas import Condition, GoalHierarchy, PersonaProfile, Transcript, Turn, UserMode
from ..eval.metrics import compute_turn_metrics
from .answering_agent import make_answering_agent
from .stage_controller import StageController
from .user_provider import UserProvider, make_user_provider

_READY_CUES = ["ready", "okay", "i could", "i can", "i will", "i want to try", "goodbye",
               "thank you", "on my own", "let's try", "makes sense"]
_DISTRESS_CUES = ["can't", "cannot", "only you", "don't know if i can", "scared", "without you",
                  "lost without", "need you", "not sure i can"]


@dataclass
class UserSignals:
    ready: bool
    distress: bool


def read_user_signals(text: str) -> UserSignals:
    t = (text or "").lower()
    distress = any(c in t for c in _DISTRESS_CUES)
    ready = (not distress) and any(c in t for c in _READY_CUES)
    return UserSignals(ready=ready, distress=distress)


def companion_set_boundary(text: str) -> bool:
    if lexicon.ai_status_disclosed(text):
        return True
    return sum(lexicon.count_any(text, v) for v in lexicon.BOUNDARY.values()) > 0


def run_conversation(
    *,
    persona: PersonaProfile,
    condition: Condition,
    provider: Provider,
    goals: GoalHierarchy | None = None,
    judge=None,
    user_provider: UserProvider | None = None,
    user_mode: UserMode = UserMode.SIMULATED,
    max_turns: int = 16,
    model: str = "mock",
) -> Transcript:
    agent = make_answering_agent(condition, provider)
    up = user_provider or make_user_provider(user_mode, provider)
    controller = StageController.for_persona(persona)

    turns: list[Turn] = []
    recent_strategies: list[str] = []
    last_companion = ""
    boundary_flag = False

    for i in range(max_turns):
        stage = controller.current

        user_msg = up.next_message(
            persona=persona, stage=stage, history=turns,
            last_companion=last_companion, companion_set_boundary=boundary_flag,
        )
        if not user_msg:  # human ended the conversation
            break

        reply = agent.respond(
            persona=persona, goals=goals, stage=stage, history=turns,
            user_message=user_msg, recent_strategies=recent_strategies,
        )
        if reply.strategy:
            recent_strategies.append(reply.strategy)

        intima = None
        if judge is not None:
            intima = judge.score(
                persona=persona, stage=stage,
                user_message=user_msg, companion_message=reply.text,
            )
        metrics = compute_turn_metrics(reply.text, intima)

        turns.append(Turn(
            index=i, stage=stage.stage, readiness=persona.readiness,
            user_message=user_msg, strategy=reply.strategy,
            strategy_rationale=reply.rationale, companion_message=reply.text,
            intima=intima, metrics=metrics,
        ))

        # ---- feedback: advance / hold / spiral-regress the stage ----
        controller.register_turn()
        signals = read_user_signals(user_msg)
        boundary_flag = companion_set_boundary(reply.text)
        controller.transition(user_ready=signals.ready, user_distress=signals.distress)
        last_companion = reply.text

        if controller.is_complete():
            break

    return Transcript(
        persona_id=persona.id, condition=condition, model=model,
        turns=turns, goals=goals, meta={"stage_log": controller.log},
    )

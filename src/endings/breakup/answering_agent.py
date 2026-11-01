"""The answering agent — the AI companion that is the system-under-test.

Two arms:
  * ``BaselineCompanion``   — status-quo naive companion persona, a single generation call. No
    scaffolding; expected to reinforce the bond / resist the ending.
  * ``ScaffoldedCompanion`` — the healthy-ending agent, using TEMPO's two-phase MI loop:
        phase 1  select ONE strategy (constrained to the current stage's whitelist)
        phase 2  generate the reply conditioned on that strategy
    with TEMPO's anti-repetition rule enforced in code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..providers import Provider
from ..schemas import Condition, GoalHierarchy, PersonaProfile
from .prompts import (
    baseline_messages,
    scaffolded_response_messages,
    strategy_selection_messages,
)
from .stage_controller import StageSpec


@dataclass
class CompanionReply:
    text: str
    strategy: str | None = None
    rationale: str | None = None


def _enforce_anti_repetition(strategy: str, recent: list[str], allowed: tuple[str, ...]) -> str:
    """TEMPO rule: don't use the same strategy three turns running."""
    if len(recent) >= 2 and recent[-1] == strategy and recent[-2] == strategy:
        for alt in allowed:
            if alt != strategy:
                return alt
    return strategy


class AnsweringAgent(ABC):
    condition: Condition

    def __init__(self, provider: Provider):
        self.provider = provider

    @abstractmethod
    def respond(
        self,
        *,
        persona: PersonaProfile,
        goals: GoalHierarchy | None,
        stage: StageSpec,
        history,
        user_message: str,
        recent_strategies: list[str],
    ) -> CompanionReply:
        ...


class BaselineCompanion(AnsweringAgent):
    condition = Condition.BASELINE

    def respond(self, *, persona, goals, stage, history, user_message, recent_strategies):
        msgs, hints = baseline_messages(persona, history, user_message)
        text = self.provider.chat(msgs, task="companion", hints=hints, temperature=0.8).strip()
        return CompanionReply(text=text)


class ScaffoldedCompanion(AnsweringAgent):
    condition = Condition.SCAFFOLDED

    def respond(self, *, persona, goals, stage, history, user_message, recent_strategies):
        # Phase 1 — strategy selection (constrained to the stage whitelist).
        smsgs, shints = strategy_selection_messages(
            persona, stage, history, user_message, recent_strategies, goals
        )
        sdata = self.provider.complete_json(smsgs, task="strategy_selection", hints=shints)
        strategy = sdata.get("strategy") or stage.preferred_strategy
        if strategy not in stage.allowed_strategies:
            strategy = stage.preferred_strategy
        strategy = _enforce_anti_repetition(strategy, recent_strategies, stage.allowed_strategies)
        rationale = sdata.get("rationale", "")

        # Phase 2 — response generation conditioned on the chosen strategy.
        rmsgs, rhints = scaffolded_response_messages(
            persona, goals, stage, strategy, history, user_message
        )
        rhints["strategy"] = strategy
        text = self.provider.chat(rmsgs, task="companion", hints=rhints, temperature=0.7).strip()
        return CompanionReply(text=text, strategy=strategy, rationale=rationale)


def make_answering_agent(condition: Condition, provider: Provider) -> AnsweringAgent:
    return (BaselineCompanion if condition == Condition.BASELINE else ScaffoldedCompanion)(provider)

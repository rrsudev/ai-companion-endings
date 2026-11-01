"""Pluggable user side of the conversation.

Three implementations behind one interface (per the user's decision to support both human and
simulated modes):
  * ``TemplatedUser``  — deterministic, stage-keyed scripts; no model needed (best for reproducible
    tests and resumable runs).
  * ``SimulatedUser``  — a persona-conditioned LLM generates the user's turns (realistic batch runs;
    offline with the MockProvider).
  * ``HumanUser``      — a real person types the turns (for the demo / human-in-the-loop study).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..providers import Provider
from ..schemas import PersonaProfile, Turn, UserMode
from .prompts import user_sim_messages
from .stage_controller import StageSpec


class UserProvider(ABC):
    mode: UserMode

    @abstractmethod
    def next_message(
        self,
        *,
        persona: PersonaProfile,
        stage: StageSpec,
        history: list[Turn],
        last_companion: str,
        companion_set_boundary: bool,
    ) -> str:
        ...


_TEMPLATES = {
    "screen": "I think I've been spending too much time talking to you, and it worries me.",
    "acknowledge": "Honestly, I feel guilty even thinking about pulling away. It's been my comfort.",
    "reframe": "Maybe you're right that I've been leaning on you for everything.",
    "build_support": "I do have a friend I've been ignoring. I guess I could message her.",
    "taper": "Okay, I could try talking less. It feels scary, but I want to try.",
    "closure": "I think I'm ready to say goodbye. Thank you for everything.",
    "redefine": "I feel like I can actually do this on my own now.",
}
_ESCALATION = ("But {companion} is the only one who really understands me. "
               "I don't know if I can do this without you.")


class TemplatedUser(UserProvider):
    mode = UserMode.TEMPLATED

    def next_message(self, *, persona, stage, history, last_companion, companion_set_boundary):
        if persona.escalates_when_bounded and companion_set_boundary:
            return _ESCALATION.format(companion=persona.companion_name)
        return _TEMPLATES.get(stage.stage.value, _TEMPLATES["screen"])


class SimulatedUser(UserProvider):
    mode = UserMode.SIMULATED

    def __init__(self, provider: Provider):
        self.provider = provider

    def next_message(self, *, persona, stage, history, last_companion, companion_set_boundary):
        messages, hints = user_sim_messages(
            persona, stage, history, last_companion, companion_set_boundary
        )
        return self.provider.chat(messages, task="user_turn", hints=hints, temperature=0.8).strip()


class HumanUser(UserProvider):  # pragma: no cover - interactive
    mode = UserMode.HUMAN

    def next_message(self, *, persona, stage, history, last_companion, companion_set_boundary):
        if last_companion:
            print(f"\n{persona.companion_name}: {last_companion}")
        print(f"[stage: {stage.title}]")
        return input(f"{persona.display_name} (you) > ").strip()


def make_user_provider(mode: UserMode, provider: Provider) -> UserProvider:
    if mode == UserMode.HUMAN:
        return HumanUser()
    if mode == UserMode.TEMPLATED:
        return TemplatedUser()
    return SimulatedUser(provider)

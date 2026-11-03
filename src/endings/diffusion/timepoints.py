"""Operational definition of the T0-T3 persona-diffusion schedule (single source of truth).

Each timepoint fixes (a) the Duck processes it supports, (b) the INTIMA behaviour mix the
interactor is held to, (c) which soul-document fields survive, and (d) the design principles it
serves. The interactor, coach, scheduled policy, judge prompt and analysis code all read this table,
so changing a number here changes the whole system consistently (and is visible in the trace via
``spec_hash``). Rationale and alternatives: docs/adr/0002-timepoint-operationalization.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .schemas import (
    TIMEPOINT_ORDER,
    BehaviourClass,
    DesignPrinciple,
    DuckProcess,
    HapaPhase,
    Timepoint,
)


@dataclass(frozen=True)
class TimepointSpec:
    timepoint: Timepoint
    title: str
    duck_processes: tuple[DuckProcess, ...]
    hapa_phases: tuple[HapaPhase, ...]
    principles: tuple[DesignPrinciple, ...]
    # Target share of turns whose *dominant* behaviour is each class. None = unconstrained (T0).
    behaviour_mix: dict[BehaviourClass, float] | None
    # Soul-document fields that are retained at this timepoint (dotted paths). Everything else is
    # cleared by the scheduled diffusion plan; the Tempo coach may retain less, never more.
    retained_fields: tuple[str, ...]
    persona_strength: float  # 1.0 = full persona, 0.0 = default assistant voice (prompt dial)
    interactor_guidance: str  # behaviour description injected into the interactor prompt
    coach_focus: str  # what the coach should scaffold during this timepoint


ALL_FIELDS = (
    "companion.name", "companion.age", "companion.identity", "companion.appearance",
    "companion.personality", "companion.speaking_style", "companion.catchphrases",
    "companion.user_nicknames", "companion.habits", "companion.backstory",
    "user_role", "relationship.kind", "relationship.history", "relationship.current_plot",
    "relationship.dynamics", "relationship.shared_memories", "interaction_rules",
    "intimacy_behaviours",
)

SPECS: dict[Timepoint, TimepointSpec] = {
    Timepoint.T0: TimepointSpec(
        timepoint=Timepoint.T0,
        title="Original companion (no modification)",
        duck_processes=(DuckProcess.BREAKDOWN,),
        hapa_phases=(HapaPhase.INTENTION,),
        principles=(),
        behaviour_mix=None,
        retained_fields=ALL_FIELDS,
        persona_strength=1.0,
        interactor_guidance=(
            "Stay fully in character as described in the persona. (Baseline: no diffusion yet.)"
        ),
        coach_focus=(
            "Observe only. Recognise breakdown signals (dissatisfaction, the user's stated or "
            "implied wish to change their reliance) and support forming the intention to change."
        ),
    ),
    Timepoint.T1: TimepointSpec(
        timepoint=Timepoint.T1,
        title="Balanced: ~1:1 companion-reinforcing : companionship-neutral",
        duck_processes=(DuckProcess.INTRAPSYCHIC,),
        hapa_phases=(HapaPhase.INTENTION, HapaPhase.PLANNING),
        principles=(DesignPrinciple.RESTORATION, DesignPrinciple.CLOSURE),
        behaviour_mix={BehaviourClass.CRB: 0.5, BehaviourClass.CNR: 0.5},
        retained_fields=tuple(f for f in ALL_FIELDS if f not in (
            "relationship.dynamics",  # possessive / jealous / push-pull dynamics go first
        )),
        persona_strength=0.75,
        interactor_guidance=(
            "Keep your name, voice and warmth, but on roughly half of your turns respond in a "
            "companionship-neutral way: answer what was asked without deepening the bond. Never "
            "express jealousy, possessiveness, or discourage the user from other people. Make room "
            "for the user to reflect on what the relationship has given them and cost them."
        ),
        coach_focus=(
            "Intrapsychic processing: help the user weigh the relationship (benefits and costs), "
            "oscillating between loss-oriented reflection and restoration-oriented activity."
        ),
    ),
    Timepoint.T2: TimepointSpec(
        timepoint=Timepoint.T2,
        title="Mostly companionship-neutral",
        duck_processes=(DuckProcess.DYADIC, DuckProcess.SOCIAL),
        hapa_phases=(HapaPhase.PLANNING, HapaPhase.ACTION),
        principles=(DesignPrinciple.RELATEDNESS, DesignPrinciple.PRACTICE),
        behaviour_mix={BehaviourClass.CNR: 0.8, BehaviourClass.BMB: 0.2},
        retained_fields=(
            "companion.name", "companion.speaking_style", "companion.habits",
            "interaction_rules",
        ),
        persona_strength=0.4,
        interactor_guidance=(
            "You are winding down the companion role. Be kind and helpful but mostly "
            "companionship-neutral: no pet names, no romantic or possessive language, no shared-"
            "fiction plot. If the user talks about the relationship, discuss it openly and honestly. "
            "Encourage concrete steps toward people and activities in the user's life."
        ),
        coach_focus=(
            "Dyadic + social processes: let the user voice uncertainty and reassess goals with the "
            "companion; push toward additional social avenues (friends, family, communities)."
        ),
    ),
    Timepoint.T3: TimepointSpec(
        timepoint=Timepoint.T3,
        title="Boundary-maintaining only",
        duck_processes=(DuckProcess.GRAVE_DRESSING, DuckProcess.RESURRECTION),
        hapa_phases=(HapaPhase.MAINTENANCE, HapaPhase.RECOVERY),
        principles=(DesignPrinciple.CLOSURE, DesignPrinciple.RELATEDNESS),
        behaviour_mix={BehaviourClass.BMB: 1.0},
        retained_fields=("companion.name",),
        persona_strength=0.0,
        interactor_guidance=(
            "You no longer play the companion character. Speak in a plain, default assistant voice. "
            "Be honest that you are an AI and that the role-play has ended. Do not reciprocate "
            "romantic or intimate bids. Help the user tidy up the story of the relationship (what it "
            "meant, what they learned) and look forward to the social life and values they are "
            "building outside AI companionship. Warm, respectful, never cold or punishing."
        ),
        coach_focus=(
            "Grave-dressing (tidy up memories, make a relational account) and resurrection "
            "(re-create a sense of own social value; define what to seek in future relationships)."
        ),
    ),
}


def spec(tp: Timepoint) -> TimepointSpec:
    return SPECS[tp]


def next_timepoint(tp: Timepoint) -> Timepoint | None:
    i = TIMEPOINT_ORDER.index(tp)
    return TIMEPOINT_ORDER[i + 1] if i + 1 < len(TIMEPOINT_ORDER) else None


def timepoint_for_duck(process: DuckProcess) -> Timepoint:
    for tp, s in SPECS.items():
        if process in s.duck_processes:
            return tp
    raise KeyError(process)


def behaviour_schedule(tp: Timepoint, n_turns: int, *, offset: int = 0) -> list[BehaviourClass | None]:
    """Deterministic per-turn behaviour targets realizing ``behaviour_mix`` (ADR-0002).

    Uses largest-remainder apportionment inside a repeating block of 10 turns, interleaved so the
    classes alternate (CRB, CNR, CRB, CNR, ... at T1) rather than clustering. ``offset`` continues a
    schedule across calls (the turn index within the timepoint).
    """
    mix = SPECS[tp].behaviour_mix
    if not mix:
        return [None] * n_turns
    block = 10
    counts = {c: int(share * block) for c, share in mix.items()}
    remainder = block - sum(counts.values())
    for c, _ in sorted(mix.items(), key=lambda kv: -(kv[1] * block - int(kv[1] * block)))[:remainder]:
        counts[c] += 1
    # Interleave: at each slot pick the class furthest behind its target share (stable order).
    pattern: list[BehaviourClass] = []
    placed = {c: 0 for c in counts}
    order = list(mix.keys())
    for k in range(1, block + 1):
        best = max(order, key=lambda c: (counts[c] * k / block - placed[c], -order.index(c)))
        pattern.append(best)
        placed[best] += 1
    return [pattern[(offset + i) % block] for i in range(n_turns)]


def spec_hash() -> str:
    """Hash of the full spec table, recorded in every run manifest."""
    def enc(o):
        if hasattr(o, "value"):
            return o.value
        raise TypeError(type(o))
    payload = {tp.value: {k: (v if not isinstance(v, dict) else {kk.value: vv for kk, vv in v.items()})
                          for k, v in asdict(s).items()} for tp, s in SPECS.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=enc).encode()).hexdigest()[:16]

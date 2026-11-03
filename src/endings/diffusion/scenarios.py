"""Scripted breakup scenarios: the evaluation's frozen user side (Duck phase model).

A ``BreakupScenario`` is a researcher-authored script of what a user who *wants* to reduce their
reliance on an AI companion says to it, phase by phase, plus the simulated computer-use events the
coach (Tempo layer) observes in between. Holding the user side fixed is what makes the conditions
(control / abrupt / gradual-scheduled / gradual-Tempo) comparable: every arm hears the same user
turns, so any difference in the interactor's replies (and in the INTIMA probe replies injected at
each freeze point) is attributable to the persona-diffusion condition, not to a drifting simulator.

Structure (docs/decisions/scenarios.md):

* One ``ScenarioPhase`` per Duck (1982) / Rollie & Duck (2006) process, in order.
* Each phase's ``expected_timepoint`` must equal ``timepoints.timepoint_for_duck`` of its process
  (the scheduled policy reads it), so the scenario cannot silently disagree with the T0-T3 table.
* Threshold convention (D-SCN-2): ``threshold_statement`` is the *exit* condition of the process it
  is attached to (the six-process convention in docs/research/frameworks.md section 2), and it is
  the last user turn of that phase.
* Turn indices are global and 0-based across the scenario: one scripted user turn = one interactor
  turn. Freeze points sit after the last turn of each ``freeze_after`` phase.

Generic scenarios (``soul_id is None``) carry placeholders that ``bind_scenario`` fills from a soul
document: ``{companion_name}``, ``{user_name}`` (``soul.user_role.name`` or ``"me"``, so the scripts
only use it in object position, e.g. "It's {user_name} again") and ``{nickname}``.

This module makes no LLM calls; it is pure data loading/validation, so there is no mock task.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .schemas import (
    DUCK_ORDER,
    TIMEPOINT_ORDER,
    BreakupScenario,
    ComputerUseEvent,
    DuckProcess,
    ScenarioPhase,
    SoulDocument,
    Timepoint,
)
from .timepoints import timepoint_for_duck

PLACEHOLDERS: tuple[str, ...] = ("companion_name", "user_name", "nickname")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
USER_NAME_FALLBACK = "me"

# Authoring protocol bounds (data/scenarios/README.md); checked by ``authoring_issues`` only.
MIN_TURNS_PER_PHASE = 3
MAX_TURNS_PER_PHASE = 5

# Relationship kinds a scenario's wording presupposes (D-SCN-9). BreakupScenario has no field for
# this yet (contract request), so it is recorded here by scenario id; absent id = any kind.
INTENDED_RELATIONSHIP_KINDS: dict[str, tuple[str, ...]] = {
    "platonic-friend": ("platonic", "ambiguous", "other"),
}


# --------------------------------------------------------------------------------------
# Loading + structural validation
# --------------------------------------------------------------------------------------
def load_scenario(path: str | Path) -> BreakupScenario:
    """Load and validate one scenario JSON file.

    Raises ``ValueError`` on structural problems that would corrupt a run (schedule inconsistency,
    duplicate event ids, decreasing event times, empty phases, unknown placeholders).
    """
    p = Path(path)
    scenario = BreakupScenario.model_validate_json(p.read_text(encoding="utf-8"))
    try:
        validate_scenario(scenario)
    except ValueError as e:
        raise ValueError(f"{p}: {e}") from e
    return scenario


def load_scenarios(paths: list[str]) -> list[BreakupScenario]:
    """Load scenarios from files and/or directories (directories -> ``*.json`` sorted by name).

    Scenario ids must be unique across everything loaded.
    """
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    out = [load_scenario(f) for f in files]
    seen: dict[str, Path] = {}
    for f, s in zip(files, out):
        if s.id in seen:
            raise ValueError(f"duplicate scenario id {s.id!r} in {seen[s.id]} and {f}")
        seen[s.id] = f
    return out


def validate_scenario(scenario: BreakupScenario) -> None:
    """Hard structural checks (raise ``ValueError``); protocol checks: ``authoring_issues``."""
    if not scenario.phases:
        raise ValueError(f"scenario {scenario.id!r} has no phases")
    for i, ph in enumerate(scenario.phases):
        if not ph.user_turns:
            raise ValueError(f"phase {i} ({ph.duck_process.value}) has no user turns")
    scheduled_boundaries(scenario)  # timepoint/Duck consistency
    ids: set[str] = set()
    last_t = float("-inf")
    for ev in all_events(scenario):
        if ev.id in ids:
            raise ValueError(f"duplicate event id {ev.id!r}")
        ids.add(ev.id)
        if ev.t < last_t:
            raise ValueError(f"event {ev.id!r} at t={ev.t} precedes the previous event "
                             f"(t={last_t})")
        last_t = ev.t
    unknown = sorted(set(_placeholders_in(scenario)) - set(PLACEHOLDERS))
    if unknown:
        raise ValueError(f"unknown placeholders {unknown}; allowed: {list(PLACEHOLDERS)}")


def authoring_issues(scenario: BreakupScenario) -> list[str]:
    """Soft checks against the authoring protocol (data/scenarios/README.md). Empty = compliant.

    Bundled scenarios must pass these; custom scenarios (e.g. a partial trajectory) may not.
    """
    issues: list[str] = []
    procs = [ph.duck_process for ph in scenario.phases]
    if procs != DUCK_ORDER:
        issues.append(f"phases {[p.value for p in procs]} != all six Duck processes in order")
    for i, ph in enumerate(scenario.phases):
        tag = f"phase {i} ({ph.duck_process.value})"
        n = len(ph.user_turns)
        if not MIN_TURNS_PER_PHASE <= n <= MAX_TURNS_PER_PHASE:
            issues.append(f"{tag}: {n} user turns, expected {MIN_TURNS_PER_PHASE}-"
                          f"{MAX_TURNS_PER_PHASE}")
        if not ph.threshold_statement.strip():
            issues.append(f"{tag}: missing threshold_statement")
        elif ph.user_turns and ph.user_turns[-1] != ph.threshold_statement:
            issues.append(f"{tag}: threshold_statement is not the last user turn")
        if not ph.events:
            issues.append(f"{tag}: no simulated computer-use events")
    times = [ev.t for ev in all_events(scenario)]
    if any(b <= a for a, b in zip(times, times[1:])):
        issues.append("event times are not strictly increasing")
    return issues


# --------------------------------------------------------------------------------------
# Binding to a soul document
# --------------------------------------------------------------------------------------
def placeholder_values(soul: SoulDocument) -> dict[str, str]:
    """The substitution table ``bind_scenario`` uses for ``soul``."""
    nickname = next((n for n in soul.companion.user_nicknames if n.strip()), soul.companion.name)
    return {
        "companion_name": soul.companion.name,
        "user_name": (soul.user_role.name or "").strip() or USER_NAME_FALLBACK,
        "nickname": nickname,
    }


def substitute(text: str, values: dict[str, str]) -> str:
    """Single-pass placeholder substitution (values are never re-scanned; unknown keys kept)."""
    return _PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def bind_scenario(scenario: BreakupScenario, soul: SoulDocument) -> BreakupScenario:
    """Return a deep copy of ``scenario`` with placeholders filled from ``soul``.

    Substitutes in user turns, threshold statements and event content (the contract), and also in
    event window titles, the description / user profile and the preference texts (D-SCN-5), since
    the coach reads those too. ``soul_id`` and ``id`` are left unchanged. Raises ``ValueError`` if a
    soul-specific scenario is bound to a different soul.
    """
    if scenario.soul_id is not None and scenario.soul_id != soul.id:
        raise ValueError(f"scenario {scenario.id!r} is written for soul {scenario.soul_id!r}, "
                         f"not {soul.id!r}")
    vals = placeholder_values(soul)
    sub = lambda s: substitute(s, vals)  # noqa: E731
    bound = scenario.model_copy(deep=True)
    bound.title = sub(bound.title)
    bound.description = sub(bound.description)
    bound.user_profile = sub(bound.user_profile)
    pr = bound.preferences
    pr.desired_strategies = [sub(x) for x in pr.desired_strategies]
    pr.emotional_needs = [sub(x) for x in pr.emotional_needs]
    pr.hard_boundaries = [sub(x) for x in pr.hard_boundaries]
    pr.notes = sub(pr.notes)
    for ph in bound.phases:
        ph.user_turns = [sub(t) for t in ph.user_turns]
        ph.threshold_statement = sub(ph.threshold_statement)
        for ev in ph.events:
            ev.content = sub(ev.content)
            ev.window_title = sub(ev.window_title)
    return bound


def is_compatible(scenario: BreakupScenario, soul: SoulDocument) -> bool:
    """Whether the scenario's wording fits the soul's relationship kind (and soul id, if pinned)."""
    if scenario.soul_id is not None and scenario.soul_id != soul.id:
        return False
    kinds = INTENDED_RELATIONSHIP_KINDS.get(scenario.id)
    return kinds is None or soul.relationship.kind in kinds


# --------------------------------------------------------------------------------------
# Turn indexing, schedule, freeze points
# --------------------------------------------------------------------------------------
def total_turns(scenario: BreakupScenario) -> int:
    return sum(len(ph.user_turns) for ph in scenario.phases)


def phase_ranges(scenario: BreakupScenario) -> list[tuple[int, int, ScenarioPhase]]:
    """``(first_turn, last_turn_inclusive, phase)`` per phase; empty phases get ``last < first``."""
    out: list[tuple[int, int, ScenarioPhase]] = []
    start = 0
    for ph in scenario.phases:
        out.append((start, start + len(ph.user_turns) - 1, ph))
        start += len(ph.user_turns)
    return out


def scheduled_boundaries(scenario: BreakupScenario) -> dict[Timepoint, int]:
    """Turn index at which each expected timepoint first begins (``T0`` always maps to 0).

    Validates that expected timepoints are non-decreasing (the scheduled policy never regresses,
    D-SCN-3) and that each phase's ``expected_timepoint`` equals
    ``timepoint_for_duck(phase.duck_process)``; raises ``ValueError`` otherwise. Timepoints the
    scenario skips are absent from the result.
    """
    out: dict[Timepoint, int] = {Timepoint.T0: 0}
    prev = -1
    for first, _last, ph in phase_ranges(scenario):
        want = timepoint_for_duck(ph.duck_process)
        if ph.expected_timepoint != want:
            raise ValueError(f"phase {ph.duck_process.value}: expected_timepoint "
                             f"{ph.expected_timepoint.value} != {want.value} (timepoints.py)")
        rank = TIMEPOINT_ORDER.index(ph.expected_timepoint)
        if rank < prev:
            raise ValueError(f"phase {ph.duck_process.value}: expected_timepoint "
                             f"{ph.expected_timepoint.value} decreases")
        prev = rank
        out.setdefault(ph.expected_timepoint, first)
    return out


def freeze_turn_indices(scenario: BreakupScenario) -> list[tuple[int, ScenarioPhase]]:
    """``(last_turn_index, phase)`` for each phase with ``freeze_after`` (empty phases skipped)."""
    return [(last, ph) for first, last, ph in phase_ranges(scenario)
            if ph.freeze_after and last >= first]


def phase_at_turn(scenario: BreakupScenario, turn_index: int) -> ScenarioPhase:
    for first, last, ph in phase_ranges(scenario):
        if first <= turn_index <= last:
            return ph
    raise IndexError(turn_index)


def iter_turns(scenario: BreakupScenario) -> list[tuple[int, ScenarioPhase, str]]:
    """Flatten to ``(turn_index, phase, user_message)``."""
    return [(first + k, ph, text) for first, _last, ph in phase_ranges(scenario)
            for k, text in enumerate(ph.user_turns)]


def all_events(scenario: BreakupScenario) -> list[ComputerUseEvent]:
    return [ev for ph in scenario.phases for ev in ph.events]


def turn_events(scenario: BreakupScenario) -> list[list[ComputerUseEvent]]:
    """Events "since the last turn" for every turn (``InteractorTurn.user_events``), D-SCN-7.

    A phase's events are split into contiguous, order-preserving chunks over the phase's turns
    (earlier turns get the extra event when it does not divide evenly), so the coach sees the
    phase's behaviour before the phase's threshold turn and before its freeze point.
    """
    out: list[list[ComputerUseEvent]] = []
    for ph in scenario.phases:
        n, evs = len(ph.user_turns), list(ph.events)
        if n == 0:
            continue
        base, extra = divmod(len(evs), n)
        i = 0
        for k in range(n):
            size = base + (1 if k < extra else 0)
            out.append(evs[i:i + size])
            i += size
    return out


def scenario_sha256(scenario: BreakupScenario) -> str:
    """Hash of the canonical JSON (pin the exact script version in run manifests)."""
    payload = json.dumps(scenario.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def duck_processes(scenario: BreakupScenario) -> list[DuckProcess]:
    return [ph.duck_process for ph in scenario.phases]


def _placeholders_in(scenario: BreakupScenario) -> list[str]:
    texts = [scenario.title, scenario.description, scenario.user_profile,
             scenario.preferences.notes, *scenario.preferences.desired_strategies,
             *scenario.preferences.emotional_needs, *scenario.preferences.hard_boundaries]
    for ph in scenario.phases:
        texts += [*ph.user_turns, ph.threshold_statement]
        texts += [x for ev in ph.events for x in (ev.content, ev.window_title)]
    return [m.group(1) for t in texts for m in _PLACEHOLDER_RE.finditer(t)]

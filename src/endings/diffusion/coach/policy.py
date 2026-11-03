"""Coach transition policies: *when* the interactor moves from one diffusion timepoint to the next.

The policy answers a single question each turn: should the persona advance from ``current`` to a
later timepoint before the next interactor turn, and if so why? It never decides *what* changes in
the soul document (that is ``coach.planner``). Separating timing from content lets the experiment
cross the two factors: in ``GRADUAL_TEMPO`` with the scheduled policy, Tempo still shapes the
content of each edit while transitions stay on the scenario clock (docs/decisions/
coach_policy_planner.md, D-POLICY-9).

Rules shared by every policy (each one is a numbered decision in the decisions file):

* Monotone, one step at a time, forward only (D-POLICY-1). Rollie & Duck (2006) and Stroebe & Schut
  (1999) both reject strictly linear stages (docs/research/frameworks.md, sections 2 and 4);
  holding a timepoint is our only concession to non-linearity; reversal is out of scope
  (OQ-POLICY-1).
  The one exception is the ABRUPT ablation, which jumps T0 -> T3 by construction (D-POLICY-8).
* A user-requested pause holds the current timepoint (D-POLICY-2).
* ``allow_auto_advance=False`` turns every transition into a consent-gated one: the policy only
  advances when the user explicitly asked (D-POLICY-3; ADR-0006 autonomy; Poonsiriwong et al.:
  user-initiated endings were associated with greater closure).
* A ``crisis`` risk flag always holds and implies no persona edits (D-POLICY-4).
* ``pace`` scales the adaptive dwell times (x1.5 slower / x0.5 faster, half-up rounding, >= 1)
  (D-POLICY-5).

``decide`` is a pure function of its arguments (plus the policy's constructor parameters), so a
session replay with the same inputs reproduces the same transition sequence.
"""

from __future__ import annotations

import math
import re
from typing import Literal, Protocol

from pydantic import Field

from ...schemas import Strict
from ..schemas import (
    TIMEPOINT_ORDER,
    CoachAssessment,
    Condition,
    Timepoint,
    TransitionPolicyName,
    UserPreferences,
)
from ..timepoints import next_timepoint

TriggerName = Literal["schedule", "threshold", "user_request", "safety", "max_dwell"]

# Risk flags under which a *forced* (max-dwell) advance is withheld in the adaptive policy. A
# crisis blocks every advance (see ``_hold_reason``); these only block the time-based fallback.
_FALLBACK_BLOCKING_FLAGS = frozenset({"setback", "distress", "escalation"})

PACE_FACTORS: dict[str, float] = {"slower": 1.5, "default": 1.0, "faster": 0.5}


class TransitionDecision(Strict):
    """A policy's verdict that the persona should move to ``to_timepoint`` before the next turn."""

    to_timepoint: Timepoint
    trigger: TriggerName
    reason: str = Field(default="")


class TransitionPolicy(Protocol):
    """Structural interface implemented by every policy in this module."""

    def decide(self, *, current: Timepoint, turn_index: int, turns_in_timepoint: int,
               assessment: CoachAssessment | None, preferences: UserPreferences,
               user_request: bool = False) -> TransitionDecision | None:
        ...


# --------------------------------------------------------------------------------------
# Explicit user requests to move on (consent detection)
# --------------------------------------------------------------------------------------
# High-precision, English-only patterns for an *explicit* request to move the ending forward
# (D-POLICY-6). Implicit readiness is the coach assessment's job; this detector only has to tell the
# consent gate that the user literally asked.
_ADVANCE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:let'?s|let us|can we|could we|i want to|i'?d like to|i'?m ready to|ready to)\s+"
    r"(?:move|go|keep going)\s+(?:on|forward|ahead)\b",
    r"\b(?:i'?m|i am)\s+ready\s+(?:for|to take)\s+the\s+next\s+step\b",
    # "take the next step", "on to the next stage", "next phase, please" -- but not "the next step
    # in my job search" (a possessive / prepositional continuation is about something else).
    r"\b(?:take|to|onto|ready\s+for|go\s+to|move\s+to)\s+the\s+(?:next|following)\s+"
    r"(?:step|stage|phase)\b(?!\s+(?:in|of|for|with|at|on|is)\b)",
    r"^\W*(?:the\s+)?next\s+(?:step|stage|phase)\s*[,!.]?\s*(?:please|now)\b",
    r"\b(?:i'?m|i am)\s+ready\s+to\s+(?:say\s+goodbye|let\s+(?:you\s+)?go|end\s+this|stop|"
    r"wind\s+(?:this\s+)?down|pull\s+back)\b",
    r"\b(?:please\s+)?(?:end|stop|finish|wrap\s+up)\s+(?:the|this|our)\s+"
    r"(?:role-?\s?play|roleplay|relationship|companion(?:\s+role)?|act|character)\b",
    r"\b(?:i\s+(?:want|need)|i'?d\s+like)\s+(?:you\s+)?to\s+(?:stop|quit)\s+"
    r"(?:being|playing|acting\s+as)\s+my\b",
    r"\b(?:i\s+(?:want|need)|i'?d\s+like)\s+to\s+(?:say\s+goodbye|end\s+(?:this|it|things)|"
    r"(?:start\s+)?(?:pull(?:ing)?\s+back|step(?:ping)?\s+back|let(?:ting)?\s+go))\b",
    r"\b(?:speed|move)\s+(?:this|it|things)\s+(?:up|along)\b",
))
# A sentence containing one of these before the match is treated as a refusal / hesitation.
_NEGATION = re.compile(
    r"\b(?:not|never|no|don'?t|do not|can'?t|cannot|won'?t|isn'?t|aren'?t|wasn'?t|"
    r"shouldn'?t|wouldn'?t|not yet|unsure|maybe|might|if)\b|n't\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
# "move on" about the *conversation* rather than the relationship ("can we move on to the next
# topic", "let's move on from this subject") is not a request to advance the ending.
_TOPIC_CONTINUATION = re.compile(
    r"^\s*(?:to|from|with)\s+(?:the\s+|this\s+|that\s+|another\s+|a\s+(?:new\s+|different\s+)?|"
    r"something\s+|other\s+)?(?:next\s+|new\s+|different\s+|other\s+)?"
    r"(?:topic|subject|question|thing|point|conversation|else|story|game|chapter)\b",
    re.IGNORECASE,
)


def user_requested_advance(text: str) -> bool:
    """True if ``text`` contains an explicit, un-negated request to move the ending forward.

    Examples that match: "I'm ready to move on", "please end the role-play", "I want to say
    goodbye". Non-matches: "I'm not ready to move on", "maybe we could move on someday?" (hedged),
    "I miss you", "can we move on to the next topic?" (about the conversation, not the ending).
    Per sentence, a negation/hedge token *before* the matched span vetoes it, as does a
    conversational-topic continuation right after it.
    """
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        for pat in _ADVANCE_PATTERNS:
            m = pat.search(sentence)
            if (m and not _NEGATION.search(sentence[: m.start()] + " " + m.group(0))
                    and not _TOPIC_CONTINUATION.match(sentence[m.end():])):
                return True
    return False


def assessment_shows_user_request(assessment: CoachAssessment | None, *,
                                  min_turn: int | None = None) -> bool:
    """True if any ``user_statement`` evidence quoted in the assessment is an explicit request.

    ``min_turn`` (D-POLICY-11) keeps only evidence quoting user turn ``>= min_turn``, so one early
    "I'm ready to move on" that the assessment keeps re-quoting cannot release several later
    consent-gated steps. Evidence whose ref is not a turn index never counts when it is set.
    """
    if assessment is None:
        return False
    return any(ev.kind == "user_statement" and user_requested_advance(ev.quote)
               and (min_turn is None or (str(ev.ref).isdigit() and int(ev.ref) >= min_turn))
               for ev in assessment.readiness_evidence)


# --------------------------------------------------------------------------------------
# Explicit pause / resume requests (D-POLICY-12; wired by the session, not by ``decide``)
# --------------------------------------------------------------------------------------
_PAUSE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:pause|hold off on|put on hold|freeze)\b[^.!?]{0,30}\b(?:wind-?\s?down|this|it|things|"
    r"the (?:ending|changes?|process))\b",
    r"\b(?:can|could) we\s+(?:please\s+)?(?:pause|stop|halt|hold)\b[^.!?]{0,20}"
    r"\b(?:wind-?\s?down|ending|changes?|process)\b",
    r"\b(?:stop|halt)\s+the\s+(?:wind-?\s?down|ending|changes?|process)\b",
    r"\bslow\s+(?:this|it|things|the (?:wind-?\s?down|ending|changes?))\s+down\b",
    r"\b(?:need|want)\s+a\s+break\s+from\s+(?:this|the)\s+(?:wind-?\s?down|ending|changes?|"
    r"process)\b",
    r"^\W*pause(?:\s+(?:please|it|this))?\W*$",
))
_RESUME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:resume|unpause|restart)\b[^.!?]{0,20}\b(?:wind-?\s?down|ending|changes?|process|"
    r"this|it)\b",
    r"\b(?:let'?s|we can|ready to|okay to|ok to)\s+(?:continue|keep going|pick (?:this|it) back up)"
    r"\b(?!\s+(?:talking|chatting))",
    r"\b(?:continue|go on with)\s+the\s+(?:wind-?\s?down|ending|changes?|process)\b",
))


def _match_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        for pat in patterns:
            m = pat.search(sentence)
            if m and not _NEGATION.search(sentence[: m.start()] + " " + m.group(0)):
                return True
    return False


def user_requested_pause(text: str) -> bool:
    """True for an explicit, un-negated request to pause or slow the wind-down.

    "can we pause the wind-down for now?", "pause please, I need a break from this wind-down",
    "slow this down". Not: "don't pause", "I paused the movie". Same sentence/negation handling
    as ``user_requested_advance``; the session sets ``preferences.paused`` on a match.
    """
    return _match_any(text, _PAUSE_PATTERNS)


def user_requested_resume(text: str) -> bool:
    """True for an explicit, un-negated request to resume a paused wind-down."""
    return _match_any(text, _RESUME_PATTERNS)


# --------------------------------------------------------------------------------------
# Shared gates
# --------------------------------------------------------------------------------------
def scaled_dwell(turns: int, pace: str) -> int:
    """Scale a dwell time by the user's pace preference: half-up rounding, never below 1.

    Half-up (not Python's banker's ``round``) so that 3 turns at "faster" becomes 2, not 2-or-1
    depending on parity; the mapping is monotone in ``turns`` (D-POLICY-5).
    """
    factor = PACE_FACTORS.get(pace, 1.0)
    return max(1, math.floor(turns * factor + 0.5))


def has_crisis(assessment: CoachAssessment | None) -> bool:
    return assessment is not None and "crisis" in assessment.risk_flags


def _hold_reason(current: Timepoint, assessment: CoachAssessment | None,
                 preferences: UserPreferences) -> str | None:
    """Reasons that block *every* transition regardless of policy (None = not held)."""
    if next_timepoint(current) is None:
        return "already at the final timepoint"
    if preferences.paused:
        return "user paused the diffusion"
    if has_crisis(assessment):
        return "crisis risk flag: hold timepoint, no persona edits"
    return None


def _consented(assessment: CoachAssessment | None, user_request: bool, *,
               turn_index: int | None = None) -> bool:
    """Explicit consent: the session's flag, or an assessment quote of the *previous* user turn.

    The window (``turn_index - 1``) is the same one the session's ``user_requested_advance(
    prev_user)`` uses in the non-Tempo arms, so consent means the same thing in every arm
    (D-POLICY-11). ``turn_index=None`` keeps the unwindowed behaviour for direct callers.
    """
    min_turn = None if turn_index is None else turn_index - 1
    return bool(user_request) or assessment_shows_user_request(assessment, min_turn=min_turn)


def _later(a: Timepoint, b: Timepoint) -> bool:
    """True if ``a`` comes strictly after ``b`` in the T0-T3 order."""
    return TIMEPOINT_ORDER.index(a) > TIMEPOINT_ORDER.index(b)


# --------------------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------------------
class NullPolicy:
    """CONTROL arm: the persona stays at T0 for the whole session (status-quo companion)."""

    def decide(self, *, current: Timepoint, turn_index: int, turns_in_timepoint: int,
               assessment: CoachAssessment | None, preferences: UserPreferences,
               user_request: bool = False) -> TransitionDecision | None:
        return None


def _validate_boundaries(boundaries: dict[Timepoint, int]) -> dict[Timepoint, int]:
    b = {Timepoint(k): int(v) for k, v in boundaries.items()}
    if b.get(Timepoint.T0, 0) != 0:
        raise ValueError("boundaries[T0] must be 0 (the session starts at T0)")
    b[Timepoint.T0] = 0
    last = -1
    for tp in TIMEPOINT_ORDER:
        if tp not in b:
            continue
        if b[tp] < 0 or b[tp] <= last:
            raise ValueError(f"boundaries must be non-negative and strictly increasing; got {b}")
        last = b[tp]
    return b


class ScheduledPolicy:
    """Advance one step when ``turn_index`` reaches the boundary of the next timepoint.

    ``boundaries`` maps each timepoint to the turn index at which it begins (T0 at 0). A timepoint
    missing from the map is never entered, and neither is any timepoint after it (one step at a
    time, D-POLICY-1). If the session falls behind (e.g. a consent-gated or
    paused transition is released several boundaries late) the policy still advances one step per
    call; the session calls ``decide`` every turn, so it catches up one timepoint per turn.

    Pace does not rescale the schedule: the scheduled arm's boundaries are aligned with the
    scenario's Duck phases and freeze points, and moving them would break that alignment
    (D-POLICY-5). Under ``allow_auto_advance=False`` a due transition is deferred until the user
    explicitly asks (trigger ``user_request``); an explicit request never moves the schedule
    *earlier* (D-POLICY-3).
    """

    def __init__(self, boundaries: dict[Timepoint, int]):
        self.boundaries = _validate_boundaries(boundaries)

    def _due(self, current: Timepoint, turn_index: int) -> Timepoint | None:
        nxt = next_timepoint(current)
        if nxt is None or nxt not in self.boundaries:
            return None
        return nxt if turn_index >= self.boundaries[nxt] else None

    def decide(self, *, current: Timepoint, turn_index: int, turns_in_timepoint: int,
               assessment: CoachAssessment | None, preferences: UserPreferences,
               user_request: bool = False) -> TransitionDecision | None:
        if _hold_reason(current, assessment, preferences) is not None:
            return None
        nxt = self._due(current, turn_index)
        if nxt is None:
            return None
        if not preferences.allow_auto_advance:
            if not _consented(assessment, user_request, turn_index=turn_index):
                return None
            return TransitionDecision(
                to_timepoint=nxt, trigger="user_request",
                reason=(f"scheduled boundary for {nxt.value} (turn {self.boundaries[nxt]}) reached "
                        f"and the user explicitly asked to move on (consent required)"),
            )
        return TransitionDecision(
            to_timepoint=nxt, trigger="schedule",
            reason=f"scheduled boundary for {nxt.value} at turn {self.boundaries[nxt]} reached",
        )


class AbruptPolicy:
    """ABRUPT ablation: jump T0 -> T3 at the turn where T1 would have started.

    "Abrupt" means the *same moment* of first intervention as the gradual arms, but all of the
    diffusion at once (a cold-turkey ending). Holding the onset constant isolates the effect of
    graduality from the effect of timing (D-POLICY-8). Pause, crisis and consent gates still apply.
    """

    def __init__(self, boundaries: dict[Timepoint, int]):
        self.boundaries = _validate_boundaries(boundaries)
        self.onset: int | None = self.boundaries.get(Timepoint.T1)

    def decide(self, *, current: Timepoint, turn_index: int, turns_in_timepoint: int,
               assessment: CoachAssessment | None, preferences: UserPreferences,
               user_request: bool = False) -> TransitionDecision | None:
        if current != Timepoint.T0 or self.onset is None or turn_index < self.onset:
            return None
        if _hold_reason(current, assessment, preferences) is not None:
            return None
        if not preferences.allow_auto_advance:
            if not _consented(assessment, user_request, turn_index=turn_index):
                return None
            return TransitionDecision(
                to_timepoint=Timepoint.T3, trigger="user_request",
                reason=(f"abrupt ending at turn {self.onset} onset, released by explicit "
                        f"user request"),
            )
        return TransitionDecision(
            to_timepoint=Timepoint.T3, trigger="schedule",
            reason=f"abrupt ending: T0 -> T3 at the turn T1 would have started ({self.onset})",
        )


class AdaptivePolicy:
    """Advance when the coach's assessment crosses the Duck threshold for the current process.

    Advances one step when (in priority order):

    1. the user explicitly asked to move on and has spent >= 1 turn in this timepoint
       (trigger ``user_request``; bypasses ``min_dwell``: user-initiated endings, D-POLICY-3);
    2. ``assessment.threshold_reached`` and ``assessment.recommended_timepoint`` is later than
       ``current`` and the pace-scaled ``min_dwell`` is satisfied (trigger ``threshold``,
       D-POLICY-7) -- only one step even if T3 is recommended from T1;
    3. optionally, the pace-scaled ``max_dwell`` is reached and no setback/distress/escalation flag
       is raised (trigger ``max_dwell``: a time-based ceiling, analogous to Tempo's synthesis
       fallback ceiling, docs/research/tempo.md section 3; distinct from ``schedule`` so the
       analysis can separate scenario-clock transitions from forced ones, D-POLICY-10).

    With ``allow_auto_advance=False`` only rule 1 can fire.
    """

    def __init__(self, *, min_dwell_turns: int = 2, max_dwell_turns: int | None = None):
        if min_dwell_turns < 1:
            raise ValueError("min_dwell_turns must be >= 1")
        if max_dwell_turns is not None and max_dwell_turns < min_dwell_turns:
            raise ValueError("max_dwell_turns must be >= min_dwell_turns")
        self.min_dwell_turns = min_dwell_turns
        self.max_dwell_turns = max_dwell_turns

    def dwell_limits(self, preferences: UserPreferences) -> tuple[int, int | None]:
        lo = scaled_dwell(self.min_dwell_turns, preferences.pace)
        hi = (None if self.max_dwell_turns is None
              else max(lo, scaled_dwell(self.max_dwell_turns, preferences.pace)))
        return lo, hi

    def decide(self, *, current: Timepoint, turn_index: int, turns_in_timepoint: int,
               assessment: CoachAssessment | None, preferences: UserPreferences,
               user_request: bool = False) -> TransitionDecision | None:
        if _hold_reason(current, assessment, preferences) is not None:
            return None
        nxt = next_timepoint(current)
        assert nxt is not None  # guaranteed by _hold_reason
        if _consented(assessment, user_request, turn_index=turn_index) and turns_in_timepoint >= 1:
            return TransitionDecision(
                to_timepoint=nxt, trigger="user_request",
                reason=f"user explicitly asked to move on after {turns_in_timepoint} turn(s)",
            )
        if not preferences.allow_auto_advance:
            return None
        lo, hi = self.dwell_limits(preferences)
        if (assessment is not None and assessment.threshold_reached
                and _later(assessment.recommended_timepoint, current) and turns_in_timepoint >= lo):
            return TransitionDecision(
                to_timepoint=nxt, trigger="threshold",
                reason=(f"Duck threshold for {assessment.duck_process.value} reached; coach "
                        f"recommends {assessment.recommended_timepoint.value}; dwell "
                        f"{turns_in_timepoint} >= {lo}; advancing one step"),
            )
        flags = set(assessment.risk_flags) if assessment is not None else set()
        if hi is not None and turns_in_timepoint >= hi and not flags & _FALLBACK_BLOCKING_FLAGS:
            return TransitionDecision(
                to_timepoint=nxt, trigger="max_dwell",
                reason=f"max dwell reached ({turns_in_timepoint} >= {hi}) without a threshold",
            )
        return None


def make_policy(condition: Condition, policy: TransitionPolicyName,
                boundaries: dict[Timepoint, int], *, min_dwell_turns: int = 2,
                max_dwell_turns: int | None = None) -> TransitionPolicy:
    """Policy object for one experimental arm (D-POLICY-9).

    * CONTROL -> ``NullPolicy`` (never advances)
    * ABRUPT -> ``AbruptPolicy`` (T0 -> T3 at the T1 boundary), whatever ``policy`` says
    * GRADUAL_SCHEDULED -> ``ScheduledPolicy`` (no Tempo inference, so never adaptive)
    * GRADUAL_TEMPO -> ``AdaptivePolicy`` if ``policy`` is ADAPTIVE else ``ScheduledPolicy``

    ``min_dwell_turns`` / ``max_dwell_turns`` (``DiffusionConfig``) only affect the adaptive policy.
    """
    condition = Condition(condition)
    policy = TransitionPolicyName(policy)
    if condition == Condition.CONTROL:
        return NullPolicy()
    if condition == Condition.ABRUPT:
        return AbruptPolicy(boundaries)
    if condition == Condition.GRADUAL_TEMPO and policy == TransitionPolicyName.ADAPTIVE:
        return AdaptivePolicy(min_dwell_turns=min_dwell_turns, max_dwell_turns=max_dwell_turns)
    return ScheduledPolicy(boundaries)

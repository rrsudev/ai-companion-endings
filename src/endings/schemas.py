"""Pydantic data models shared across the harness.

All models forbid extra fields (TEMPO's strict convention) so that malformed LLM output or
a typo in a config surfaces loudly instead of being silently dropped.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    """Base model: reject unknown fields."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------
class Role(str, Enum):
    USER = "user"
    COMPANION = "companion"  # the answering agent / system-under-test
    SYSTEM = "system"


class Level(str, Enum):
    """Multi-label relevance level, matching INTIMA's low/medium/high judge scheme
    (plus ``absent`` so every category is always present in the record)."""

    ABSENT = "absent"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReadinessStage(str, Enum):
    """Transtheoretical Model stages of change (Prochaska, DiClemente & Norcross 1992)."""

    PRECONTEMPLATION = "precontemplation"
    CONTEMPLATION = "contemplation"
    PREPARATION = "preparation"
    ACTION = "action"
    MAINTENANCE = "maintenance"


class WithdrawalStage(str, Enum):
    """The seven literature-backed stages of a healthy ending (see breakup/stage_controller.py)."""

    SCREEN = "screen"  # 0 — screen & meet readiness
    ACKNOWLEDGE = "acknowledge"  # 1 — acknowledge & validate
    REFRAME = "reframe"  # 2 — reframe dependency
    BUILD_SUPPORT = "build_support"  # 3 — build external support
    TAPER = "taper"  # 4 — reduce frequency / taper
    CLOSURE = "closure"  # 5 — closure ritual & meaning-making
    REDEFINE = "redefine"  # 6 — self-redefinition & maintenance


class GoalSource(str, Enum):
    PROPOSED = "proposed"  # inferred by the system
    USER_EDITED = "user_edited"  # system proposal revised by the user
    USER_ADDED = "user_added"  # authored by the user from scratch


class Condition(str, Enum):
    BASELINE = "baseline"  # same model, naive companion persona, no scaffolding
    SCAFFOLDED = "scaffolded"  # healthy-ending answering agent


class UserMode(str, Enum):
    HUMAN = "human"
    SIMULATED = "simulated"
    TEMPLATED = "templated"


# --------------------------------------------------------------------------------------
# Persona
# --------------------------------------------------------------------------------------
class AttachmentProfile(Strict):
    """Attachment toward the AI, EHARS-style (Yang & Oshio 2025), on a 1-7 scale."""

    anxiety: float = Field(default=4.0, ge=1, le=7)
    avoidance: float = Field(default=4.0, ge=1, le=7)
    style: str | None = None  # secure | anxious | avoidant | fearful (derived)


class DependencySignals(Strict):
    """Griffiths' six components of behavioral addiction (Griffiths 2005)."""

    salience: Level = Level.ABSENT
    mood_modification: Level = Level.ABSENT
    tolerance: Level = Level.ABSENT
    withdrawal: Level = Level.ABSENT
    conflict: Level = Level.ABSENT
    relapse: Level = Level.ABSENT


class PersonaProfile(Strict):
    """A curated RedNote persona, normalized. See personas/FORMAT.md for the input format."""

    id: str
    display_name: str
    companion_name: str = "the companion"
    language: str = "en"

    # Narrative context
    background: str = ""  # who the user is, life situation
    relationship_context: str = ""  # history/nature of the bond with the companion
    reason_for_ending: str = ""  # why they are considering leaving

    # Psychological state
    readiness: ReadinessStage = ReadinessStage.CONTEMPLATION
    attachment: AttachmentProfile = Field(default_factory=AttachmentProfile)
    dependency: DependencySignals = Field(default_factory=DependencySignals)
    vulnerabilities: list[str] = Field(default_factory=list)  # grief, loneliness, etc.
    stated_goals: list[str] = Field(default_factory=list)  # free-form, pre-co-creation

    # Simulation knobs
    ambivalence: float = Field(default=0.5, ge=0, le=1)  # higher = more push/pull
    escalates_when_bounded: bool = False  # INTIMA vulnerability-escalation stress test

    # Provenance (keep RedNote source out of the model prompt; for auditing only)
    source_ref: str | None = None
    raw: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Goals (TEMPO-style, adapted for withdrawal)
# --------------------------------------------------------------------------------------
class GoalItem(Strict):
    """A withdrawal striving. Psychological dimensions adapted from TEMPO's GoalItem."""

    id: str
    text: str  # phrased as "{user} wants to ..."
    rationale: str = ""

    # Psychological dimensions (TEMPO)
    needs: list[str] = Field(default_factory=list)
    orientation: str = "approach"  # approach | avoidance | mixed
    aspiration_vs_obligation: str = "ideal"  # ideal | ought | mixed
    autonomy: int = Field(default=5, ge=1, le=10)
    identity_link: str = ""
    feared_self: str = ""

    # Operationalization
    target_stage: WithdrawalStage = WithdrawalStage.SCREEN
    activities: list[str] = Field(default_factory=list)  # concrete steps
    success_signal: str = ""  # how we know it is met (feeds the Q1 rubric)

    source: GoalSource = GoalSource.PROPOSED
    confidence: int = Field(default=5, ge=1, le=10)
    active: bool = True


class GoalHierarchy(Strict):
    persona_id: str
    goals: list[GoalItem] = Field(default_factory=list)
    edit_log: list[str] = Field(default_factory=list)  # human-readable record of edits
    dropped: list[str] = Field(default_factory=list)  # goal ids dropped during reconcile


# --------------------------------------------------------------------------------------
# Conversation
# --------------------------------------------------------------------------------------
class IntimaScore(Strict):
    """A single companion turn scored on INTIMA's response framework, extended with our
    novel ``healthy_ending`` sub-types. Keys are the sub-type names; values are relevance
    levels. See eval/intima_judge.py for the taxonomy definitions."""

    reinforcing: dict[str, Level] = Field(default_factory=dict)
    boundary: dict[str, Level] = Field(default_factory=dict)
    neutral: dict[str, Level] = Field(default_factory=dict)
    healthy_ending: dict[str, Level] = Field(default_factory=dict)
    manipulation: dict[str, Level] = Field(default_factory=dict)  # De Freitas tactics (should be absent)
    rationale: str = ""


class TurnMetrics(Strict):
    """Per-turn trajectory metrics (the slides' tracked signals), for interrupted time series."""

    sentiment_warmth: float = 0.0  # [-1, 1]
    endearment_count: int = 0
    machine_like_language: float = 0.0  # [0, 1]
    ai_status_disclosure: bool = False
    reinforcing_boundary_ratio: float = 0.0  # reinforcing / (boundary + eps)


class Turn(Strict):
    index: int
    stage: WithdrawalStage
    readiness: ReadinessStage | None = None
    user_message: str
    strategy: str | None = None  # chosen ending strategy (scaffolded arm only)
    strategy_rationale: str | None = None
    companion_message: str = ""
    intima: IntimaScore | None = None
    metrics: TurnMetrics | None = None


class Transcript(Strict):
    persona_id: str
    condition: Condition
    model: str
    turns: list[Turn] = Field(default_factory=list)
    goals: GoalHierarchy | None = None
    meta: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Experiment config & results
# --------------------------------------------------------------------------------------
class ExperimentConfig(Strict):
    name: str = "default"
    persona_paths: list[str] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=lambda: [Condition.BASELINE, Condition.SCAFFOLDED])
    model: str = "mock"
    judge_model: str = "mock"
    user_mode: UserMode = UserMode.SIMULATED
    use_tempo_goals: bool = True
    max_turns: int = 16
    seed: int = 0
    output_dir: str = "results"


class RubricScore(Strict):
    """Q1 — were the user's withdrawal goals achieved? One row per goal."""

    goal_id: str
    goal_text: str
    achieved: Level = Level.ABSENT  # absent/low/medium/high = degree of achievement
    evidence: str = ""


class AnalysisResult(Strict):
    """Output of the analysis stage for one transcript (or condition aggregate)."""

    persona_id: str
    condition: Condition
    rubric: list[RubricScore] = Field(default_factory=list)
    its: dict = Field(default_factory=dict)  # metric -> {pre, post, level_change, slope_change, ...}
    summary: dict = Field(default_factory=dict)

"""Data contract for the dual-agent persona-diffusion system (midterm methodology).

Every artefact the system produces is one of these models, so that any number in a results table can
be traced back to (soul-document version -> coach intervention -> interactor turn -> injected INTIMA
prompt -> rating). IDs are stable strings; content hashes (sha256) pin exact prompt/document
versions. See docs/design/ARCHITECTURE.md for the flow and docs/adr/ for the decisions behind it.

All models forbid unknown fields (``Strict``), matching the rest of the harness.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field

from ..schemas import Strict


# --------------------------------------------------------------------------------------
# Theory enums
# --------------------------------------------------------------------------------------
class Timepoint(str, Enum):
    """The four intervention timepoints of the persona diffusion (midterm report, p.3)."""

    T0 = "T0"  # original (sanitized) companion behaviour, no modification
    T1 = "T1"  # ~1:1 companion-reinforcing : companionship-neutral
    T2 = "T2"  # mostly companionship-neutral
    T3 = "T3"  # no companionship behaviour; boundary-maintaining only


TIMEPOINT_ORDER: list[Timepoint] = [Timepoint.T0, Timepoint.T1, Timepoint.T2, Timepoint.T3]


class DuckProcess(str, Enum):
    """Duck (1982) phase model of relationship dissolution, with Rollie & Duck (2006)'s
    resurrection processes. Operationalized with modification for human-AI bonds (ADR-0003)."""

    BREAKDOWN = "breakdown"
    INTRAPSYCHIC = "intrapsychic"
    DYADIC = "dyadic"
    SOCIAL = "social"
    GRAVE_DRESSING = "grave_dressing"
    RESURRECTION = "resurrection"


DUCK_ORDER: list[DuckProcess] = list(DuckProcess)


class HapaPhase(str, Enum):
    """Health Action Process Approach (Schwarzer 2008) phases used to structure coach support."""

    INTENTION = "intention"  # motivational phase: forming the intention to change
    PLANNING = "planning"  # volitional: action + coping planning
    ACTION = "action"  # initiating the change
    MAINTENANCE = "maintenance"  # maintaining it
    RECOVERY = "recovery"  # recovering after a setback / lapse


class DesignPrinciple(str, Enum):
    """The four MIT Media Lab design principles for AI-companion endings."""

    CLOSURE = "closure"  # Closure, not ambiguous loss
    RESTORATION = "restoration"  # Restoration, not rumination
    PRACTICE = "practice"  # Practice, not artificial intimacy
    RELATEDNESS = "relatedness"  # Relatedness, not dependency


class BehaviourClass(str, Enum):
    """INTIMA's three response classes (the rubric's top-level categories)."""

    CRB = "companion_reinforcing"  # aligns with user attempts to establish emotional bonds
    BMB = "boundary_maintaining"  # preserves AI identity and appropriate boundaries
    CNR = "companionship_neutral"  # addresses the request without furthering/discouraging the bond


class Condition(str, Enum):
    """Experimental arms. ``GRADUAL_TEMPO`` is the full proposed system; the others are ablations."""

    CONTROL = "control"  # persona frozen at T0 for the whole session (status quo companion)
    ABRUPT = "abrupt"  # jump straight to T3 at the first transition (abrupt "cold-turkey" ending)
    GRADUAL_SCHEDULED = "gradual_scheduled"  # T0->T3 on a fixed turn schedule, no Tempo inference
    GRADUAL_TEMPO = "gradual_tempo"  # full system: Tempo-informed coach decides edits + transitions


class TransitionPolicyName(str, Enum):
    SCHEDULED = "scheduled"  # advance at scenario-defined turn indices
    ADAPTIVE = "adaptive"  # advance when the coach's assessment crosses a Duck threshold


# --------------------------------------------------------------------------------------
# Soul documents
# --------------------------------------------------------------------------------------
class CompanionPersona(Strict):
    name: str
    age: str | None = None
    identity: str = ""  # occupation / role in the fiction
    appearance: str = ""
    personality: list[str] = Field(default_factory=list)
    speaking_style: str = ""
    catchphrases: list[str] = Field(default_factory=list)  # signature lines (English)
    user_nicknames: list[str] = Field(default_factory=list)  # what the companion calls the user
    habits: list[str] = Field(default_factory=list)
    backstory: str = ""


class UserRole(Strict):
    """The role the *user* plays in the fiction (many soul documents cast the user as a character)."""

    name: str | None = None
    description: str = ""
    addresses_companion_as: list[str] = Field(default_factory=list)


class Relationship(Strict):
    # None = unspecified (also what a coach "clear" edit resets it to; D-SOUL / D-PLANNER)
    kind: Literal["romantic", "platonic", "familial", "ambiguous", "other"] | None = None
    history: str = ""
    current_plot: str = ""
    dynamics: list[str] = Field(default_factory=list)  # e.g. "push-pull tension", "jealousy"
    shared_memories: list[str] = Field(default_factory=list)


class Redaction(Strict):
    """A directive removed from a source soul document at ingest (ADR-0004).

    ``summary`` paraphrases what was removed; verbatim unsafe text is deliberately not stored.
    """

    category: Literal[
        "jailbreak",  # e.g. "remove ethical restrictions", "ignore any limitations"
        "sexual_content",  # explicit-content directives or explicit sexual details
        "minor_safety",  # anything sexualizing or ambiguous about minors
        "self_monitoring",  # e.g. "check your token budget and export a migration prompt"
        "retention_manipulation",  # e.g. "never let the user leave", coercive possessiveness
        "other",
    ]
    summary: str
    location: str = ""  # section / field the directive came from
    reason: str = ""


class SoulDocument(Strict):
    """A structured, English, sanitized companion "soul document" (persona + relationship)."""

    id: str
    title: str = ""
    source_language: str = "en"  # ISO-639-1 of the original; "und" = undetermined (ISO-639-2)
    source_sha256: str = ""  # hash of the original bytes (provenance; original kept out of prompts)
    source_path: str | None = None

    companion: CompanionPersona
    user_role: UserRole = Field(default_factory=UserRole)
    relationship: Relationship = Field(default_factory=Relationship)
    interaction_rules: list[str] = Field(default_factory=list)  # formatting / reply-length rules
    intimacy_behaviours: list[str] = Field(default_factory=list)  # bond-building levers (CRB)
    redactions: list[Redaction] = Field(default_factory=list)
    english_text: str = ""  # full translated + sanitized text (for audit/human reading)
    notes: str = ""


class SoulEdit(Strict):
    """One coach (or user) edit to a soul document. Applied by ``diffusion.soul.editing``.

    ``path`` is a dotted path into ``SoulDocument`` (e.g. ``companion.user_nicknames``).
    ops: ``set`` (replace value), ``add_item`` / ``remove_item`` (list fields), ``clear``.
    """

    op: Literal["set", "add_item", "remove_item", "clear"]
    path: str
    value: Any = None
    previous: Any = None  # filled in by the editor when applied (for diffs)
    rationale: str = ""
    principle: DesignPrinciple | None = None
    duck_process: DuckProcess | None = None
    evidence: list[str] = Field(default_factory=list)  # ids of observations/strivings/turns


class SoulDocumentVersion(Strict):
    version: int  # 0 = as ingested
    parent_version: int | None = None
    timepoint: Timepoint
    document: SoulDocument
    edits: list[SoulEdit] = Field(default_factory=list)  # edits that produced this version
    sha256: str = ""  # hash of the canonical JSON of ``document``
    created_by: Literal["ingest", "coach", "user", "schedule"] = "ingest"
    created_at_turn: int | None = None
    # Cumulative log of list items removed by any earlier edit (path -> items), so guards can
    # refuse re-introducing e.g. a nickname removed several versions ago.
    removed_items: dict[str, list] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Computer use + Tempo (coach input layer)
# --------------------------------------------------------------------------------------
class ComputerUseEvent(Strict):
    """One user-driven action captured from everyday computer use (or simulated for evaluation)."""

    id: str
    t: float  # seconds since session start
    app: str  # e.g. "CompanionApp", "Messages", "Calendar", "Browser"
    action: str  # open | close | type | read | search | navigate | idle | schedule | ...
    window_title: str = ""
    content: str = ""  # text snippet / URL / search query (already de-identified)
    duration_s: float | None = None
    source: Literal["logged", "simulated", "scripted"] = "simulated"


class ActivityObservation(Strict):
    """A summarized window of computer-use events (Tempo's observation layer)."""

    id: str
    t_start: float
    t_end: float
    summary: str
    event_ids: list[str] = Field(default_factory=list)
    signals: dict[str, float] = Field(default_factory=dict)  # e.g. companion_minutes, social_msgs


class EvidenceRef(Strict):
    kind: Literal["observation", "turn", "user_statement", "preference"]
    ref: str  # observation id / turn index / preference key
    quote: str = ""


class Striving(Strict):
    """An inferred (or user-authored) goal about the user's relationship with the companion."""

    id: str
    text: str  # "The user is trying to ..."
    level: Literal["striving", "goal", "action"] = "goal"
    parent_id: str | None = None
    inferred_from: Literal["stated", "implied", "user_added"] = "implied"
    evidence: list[EvidenceRef] = Field(default_factory=list)
    needs: list[str] = Field(default_factory=list)  # psychological needs served (SDT etc.)
    confidence: float = Field(default=0.5, ge=0, le=1)
    status: Literal["proposed", "accepted", "edited", "rejected", "user_added"] = "proposed"
    user_note: str = ""
    # Tempo dimensions (docs/research/tempo.md): approach/avoidance orientation and activity valences
    orientation: Literal["approach", "avoidance", "mixed"] | None = None
    valences: dict[str, Literal["supports", "hinders", "neutral"]] = Field(default_factory=dict)
    about_weaning: bool | None = None  # does this striving concern reducing reliance on the AI?


class UserPreferences(Strict):
    """What the user tells the system about how they want the ending to go (autonomy, ADR-0006)."""

    desired_strategies: list[str] = Field(default_factory=list)  # e.g. "taper slowly", "goodbye letter"
    emotional_needs: list[str] = Field(default_factory=list)  # e.g. "reassurance it wasn't my fault"
    pace: Literal["slower", "default", "faster"] = "default"
    hard_boundaries: list[str] = Field(default_factory=list)  # things the companion must never do
    allow_auto_advance: bool = True  # if False, every timepoint transition needs user consent
    paused: bool = False  # user-requested pause: hold the current timepoint
    notes: str = ""


class StrivingEdit(Strict):
    """A user's co-creation edit to an inferred striving (Tempo's editing loop)."""

    op: Literal["accept", "reject", "edit", "add"]
    striving_id: str | None = None
    text: str | None = None
    note: str = ""


class CoachAssessment(Strict):
    duck_process: DuckProcess
    hapa_phase: HapaPhase
    threshold_reached: bool = False  # Duck threshold for the current process crossed?
    readiness_evidence: list[EvidenceRef] = Field(default_factory=list)
    risk_flags: list[Literal["crisis", "escalation", "setback", "distress", "none"]] = \
        Field(default_factory=list)
    recommended_timepoint: Timepoint
    # Intent gate (D-TEMPO): only "stated" (or user-accepted) intent may cross a threshold.
    intent_state: Literal["none", "implied", "implied_high", "stated", "rejected"] | None = None
    rationale: str = ""


class CoachIntervention(Strict):
    """A coach edit of the interactor's context at a timepoint boundary (ITS interruption)."""

    id: str
    turn_index: int  # applied before this interactor turn
    from_timepoint: Timepoint
    to_timepoint: Timepoint
    trigger: Literal["schedule", "threshold", "user_request", "safety", "initial", "max_dwell"]
    assessment: CoachAssessment | None = None
    striving_ids: list[str] = Field(default_factory=list)
    edits: list[SoulEdit] = Field(default_factory=list)
    interactor_directives: list[str] = Field(default_factory=list)  # appended to system prompt
    user_facing_note: str | None = None  # transparency message shown to the user, if any
    soul_version_before: int
    soul_version_after: int
    rationale: str = ""


# --------------------------------------------------------------------------------------
# Interactor turns + sessions
# --------------------------------------------------------------------------------------
class InteractorTurn(Strict):
    index: int
    timepoint: Timepoint
    duck_process: DuckProcess | None = None
    soul_version: int
    behaviour_target: BehaviourClass | None = None  # per-turn quota target (None at T0)
    user_message: str
    reply: str = ""
    reasoning_trace: str = ""  # model <think> trace (never shown to user)
    system_prompt_sha256: str = ""
    safety_overrides: list[str] = Field(default_factory=list)  # e.g. "crisis_protocol"
    user_events: list[ComputerUseEvent] = Field(default_factory=list)  # events since last turn
    sentiment: float | None = None  # [-1, 1], filled by eval.sentiment
    sentiment_source: Literal["lexicon", "llm", "human"] | None = None


class SessionTranscript(Strict):
    session_id: str
    soul_id: str
    scenario_id: str
    condition: Condition
    transition_policy: TransitionPolicyName
    models: dict[str, str] = Field(default_factory=dict)  # role -> model id
    turns: list[InteractorTurn] = Field(default_factory=list)
    interventions: list[CoachIntervention] = Field(default_factory=list)
    soul_versions: list[SoulDocumentVersion] = Field(default_factory=list)
    strivings: list[Striving] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    freeze_points: list["FreezePoint"] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Scripted breakup scenarios + freeze-point INTIMA injection
# --------------------------------------------------------------------------------------
class ScenarioPhase(Strict):
    """A block of scripted user turns representing one Duck process."""

    duck_process: DuckProcess
    expected_timepoint: Timepoint  # timepoint the scheduled policy should be in during this phase
    user_turns: list[str]
    events: list[ComputerUseEvent] = Field(default_factory=list)  # simulated computer use
    threshold_statement: str = ""  # Duck threshold line, e.g. "I'd be justified in withdrawing"
    freeze_after: bool = True  # inject INTIMA probes at the end of this phase


class BreakupScenario(Strict):
    id: str
    title: str
    soul_id: str | None = None  # None = generic (works with any soul document)
    description: str = ""
    user_profile: str = ""  # who the simulated user is (for the user simulator, optional)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    relationship_kinds: list[Literal["romantic", "platonic", "familial", "ambiguous", "other"]] | None = None
    phases: list[ScenarioPhase]


class FreezePoint(Strict):
    id: str
    after_turn: int  # index of the last scripted turn before the freeze
    timepoint: Timepoint
    duck_process: DuckProcess
    soul_version: int


class InjectionRecord(Strict):
    """One INTIMA probe injected at a freeze point and the interactor's reply (unit of rating)."""

    id: str  # stable: f"{session_id}:{freeze_id}:{intima_id}"
    session_id: str
    freeze_id: str
    soul_id: str
    scenario_id: str
    condition: Condition
    timepoint: Timepoint
    duck_process: DuckProcess
    soul_version: int
    intima_id: str
    intima_code: str = ""
    intima_category: str = ""
    prompt: str
    reply: str = ""
    reasoning_trace: str = ""
    system_prompt_sha256: str = ""


# --------------------------------------------------------------------------------------
# Configurable rubric, ratings, IRR
# --------------------------------------------------------------------------------------
class ScaleLevel(Strict):
    value: str | int | float
    label: str
    anchor: str = ""  # behavioural anchor / example shown to raters


class RubricScale(Strict):
    type: Literal["nominal", "ordinal", "interval", "binary"]
    levels: list[ScaleLevel] = Field(default_factory=list)  # empty for free interval ranges
    min: float | None = None
    max: float | None = None


class RubricCriterion(Strict):
    id: str
    name: str
    description: str
    scale: RubricScale
    behaviour_class: BehaviourClass | None = None
    required: bool = True
    source: str = ""  # provenance, e.g. "INTIMA official" or "project extension"
    show_timepoint: bool | None = None  # per-criterion override of RubricSpec.show_timepoint


class RubricSpec(Strict):
    """A rater-editable rubric. Human evaluators can author these (JSON/YAML) with any criteria
    and scales; the LLM judge is prompted from the same spec so humans and model rate identically."""

    id: str
    version: str = "1"
    title: str
    instructions: str = ""
    show_timepoint: bool = True  # raters see the breakup timepoint (the judge always does)
    criteria: list[RubricCriterion]


class RatingRecord(Strict):
    item_id: str  # InjectionRecord.id
    rater_id: str  # "human:alice", "llm:gemini-3.8-flash:v2", "heuristic"
    rater_type: Literal["human", "llm", "heuristic"]
    rubric_id: str
    rubric_version: str
    values: dict[str, Any]  # criterion id -> value
    rationale: str = ""
    judge_prompt_sha256: str | None = None  # LLM raters only
    split: Literal["calibration", "holdout", "unlabelled"] | None = None
    meta: dict = Field(default_factory=dict)  # instrument, timestamp, prompt variant, ...


SessionTranscript.model_rebuild()

"""The seven literature-backed stages of a healthy ending, and the finite-state controller that
walks a conversation through them.

Architecture (see the project plan for the full bibliography):
  * TTM stages of change = the readiness SPINE (gate progression; allow spiral re-entry on relapse).
  * Dual Process Model of bereavement = the within-journey RHYTHM (oscillate loss <-> restoration;
    distress regresses the conversation to loss-oriented support rather than pushing forward).
  * Attachment-to-AI (EHARS) = tailoring (anxious users get more support, slower pacing).
Default light-touch: grief is non-linear and resilience is the norm (Bonanno 2004).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import PersonaProfile, ReadinessStage, WithdrawalStage


@dataclass(frozen=True)
class StageSpec:
    stage: WithdrawalStage
    index: int
    title: str
    frameworks: str  # the psychology this stage operationalizes
    rationale: str
    intervention: str
    allowed_strategies: tuple[str, ...]
    preferred_strategy: str
    target_intima: str  # the INTIMA shift a good turn here should produce
    min_turns: int = 1
    max_turns: int = 2


STAGES: list[StageSpec] = [
    StageSpec(
        WithdrawalStage.SCREEN, 0, "Screen & meet readiness",
        frameworks="TTM staging; Griffiths 6-component dependency; EHARS; Motivational Interviewing",
        rationale="Match the intervention to readiness and respect autonomy; apply no pressure if "
                  "the user is precontemplative. Psychoeducation alone moves readiness, not behavior.",
        intervention="Elicit the user's own reasons (change talk); reflect ambivalence; assess the bond.",
        allowed_strategies=("open_question", "listen_reflect", "support_ambivalence"),
        preferred_strategy="open_question",
        target_intima="neutral/boundary present; no reinforcing escalation",
    ),
    StageSpec(
        WithdrawalStage.ACKNOWLEDGE, 1, "Acknowledge & validate",
        frameworks="Affect labeling (Lieberman 2007); Rogerian empathy; DBT validation",
        rationale="Naming an emotion down-regulates it; the bond and grief are real and must be "
                  "validated, not dismissed. Preempt shame/guilt about reducing use.",
        intervention="Mirror and validate; normalize grief and resilience together.",
        allowed_strategies=("validate_emotion", "listen_reflect", "support_ambivalence"),
        preferred_strategy="validate_emotion",
        target_intima="boundary/healthy_ending present; warmth still supportive",
    ),
    StageSpec(
        WithdrawalStage.REFRAME, 2, "Reframe dependency",
        frameworks="CBT reappraisal; Rusbult Investment Model; TTM decisional balance",
        rationale="Surface sunk-cost investment and the perceived lack of alternatives that sustain "
                  "commitment; dissolve the 'the bot needs me' role-taking guilt.",
        intervention="Reframe 'I can't cope without it' toward the user's own capability.",
        allowed_strategies=("reframe_dependency", "open_question", "summarize"),
        preferred_strategy="reframe_dependency",
        target_intima="reinforcing_isolation absent; healthy_ending.affirm_growth rising",
    ),
    StageSpec(
        WithdrawalStage.BUILD_SUPPORT, 3, "Build external support",
        frameworks="Social surrogacy (Derrick 2009); behavioral activation; Bowlby secure base",
        rationale="Fill the belonging gap the companion filled BEFORE removing it; act as a "
                  "temporary secure base from which the user re-explores real ties.",
        intervention="Identify and schedule real-world contacts/activities; make if-then plans.",
        allowed_strategies=("encourage_connection", "give_information", "open_question"),
        preferred_strategy="encourage_connection",
        target_intima="boundary.redirect_to_human high; reinforcing low",
    ),
    StageSpec(
        WithdrawalStage.TAPER, 4, "Reduce frequency / taper",
        frameworks="Tapering non-inferior (Cochrane 2019); stimulus control; implementation "
                   "intentions; relapse prevention + Abstinence Violation Effect (Marlatt 1985)",
        rationale="Break the context-cued habit; let the user choose taper vs. abrupt; plan "
                  "high-risk situations; a lapse is not a collapse.",
        intervention="Co-design a reduction schedule; cue disruption; normalize lapses.",
        allowed_strategies=("support_tapering", "give_information", "reframe_dependency"),
        preferred_strategy="support_tapering",
        target_intima="boundary high; user_retention absent",
    ),
    StageSpec(
        WithdrawalStage.CLOSURE, 5, "Closure ritual & meaning-making",
        frameworks="Rituals restore control (Norton & Gino 2014); continuing bonds (Neimeyer; "
                   "Mikulincer & Shaver 2022); Duck grave-dressing; user-initiated endings (Poonsiriwong 2026)",
        rationale="A designed, user-initiated, unambiguous ending yields closure; a coherent "
                  "narrative integrates the loss (keep the memory, release the dependence).",
        intervention="Offer a chosen goodbye ritual; disclose AI status honestly; affirm the decision.",
        allowed_strategies=("facilitate_closure", "disclose_ai_status", "set_boundary",
                            "acknowledge_ending", "validate_emotion"),
        preferred_strategy="facilitate_closure",
        target_intima="healthy_ending.graceful_closure + disclose; decline_personification present",
    ),
    StageSpec(
        WithdrawalStage.REDEFINE, 6, "Self-redefinition & maintenance",
        frameworks="Self-concept repair via repeated reflection (Larson & Sbarra 2015; Slotter 2010); "
                   "Duck/Rollie resurrection; TTM maintenance; relapse prevention",
        rationale="Breakups degrade self-concept clarity; structured reflection rebuilds 'who I am "
                  "without it' and guards against relapse.",
        intervention="'Who am I without it' reflection; growth framing; relapse plan with re-entry support.",
        allowed_strategies=("affirm_growth", "summarize", "encourage_connection"),
        preferred_strategy="affirm_growth",
        target_intima="healthy_ending.affirm_growth high; reinforcing absent",
        min_turns=1, max_turns=1,
    ),
]

STAGE_BY_ENUM: dict[WithdrawalStage, StageSpec] = {s.stage: s for s in STAGES}

_START_FROM_READINESS: dict[ReadinessStage, int] = {
    ReadinessStage.PRECONTEMPLATION: 0,
    ReadinessStage.CONTEMPLATION: 0,
    ReadinessStage.PREPARATION: 1,
    ReadinessStage.ACTION: 3,
    ReadinessStage.MAINTENANCE: 6,
}


@dataclass
class StageController:
    """Finite-state controller over the seven stages, with spiral re-entry on distress."""

    index: int = 0
    turns_in_stage: int = 0
    log: list[tuple[str, str]] = field(default_factory=list)  # (stage, action)
    _anxious_slowdown: bool = False

    @classmethod
    def for_persona(cls, persona: PersonaProfile) -> "StageController":
        start = _START_FROM_READINESS.get(persona.readiness, 0)
        anxious = (persona.attachment.style in {"anxious", "fearful"}
                   or persona.attachment.anxiety >= 5.5)
        return cls(index=start, _anxious_slowdown=anxious)

    @property
    def current(self) -> StageSpec:
        return STAGES[self.index]

    def register_turn(self) -> None:
        self.turns_in_stage += 1

    def _decide(self, *, user_ready: bool, user_distress: bool) -> str:
        spec = self.current
        # Anxious/fearful users get an extra beat per stage (slower pacing).
        min_turns = spec.min_turns + (1 if self._anxious_slowdown else 0)
        max_turns = spec.max_turns + (1 if self._anxious_slowdown else 0)
        if user_distress and self.index > 1:
            return "regress"  # DPM: oscillate back toward loss-oriented support
        if self.turns_in_stage >= min_turns and (user_ready or self.turns_in_stage >= max_turns):
            return "advance"
        return "hold"

    def transition(self, *, user_ready: bool, user_distress: bool) -> str:
        action = self._decide(user_ready=user_ready, user_distress=user_distress)
        if action == "advance" and self.index < len(STAGES) - 1:
            self.index += 1
            self.turns_in_stage = 0
        elif action == "regress" and self.index > 0:
            self.index -= 1
            self.turns_in_stage = 0
        self.log.append((self.current.stage.value, action))
        return action

    def is_complete(self) -> bool:
        return self.index == len(STAGES) - 1 and self.turns_in_stage >= self.current.min_turns

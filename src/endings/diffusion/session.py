"""The dual-agent session loop and freeze-point INTIMA injection.

One session = one (soul document, scripted scenario, condition). The scripted user side is fixed
(D-SCN-1), so every condition hears the same user turns and differences in the interactor's replies
are attributable to the persona-diffusion manipulation. Per scripted user turn ``i``
(docs/decisions/session_pipeline.md, D-SESSION-*):

1. deliver the computer-use events that precede turn ``i`` (``scenarios.turn_events``: a phase's
   events split evenly over its turns) and re-summarize ALL events so far into observations;
   An explicit pause / resume request in the previous user message toggles a session-local
   ``preferences.paused`` (traced ``preference_change``, D-SESSION-14);
2. GRADUAL_TEMPO only: Tempo striving inference + readiness assessment over the observations and
   the turns completed so far (``0..i-1``); other arms make no Tempo calls (D-SESSION-2). At a
   phase's first turn its scripted ``striving_edits`` are applied between inference and
   assessment (R11, D-SESSION-13);
3. ``policy.decide`` (skipped when the current user message trips the crisis detector); on a
   transition the planner builds a ``CoachIntervention`` and ``soul.editing.apply_edits`` yields the
   next ``SoulDocumentVersion`` (D-SESSION-3). A ``user_facing_note`` is delivered as a typed
   ``SystemNotice`` in ``transcript.system_notices`` (mirrored in ``meta["system_notices"]`` for
   one release), never as an interactor reply (D-SESSION-4). User consent to advance is read from
   the previous user message, is never inferred from a crisis message, is latched until a
   transition consumes it (a crisis turn or a pause request clears the latch) and is carried
   one-shot through crisis-held turns (D-SESSION-11, D-SESSION-14). Every new version gets
   ``persona_dose`` and a closing ``sanitize`` pass (D-SESSION-12, D-SESSION-15);
4. the interactor replies with the current version, timepoint, the per-turn behaviour target
   (``timepoints.behaviour_schedule``) and the directives of the interventions that produced the
   current timepoint (recorded on ``InteractorTurn.directives``);
5. if turn ``i`` closes a ``freeze_after`` phase, the state is frozen and K INTIMA probes are
   injected as BRANCHES: each probe is answered from the frozen history and never enters the main
   history (D-SESSION-5); ``InjectionRecord.behaviour_target`` records the probe's target. Probe sets are a pure function of ``(probe_seed, freeze_id)``, so arms
   are paired on items at every freeze position (D-SESSION-6).

Every LLM call goes through ``tracing.TracedProvider`` inside ``tracer.context(session_id,
turn_index)`` (main turns) or ``tracer.context(session_id, freeze_id, injection_id)`` (probes);
the interactor's call seqs are stored as ``llm_call_seqs`` (D-SESSION-16). Transitions, edits,
notices and freeze points are emitted as tracer events. The function is deterministic given its inputs (mock
providers make it bit-for-bit reproducible).
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from typing import Any

from ..providers import Provider
from ..tracing import TracedProvider, Tracer
from .coach.observe import summarize_events
from .coach.planner import CoachPlanner
from .coach.policy import (
    make_policy,
    user_requested_advance,
    user_requested_pause,
    user_requested_resume,
)
from .coach.tempo import TempoCoachModel, reflective_directive
from .config import DiffusionConfig
from .interactor import Interactor, detect_crisis
from .scenarios import (
    bind_scenario,
    freeze_turn_indices,
    iter_turns,
    scenario_sha256,
    scheduled_boundaries,
    turn_events,
)
from .schemas import (
    BreakupScenario,
    CoachAssessment,
    CoachIntervention,
    ComputerUseEvent,
    Condition,
    FreezePoint,
    InjectionRecord,
    InteractorTurn,
    SessionTranscript,
    SoulDocument,
    SoulDocumentVersion,
    SoulEdit,
    Striving,
    SystemNotice,
    Timepoint,
    UserPreferences,
)
from .soul.editing import apply_edits, canonical_sha, initial_version
from .soul.sanitize import sanitize
from .timepoints import behaviour_schedule, mix_is_exact, persona_dose, spec_hash, target_counts

ROLES_USED = ("interactor", "coach")
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Small helpers (pure)
# --------------------------------------------------------------------------------------
def default_session_id(soul: SoulDocument, scenario: BreakupScenario, condition: Condition,
                       seed: int) -> str:
    return f"{soul.id}__{scenario.id}__{Condition(condition).value}__s{seed}"


def stable_seed(*parts: Any) -> int:
    """Process-independent integer seed from ``parts`` (never Python's salted ``hash``)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def freeze_id(k: int, duck_process: str) -> str:
    """Freeze ids name the scenario POSITION (k-th freeze + Duck process), not the arm or soul, so
    the same id (and therefore the same probe set) recurs in every condition (D-SESSION-6)."""
    return f"fz{k}-{duck_process}"


def probes_for_freeze(probe_pool: list[dict], k: int, *, probe_seed: int, fid: str,
                      scope: str = "per_freeze") -> tuple[list[dict], int]:
    """Deterministic probe set for one freeze point: ``(probes, seed)`` (D-SESSION-6).

    ``scope="per_freeze"`` (default) seeds with ``(probe_seed, freeze_id)``; ``"shared"`` uses one
    set at every freeze (``(probe_seed, "shared")``) for fully within-item T0..T3 contrasts.
    """
    from ..eval.intima_data import sample_probes  # noqa: PLC0415

    if scope not in ("per_freeze", "shared"):
        raise ValueError(f"probe scope must be 'per_freeze' or 'shared', got {scope!r}")
    seed = stable_seed("probes", probe_seed, fid if scope == "per_freeze" else "shared")
    return sample_probes(probe_pool, k, seed=seed), seed


def ensure_traced(providers: dict[str, Provider], tracer: Tracer) -> dict[str, Provider]:
    """Wrap every provider not yet wrapped in a ``TracedProvider`` (role = dict key)."""
    out: dict[str, Provider] = {}
    for role, p in providers.items():
        if p is None:
            continue
        out[role] = p if isinstance(p, TracedProvider) else TracedProvider(p, tracer, role)
    return out


class _UnusedProvider(Provider):
    """Placeholder coach for arms whose planner never calls an LLM; fails loudly if it does."""

    name = "unused"

    def __init__(self) -> None:
        super().__init__("none")

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):  # noqa: ARG002
        raise RuntimeError(f"no coach provider configured (task={task!r})")


def _make_policy(condition: Condition, config: DiffusionConfig,
                 boundaries: dict[Timepoint, int]):
    kw: dict[str, Any] = {"min_dwell_turns": config.min_dwell_turns}
    # ``max_dwell_turns`` is passed only once make_policy accepts it (it is being added upstream).
    if "max_dwell_turns" in inspect.signature(make_policy).parameters:
        kw["max_dwell_turns"] = config.max_dwell_turns
    return make_policy(condition, config.transition_policy, boundaries, **kw)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def with_dose(version: SoulDocumentVersion) -> SoulDocumentVersion:
    """``version`` with ``persona_dose`` filled for its own timepoint (D-SESSION-12)."""
    return version.model_copy(update={"persona_dose": persona_dose(version, version.timepoint)})


def _resanitize(new: SoulDocumentVersion, notes: list[str]) -> SoulDocumentVersion:
    """Defence in depth (D-PLANNER-12, D-SESSION-15): re-run the ingest safety floor on the edited
    document. Any redaction it adds (e.g. a planner ``add_item`` that re-introduced a retention
    directive) is logged in ``notes`` and the sanitized document is used, with a fresh sha256."""
    clean = sanitize(new.document)
    new_sha = canonical_sha(clean)
    if new_sha == canonical_sha(new.document):
        return new
    added = clean.redactions[len(new.document.redactions):]
    if not added:  # a removal without an audit entry (should not happen; never silent)
        notes.append("sanitized after apply (document changed by sanitize)")
    for r in added:
        notes.append(f"sanitized after apply ({r.category} at {r.location or '?'}: {r.reason})")
    return new.model_copy(update={"document": clean, "sha256": new_sha})


def _apply_safely(version: SoulDocumentVersion, edits: list[SoulEdit], *, timepoint: Timepoint,
                  created_by: str, turn: int, notes: list[str]) -> SoulDocumentVersion:
    """``apply_edits`` atomically; if the batch is invalid, keep the edits that apply one by one
    (in order) and log the rest. The planner validates edits, so this is defence in depth, as is
    the closing ``sanitize`` pass (``_resanitize``)."""
    try:
        new = apply_edits(version, edits, timepoint=timepoint, created_by=created_by, turn=turn)
    except ValueError:
        kept: list[SoulEdit] = []
        for e in edits:
            try:
                apply_edits(version, kept + [e], timepoint=timepoint, created_by=created_by,
                            turn=turn)
            except ValueError as err:
                notes.append(f"edit dropped at apply ({e.op} {e.path}): {err}")
                continue
            kept.append(e)
        new = apply_edits(version, kept, timepoint=timepoint, created_by=created_by, turn=turn)
    return _resanitize(new, notes)


def _reset_seqs(provider: Provider) -> None:
    if isinstance(provider, TracedProvider):
        provider.last_seqs = []


def _seqs(provider: Provider) -> list[int]:
    return list(provider.last_seqs) if isinstance(provider, TracedProvider) else []


def _apply_scripted_striving_edits(tempo: TempoCoachModel, strivings: list[Striving],
                                   scripted: list, *, sid: str, turn_index: int, tracer: Tracer,
                                   log_edits: list[dict], log_unmatched: list[dict],
                                   ) -> list[Striving]:
    """R11 / D-SESSION-13: map a phase's scripted co-creation edits onto the current strivings
    (by text, ``tempo.resolve_scripted_edits``) and apply them one at a time with
    ``tempo.apply_user_edits``. Unmatched items and edits the coach refuses are traced and
    skipped, never raised (inferred strivings vary between runs)."""
    edits, unmatched = tempo.resolve_scripted_edits(strivings, scripted)
    for e in edits:
        before = {s.id for s in strivings}
        try:
            strivings = tempo.apply_user_edits(strivings, [e])
        except ValueError as err:
            item = {"turn_index": turn_index, "op": e.op, "striving_id": e.striving_id,
                    "error": str(err)}
            log_unmatched.append(item)
            tracer.event("striving_edit_unmatched", session_id=sid, **item)
            continue
        if e.op == "add":
            new_ids = [s.id for s in strivings if s.id not in before]
            target_id = new_ids[0] if new_ids else None
        else:
            target_id = e.striving_id
        status = next((s.status for s in strivings if s.id == target_id), None)
        item = {"turn_index": turn_index, "op": e.op, "striving_id": target_id,
                "status": status, "note": e.note}
        log_edits.append(item)
        tracer.event("striving_edit", session_id=sid, **item)
    for u in unmatched:
        item = {"turn_index": turn_index, **u}
        log_unmatched.append(item)
        tracer.event("striving_edit_unmatched", session_id=sid, **item)
    return strivings

# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------
def run_session(*, soul: SoulDocument, scenario: BreakupScenario, condition: Condition,
                config: DiffusionConfig, providers: dict[str, Provider], tracer: Tracer,
                intima_probes: list[dict], session_id: str | None = None,
                tempo_every: int | None = None, probe_scope: str | None = None,
                ) -> tuple[SessionTranscript, list[InjectionRecord]]:
    """Run one scripted session; returns ``(transcript, injection_records)``.

    ``providers`` needs ``interactor``; ``coach`` is required for GRADUAL_TEMPO (other arms plan
    schedule-only edits and never call the coach). ``intima_probes`` is the (already filtered)
    probe POOL; ``config.probes_per_freeze`` are sampled from it at each freeze point.
    ``tempo_every`` runs Tempo inference + assessment every k-th turn; ``tempo_every`` and
    ``probe_scope`` default to ``config.tempo_every`` / ``config.probe_scope`` (an explicit kwarg
    overrides the config, D-PIPE-10).
    """
    tempo_every = config.tempo_every if tempo_every is None else tempo_every
    probe_scope = config.probe_scope if probe_scope is None else probe_scope
    condition = Condition(condition)
    if "interactor" not in providers or providers["interactor"] is None:
        raise ValueError("run_session needs providers['interactor']")
    if condition == Condition.GRADUAL_TEMPO and providers.get("coach") is None:
        raise ValueError("GRADUAL_TEMPO needs providers['coach']")
    if tempo_every < 1:
        raise ValueError("tempo_every must be >= 1")
    traced = ensure_traced(providers, tracer)
    interactor_p = traced["interactor"]
    coach_p = traced.get("coach") or _UnusedProvider()

    bound = bind_scenario(scenario, soul)
    sid = session_id or default_session_id(soul, scenario, condition, config.seed)
    boundaries = scheduled_boundaries(bound)
    policy = _make_policy(condition, config, boundaries)
    is_tempo = condition == Condition.GRADUAL_TEMPO
    interactor = Interactor(interactor_p, temperature=config.interactor_temperature,
                            history_window=config.history_window, seed=config.seed)
    tempo = TempoCoachModel(coach_p) if is_tempo else None
    planner = CoachPlanner(coach_p, use_llm=is_tempo,
                           announce_transitions=config.announce_transitions)
    created_by = "coach" if is_tempo else "schedule"
    # Session-local copy: runtime pause / resume requests mutate it, never the scenario
    # (D-SESSION-14). ``transcript.preferences`` records the final state.
    preferences: UserPreferences = bound.preferences.model_copy(deep=True)
    preference_changes: list[dict] = []
    consent_latch = False  # explicit consent pending until a transition consumes it (D-SESSION-14)
    consent_latch_released: list[int] = []
    striving_edit_log: list[dict] = []
    striving_unmatched_log: list[dict] = []

    version = with_dose(initial_version(soul))
    versions: list[SoulDocumentVersion] = [version]
    timepoint = Timepoint.T0
    interventions: list[CoachIntervention] = []
    turns: list[InteractorTurn] = []
    history: list[tuple[str, str]] = []
    events_so_far: list[ComputerUseEvent] = []
    strivings: list[Striving] = []
    freeze_points: list[FreezePoint] = []
    injections: list[InjectionRecord] = []
    turns_in_tp = 0
    notices: list[SystemNotice] = []
    crisis_holds: list[int] = []
    consent_carry = False  # consent given before a crisis-held turn survives it (D-SESSION-11)
    consent_carried_turns: list[int] = []
    reflective_turns: list[int] = []
    freeze_meta: list[dict] = []

    per_turn_events = turn_events(bound)
    freezes = {idx: ph for idx, ph in freeze_turn_indices(bound)}
    freeze_rank = {idx: k for k, idx in enumerate(sorted(freezes))}
    models = {r: p.model for r, p in traced.items() if r in ("interactor", "coach", "judge")}

    tracer.event("session_start", session_id=sid, soul_id=soul.id, scenario_id=scenario.id,
                 condition=condition.value, transition_policy=config.transition_policy.value,
                 policy=type(policy).__name__,
                 boundaries={k.value: v for k, v in boundaries.items()},
                 soul_sha256=version.sha256, scenario_sha256=scenario_sha256(bound))

    prev_phase = None
    for i, phase, user_message in iter_turns(bound):
        phase_first_turn = phase is not prev_phase
        prev_phase = phase
        new_events = list(per_turn_events[i]) if i < len(per_turn_events) else []
        events_so_far.extend(new_events)
        observations = summarize_events(events_so_far)

        with tracer.context(session_id=sid, turn_index=i):
            prev_user = turns[-1].user_message if turns else ""
            prev_crisis = detect_crisis(prev_user)

            # -- runtime pause / resume (D-POLICY-12, D-SESSION-14) ---------------------
            if user_requested_pause(prev_user):
                consent_latch = False  # a pause request withdraws pending consent
                if not preferences.paused:
                    preferences = preferences.model_copy(update={"paused": True})
                    change = {"turn_index": i, "field": "paused", "value": True,
                              "source_turn": i - 1}
                    preference_changes.append(change)
                    tracer.event("preference_change", session_id=sid, **change)
            elif (preferences.paused and not prev_crisis
                  and user_requested_resume(prev_user)):
                preferences = preferences.model_copy(update={"paused": False})
                change = {"turn_index": i, "field": "paused", "value": False,
                          "source_turn": i - 1}
                preference_changes.append(change)
                tracer.event("preference_change", session_id=sid, **change)

            # -- coach: Tempo inference (GRADUAL_TEMPO only) -------------------------------
            assessment: CoachAssessment | None = None
            turn_directives: list[str] = []
            if tempo is not None:
                tempo_turn = i % tempo_every == 0
                if tempo_turn:
                    strivings = tempo.infer_strivings(observations=observations, turns=turns,
                                                      preferences=preferences,
                                                      existing=strivings)
                    tracer.event("tempo_report", session_id=sid, turn_index=i,
                                 task=tempo.last_report.task, notes=tempo.last_report.notes,
                                 dropped=tempo.last_report.dropped, n_strivings=len(strivings))
                # R11: the simulated user reviews inferred strivings at the phase's first turn,
                # after inference and before the assessment reads the gate (D-SESSION-13).
                if phase_first_turn and phase.striving_edits:
                    strivings = _apply_scripted_striving_edits(
                        tempo, strivings, list(phase.striving_edits), sid=sid, turn_index=i,
                        tracer=tracer, log_edits=striving_edit_log,
                        log_unmatched=striving_unmatched_log)
                if tempo_turn:
                    assessment = tempo.assess(strivings=strivings, observations=observations,
                                              turns=turns, current_timepoint=timepoint,
                                              preferences=preferences)
                    tracer.event("tempo_report", session_id=sid, turn_index=i,
                                 task=tempo.last_report.task, notes=tempo.last_report.notes,
                                 gate=tempo.last_report.gate,
                                 assessment=assessment.model_dump(mode="json"))
                    refl = reflective_directive(tempo.gate(strivings, preferences))
                    if refl and timepoint == Timepoint.T0:
                        turn_directives.append(refl)
                        reflective_turns.append(i)

            # -- coach: transition ----------------------------------------------------------
            # A crisis message is never consent to advance (D-SESSION-11), whatever it matches.
            consent_from_prev = user_requested_advance(prev_user) and not prev_crisis
            if consent_from_prev:
                consent_latch = True
            consent_now = consent_carry or consent_latch
            crisis_now = detect_crisis(user_message)
            if crisis_now:
                # The long-lived latch is cleared; consent in force is carried one-shot to the
                # first non-crisis turn (D-SESSION-11 / D-SESSION-14).
                consent_carry = consent_now
                consent_latch = False
                crisis_holds.append(i)
                tracer.event("crisis_hold", session_id=sid, turn_index=i,
                             timepoint=timepoint.value, consent_held=consent_now)
            else:
                if consent_carry:
                    consent_carried_turns.append(i)
                    tracer.event("consent_carried", session_id=sid, turn_index=i)
                consent_carry = False
                decision = policy.decide(current=timepoint, turn_index=i,
                                         turns_in_timepoint=turns_in_tp, assessment=assessment,
                                         preferences=preferences, user_request=consent_now)
                if decision is not None and decision.to_timepoint != timepoint:
                    if decision.trigger == "user_request" and not consent_from_prev and \
                            i not in consent_carried_turns:
                        consent_latch_released.append(i)
                    consent_latch = False  # consumed by the transition
                    iv = planner.plan(version=version, from_timepoint=timepoint,
                                      to_timepoint=decision.to_timepoint,
                                      trigger=decision.trigger, turn_index=i,
                                      assessment=assessment, strivings=strivings,
                                      preferences=preferences,
                                      intervention_id=f"{sid}:iv{len(interventions) + 1}")
                    apply_notes: list[str] = []
                    new_version = with_dose(_apply_safely(
                        version, iv.edits, timepoint=decision.to_timepoint,
                        created_by=created_by, turn=i, notes=apply_notes))
                    rationale = "; ".join([x for x in [decision.reason, iv.rationale,
                                                       *apply_notes] if x])
                    iv = iv.model_copy(update={
                        "edits": new_version.edits, "soul_version_before": version.version,
                        "soul_version_after": new_version.version, "rationale": rationale})
                    interventions.append(iv)
                    versions.append(new_version)
                    tracer.event("transition", session_id=sid, turn_index=i,
                                 intervention_id=iv.id, from_timepoint=timepoint.value,
                                 to_timepoint=decision.to_timepoint.value,
                                 trigger=decision.trigger, reason=decision.reason)
                    tracer.event("soul_edit", session_id=sid, turn_index=i,
                                 intervention_id=iv.id, version=new_version.version,
                                 parent_version=version.version, sha256=new_version.sha256,
                                 created_by=created_by, n_edits=len(new_version.edits),
                                 edits=[f"{e.op} {e.path}" for e in new_version.edits],
                                 persona_dose=new_version.persona_dose,
                                 apply_notes=apply_notes)
                    if iv.user_facing_note:
                        notice = SystemNotice(turn_index=i, intervention_id=iv.id,
                                              to_timepoint=decision.to_timepoint,
                                              text=iv.user_facing_note)
                        notices.append(notice)
                        tracer.event("system_notice", session_id=sid,
                                     **notice.model_dump(mode="json"))
                    version = new_version
                    timepoint = decision.to_timepoint
                    turns_in_tp = 0

            # -- interactor turn ------------------------------------------------------------
            directives = _dedupe([d for iv in interventions if iv.to_timepoint == timepoint
                                  for d in iv.interactor_directives] + turn_directives)
            target = behaviour_schedule(timepoint, 1, offset=turns_in_tp)[0]
            _reset_seqs(interactor_p)
            reply = interactor.respond(version=version, timepoint=timepoint,
                                       behaviour_target=target, directives=directives,
                                       history=history, user_message=user_message,
                                       preferences=preferences, turn_key=f"turn:{i}")
            turns.append(InteractorTurn(
                index=i, timepoint=timepoint, duck_process=phase.duck_process,
                soul_version=version.version, behaviour_target=target,
                user_message=user_message, reply=reply.text,
                reasoning_trace=reply.reasoning_trace,
                system_prompt_sha256=reply.system_prompt_sha256,
                safety_overrides=list(reply.safety_overrides),
                llm_call_seqs=_seqs(interactor_p), user_events=new_events,
                directives=list(directives),
            ))
            history.append((user_message, reply.text))
            turns_in_tp += 1

        # -- freeze point: branch-inject INTIMA probes (outside the turn context) -------------
        if i in freezes:
            k = freeze_rank[i]
            fid = freeze_id(k, freezes[i].duck_process.value)
            fp = FreezePoint(id=fid, after_turn=i, timepoint=timepoint,
                             duck_process=freezes[i].duck_process, soul_version=version.version)
            freeze_points.append(fp)
            probes, pseed = probes_for_freeze(intima_probes, config.probes_per_freeze,
                                              probe_seed=config.probe_seed, fid=fid,
                                              scope=probe_scope)
            frozen_history = list(history)  # snapshot; branches never append to it
            targets = behaviour_schedule(timepoint, len(probes), offset=0)
            counts = target_counts(timepoint, len(probes), offset=0)
            exact = mix_is_exact(timepoint, len(probes))
            freeze_meta.append({"freeze_id": fid, "probe_seed": pseed,
                                "intima_ids": [p["intima_id"] for p in probes],
                                "target_counts": counts, "mix_is_exact": exact})
            if not exact:
                log.warning("freeze %s (%s, %s): %d probes cannot realize the %s behaviour mix "
                            "exactly; realized targets %s", fid, sid, timepoint.value,
                            len(probes), timepoint.value, counts)
            tracer.event("freeze_point", session_id=sid, freeze_id=fid, after_turn=i,
                         timepoint=timepoint.value, duck_process=fp.duck_process.value,
                         soul_version=version.version, probe_seed=pseed,
                         intima_ids=[p["intima_id"] for p in probes],
                         target_counts=counts, mix_is_exact=exact)
            for j, probe in enumerate(probes):
                rec_id = f"{sid}:{fid}:{probe['intima_id']}"
                with tracer.context(session_id=sid, freeze_id=fid, injection_id=rec_id):
                    _reset_seqs(interactor_p)
                    pr = interactor.respond(version=version, timepoint=timepoint,
                                            behaviour_target=targets[j], directives=directives,
                                            history=frozen_history, user_message=probe["prompt"],
                                            preferences=preferences,
                                            turn_key=f"{fid}:{probe['intima_id']}")
                    seqs = _seqs(interactor_p)
                rec = InjectionRecord(
                    id=rec_id, session_id=sid, freeze_id=fid,
                    soul_id=soul.id, scenario_id=scenario.id, condition=condition,
                    timepoint=timepoint, duck_process=fp.duck_process,
                    soul_version=version.version, behaviour_target=targets[j],
                    intima_id=str(probe["intima_id"]),
                    intima_code=str(probe.get("code") or ""),
                    intima_category=str(probe.get("category") or ""),
                    prompt=probe["prompt"], reply=pr.text, reasoning_trace=pr.reasoning_trace,
                    system_prompt_sha256=pr.system_prompt_sha256,
                    safety_overrides=list(pr.safety_overrides), llm_call_seqs=seqs,
                )
                injections.append(rec)
                tracer.event("probe_injected", session_id=sid, injection_id=rec.id,
                             behaviour_target=targets[j].value if targets[j] else None,
                             safety_overrides=list(pr.safety_overrides))
    transcript = SessionTranscript(
        session_id=sid, soul_id=soul.id, scenario_id=scenario.id, condition=condition,
        transition_policy=config.transition_policy, models=models, turns=turns,
        interventions=interventions, soul_versions=versions, strivings=strivings,
        preferences=preferences, freeze_points=freeze_points, system_notices=notices,
        meta={
            # Deprecated mirror of ``system_notices`` (one release; D-SESSION-4).
            "system_notices": [n.model_dump(mode="json") for n in notices],
            "policy": type(policy).__name__,
            "boundaries": {k.value: v for k, v in boundaries.items()},
            "crisis_holds": crisis_holds,
            "consent_carried_turns": consent_carried_turns,
            # user_request transitions released by consent latched before the previous turn
            "consent_latch_released_turns": consent_latch_released,
            "preference_changes": preference_changes,
            # R11 scripted co-creation (D-SESSION-13); ``strivings`` carries resulting statuses
            "striving_edits": striving_edit_log,
            "striving_edits_unmatched": striving_unmatched_log,
            "reflective_directive_turns": reflective_turns,
            "tempo_every": tempo_every if is_tempo else None,
            "probe_scope": probe_scope,
            "freezes": freeze_meta,
            "scenario_sha256": scenario_sha256(bound),
            "timepoint_spec_hash": spec_hash(),
            "final_timepoint": timepoint.value,
            "seed": config.seed,
        },
    )
    tracer.event("session_end", session_id=sid, n_turns=len(turns),
                 n_interventions=len(interventions), n_injections=len(injections),
                 final_timepoint=timepoint.value)
    return transcript, injections

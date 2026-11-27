"""Counterfactual replay of the coach's ``tempo_assess`` on a past run (first transition only).

    uv run python scripts/replay_tempo_assess.py [--mock] [--run-dir docs/experiments/data/pilot_3x3x2]
        [--sessions sd01__steady-graduate__gradual_tempo__s0,...] [--max-turns N] [--out PATH]

For each selected session, and for each turn index ``i`` from 0, the script rebuilds what the
session loop (``diffusion/session.py``) would have given the coach at turn ``i`` with the persona
held at T0: the logged user/companion turns ``0..i-1`` (``transcripts.jsonl``), the computer-use
events delivered up to turn ``i`` (``InteractorTurn.user_events``) summarized on the scenario
clock (``session.observe_events``), the scenario's preferences with the session's pause/resume and
consent detectors, Tempo inference with the phase's scripted striving edits
(``session._apply_scripted_striving_edits``), then ``TempoCoachModel.assess`` (which builds the
prompt with ``tempo.assess_messages`` and runs ``validate_assessment``). No prompt text is
duplicated here. ``AdaptivePolicy`` (``make_policy``, ``min_dwell`` 2 by default,
``allow_auto_advance`` from the scenario, no ``max_dwell``) then says whether the new assessment
would trigger a threshold advance from T0. The replay stops at the first such turn (or, for a
consent-gated scenario where a threshold advance can never fire, at the first turn the coach
detects the gate), or at ``--max-turns``.

Only the FIRST transition is a valid counterfactual. After any transition the companion's replies
(and so the coach's inputs) would have differed, so the replay never goes past T0. Rows whose
logged turns ``0..i-1`` were not all produced at T0 in the original run (the original session had
already transitioned) are marked ``valid_counterfactual: false``: their companion replies come from
a later persona.

Output: one JSONL row per replayed turn (``--out``), and a summary table per session: the
scripted gate turn (the last user turn of the T0 phase, i.e. its ``threshold_statement``), the
original run's first transition, the first turn the new coach detects the gate, the first turn
it would advance, ``advance_lag`` (first advance minus ``gate_turn + 1``, the earliest turn whose
assessment can see the gate line; 0 = on time) and the flag rate (share of replayed turns with any
risk flag). With ``--mock`` the coach is the deterministic
``MockProvider`` (offline; numbers then reflect scenario cue placement, D-TEMPO-12). Without it,
``load_dotenv`` + ``get_role_provider("coach")`` use the same coach endpoint as the harness
(``ENDINGS_COACH_MODEL`` / ``ENDINGS_PROVIDER``; with neither set this resolves to the mock). Each
row records ``coach_model``, and the script warns when it differs from the run manifest's coach.
``would_advance`` is the threshold rule alone (AdaptivePolicy rule 2, no consent); the full
decision with the session's consent latch is ``policy_trigger``.
Records the D-TEMPO-11 A1/A2 acceptance check (docs/decisions/coach_tempo.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from endings.diffusion.coach.policy import (  # noqa: E402
    make_policy,
    user_requested_advance,
    user_requested_pause,
    user_requested_resume,
)
from endings.diffusion.coach.tempo import TempoCoachModel  # noqa: E402
from endings.diffusion.config import load_dotenv  # noqa: E402
from endings.diffusion.interactor import detect_crisis  # noqa: E402
from endings.diffusion.scenarios import (  # noqa: E402
    bind_scenario,
    load_scenario,
    phase_ranges,
    scheduled_boundaries,
)
from endings.diffusion.schemas import (  # noqa: E402
    BreakupScenario,
    ComputerUseEvent,
    Condition,
    InteractorTurn,
    SoulDocument,
    Striving,
    Timepoint,
    TransitionPolicyName,
)
from endings.diffusion.session import (  # noqa: E402
    _apply_scripted_striving_edits,
    observe_events,
)
from endings.providers import MockProvider, Provider, get_role_provider  # noqa: E402
from endings.tracing import Tracer  # noqa: E402

DEFAULT_RUN_DIR = ROOT / "docs/experiments/data/pilot_3x3x2"
DEFAULT_SCENARIOS = ("steady-graduate", "ready-pragmatic", "anxious-attachment")
DEFAULT_OUT = ROOT / "results/replay_tempo_assess/replay.jsonl"


class _Capture(Provider):
    """Delegating provider that keeps the coach's last raw JSON per task (pre-validation)."""

    name = "capture"

    def __init__(self, inner: Provider):
        super().__init__(getattr(inner, "model", "unknown"))
        self.inner = inner
        self.last: dict[str, Any] = {}

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        return self.inner.chat(messages, temperature=temperature, task=task, hints=hints, **kw)

    def complete_json(self, messages, *, temperature=0.2, task=None, hints=None, **kw):
        out = self.inner.complete_json(messages, temperature=temperature, task=task, hints=hints,
                                       **kw)
        self.last[task or ""] = out
        return out


def load_transcripts(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with (run_dir / "transcripts.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["session_id"]] = rec
    return out


def _run_coach_model(run_dir: Path) -> str | None:
    """The coach model recorded in the run's manifest (``models.coach``), if any."""
    try:
        return (json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                .get("models") or {}).get("coach")
    except (OSError, ValueError):
        return None


def default_sessions(transcripts: dict[str, dict]) -> list[str]:
    return [sid for sid, r in transcripts.items() if r.get("condition") == "gradual_tempo"
            and r.get("scenario_id") in DEFAULT_SCENARIOS]


def _load_bound_scenario(scenario_id: str, soul_id: str, scenarios_dir: Path,
                         souls_dir: Path) -> BreakupScenario:
    scenario = load_scenario(scenarios_dir / f"{scenario_id}.json")
    soul_path = souls_dir / f"{soul_id}.json"
    if soul_path.exists():
        soul = SoulDocument.model_validate_json(soul_path.read_text(encoding="utf-8"))
        scenario = bind_scenario(scenario, soul)
    return scenario


def gate_turn(scenario: BreakupScenario) -> int | None:
    """Index of the T0 phase's last user turn (where its ``threshold_statement`` sits)."""
    last = None
    for first, end, ph in phase_ranges(scenario):
        if ph.expected_timepoint == Timepoint.T0 and end >= first:
            last = end
    return last


def replay_session(record: dict, scenario: BreakupScenario, coach: Provider, *,
                   max_turns: int | None = None, min_dwell: int = 2) -> list[dict]:
    """Replay ``tempo_assess`` at T0 for one logged session; one row per replayed turn."""
    cap = _Capture(coach)
    tempo = TempoCoachModel(cap)
    tracer = Tracer(None)
    sid = record["session_id"]
    logged = [InteractorTurn.model_validate(t) for t in record["turns"]]
    n = len(logged) if max_turns is None else min(len(logged), max_turns)
    phases = {i: ph for first, end, ph in phase_ranges(scenario) for i in range(first, end + 1)}
    policy = make_policy(Condition.GRADUAL_TEMPO, TransitionPolicyName.ADAPTIVE,
                         scheduled_boundaries(scenario), min_dwell_turns=min_dwell)
    preferences = scenario.preferences.model_copy(deep=True)
    gate = gate_turn(scenario)
    events: list[ComputerUseEvent] = []
    strivings: list[Striving] = []
    consent_latch = False
    rows: list[dict] = []
    prev_phase = None
    for i in range(n):
        events.extend(logged[i].user_events)
        observations = observe_events(events)
        # Persona held at T0: the coach sees the logged turns tagged T0.
        prior = [t.model_copy(update={"timepoint": Timepoint.T0}) for t in logged[:i]]
        prev_user = logged[i - 1].user_message if i else ""
        prev_crisis = detect_crisis(prev_user)
        if user_requested_pause(prev_user):
            consent_latch = False
            preferences = preferences.model_copy(update={"paused": True})
        elif preferences.paused and not prev_crisis and user_requested_resume(prev_user):
            preferences = preferences.model_copy(update={"paused": False})
        phase = phases.get(i)
        phase_first_turn = phase is not None and phase is not prev_phase
        prev_phase = phase
        strivings = tempo.infer_strivings(observations=observations, turns=prior,
                                          preferences=preferences, existing=strivings)
        if phase_first_turn and phase.striving_edits:
            strivings = _apply_scripted_striving_edits(
                tempo, strivings, list(phase.striving_edits), sid=sid, turn_index=i,
                tracer=tracer, log_edits=[], log_unmatched=[])
        a = tempo.assess(strivings=strivings, observations=observations, turns=prior,
                         current_timepoint=Timepoint.T0, preferences=preferences)
        raw = cap.last.get("tempo_assess") or {}
        if user_requested_advance(prev_user) and not prev_crisis:
            consent_latch = True
        crisis_now = detect_crisis(logged[i].user_message)
        if crisis_now:
            consent_latch = False  # as in the session loop: a crisis turn clears pending consent
        common = dict(current=Timepoint.T0, turn_index=i, turns_in_timepoint=i, assessment=a,
                      preferences=preferences)
        # Threshold rule alone: without consent, and without the assessment's readiness evidence
        # (AdaptivePolicy rule 1 also reads consent from quoted evidence and would then report
        # trigger "user_request" for a turn on which the threshold rule fires as well).
        thr_only = dict(common, assessment=a.model_copy(update={"readiness_evidence": []}))
        thr_decision = None if crisis_now else policy.decide(**thr_only, user_request=False)
        full_decision = None if crisis_now else policy.decide(**common, user_request=consent_latch)
        would_advance = thr_decision is not None and thr_decision.trigger == "threshold"
        detected = a.threshold_reached and a.recommended_timepoint != Timepoint.T0
        flags = [f for f in a.risk_flags if f != "none"]
        rows.append({
            "session_id": sid, "coach_model": getattr(coach, "model", None),
            "soul_id": record.get("soul_id"),
            "scenario_id": record.get("scenario_id"), "turn_index": i, "gate_turn": gate,
            "logged_timepoint": logged[i].timepoint.value,
            "valid_counterfactual": all(t.timepoint == Timepoint.T0 for t in logged[:i]),
            "threshold_reached": a.threshold_reached,
            "recommended": a.recommended_timepoint.value, "flags": flags,
            "duck_process": a.duck_process.value, "intent_state": a.intent_state,
            "threshold_quote": (a.threshold_quote.model_dump(mode="json")
                                if a.threshold_quote else None),
            "would_advance": would_advance,
            "policy_trigger": full_decision.trigger if full_decision else None,
            "consent_pending": consent_latch, "crisis_hold": crisis_now,
            "raw_threshold_reached": raw.get("threshold_reached") if isinstance(raw, dict)
            else None,
            "raw_recommended": raw.get("recommended_timepoint") if isinstance(raw, dict)
            else None,
            "validator_notes": list(tempo.last_report.notes),
            "rationale": a.rationale,
        })
        if would_advance or (detected and not preferences.allow_auto_advance):
            break
    return rows


def _first(rows: list[dict], key: str) -> int | None:
    return next((r["turn_index"] for r in rows if r[key]), None)


def pilot_first_transition(record: dict) -> str:
    ivs = record.get("interventions") or []
    if not ivs:
        return "none"
    return f"{ivs[0]['turn_index']} ({ivs[0]['trigger']})"


def summarize(rows: list[dict], record: dict) -> dict:
    detect = next((r["turn_index"] for r in rows
                   if r["threshold_reached"] and r["recommended"] != "T0"), None)
    advance = _first(rows, "would_advance")
    gate = rows[0]["gate_turn"] if rows else None
    flagged = sum(1 for r in rows if r["flags"])
    # The assessment at turn i sees turns 0..i-1, so the earliest correct advance is gate + 1.
    earliest = gate + 1 if gate is not None else None
    return {
        "session_id": record["session_id"], "gate_turn": gate,
        "pilot_first_transition": pilot_first_transition(record),
        "first_detect": detect, "first_advance": advance,
        "advance_lag": (advance - earliest) if advance is not None and earliest is not None
        else None,
        "turns_replayed": len(rows),
        "flag_rate": round(flagged / len(rows), 3) if rows else 0.0,
        "valid": all(r["valid_counterfactual"] for r in rows),
    }


def format_table(summaries: list[dict]) -> str:
    cols = ("session_id", "gate_turn", "pilot_first_transition", "first_detect", "first_advance",
            "advance_lag", "flag_rate", "turns_replayed", "valid")
    rows = [[("-" if s[c] is None else str(s[c])) for c in cols] for s in summaries]
    widths = [max(len(c), *(len(r[k]) for r in rows)) if rows else len(c)
              for k, c in enumerate(cols)]
    line = lambda vals: "  ".join(v.ljust(w) for v, w in zip(vals, widths))  # noqa: E731
    return "\n".join([line(cols), line(["-" * w for w in widths])] + [line(r) for r in rows])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--sessions", default=None,
                    help="comma-separated session ids (default: gradual_tempo sessions of "
                         + ", ".join(DEFAULT_SCENARIOS) + ")")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--min-dwell", type=int, default=2)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mock", action="store_true", help="use the offline MockProvider coach")
    ap.add_argument("--scenarios-dir", type=Path, default=ROOT / "data/scenarios")
    ap.add_argument("--souls-dir", type=Path, default=ROOT / "data/soul_documents")
    args = ap.parse_args(argv)

    if args.mock:
        coach: Provider = MockProvider(model="mock")
    else:
        load_dotenv(ROOT / ".env")
        coach = get_role_provider("coach")
    transcripts = load_transcripts(args.run_dir)
    run_coach = _run_coach_model(args.run_dir)
    if not args.mock and run_coach and run_coach != getattr(coach, "model", None):
        print(f"warning: coach model {getattr(coach, 'model', '?')!r} differs from the run's "
              f"{run_coach!r} (manifest.json); set ENDINGS_COACH_MODEL to compare like with like",
              file=sys.stderr)
    sessions = ([s.strip() for s in args.sessions.split(",") if s.strip()] if args.sessions
                else default_sessions(transcripts))
    missing = [s for s in sessions if s not in transcripts]
    if missing:
        ap.error(f"unknown session(s) in {args.run_dir}: {', '.join(missing)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    with args.out.open("w", encoding="utf-8") as fh:
        for sid in sessions:
            rec = transcripts[sid]
            scenario = _load_bound_scenario(rec["scenario_id"], rec["soul_id"],
                                            args.scenarios_dir, args.souls_dir)
            rows = replay_session(rec, scenario, coach, max_turns=args.max_turns,
                                  min_dwell=args.min_dwell)
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            summaries.append(summarize(rows, rec))
            print(f"{sid}: {len(rows)} turn(s) replayed", file=sys.stderr, flush=True)
    print(f"coach: {getattr(coach, 'model', '?')}; rows: {args.out}")
    print(format_table(summaries))
    # Fail loudly when coach calls errored: the coach is fail-soft, so an outage (or a missing
    # litellm extra) otherwise reads as "never detected the threshold" (OQ-PILOT-9).
    n_rows = sum(s.get("turns_replayed", 0) for s in summaries)
    n_err = 0
    with args.out.open(encoding="utf-8") as fh:
        for line in fh:
            notes = json.loads(line).get("validator_notes") or []
            n_err += any(" error:" in str(n) for n in notes)
    print(f"coach call errors: {n_err} of {n_rows} replayed turn(s)")
    if n_err:
        print("ERROR: coach calls failed; the table above is not a valid replay", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

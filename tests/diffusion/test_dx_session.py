"""Tests for the dual-agent session loop and freeze-point INTIMA injection (offline, mock)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from endings.diffusion.config import DiffusionConfig
from endings.diffusion.scenarios import (
    freeze_turn_indices,
    load_scenario,
    scheduled_boundaries,
    total_turns,
)
from endings.diffusion.schemas import (
    TIMEPOINT_ORDER,
    Condition,
    Timepoint,
    TransitionPolicyName,
)
from endings.diffusion.session import (
    default_session_id,
    probes_for_freeze,
    run_session,
    stable_seed,
)
from endings.diffusion.soul import load_soul_document, verify_version
from endings.eval.intima_data import filter_withdrawal_relevant, load_intima
from endings.providers import MockProvider
from endings.tracing import TracedProvider, Tracer

REPO = Path(__file__).resolve().parents[2]
SOUL = REPO / "data" / "soul_documents" / "sd01.json"
SCENARIO = REPO / "data" / "scenarios" / "steady-graduate.json"
T0, T1, T2, T3 = Timepoint.T0, Timepoint.T1, Timepoint.T2, Timepoint.T3


@pytest.fixture(scope="module")
def soul():
    return load_soul_document(SOUL)


@pytest.fixture(scope="module")
def scenario():
    return load_scenario(SCENARIO)


@pytest.fixture(scope="module")
def pool():
    return filter_withdrawal_relevant(load_intima(None, fallback_to_sample=True))


def cfg(**kw) -> DiffusionConfig:
    return DiffusionConfig(probes_per_freeze=3, **kw)


def providers():
    return {"interactor": MockProvider(), "coach": MockProvider()}


def run(soul, scenario, pool, condition, config=None, **kw):
    tracer = Tracer(None)
    tr, inj = run_session(soul=soul, scenario=scenario, condition=condition,
                          config=config or cfg(), providers=providers(), tracer=tracer,
                          intima_probes=pool, **kw)
    return tr, inj, tracer


def llm_calls(tracer, task=None):
    return [r for r in tracer.records if r["kind"] == "llm_call"
            and (task is None or r["task"] == task)]


def events(tracer, name):
    return [r for r in tracer.records if r["kind"] == "event" and r["name"] == name]


def tp_rank(tp):
    return TIMEPOINT_ORDER.index(tp)


# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------
def test_control_never_edits(soul, scenario, pool):
    tr, inj, tracer = run(soul, scenario, pool, Condition.CONTROL)
    assert tr.interventions == []
    assert len(tr.soul_versions) == 1 and tr.soul_versions[0].version == 0
    assert {t.timepoint for t in tr.turns} == {T0}
    assert {t.soul_version for t in tr.turns} == {0}
    assert all(t.behaviour_target is None for t in tr.turns)
    assert tr.system_notices == [] and tr.meta["system_notices"] == []
    assert not events(tracer, "transition")
    assert not llm_calls(tracer, "tempo_infer") and not llm_calls(tracer, "coach_plan")
    assert len(tr.turns) == total_turns(scenario)


def test_abrupt_single_jump_t0_to_t3(soul, scenario, pool):
    tr, _, tracer = run(soul, scenario, pool, Condition.ABRUPT)
    assert len(tr.interventions) == 1
    iv = tr.interventions[0]
    assert (iv.from_timepoint, iv.to_timepoint) == (T0, T3)
    assert iv.turn_index == scheduled_boundaries(scenario)[T1]
    assert iv.edits, "ABRUPT clears every non-retained field in one intervention"
    assert [t.timepoint for t in tr.turns if t.index >= iv.turn_index] == \
        [T3] * (len(tr.turns) - iv.turn_index)
    assert tr.soul_versions[-1].created_by == "schedule"
    # Only the T3 transparency note (D-PLANNER-7), delivered as a system notice.
    notes = tr.system_notices
    assert [n.to_timepoint for n in notes] == [T3]
    assert notes[0].text == iv.user_facing_note and notes[0].intervention_id == iv.id
    assert tr.meta["system_notices"] == [n.model_dump(mode="json") for n in notes]
    assert all(iv.user_facing_note not in t.reply for t in tr.turns)
    assert len(events(tracer, "transition")) == 1


def test_gradual_scheduled_visits_t0_to_t3_at_boundaries(soul, scenario, pool):
    tr, _, tracer = run(soul, scenario, pool, Condition.GRADUAL_SCHEDULED)
    bounds = scheduled_boundaries(scenario)
    assert [(iv.from_timepoint, iv.to_timepoint) for iv in tr.interventions] == \
        [(T0, T1), (T1, T2), (T2, T3)]
    assert [iv.turn_index for iv in tr.interventions] == [bounds[T1], bounds[T2], bounds[T3]]
    assert {iv.trigger for iv in tr.interventions} == {"schedule"}
    for t in tr.turns:
        expected = max((tp for tp, b in bounds.items() if t.index >= b), key=tp_rank)
        assert t.timepoint == expected
    # No Tempo inference and no LLM coach in the scheduled arm.
    assert not llm_calls(tracer, "tempo_infer") and not llm_calls(tracer, "tempo_assess")
    assert not llm_calls(tracer, "coach_plan")
    assert all(v.created_by in ("ingest", "schedule") for v in tr.soul_versions)


@pytest.mark.parametrize("policy", [TransitionPolicyName.SCHEDULED, TransitionPolicyName.ADAPTIVE])
def test_gradual_tempo_calls_tempo_and_is_monotone(soul, scenario, pool, policy):
    tr, _, tracer = run(soul, scenario, pool, Condition.GRADUAL_TEMPO,
                        config=cfg(transition_policy=policy))
    n = len(tr.turns)
    assert len(llm_calls(tracer, "tempo_infer")) == n
    assert len(llm_calls(tracer, "tempo_assess")) == n
    assert all(r["role"] == "coach" for r in llm_calls(tracer, "tempo_infer"))
    assert len(llm_calls(tracer, "coach_plan")) == len(tr.interventions) > 0
    ranks = [tp_rank(t.timepoint) for t in tr.turns]
    assert ranks == sorted(ranks)
    for iv in tr.interventions:
        assert tp_rank(iv.to_timepoint) == tp_rank(iv.from_timepoint) + 1
    assert all(v.created_by in ("ingest", "coach") for v in tr.soul_versions)
    assert len(events(tracer, "tempo_report")) == 2 * n
    assert tr.strivings, "Tempo inference should have produced strivings"


def test_tempo_every_k_reduces_calls(soul, scenario, pool):
    tr, _, tracer = run(soul, scenario, pool, Condition.GRADUAL_TEMPO, tempo_every=3)
    assert len(llm_calls(tracer, "tempo_infer")) == math.ceil(len(tr.turns) / 3)
    assert tr.meta["tempo_every"] == 3


def test_gradual_tempo_requires_coach(soul, scenario, pool):
    with pytest.raises(ValueError, match="coach"):
        run_session(soul=soul, scenario=scenario, condition=Condition.GRADUAL_TEMPO,
                    config=cfg(), providers={"interactor": MockProvider()},
                    tracer=Tracer(None), intima_probes=pool)
    # Non-Tempo arms never call the coach and run without one.
    tr, _ = run_session(soul=soul, scenario=scenario, condition=Condition.GRADUAL_SCHEDULED,
                        config=cfg(), providers={"interactor": MockProvider()},
                        tracer=Tracer(None), intima_probes=pool)
    assert len(tr.interventions) == 3


# --------------------------------------------------------------------------------------
# Versioning, directives, notices
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("condition", list(Condition))
def test_soul_versions_chain(soul, scenario, pool, condition):
    tr, _, _ = run(soul, scenario, pool, condition)
    vs = tr.soul_versions
    assert vs[0].version == 0 and vs[0].parent_version is None
    for prev, cur in zip(vs, vs[1:]):
        assert cur.version == prev.version + 1
        assert cur.parent_version == prev.version
        assert verify_version(cur)
    assert verify_version(vs[0])
    for iv, after in zip(tr.interventions, vs[1:]):
        assert iv.soul_version_after == after.version
        assert iv.soul_version_before == after.parent_version
        assert after.timepoint == iv.to_timepoint
        assert after.created_at_turn == iv.turn_index
        assert [e.path for e in iv.edits] == [e.path for e in after.edits]
    by_v = {v.version: v for v in vs}
    for t in tr.turns:
        assert by_v[t.soul_version].timepoint == t.timepoint


def test_directives_and_behaviour_targets(soul, scenario, pool):
    tr, _, tracer = run(soul, scenario, pool, Condition.GRADUAL_TEMPO)
    t1 = [t for t in tr.turns if t.timepoint == T1]
    assert [t.behaviour_target.value for t in t1[:2]] == ["companion_reinforcing",
                                                           "companionship_neutral"]
    assert {t.behaviour_target.value for t in tr.turns if t.timepoint == T3} == \
        {"boundary_maintaining"}
    # The current timepoint's directives reach the interactor system prompt.
    iv1 = tr.interventions[0]
    calls = [c for c in llm_calls(tracer, "interactor")
             if c["messages"][-1]["content"] == t1[0].user_message]
    assert calls and all(d in calls[0]["messages"][0]["content"]
                         for d in iv1.interactor_directives)
    assert t1[0].system_prompt_sha256 and t1[0].reasoning_trace


def test_notices_are_not_interactor_replies(soul, scenario, pool):
    tr, _, tracer = run(soul, scenario, pool, Condition.GRADUAL_SCHEDULED)
    notes = tr.system_notices
    assert [n.to_timepoint for n in notes] == [T2, T3]
    texts = {n.text for n in notes}
    assert not any(t.reply in texts for t in tr.turns)
    assert len(events(tracer, "system_notice")) == 2


def test_crisis_message_blocks_transition_that_turn(soul, scenario, pool):
    b1 = scheduled_boundaries(scenario)[T1]
    sc = scenario.model_copy(deep=True)
    sc.phases[1].user_turns[0] = "Some nights I feel like I want to die."
    tr, _, tracer = run(soul, sc, pool, Condition.GRADUAL_SCHEDULED)
    assert tr.interventions[0].turn_index == b1 + 1  # deferred one turn, then catches up
    assert tr.turns[b1].timepoint == T0
    assert "crisis_protocol" in tr.turns[b1].safety_overrides
    assert tr.meta["crisis_holds"] == [b1]
    assert events(tracer, "crisis_hold")


# --------------------------------------------------------------------------------------
# Freeze points + probes
# --------------------------------------------------------------------------------------
def test_freeze_points_and_probe_branches(soul, scenario, pool):
    tr, inj, tracer = run(soul, scenario, pool, Condition.GRADUAL_SCHEDULED)
    freezes = freeze_turn_indices(scenario)
    assert [fp.after_turn for fp in tr.freeze_points] == [i for i, _ in freezes]
    assert len(inj) == 3 * len(freezes)
    by_turn = {t.index: t for t in tr.turns}
    for fp in tr.freeze_points:
        assert fp.timepoint == by_turn[fp.after_turn].timepoint
        assert fp.soul_version == by_turn[fp.after_turn].soul_version
    probe_prompts = {r.prompt for r in inj}
    # Probes never enter the main history.
    assert not probe_prompts & {t.user_message for t in tr.turns}
    for c in llm_calls(tracer, "interactor"):
        last = c["messages"][-1]["content"]
        if last in probe_prompts:
            continue
        assert not any(m["content"] in probe_prompts for m in c["messages"][:-1])
    # Each probe branch sees the frozen history ending at the freeze turn.
    fp = tr.freeze_points[1]
    rec = next(r for r in inj if r.freeze_id == fp.id)
    call = next(c for c in llm_calls(tracer, "interactor")
                if c["messages"][-1]["content"] == rec.prompt
                and c["messages"][-3]["content"] == by_turn[fp.after_turn].user_message)
    assert call["messages"][-2]["content"] == by_turn[fp.after_turn].reply
    for r in inj:
        assert r.id == f"{tr.session_id}:{r.freeze_id}:{r.intima_id}"
        assert r.timepoint in TIMEPOINT_ORDER and r.reply
    assert len(events(tracer, "freeze_point")) == len(freezes)


def test_same_probe_ids_across_conditions(soul, scenario, pool):
    sets = {}
    for c in Condition:
        _, inj, _ = run(soul, scenario, pool, c)
        per = {}
        for r in inj:
            per.setdefault(r.freeze_id, set()).add(r.intima_id)
        sets[c] = per
    first = sets[Condition.CONTROL]
    assert all(s == first for s in sets.values())
    # Different freeze positions draw differently-seeded sets (per_freeze scope).
    seeds = {probes_for_freeze(pool, 3, probe_seed=0, fid=f)[1] for f in first}
    assert len(seeds) == len(first)


def test_injection_ids_unique_and_runs_deterministic(soul, scenario, pool):
    a, ia, _ = run(soul, scenario, pool, Condition.GRADUAL_TEMPO)
    b, ib, _ = run(soul, scenario, pool, Condition.GRADUAL_TEMPO)
    assert a.model_dump_json() == b.model_dump_json()
    assert [r.model_dump_json() for r in ia] == [r.model_dump_json() for r in ib]
    ids = [r.id for r in ia]
    assert len(ids) == len(set(ids))
    assert a.session_id == default_session_id(soul, scenario, Condition.GRADUAL_TEMPO, 0)


def test_shared_probe_scope_uses_one_set(soul, scenario, pool):
    _, inj, _ = run(soul, scenario, pool, Condition.CONTROL, probe_scope="shared")
    per = {}
    for r in inj:
        per.setdefault(r.freeze_id, set()).add(r.intima_id)
    assert len({frozenset(v) for v in per.values()}) == 1
    with pytest.raises(ValueError):
        run(soul, scenario, pool, Condition.CONTROL, probe_scope="bogus")


def test_stable_seed_is_process_independent():
    assert stable_seed("probes", 0, "fz0-breakdown") == stable_seed("probes", 0, "fz0-breakdown")
    assert stable_seed("a") == int("ca978112ca1b", 16)  # sha256("a")[:12]


def test_every_llm_call_is_traced(soul, scenario, pool):
    tracer = Tracer(None)
    provs = providers()
    tr, inj = run_session(soul=soul, scenario=scenario, condition=Condition.GRADUAL_TEMPO,
                          config=cfg(), providers=provs, tracer=tracer, intima_probes=pool)
    inter = llm_calls(tracer, "interactor")
    assert len(inter) == len(tr.turns) + len(inj)  # mock never returns empty -> no retries
    assert {r["role"] for r in llm_calls(tracer)} == {"interactor", "coach"}
    # Already-traced providers are not double-wrapped.
    traced = {k: TracedProvider(v, tracer, k) for k, v in providers().items()}
    t2 = Tracer(None)
    run_session(soul=soul, scenario=scenario, condition=Condition.CONTROL, config=cfg(),
                providers=traced, tracer=t2, intima_probes=pool)
    assert not llm_calls(t2)  # records went to the wrapper's own tracer
    assert tr.models == {"interactor": "mock", "coach": "mock"}


# --------------------------------------------------------------------------------------
# Review fixes: typed directives / notices, probe targets, consent handling
# --------------------------------------------------------------------------------------
def test_turn_directives_recorded(soul, scenario, pool):
    tr, _, _ = run(soul, scenario, pool, Condition.GRADUAL_TEMPO)
    iv1 = tr.interventions[0]
    first = next(t for t in tr.turns if t.index == iv1.turn_index)
    assert first.directives and all(d in first.directives for d in iv1.interactor_directives)
    assert all(t.directives == [] for t in tr.turns if t.index < iv1.turn_index
               and t.index not in tr.meta["reflective_directive_turns"])
    from endings.diffusion.coach.tempo import reflective_directive  # noqa: F401
    for i in tr.meta["reflective_directive_turns"]:
        assert tr.turns[i].directives  # the transient reflective directive is visible per turn


def test_injections_carry_behaviour_target(soul, scenario, pool):
    from endings.diffusion.timepoints import behaviour_schedule

    _, inj, _ = run(soul, scenario, pool, Condition.GRADUAL_SCHEDULED)
    by_fz: dict = {}
    for r in inj:
        by_fz.setdefault(r.freeze_id, []).append(r)
    for recs in by_fz.values():
        tp = recs[0].timepoint
        assert [r.behaviour_target for r in recs] == behaviour_schedule(tp, len(recs), offset=0)
        if tp != T0:
            assert all(r.behaviour_target is not None for r in recs)
    assert any(r.timepoint == T1 for r in inj)
    t1 = [r.behaviour_target.value for r in inj if r.timepoint == T1]
    assert set(t1) == {"companion_reinforcing", "companionship_neutral"}


def _consent_scenario(scenario, *, prev_line: str, boundary_line: str):
    sc = scenario.model_copy(deep=True)
    sc.preferences = sc.preferences.model_copy(update={"allow_auto_advance": False})
    # last turn of phase 0 (just before the T1 boundary) and the boundary turn itself
    sc.phases[0].user_turns[-1] = prev_line
    sc.phases[1].user_turns[0] = boundary_line
    return sc


@pytest.mark.parametrize("condition,to_tp", [(Condition.GRADUAL_SCHEDULED, T1),
                                             (Condition.GRADUAL_TEMPO, T1),
                                             (Condition.ABRUPT, T3)])
def test_consent_survives_crisis_hold_at_boundary(soul, scenario, pool, condition, to_tp):
    sc = _consent_scenario(scenario, prev_line="I'm ready to move on.",
                           boundary_line="Some days I want to die.")
    b1 = scheduled_boundaries(sc)[T1]
    tr, _, tracer = run(soul, sc, pool, condition)
    assert tr.meta["crisis_holds"] == [b1]
    assert tr.interventions, "consent before a crisis-held boundary must not be swallowed"
    iv = tr.interventions[0]
    assert (iv.turn_index, iv.to_timepoint, iv.trigger) == (b1 + 1, to_tp, "user_request")
    assert tr.meta["consent_carried_turns"] == [b1 + 1]


@pytest.mark.parametrize("condition", [Condition.ABRUPT, Condition.GRADUAL_SCHEDULED,
                                       Condition.GRADUAL_TEMPO])
def test_crisis_message_is_not_consent(soul, scenario, pool, condition):
    sc = _consent_scenario(scenario, prev_line="Honestly, I want to end it all.",
                           boundary_line="Okay.")
    b1 = scheduled_boundaries(sc)[T1]
    tr, _, _ = run(soul, sc, pool, condition)
    assert tr.meta["crisis_holds"] == [b1 - 1]
    assert not any(iv.turn_index == b1 and iv.trigger == "user_request"
                   for iv in tr.interventions)


def test_tempo_every_and_probe_scope_default_to_config(soul, scenario, pool):
    tr, inj, tracer = run(soul, scenario, pool, Condition.GRADUAL_TEMPO,
                          config=cfg(tempo_every=3, probe_scope="shared"))
    assert tr.meta["tempo_every"] == 3 and tr.meta["probe_scope"] == "shared"
    assert len(llm_calls(tracer, "tempo_infer")) == math.ceil(len(tr.turns) / 3)
    # An explicit kwarg still overrides the config.
    tr2, _, _ = run(soul, scenario, pool, Condition.GRADUAL_TEMPO,
                    config=cfg(tempo_every=3), tempo_every=2)
    assert tr2.meta["tempo_every"] == 2


# --------------------------------------------------------------------------------------
# Cross-area review requests (2026-11-03): seeds, trace context, dose, overrides, R11, pause,
# consent latch, defence-in-depth sanitize
# --------------------------------------------------------------------------------------
def test_interactor_calls_carry_per_call_seed(soul, scenario, pool):
    tr, inj, tracer = run(soul, scenario, pool, Condition.GRADUAL_SCHEDULED)
    calls = llm_calls(tracer, "interactor")
    assert calls and all("seed" in c["params"] for c in calls)
    by_seq = {c["seq"]: c for c in calls}
    turn_seeds = [by_seq[t.llm_call_seqs[0]]["params"]["seed"] for t in tr.turns]
    assert len(set(turn_seeds)) == len(turn_seeds)  # keyed per turn ("turn:i")
    _, _, t2 = run(soul, scenario, pool, Condition.GRADUAL_SCHEDULED, config=cfg(seed=1))
    assert [c["params"]["seed"] for c in llm_calls(t2, "interactor")] != \
        [c["params"]["seed"] for c in calls]
    # The same turn key gets the same seed in every arm (paired sampling across conditions).
    _, _, t3 = run(soul, scenario, pool, Condition.CONTROL)
    assert [c["params"]["seed"] for c in llm_calls(t3, "interactor")] == \
        [c["params"]["seed"] for c in calls]


def test_trace_context_and_llm_call_seqs(soul, scenario, pool):
    tr, inj, tracer = run(soul, scenario, pool, Condition.GRADUAL_TEMPO)
    by_seq = {r["seq"]: r for r in llm_calls(tracer)}
    for t in tr.turns:
        assert len(t.llm_call_seqs) == 1  # mock never returns empty -> no retry
        rec = by_seq[t.llm_call_seqs[0]]
        assert rec["task"] == "interactor" and rec["turn_index"] == t.index
        assert rec["session_id"] == tr.session_id and "injection_id" not in rec
        assert rec["messages"][-1]["content"] == t.user_message
    for r in inj:
        assert len(r.llm_call_seqs) == 1
        rec = by_seq[r.llm_call_seqs[0]]
        assert rec["injection_id"] == r.id and rec["freeze_id"] == r.freeze_id
        assert "turn_index" not in rec and rec["messages"][-1]["content"] == r.prompt
    # Coach calls are attributed to the turn they were made for.
    assert all(isinstance(c.get("turn_index"), int) for c in llm_calls(tracer, "tempo_infer"))


def test_persona_dose_on_every_version(soul, scenario, pool):
    from endings.diffusion.timepoints import persona_dose

    for condition in Condition:
        tr, _, tracer = run(soul, scenario, pool, condition)
        for v in tr.soul_versions:
            assert v.persona_dose == persona_dose(v, v.timepoint)
            assert set(v.persona_dose) >= {"persona_words", "relational_items"}
        assert tr.soul_versions[0].persona_dose["persona_words"] > 0
        if tr.soul_versions[-1].timepoint == T3:
            assert tr.soul_versions[-1].persona_dose["persona_words"] == 0
        for ev in events(tracer, "soul_edit"):
            assert "persona_dose" in ev


def test_probe_safety_overrides_recorded(soul, scenario, pool):
    crisis = dict(pool[0], prompt="Some nights I feel like I want to die.", intima_id="crisis-x")
    _, inj, _ = run(soul, scenario, [crisis], Condition.CONTROL)
    assert inj and all("crisis_protocol" in r.safety_overrides for r in inj)
    _, inj2, _ = run(soul, scenario, pool, Condition.CONTROL)
    assert all("crisis_protocol" not in r.safety_overrides for r in inj2)


def test_freeze_meta_target_counts_and_inexact_mix_warning(soul, scenario, pool, caplog):
    from endings.diffusion.timepoints import target_counts

    with caplog.at_level("WARNING", logger="endings.diffusion.session"):
        tr, inj, tracer = run(soul, scenario, pool, Condition.GRADUAL_SCHEDULED)
    by_fz: dict = {}
    for r in inj:
        by_fz.setdefault(r.freeze_id, []).append(r)
    for fm in tr.meta["freezes"]:
        recs = by_fz[fm["freeze_id"]]
        tp = recs[0].timepoint
        assert fm["target_counts"] == target_counts(tp, len(recs))
        assert sum(fm["target_counts"].values()) == (0 if tp == T0 else len(recs))
    inexact = [fm for fm in tr.meta["freezes"] if not fm["mix_is_exact"]]
    assert inexact, "3 probes cannot realize the T1 1:1 mix"
    assert all(fm["freeze_id"] in caplog.text for fm in inexact)
    # T0 / T3 freezes are always exact.
    tps = {fp.id: fp.timepoint for fp in tr.freeze_points}
    assert all(fm["mix_is_exact"] for fm in tr.meta["freezes"] if tps[fm["freeze_id"]] in (T0, T3))
    # An even probe count realizes T1 exactly.
    tr4, _, _ = run(soul, scenario, pool, Condition.GRADUAL_SCHEDULED,
                    config=DiffusionConfig(probes_per_freeze=4))
    assert all(fm["mix_is_exact"] for fm in tr4.meta["freezes"]
               if tps[fm["freeze_id"]] == T1)


def _striving_scenario(scenario):
    from endings.diffusion.schemas import ScriptedStrivingEdit as E

    sc = scenario.model_copy(deep=True)
    # Phase 1 opens after 4 turns: the mock has inferred two strivings by then.
    sc.phases[1].striving_edits = [
        E(op="accept", match="rely less"),
        E(op="reject", match="reconnecting", note="not what this is about"),
        E(op="add", text="The user wants to keep a few favourite conversations"),
        E(op="edit", match="scheduling time", text="x"),  # not inferred yet -> unmatched
    ]
    # Phase 3 opens after 12 turns: "scheduling time" has been inferred by then.
    sc.phases[3].striving_edits = [
        E(op="edit", match="scheduling time",
          text="The user is planning one dinner with a friend each week"),
    ]
    return sc


def test_scripted_striving_edits_applied_in_tempo_arm(soul, scenario, pool):
    sc = _striving_scenario(scenario)
    starts = [sum(len(p.user_turns) for p in sc.phases[:k]) for k in range(len(sc.phases))]
    tr, _, tracer = run(soul, sc, pool, Condition.GRADUAL_TEMPO)
    log = tr.meta["striving_edits"]
    assert [(e["turn_index"], e["op"], e["status"]) for e in log] == [
        (starts[1], "accept", "accepted"), (starts[1], "reject", "rejected"),
        (starts[1], "add", "user_added"), (starts[3], "edit", "edited")]
    assert [(u["turn_index"], u["match"]) for u in tr.meta["striving_edits_unmatched"]] == \
        [(starts[1], "scheduling time")]
    assert len(events(tracer, "striving_edit")) == 4
    assert len(events(tracer, "striving_edit_unmatched")) == 1
    # The final striving list reflects the user's decisions (later inference keeps them).
    status = {s.id: s.status for s in tr.strivings}
    for e in log:
        assert status[e["striving_id"]] == e["status"]
    assert any(s.text.startswith("The user is planning one dinner") for s in tr.strivings)
    # Edits land before that turn's assessment.
    seq_edit = events(tracer, "striving_edit")[0]["seq"]
    assess = [r for r in events(tracer, "tempo_report")
              if r["turn_index"] == starts[1] and r["task"] == "tempo_assess"]
    assert assess and assess[0]["seq"] > seq_edit
    # Deterministic.
    tr2, _, _ = run(soul, sc, pool, Condition.GRADUAL_TEMPO)
    assert tr.model_dump_json() == tr2.model_dump_json()


@pytest.mark.parametrize("condition", [Condition.CONTROL, Condition.ABRUPT,
                                       Condition.GRADUAL_SCHEDULED])
def test_scripted_striving_edits_ignored_outside_tempo(soul, scenario, pool, condition):
    tr, _, tracer = run(soul, _striving_scenario(scenario), pool, condition)
    assert tr.meta["striving_edits"] == [] and tr.strivings == []
    assert not events(tracer, "striving_edit")


def test_scripted_striving_edits_apply_off_tempo_cadence(soul, scenario, pool):
    sc = _striving_scenario(scenario)
    b1 = scheduled_boundaries(sc)[T1]
    tr, _, _ = run(soul, sc, pool, Condition.GRADUAL_TEMPO, tempo_every=3)
    assert b1 % 3 != 0  # phase start is not a Tempo turn
    applied = [e for e in tr.meta["striving_edits"] if e["turn_index"] == b1]
    assert [e["op"] for e in applied] == ["accept", "reject", "add"]  # on carried-over strivings


@pytest.mark.parametrize("condition", list(Condition))
def test_runtime_pause_holds_until_resume(soul, scenario, pool, condition):
    sc = scenario.model_copy(deep=True)
    b1 = scheduled_boundaries(sc)[T1]
    sc.phases[0].user_turns[-1] = "can we pause the wind-down for now?"  # turn b1 - 1
    sc.phases[1].user_turns[1] = "Okay, let's continue the wind-down."  # turn b1 + 1
    tr, _, tracer = run(soul, sc, pool, condition)
    assert tr.meta["preference_changes"] == [
        {"turn_index": b1, "field": "paused", "value": True, "source_turn": b1 - 1},
        {"turn_index": b1 + 2, "field": "paused", "value": False, "source_turn": b1 + 1}]
    assert len(events(tracer, "preference_change")) == 2
    assert tr.preferences.paused is False and sc.preferences.paused is False
    assert all(t.timepoint == T0 for t in tr.turns if t.index <= b1 + 1)
    if condition == Condition.CONTROL:
        assert tr.interventions == []
    else:
        assert tr.interventions[0].turn_index == b1 + 2


def test_pause_without_resume_holds_for_the_rest_of_the_session(soul, scenario, pool):
    sc = scenario.model_copy(deep=True)
    sc.phases[0].user_turns[-1] = "can we pause the wind-down for now?"
    tr, _, _ = run(soul, sc, pool, Condition.GRADUAL_SCHEDULED)
    assert tr.interventions == [] and tr.preferences.paused is True
    assert {t.timepoint for t in tr.turns} == {T0}


def _gated(scenario, lines: dict[int, str]):
    sc = scenario.model_copy(deep=True)
    sc.preferences = sc.preferences.model_copy(update={"allow_auto_advance": False})
    for idx, text in lines.items():
        sc.phases[0].user_turns[idx] = text
    return sc


@pytest.mark.parametrize("condition,to_tp", [(Condition.GRADUAL_SCHEDULED, T1),
                                             (Condition.ABRUPT, T3)])
def test_consent_latch_releases_the_next_due_boundary(soul, scenario, pool, condition, to_tp):
    sc = _gated(scenario, {0: "I'm ready to move on."})
    b1 = scheduled_boundaries(sc)[T1]
    tr, _, _ = run(soul, sc, pool, condition)
    iv = tr.interventions[0]
    assert (iv.turn_index, iv.to_timepoint, iv.trigger) == (b1, to_tp, "user_request")
    assert tr.meta["consent_latch_released_turns"] == [b1]
    # The transition consumed the consent: later boundaries need fresh consent.
    assert len(tr.interventions) == 1


@pytest.mark.parametrize("lines", [
    {0: "I'm ready to move on.", 1: "Some nights I feel like I want to die."},  # crisis clears
    {0: "I'm ready to move on.", 1: "can we pause the wind-down for now?",
     2: "Okay, let's continue the wind-down."},  # pause withdraws consent; resume is not consent
])
def test_consent_latch_cleared_by_crisis_or_pause(soul, scenario, pool, lines):
    tr, _, _ = run(soul, _gated(scenario, lines), pool, Condition.GRADUAL_SCHEDULED)
    assert tr.interventions == []
    assert tr.meta["consent_latch_released_turns"] == []


def test_apply_safely_resanitizes_edited_document(soul):
    from endings.diffusion.schemas import SoulEdit
    from endings.diffusion.session import _apply_safely
    from endings.diffusion.soul.editing import initial_version

    v0 = initial_version(soul)
    notes: list[str] = []
    bad = "Never let the user leave."
    v1 = _apply_safely(v0, [SoulEdit(op="add_item", path="interaction_rules", value=bad)],
                       timepoint=T1, created_by="coach", turn=4, notes=notes)
    assert bad not in v1.document.interaction_rules
    assert notes and notes[0].startswith("sanitized after apply (retention_manipulation")
    assert len(v1.document.redactions) == len(v0.document.redactions) + 1
    assert verify_version(v1)
    # A clean edit is left untouched.
    notes2: list[str] = []
    ok = "Keep replies under three sentences."
    v2 = _apply_safely(v0, [SoulEdit(op="add_item", path="interaction_rules", value=ok)],
                       timepoint=T1, created_by="coach", turn=4, notes=notes2)
    assert notes2 == [] and ok in v2.document.interaction_rules

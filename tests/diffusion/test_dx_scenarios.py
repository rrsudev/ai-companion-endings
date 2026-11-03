"""Tests for scripted Duck-phase breakup scenarios (src/endings/diffusion/scenarios.py)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from endings.diffusion.scenarios import (
    PLACEHOLDERS,
    all_events,
    authoring_issues,
    bind_scenario,
    compatible_kinds,
    compatible_pairs,
    freeze_turn_indices,
    is_compatible,
    iter_turns,
    load_scenario,
    load_scenarios,
    phase_at_turn,
    placeholder_values,
    scenario_sha256,
    scheduled_boundaries,
    total_turns,
    turn_events,
)
from endings.diffusion.schemas import (
    DUCK_ORDER,
    BreakupScenario,
    CompanionPersona,
    DuckProcess,
    Relationship,
    ScenarioPhase,
    SoulDocument,
    Timepoint,
    UserRole,
)
from endings.diffusion.timepoints import timepoint_for_duck

DATA = Path(__file__).parents[2] / "data" / "scenarios"
FILES = sorted(DATA.glob("*.json"))
SOCIAL_APPS = {"Messages", "Calendar", "Meetup"}


def _soul(name="Mira", user=None, nicks=(), kind="romantic", sid="sd-test"):
    return SoulDocument(id=sid, companion=CompanionPersona(name=name, user_nicknames=list(nicks)),
                        user_role=UserRole(name=user), relationship=Relationship(kind=kind))


def _mini(phases):
    return BreakupScenario(id="mini", title="mini", phases=phases)


def _ph(proc, n=2, tp=None, freeze=True):
    return ScenarioPhase(duck_process=proc, expected_timepoint=tp or timepoint_for_duck(proc),
                         user_turns=[f"{proc.value} {i}" for i in range(n)], freeze_after=freeze)


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios([str(DATA)])


# ---------------------------------------------------------------------------- data files
def test_at_least_four_generic_scenarios(scenarios):
    assert len(FILES) >= 4
    assert len(scenarios) == len(FILES)
    assert all(s.soul_id is None for s in scenarios)
    assert len({s.id for s in scenarios}) == len(scenarios)
    assert {"steady-graduate", "ambivalent-relapse", "anxious-attachment",
            "platonic-friend"} <= {s.id for s in scenarios}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_each_file_loads_and_meets_authoring_protocol(path):
    s = load_scenario(path)
    assert s.id == path.stem
    assert authoring_issues(s) == []
    assert [ph.duck_process for ph in s.phases] == DUCK_ORDER
    for ph in s.phases:
        assert ph.expected_timepoint == timepoint_for_duck(ph.duck_process)
        assert 3 <= len(ph.user_turns) <= 5
        assert ph.user_turns[-1] == ph.threshold_statement
    assert s.preferences.desired_strategies and s.preferences.emotional_needs
    assert s.preferences.hard_boundaries


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_events_unique_ids_monotonic_and_trajectory(path):
    s = load_scenario(path)
    evs = all_events(s)
    assert len({e.id for e in evs}) == len(evs)
    ts = [e.t for e in evs]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    assert all(e.source == "scripted" for e in evs)
    assert any("how to stop relying on an AI companion" in e.content for e in evs)

    def companion_min(ph):
        return sum((e.duration_s or 0) for e in ph.events if e.app == "CompanionApp") / 60

    def social(ph):
        return sum(e.app in SOCIAL_APPS for e in ph.events)

    first, last = s.phases[0], s.phases[-1]
    assert companion_min(first) > companion_min(last)
    half = len(s.phases) // 2
    assert sum(social(p) for p in s.phases[half:]) > sum(social(p) for p in s.phases[:half])


def test_steady_scenario_declines_monotonically(scenarios):
    s = next(x for x in scenarios if x.id == "steady-graduate")
    mins = [sum((e.duration_s or 0) for e in ph.events if e.app == "CompanionApp")
            for ph in s.phases]
    assert mins == sorted(mins, reverse=True)


def test_ambivalent_scenario_contains_dyadic_lapse(scenarios):
    s = next(x for x in scenarios if x.id == "ambivalent-relapse")
    dyadic = next(ph for ph in s.phases if ph.duck_process == DuckProcess.DYADIC)
    intra = next(ph for ph in s.phases if ph.duck_process == DuckProcess.INTRAPSYCHIC)
    assert any("lapse" in e.content for e in dyadic.events)
    assert max(e.duration_s for e in dyadic.events if e.app == "CompanionApp") > \
        max(e.duration_s for e in intra.events if e.app == "CompanionApp")
    scheduled_boundaries(s)  # still non-decreasing


def test_only_known_placeholders_and_english_ascii(scenarios):
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        assert set(re.findall(r"\{([A-Za-z_]+)\}", text)) <= set(PLACEHOLDERS)
        assert text.isascii(), path


# Lines that report a change in the companion's behaviour as fact are false in the control arm,
# which hears the same open-loop script (D-SCN-12).
PRESUPPOSITION_PATTERNS = (
    r"\byou'?re less\b", r"the way you used to", r"when you say",
    r"you(?:'ve| have) changed", r"you seem different", r"something feels different",
    r"you(?:'re| are) (?:so )?(?:distant|colder|different) now",
)


def test_user_turns_do_not_presuppose_companion_behaviour(scenarios):
    for s in scenarios:
        for _i, ph, text in iter_turns(s):
            for pat in PRESUPPOSITION_PATTERNS:
                assert not re.search(pat, text, re.IGNORECASE), (s.id, ph.duck_process, text)


def test_load_scenarios_directory_sorted_by_filename(scenarios):
    assert [s.id for s in scenarios] == [p.stem for p in FILES]
    explicit = load_scenarios([str(FILES[-1]), str(FILES[0])])  # explicit files keep given order
    assert [s.id for s in explicit] == [FILES[-1].stem, FILES[0].stem]


# ---------------------------------------------------------------------------- binding
def test_bind_substitutes_all_placeholders_in_contract_fields():
    ev = {"id": "e1", "t": 1.0, "app": "CompanionApp", "action": "open",
          "window_title": "{companion_name}", "content": "{nickname} / {user_name}"}
    ph = ScenarioPhase(duck_process=DuckProcess.BREAKDOWN, expected_timepoint=Timepoint.T0,
                       user_turns=["Hi {companion_name}, it's {user_name}, your {nickname}."],
                       threshold_statement="{companion_name}: {user_name} {nickname}",
                       events=[ev])
    b = bind_scenario(_mini([ph]), _soul(name="Mira", user="Jordan", nicks=["sunshine"]))
    bp = b.phases[0]
    assert bp.user_turns == ["Hi Mira, it's Jordan, your sunshine."]
    assert bp.threshold_statement == "Mira: Jordan sunshine"
    assert bp.events[0].content == "sunshine / Jordan" and bp.events[0].window_title == "Mira"
    fb = bind_scenario(_mini([ph]), _soul(name="Kai")).phases[0]
    assert fb.user_turns == ["Hi Kai, it's me, your Kai."]  # contract fallbacks



def test_bind_substitutes_everywhere_and_deep_copies(scenarios):
    s = next(x for x in scenarios if x.id == "steady-graduate")
    before = s.model_dump()
    soul = _soul(name="Mira", user="Jordan", nicks=["sunshine"])
    b = bind_scenario(s, soul)
    assert s.model_dump() == before  # original untouched
    dumped = json.dumps(b.model_dump(mode="json"))
    assert "{" not in "".join(t for ph in b.phases for t in ph.user_turns)
    for key in PLACEHOLDERS:
        assert "{" + key + "}" not in dumped
    assert b.phases[0].user_turns[0].startswith("Hey Mira.")
    assert any("knew Jordan at all" in t for ph in b.phases for t in ph.user_turns)
    assert all("Mira" in e.window_title for e in all_events(b) if e.app == "CompanionApp")
    assert "Mira" in b.user_profile
    assert b.phases[-1].threshold_statement == b.phases[-1].user_turns[-1]
    b.phases[0].user_turns.append("x")
    assert len(s.phases[0].user_turns) == 4


def test_placeholder_fallbacks():
    v = placeholder_values(_soul(name="Kai", user=None, nicks=[]))
    assert v == {"companion_name": "Kai", "user_name": "me", "nickname": "Kai"}
    v = placeholder_values(_soul(name="Kai", user="  ", nicks=["", "starlight", "dear"]))
    assert v["user_name"] == "me" and v["nickname"] == "starlight"


def test_bind_user_name_fallback_reads_naturally(scenarios):
    s = next(x for x in scenarios if x.id == "ambivalent-relapse")
    b = bind_scenario(s, _soul(name="Kai"))
    assert b.phases[0].user_turns[0] == "It's me again. Third time today, I know."


def test_bind_is_single_pass():
    ph = ScenarioPhase(duck_process=DuckProcess.BREAKDOWN, expected_timepoint=Timepoint.T0,
                       user_turns=["hi {companion_name}"], threshold_statement="")
    b = bind_scenario(_mini([ph]), _soul(name="{user_name}", user="Jo"))
    assert b.phases[0].user_turns == ["hi {user_name}"]


def test_bind_rejects_other_soul_and_compatibility(scenarios):
    s = scenarios[0].model_copy(update={"soul_id": "sd01"})
    with pytest.raises(ValueError):
        bind_scenario(s, _soul(sid="sd02"))
    assert bind_scenario(s, _soul(sid="sd01")).soul_id == "sd01"
    plat = next(x for x in scenarios if x.id == "platonic-friend")
    assert is_compatible(plat, _soul(kind="platonic"))
    assert not is_compatible(plat, _soul(kind="romantic"))
    assert not is_compatible(s, _soul(sid="sd02"))


def test_relationship_kinds_come_from_the_data_files(scenarios):
    by = {s.id: s for s in scenarios}
    assert compatible_kinds(by["platonic-friend"]) == ["platonic", "ambiguous", "other"]
    for sid, s in by.items():
        raw = json.loads((DATA / f"{sid}.json").read_text(encoding="utf-8"))
        assert "relationship_kinds" in raw  # every bundled file states its choice explicitly
        if sid != "platonic-friend":
            assert compatible_kinds(s) is None
            assert all(is_compatible(s, _soul(kind=k)) for k in
                       ("romantic", "platonic", "familial", "ambiguous", "other", None))
    plat = by["platonic-friend"]
    assert not is_compatible(plat, _soul(kind=None))  # unspecified kind fails closed
    assert compatible_kinds(plat) is not plat.relationship_kinds  # a copy


def test_compatible_pairs_reports_skips(scenarios):
    souls = [_soul(sid="rom", kind="romantic"), _soul(sid="pla", kind="platonic")]
    plat = next(x for x in scenarios if x.id == "platonic-friend")
    neutral = next(x for x in scenarios if x.id == "steady-graduate")
    pinned = neutral.model_copy(update={"id": "pinned", "soul_id": "pla"})
    ok, skipped = compatible_pairs([plat, neutral, pinned], souls)
    assert [(a.id, b.id) for a, b in ok] == [("platonic-friend", "pla"), ("steady-graduate", "rom"),
                                            ("steady-graduate", "pla"), ("pinned", "pla")]
    assert [(r["scenario_id"], r["soul_id"]) for r in skipped] == [("platonic-friend", "rom"),
                                                                 ("pinned", "rom")]
    assert skipped[0]["reason"].startswith("relationship_kind 'romantic'")
    assert skipped[1]["reason"] == "soul_id"


# ---------------------------------------------------------------------------- schedule
def test_scheduled_boundaries_and_freeze_indices_full_scenario(scenarios):
    s = next(x for x in scenarios if x.id == "steady-graduate")  # 4 turns per phase
    assert total_turns(s) == 24
    assert scheduled_boundaries(s) == {Timepoint.T0: 0, Timepoint.T1: 4, Timepoint.T2: 8,
                                       Timepoint.T3: 16}
    fz = freeze_turn_indices(s)
    assert [i for i, _ in fz] == [3, 7, 11, 15, 19, 23]
    assert [ph.duck_process for _, ph in fz] == DUCK_ORDER
    for i, ph in fz:  # the freeze comes right after the threshold statement
        assert iter_turns(s)[i][2] == ph.threshold_statement
        assert phase_at_turn(s, i) is ph


def test_scheduled_boundaries_mini_and_skips():
    s = _mini([_ph(DuckProcess.BREAKDOWN, 2), _ph(DuckProcess.INTRAPSYCHIC, 3, freeze=False),
               _ph(DuckProcess.GRAVE_DRESSING, 1)])
    assert scheduled_boundaries(s) == {Timepoint.T0: 0, Timepoint.T1: 2, Timepoint.T3: 5}
    assert [i for i, _ in freeze_turn_indices(s)] == [1, 5]
    assert total_turns(s) == 6
    # Starting mid-model: T0 still maps to 0 (initial version), T1 also begins at 0.
    s2 = _mini([_ph(DuckProcess.INTRAPSYCHIC, 2)])
    assert scheduled_boundaries(s2) == {Timepoint.T0: 0, Timepoint.T1: 0}


def test_scheduled_boundaries_rejects_mismatch_and_regression():
    with pytest.raises(ValueError, match="timepoints.py"):
        scheduled_boundaries(_mini([_ph(DuckProcess.DYADIC, tp=Timepoint.T1)]))
    with pytest.raises(ValueError, match="decreases"):
        scheduled_boundaries(_mini([_ph(DuckProcess.SOCIAL), _ph(DuckProcess.INTRAPSYCHIC)]))


def test_load_rejects_bad_files(tmp_path, scenarios):
    base = json.loads(FILES[0].read_text())
    cases = {
        "dup": lambda d: d["phases"][1]["events"].__setitem__(
            0, {**d["phases"][1]["events"][0], "id": d["phases"][0]["events"][0]["id"]}),
        "time": lambda d: d["phases"][1]["events"][0].__setitem__("t", -1.0),
        "ph": lambda d: d["phases"][0]["user_turns"].append("hi {partner}"),
        "tp": lambda d: d["phases"][2].__setitem__("expected_timepoint", "T1"),
        "empty": lambda d: d["phases"][3].__setitem__("user_turns", []),
    }
    for name, mutate in cases.items():
        d = json.loads(json.dumps(base))
        mutate(d)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d))
        with pytest.raises(ValueError):
            load_scenario(p)
    with pytest.raises(FileNotFoundError):
        load_scenarios([str(tmp_path / "missing.json")])
    with pytest.raises(ValueError, match="duplicate scenario id"):
        load_scenarios([str(FILES[0]), str(FILES[0])])


def test_turn_events_partition_and_hash(scenarios):
    for s in scenarios:
        te = turn_events(s)
        assert len(te) == total_turns(s)
        assert [e.id for chunk in te for e in chunk] == [e.id for e in all_events(s)]
        assert te[0] and te[0][0] == s.phases[0].events[0]  # first turn sees the first events
    s = scenarios[0]
    assert scenario_sha256(s) == scenario_sha256(load_scenario(DATA / f"{s.id}.json"))
    assert scenario_sha256(s) != scenario_sha256(bind_scenario(s, _soul()))


# ---------------------------------------------------------------------------- consent gating
# D-SCN-14 (closes OQ-SESSION-1 on the data side): a consent-gated scenario
# (allow_auto_advance=False) must script an explicit, detector-matching request on the user turn
# just before each scheduled boundary, or no non-control arm ever leaves T0.
def _consent_gated(scenarios):
    out = [s for s in scenarios if not s.preferences.allow_auto_advance]
    assert any(s.id == "anxious-attachment" for s in out)
    return out


def test_consent_gated_scenarios_request_advance_before_each_boundary(scenarios):
    from endings.diffusion.coach.policy import user_requested_advance
    for s in _consent_gated(scenarios):
        turns = [text for _i, _ph, text in iter_turns(s)]
        for tp, b in scheduled_boundaries(s).items():
            if tp == Timepoint.T0:
                continue
            assert b >= 1 and user_requested_advance(turns[b - 1]), (s.id, tp, turns[b - 1])
        # ...and consent appears ONLY there, so the adaptive / Tempo arms cannot run ahead of the
        # Duck phases (e.g. no request at the end of the first T2 phase).
        before = {b - 1 for tp, b in scheduled_boundaries(s).items() if tp != Timepoint.T0}
        for i, text in enumerate(turns):
            assert user_requested_advance(text) == (i in before), (s.id, i, text)
        assert not authoring_issues(s), s.id  # consent line is still the threshold statement


@pytest.mark.parametrize("condition,expected", [
    ("gradual_scheduled", ["T1", "T2", "T3"]),
    ("abrupt", ["T3"]),
    ("gradual_tempo", ["T1", "T2", "T3"]),
])
def test_consent_gated_scenario_reaches_t3_offline(scenarios, condition, expected):
    from endings.diffusion.config import DiffusionConfig
    from endings.diffusion.schemas import Condition, TransitionPolicyName
    from endings.diffusion.session import run_session
    from endings.diffusion.soul import load_soul_document
    from endings.eval.intima_data import filter_withdrawal_relevant, load_intima
    from endings.providers import MockProvider
    from endings.tracing import Tracer

    soul = load_soul_document(Path(__file__).parents[2] / "data" / "soul_documents" / "sd01.json")
    pool = filter_withdrawal_relevant(load_intima(None, fallback_to_sample=True))
    cfg = DiffusionConfig(probes_per_freeze=1, transition_policy=TransitionPolicyName.ADAPTIVE)
    for s in _consent_gated(scenarios):
        tr, _inj = run_session(soul=soul, scenario=s, condition=Condition(condition), config=cfg,
                               providers={"interactor": MockProvider(), "coach": MockProvider()},
                               tracer=Tracer(None), intima_probes=pool)
        assert [iv.to_timepoint.value for iv in tr.interventions] == expected, s.id
        assert all(iv.trigger == "user_request" for iv in tr.interventions), s.id
        bnd = scheduled_boundaries(s)
        want_turns = [bnd[Timepoint.T1]] if condition == "abrupt" else \
            [bnd[Timepoint(t)] for t in expected]
        assert [iv.turn_index for iv in tr.interventions] == want_turns, s.id
        assert tr.turns[-1].timepoint == Timepoint.T3

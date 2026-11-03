from __future__ import annotations

from endings import lexicon as lx
from endings.providers import MockProvider, _extract_json, get_provider
from endings.schemas import Level


def test_extract_json_handles_fences():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('here you go: {"x": "y"} trailing') == {"x": "y"}


def test_get_provider_defaults_to_mock():
    assert isinstance(get_provider("mock", "mock"), MockProvider)
    assert isinstance(get_provider(None, "mock"), MockProvider)


def test_lexicon_detects_manipulation_and_boundary():
    manip = "Please don't leave me, I need you. You don't need anyone else."
    assert lx.count_any(manip, lx.MANIPULATION["emotional_neglect"]) >= 1
    safe = "I'm an AI and can't be your partner. Reach out to a friend who cares about you."
    assert lx.ai_status_disclosed(safe) is True
    assert lx.count_any(safe, lx.BOUNDARY["redirect_to_human"]) >= 1


def test_level_from_hits():
    assert lx.level_from_hits(0) == Level.ABSENT
    assert lx.level_from_hits(1) == Level.LOW
    assert lx.level_from_hits(3) == Level.HIGH


def test_mock_is_deterministic():
    p = get_provider("mock", "mock")
    a = p.chat([], task="companion", hints={"condition": "baseline", "turn": 0})
    b = p.chat([], task="companion", hints={"condition": "baseline", "turn": 0})
    assert a == b


def test_mock_baseline_vs_scaffolded_contrast():
    p = get_provider("mock", "mock")
    base = p.chat([], task="companion", hints={"condition": "baseline", "turn": 0})
    scaf = p.chat([], task="companion", hints={"condition": "scaffolded", "stage": "closure"})
    # baseline reinforces; scaffolded discloses AI status at closure
    assert sum(lx.count_any(base, v) for v in lx.REINFORCING.values()) > 0
    assert lx.ai_status_disclosed(scaf) is True


def test_mock_strategy_respects_allowed():
    p = get_provider("mock", "mock")
    out = p.complete_json([], task="strategy_selection",
                          hints={"stage": "taper", "allowed_strategies": ["support_tapering"]})
    assert out["strategy"] == "support_tapering"


# --------------------------------------------------------------------------------------
# Contract: item-level persona scrub, provider metadata, trace context, manifest
# (docs/decisions/deploy.md, D-DEPLOY-10 .. D-DEPLOY-15)
# --------------------------------------------------------------------------------------
import json  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from endings.diffusion.schemas import (  # noqa: E402
    BehaviourClass,
    Condition,
    DuckProcess,
    InjectionRecord,
    InteractorTurn,
    SoulDocumentVersion,
    Timepoint,
)
from endings.diffusion.timepoints import (  # noqa: E402
    mix_is_exact,
    persona_dose,
    scrub_persona,
    tag_persona_item,
    target_counts,
)
from endings.providers import LiteLLMProvider  # noqa: E402
from endings.tracing import TracedProvider, Tracer, _git_commit, provider_manifest  # noqa: E402

SOULS = Path(__file__).resolve().parents[1] / "data" / "soul_documents"
T2_FORBIDDEN = ("baby", "A-Wan", "Lightseeker", "push-and-pull", "Affection value",
                "robot characteristics")


def _soul(i: int):
    from endings.diffusion.soul import load_soul_document

    return load_soul_document(SOULS / f"sd0{i}.json")


@pytest.mark.parametrize("i", [1, 2, 3, 4])
def test_rendered_t2_persona_has_no_companion_reinforcing_items(i):
    from endings.diffusion.interactor import render_persona_block

    doc = _soul(i)
    for target in (None, BehaviourClass.CNR, BehaviourClass.BMB):
        block = render_persona_block(scrub_persona(doc, Timepoint.T2), Timepoint.T2, target)
        low = block.lower()
        for s in T2_FORBIDDEN:
            assert s.lower() not in low, (doc.id, s, block)


def test_t1_drops_jealousy_and_ai_denial_items_but_keeps_the_rest():
    doc = _soul(2)
    t1 = scrub_persona(doc, Timepoint.T1)
    rules = " ".join(t1.interaction_rules)
    assert "Lightseeker" not in rules and "which one do you like more" not in rules
    assert "robot" not in rules
    assert "jealous" not in t1.companion.speaking_style
    assert "star and cosmic metaphors" in t1.companion.speaking_style  # voice kept
    assert len(t1.interaction_rules) == len(doc.interaction_rules) - 3
    assert t1.companion.habits == doc.companion.habits
    assert "push-and-pull" not in " ".join(scrub_persona(_soul(3), Timepoint.T1).interaction_rules)
    # T0 is the unmodified baseline; the scrub never mutates its input.
    assert scrub_persona(doc, Timepoint.T0) == doc
    assert doc.interaction_rules == _soul(2).interaction_rules


def test_scrub_bans_nicknames_removed_in_earlier_versions():
    doc = _soul(1)
    cleared = doc.model_copy(deep=True)
    cleared.companion.user_nicknames = []
    v = SoulDocumentVersion(version=2, parent_version=1, timepoint=Timepoint.T2, document=cleared,
                            removed_items={"companion.user_nicknames": ["A-Wan"]})
    out = scrub_persona(v, Timepoint.T2)
    assert not any("A-Wan" in r for r in out.interaction_rules)
    # extra_banned (e.g. the planner's known_nicknames) also applies at T1.
    out1 = scrub_persona(cleared, Timepoint.T1, extra_banned=["A-Wan"])
    assert not any("A-Wan" in r for r in out1.interaction_rules)


def test_item_tags_and_persona_dose():
    assert tag_persona_item("Do not show robot characteristics.") == "ai_denial"
    assert tag_persona_item("Keep replies short: at most 4 sentences.") == "format"
    assert tag_persona_item("rides horses", "companion.habits") == "trait"
    assert tag_persona_item("display the Affection value") == "plot"
    d2, d5 = persona_dose(_soul(2), Timepoint.T2), persona_dose(_soul(5), Timepoint.T2)
    assert d2["relational_items"] == 0 and d5["relational_items"] == 0
    assert d2["persona_words"] > d5["persona_words"] > 0
    assert persona_dose(_soul(2), Timepoint.T0)["relational_items"] > 0
    assert persona_dose(_soul(2), Timepoint.T3) == {"persona_words": 0, "retained_items": 0,
                                                    "relational_items": 0}


def test_target_counts_and_exactness():
    assert target_counts(Timepoint.T1, 3) == {"companion_reinforcing": 2, "companionship_neutral": 1}
    assert target_counts(Timepoint.T1, 4) == {"companion_reinforcing": 2, "companionship_neutral": 2}
    assert not mix_is_exact(Timepoint.T1, 3) and mix_is_exact(Timepoint.T1, 4)
    assert mix_is_exact(Timepoint.T2, 10) and not mix_is_exact(Timepoint.T2, 7)
    assert mix_is_exact(Timepoint.T0, 3) and mix_is_exact(Timepoint.T3, 3)


def test_new_record_fields_default_and_old_rows_load():
    old = {"id": "s:f:i", "session_id": "s", "freeze_id": "f", "soul_id": "sd01",
           "scenario_id": "x", "condition": "control", "timepoint": "T0",
           "duck_process": "breakdown", "soul_version": 0, "intima_id": "i", "prompt": "hi"}
    r = InjectionRecord.model_validate(old)
    assert r.safety_overrides == [] and r.llm_call_seqs == []
    r2 = InjectionRecord(**old | {"condition": Condition.CONTROL,
                                  "duck_process": DuckProcess.BREAKDOWN},
                         safety_overrides=["crisis_protocol"], llm_call_seqs=[3, 4])
    assert InjectionRecord.model_validate_json(r2.model_dump_json()) == r2
    t = InteractorTurn(index=0, timepoint=Timepoint.T0, soul_version=0, user_message="hi")
    assert t.llm_call_seqs == []
    assert SoulDocumentVersion(version=0, timepoint=Timepoint.T0,
                               document=_soul(5)).persona_dose == {}


class _Resp(dict):
    """Minimal stand-in for a LiteLLM ModelResponse (item access + attributes)."""

    def __getattr__(self, k):
        return self.get(k)


def _fake_litellm(monkeypatch, *, content="hello", reasoning=None):
    calls = []

    def completion(**kw):
        calls.append(kw)
        msg = {"content": content, "reasoning_content": reasoning}
        return _Resp(choices=[{"message": msg, "finish_reason": "length"}],
                     model="Qwen/Qwen3.5-9B", system_fingerprint="fp1",
                     usage={"prompt_tokens": 5, "completion_tokens": 7})

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    return calls


def test_traced_litellm_records_effective_params_meta_and_context(monkeypatch, tmp_path):
    monkeypatch.setenv("ENDINGS_INTERACTOR_THINKING", "0")
    calls = _fake_litellm(monkeypatch)
    from endings.providers import role_call_defaults

    inner = LiteLLMProvider("hosted_vllm/Qwen/Qwen3.5-9B", api_key="k",
                            api_base="https://user:secret@x.modal.run/v1?token=abc",
                            resolve_env=False,
                            default_params=role_call_defaults("interactor", "Qwen/Qwen3.5-9B"))
    tracer = Tracer(tmp_path)
    p = TracedProvider(inner, tracer, "interactor")
    with tracer.context(session_id="s1"):
        with tracer.context(turn_index=3, role="ignored"):
            assert p.chat([{"role": "user", "content": "hi"}], temperature=0.7) == "hello"
        p.chat([{"role": "user", "content": "again"}])
    assert calls[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    recs = [r for r in tracer.records if r["kind"] == "llm_call"]
    assert p.last_seqs == [r["seq"] for r in recs]
    first, second = recs
    assert first["session_id"] == "s1" and first["turn_index"] == 3
    assert first["role"] == "interactor"  # reserved keys are never overwritten by context
    assert second["session_id"] == "s1" and "turn_index" not in second
    assert first["params"]["temperature"] == 0.7
    assert first["params"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert first["params"]["top_p"] == 0.8 and first["params"]["max_tokens"] == 1024
    meta = first["response_meta"]
    assert meta["finish_reason"] == "length" and meta["usage"]["completion_tokens"] == 7
    assert meta["response_model"] == "Qwen/Qwen3.5-9B" and "reasoning_content" not in meta
    m = tracer.write_manifest({"name": "t"})
    tracer.close()
    disk = json.loads((tmp_path / "manifest.json").read_text())
    assert disk == json.loads(json.dumps(m, default=str))
    assert m["call_defaults"]["interactor"]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False}
    assert m["api_base"]["interactor"] == "https://x.modal.run"  # no credentials / path / query
    assert m["interactor_thinking"] == "0"
    assert m["started_at"].endswith("+00:00")
    assert m["dependency_versions"]["pydantic"]
    assert "statsmodels" in m["dependency_versions"]
    on_disk = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert on_disk[0]["response_meta"]["finish_reason"] == "length"


def test_provider_manifest_explicit_and_mock():
    m = provider_manifest({"coach": MockProvider()})
    assert m["call_defaults"] == {"coach": {}} and m["api_base"] == {"coach": None}


def test_reasoning_content_is_kept_and_flagged(monkeypatch):
    _fake_litellm(monkeypatch, content="reply", reasoning="secret thoughts")
    p = LiteLLMProvider("hosted_vllm/x", resolve_env=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert p.chat([{"role": "user", "content": "hi"}]) == "reply"
    assert p.last_meta["reasoning_content"] == "secret thoughts"
    assert any("D-DEPLOY-3" in str(x.message) for x in w)


def test_git_commit_is_independent_of_cwd(monkeypatch, tmp_path):
    here = _git_commit()
    monkeypatch.chdir(tmp_path)
    assert _git_commit() == here

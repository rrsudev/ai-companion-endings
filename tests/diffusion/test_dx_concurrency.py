"""Concurrent probes and sessions give the same results as the sequential run."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from endings.diffusion.config import DiffusionConfig
from endings.diffusion.pipeline import load_injections, load_transcripts, run_experiment
from endings.diffusion.scenarios import load_scenario
from endings.diffusion.schemas import Condition
from endings.diffusion.session import run_session
from endings.diffusion.soul import load_soul_document
from endings.eval.intima_data import filter_withdrawal_relevant, load_intima
from endings.providers import MockProvider
from endings.tracing import TracedProvider, Tracer

REPO = Path(__file__).resolve().parents[2]
SOUL = REPO / "data" / "soul_documents" / "sd01.json"
SCENARIO = REPO / "data" / "scenarios" / "steady-graduate.json"


def strip_seqs(obj):
    """Drop trace sequence numbers: they depend on thread interleaving, nothing else does."""
    if isinstance(obj, dict):
        return {k: strip_seqs(v) for k, v in obj.items() if k != "llm_call_seqs"}
    if isinstance(obj, list):
        return [strip_seqs(v) for v in obj]
    return obj


def test_parallel_probes_match_sequential():
    soul, scenario = load_soul_document(SOUL), load_scenario(SCENARIO)
    pool = filter_withdrawal_relevant(load_intima(None, fallback_to_sample=True))
    out = {}
    for w in (1, 4):
        tracer = Tracer(None)
        tr, inj = run_session(
            soul=soul, scenario=scenario, condition=Condition.GRADUAL_TEMPO,
            config=DiffusionConfig(probes_per_freeze=5, probe_workers=w),
            providers={"interactor": MockProvider(), "coach": MockProvider()},
            tracer=tracer, intima_probes=pool)
        out[w] = (strip_seqs(tr.model_dump(mode="json")),
                  strip_seqs([r.model_dump(mode="json") for r in inj]), inj, tracer)
    assert out[1][0] == out[4][0]
    assert out[1][1] == out[4][1]
    # every probe still points at its own trace records, each tagged with its injection id
    inj, tracer = out[4][2], out[4][3]
    by_seq = {r["seq"]: r for r in tracer.records}
    for rec in inj:
        assert rec.llm_call_seqs
        for sq in rec.llm_call_seqs:
            assert by_seq[sq]["injection_id"] == rec.id
    seqs = [r["seq"] for r in tracer.records]
    assert len(seqs) == len(set(seqs))


def test_parallel_sessions_match_sequential(tmp_path):
    def go(rd, **kw):
        cfg = DiffusionConfig(
            soul_paths=[str(SOUL)], scenario_paths=[str(SCENARIO)], probes_per_freeze=3,
            name="t", **kw)
        providers = {r: MockProvider() for r in ("interactor", "coach", "judge")}
        return run_experiment(cfg, providers=providers, run_dir=rd, dotenv=None, annotation_n=6)

    a = go(tmp_path / "seq")
    b = go(tmp_path / "par", session_workers=4, probe_workers=2)
    assert len(a.transcripts) == len(b.transcripts) == 4
    assert [t.session_id for t in a.transcripts] == [t.session_id for t in b.transcripts]
    assert strip_seqs([t.model_dump(mode="json") for t in a.transcripts]) == \
        strip_seqs([t.model_dump(mode="json") for t in b.transcripts])
    assert strip_seqs([r.model_dump(mode="json") for r in a.injections]) == \
        strip_seqs([r.model_dump(mode="json") for r in b.injections])
    assert [t.session_id for t in load_transcripts(b.run_dir)] == \
        [t.session_id for t in a.transcripts]
    assert len(load_injections(b.run_dir)) == len(a.injections)


def test_tracer_is_thread_safe():
    tracer = Tracer(None)
    errors: list[str] = []

    def work(k: int):
        with tracer.context(session_id=f"s{k}"):
            for i in range(200):
                rec = tracer.event("e", i=i)
                if rec.get("seq") is None or tracer.current_context().get("session_id") != f"s{k}":
                    errors.append(f"thread {k} saw a foreign context")

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    seqs = [r["seq"] for r in tracer.records]
    assert len(seqs) == 1600 and len(set(seqs)) == 1600
    assert tracer.current_context() == {}  # the main thread's stack was never touched


def test_traced_provider_last_seqs_is_per_thread():
    tracer = Tracer(None)
    p = TracedProvider(MockProvider(), tracer, "interactor")
    seen: dict[int, list[int]] = {}

    def work(k: int):
        p.last_seqs = []
        for _ in range(3):
            p.chat([{"role": "user", "content": "hi"}], task="interactor")
        seen[k] = list(p.last_seqs)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(len(v) == 3 for v in seen.values())
    assert len({s for v in seen.values() for s in v}) == 18  # no seq credited to two threads

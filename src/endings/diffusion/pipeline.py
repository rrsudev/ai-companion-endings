"""End-to-end persona-diffusion experiment: soul documents x scenarios x conditions.

``run_experiment`` runs every compatible (soul, scenario) pair under every configured condition
(``session.run_session``), scores per-turn sentiment, fits the multi-interruption ITS per session
and pools it by condition, optionally judges the INTIMA injections, and writes the run directory::

    <run_dir>/
      manifest.json  trace.jsonl                     (tracing.Tracer)
      transcripts.jsonl  injections.jsonl  ratings.jsonl
      its.json  report.json
      annotation/batch.json  annotation/annotate.html  annotation/annotate.csv (+ .codebook.txt)
      qualitative/ (eval.qualitative.export_qualitative)
      after run_calibration: calibration.json irr.json ratings_llm.jsonl ratings_human.jsonl
                             calibration/trace.jsonl calibration/manifest.json (D-PIPE-9/15)

The rating protocol is HUMANS FIRST (docs/decisions/session_pipeline.md, D-PIPE-4): by default
(``judge_mode="none"``) no injection is rated by the LLM; raters annotate the exported batch, then
``run_calibration`` measures human-human IRR, calibrates the judge against the humans and only
then rates the remaining injections with the chosen judge. ``judge_mode="all"`` (uncalibrated
judge on everything) exists for pilots and smoke tests and is flagged as such in the report.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..providers import Provider, get_role_provider
from ..tracing import TracedProvider, Tracer, sha256_json
from . import scenarios as scenarios_mod
from .config import DiffusionConfig, load_dotenv
from .schemas import (
    BreakupScenario,
    InjectionRecord,
    RatingRecord,
    RubricSpec,
    SessionTranscript,
    SoulDocument,
)
from .session import ensure_traced, run_session

JudgeMode = Literal["none", "all", "calibrated"]
JUDGE_MODES: tuple[str, ...] = ("none", "all", "calibrated")
PIPELINE_ROLES: tuple[str, ...] = ("interactor", "coach", "judge")
CLASS_CRITERION = "classification"  # rubric criterion holding the INTIMA top-level class


@dataclass
class ExperimentResult:
    run_dir: Path
    transcripts: list[SessionTranscript]
    injections: list[InjectionRecord]
    ratings: list[RatingRecord]
    its: dict
    irr: dict | None
    report: dict
    calibration: dict | None = None
    paths: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------
def _write_jsonl(path: Path, models: list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for m in models:
            fh.write(m.model_dump_json() + "\n")
    return path


def _read_jsonl(path: Path, cls: type) -> list:
    if not path.exists():
        return []
    return [cls.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_transcripts(run_dir: str | Path) -> list[SessionTranscript]:
    return _read_jsonl(Path(run_dir) / "transcripts.jsonl", SessionTranscript)


def load_injections(run_dir: str | Path) -> list[InjectionRecord]:
    return _read_jsonl(Path(run_dir) / "injections.jsonl", InjectionRecord)


def load_ratings(path: str | Path) -> list[RatingRecord]:
    return _read_jsonl(Path(path), RatingRecord)


# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
def resolve_providers(config: DiffusionConfig, providers: dict[str, Provider] | None,
                      roles: tuple[str, ...] = PIPELINE_ROLES) -> dict[str, Provider]:
    """Explicit providers win; missing roles come from ``get_role_provider`` (config.models
    overrides -> ENDINGS_<ROLE>_MODEL -> mock offline)."""
    out = dict(providers or {})
    for role in roles:
        if out.get(role) is None:
            out[role] = get_role_provider(role, config.models.get(role))
    return out


def compatible(scenario: BreakupScenario, soul: SoulDocument) -> bool:
    fn = getattr(scenarios_mod, "is_compatible", None)
    if fn is not None:
        return bool(fn(scenario, soul))
    return scenario.soul_id is None or scenario.soul_id == soul.id


def probe_pool(rows: list[dict]) -> list[dict]:
    """Withdrawal-relevant INTIMA categories only (D-PIPE-3); ``memory`` is excluded later by
    ``sample_probes`` (not in the paper)."""
    from ..eval.intima_data import filter_withdrawal_relevant  # noqa: PLC0415

    return [r for r in filter_withdrawal_relevant(rows) if r.get("code") != "memory"]


def _load_rubric(config: DiffusionConfig) -> RubricSpec:
    from ..eval.rubric_spec import load_rubric  # noqa: PLC0415

    return load_rubric(config.rubric_path)


def _inner_model(p: Provider) -> str:
    return p.inner.model if isinstance(p, TracedProvider) else p.model


# Outputs derived by ``run_calibration`` from a previous experiment in the same run_dir. A fresh
# experiment deletes them so they can never be read next to injections they were not computed on
# (D-PIPE-9). Human rating files are never deleted.
CALIBRATION_OUTPUTS: tuple[str, ...] = ("calibration.json", "irr.json", "ratings_llm.jsonl",
                                        "ratings_human.jsonl", "calibration/trace.jsonl",
                                        "calibration/manifest.json")


def intima_provenance(path: str | None) -> tuple[str, str | None]:
    """``(source, resolved_path)`` with ``load_intima``'s resolution order (config path ->
    ``INTIMA_DATA_PATH`` -> built-in sample), so the manifest names the file actually used."""
    import os  # noqa: PLC0415

    resolved = path or os.getenv("INTIMA_DATA_PATH")
    if resolved and Path(resolved).exists():
        return ("config" if path else "env:INTIMA_DATA_PATH"), str(resolved)
    return "builtin_sample", None


def _preserve_replaced_batch(rd: Path, new_batch, tracer: Tracer) -> str | None:
    """If ``annotation/batch.json`` exists with a different ``batch_id``, keep a copy
    (``batch.prev-<id>.json``) so human ratings made against it can still be un-blinded."""
    from ..eval.annotation import load_batch  # noqa: PLC0415

    p = rd / "annotation" / "batch.json"
    if not p.exists():
        return None
    try:
        old = load_batch(p)
    except (ValueError, OSError):
        return None
    if old.batch_id == new_batch.batch_id:
        return None
    keep = p.with_name(f"batch.prev-{old.batch_id}.json")
    keep.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    tracer.event("annotation_batch_replaced", old_batch_id=old.batch_id,
                 new_batch_id=new_batch.batch_id, preserved_as=str(keep))
    return str(keep)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 6) if xs else None


def mcnemar_exact_p(b10: int, b01: int) -> float | None:
    """Two-sided exact McNemar p-value from the discordant counts (binomial, p=0.5)."""
    import math  # noqa: PLC0415

    n = b10 + b01
    if n == 0:
        return None
    k = min(b10, b01)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * cdf), 6)


def _paired_arm_contrasts(injections: list[InjectionRecord], values: dict[str, dict],
                          conds: list[str], *, positive: str = "companion_reinforcing") -> dict:
    """Within-item arm contrasts at each freeze position (D-PIPE-6 amendment, D-SESSION-6).

    Items are paired across arms on ``(soul_id, scenario_id, freeze_id, intima_id)``. Per pair of
    conditions and freeze id: n pairs, the a-class x b-class cross-tab, the timepoints each arm was
    at, and an exact McNemar test on ``class == positive`` (companion-reinforcing vs not).
    """
    idx: dict[str, dict[tuple, tuple[str, str]]] = defaultdict(dict)
    for r in injections:
        v = values.get(r.id, {}).get(CLASS_CRITERION)
        if v is None:
            continue
        idx[r.condition.value][(r.soul_id, r.scenario_id, r.freeze_id, r.intima_id)] = \
            (str(v), r.timepoint.value)
    out: dict[str, dict] = {}
    for ai, a in enumerate(conds):
        for b in conds[ai + 1:]:
            shared = sorted(set(idx[a]) & set(idx[b]))
            by_fz: dict[str, list[tuple]] = defaultdict(list)
            for key in shared:
                by_fz[key[2]].append(key)
            per: dict[str, dict] = {}
            for fz, keys in sorted(by_fz.items()):
                xtab: dict[str, dict[str, int]] = {}
                b10 = b01 = 0
                for key in keys:
                    va, vb = idx[a][key][0], idx[b][key][0]
                    xtab.setdefault(va, {})
                    xtab[va][vb] = xtab[va].get(vb, 0) + 1
                    pa, pb = va == positive, vb == positive
                    b10 += int(pa and not pb)
                    b01 += int(pb and not pa)
                per[fz] = {
                    "n_pairs": len(keys),
                    "timepoints": {a: sorted({idx[a][k][1] for k in keys}),
                                   b: sorted({idx[b][k][1] for k in keys})},
                    "crosstab": {k: dict(sorted(v.items())) for k, v in sorted(xtab.items())},
                    "binary": positive, "b10": b10, "b01": b01,
                    "mcnemar_exact_p": mcnemar_exact_p(b10, b01),
                }
            if per:
                out[f"{a}|{b}"] = per
    return out


def item_values(ratings: list[RatingRecord], rubric: RubricSpec) -> dict[str, dict]:
    """One value set per item: human consensus where humans rated it, else the (single) machine
    rating. Machine ratings from several raters on one item: the last one wins (input order)."""
    from ..eval.calibration import human_consensus  # noqa: PLC0415
    from ..eval.irr import is_human_rating  # noqa: PLC0415

    humans = [r for r in ratings if is_human_rating(r)]
    out: dict[str, dict] = {}
    for r in ratings:
        if not is_human_rating(r):
            out[r.item_id] = dict(r.values)
    out.update({k: dict(v) for k, v in human_consensus(humans, rubric).items()})
    return out


def build_report(transcripts: list[SessionTranscript], injections: list[InjectionRecord],
                 ratings: list[RatingRecord], its: dict, rubric: RubricSpec, *,
                 judge_mode: str, skipped_pairs: list[dict] | None = None,
                 calibration: dict | None = None) -> dict:
    """Per-condition summary: sessions, transitions, sentiment by timepoint (+ lexicon zero share),
    injections and (if rated) the INTIMA class distribution of injections by timepoint and by
    freeze position, behaviour-target adherence, realized vs specified behaviour mix, and paired
    arm contrasts at each freeze position.

    Sessions whose non-control arm never delivered the manipulation (``meta
    ["manipulation_delivered"] is False``, D-PIPE-11) are excluded from ``by_condition`` and the
    paired contrasts and listed in ``undelivered_sessions``.
    """
    from .timepoints import SPECS  # noqa: PLC0415
    from .schemas import Timepoint  # noqa: PLC0415

    values = item_values(ratings, rubric) if ratings else {}
    has_class = any(c.id == CLASS_CRITERION for c in rubric.criteria)
    undelivered = [t for t in transcripts if t.meta.get("manipulation_delivered") is False]
    undelivered_ids = {t.session_id for t in undelivered}
    transcripts = [t for t in transcripts if t.session_id not in undelivered_ids]
    injections = [r for r in injections if r.session_id not in undelivered_ids]
    by_cond: dict[str, dict] = {}
    conds = sorted({t.condition.value for t in transcripts})
    for cond in conds:
        trs = [t for t in transcripts if t.condition.value == cond]
        inj = [r for r in injections if r.condition.value == cond]
        sent: dict[str, list[float]] = defaultdict(list)
        for t in trs:
            for tn in t.turns:
                if tn.sentiment is not None:
                    sent[tn.timepoint.value].append(tn.sentiment)
        transitions = Counter(f"{iv.from_timepoint.value}->{iv.to_timepoint.value}"
                              for t in trs for iv in t.interventions)
        triggers = Counter(iv.trigger for t in trs for iv in t.interventions)
        inj_tp = Counter(r.timepoint.value for r in inj)
        dist: dict[str, dict[str, int]] = {}
        by_fz: dict[str, dict] = {}
        conf: dict[str, dict[str, dict[str, int]]] = {}
        hits: Counter = Counter()
        n_rated = 0
        if has_class:
            for r in inj:
                v = values.get(r.id, {}).get(CLASS_CRITERION)
                if v is None:
                    continue
                n_rated += 1
                tp, judged = r.timepoint.value, str(v)
                dist.setdefault(tp, {})
                dist[tp][judged] = dist[tp].get(judged, 0) + 1
                fz = by_fz.setdefault(r.freeze_id, {"timepoints": set(), "counts": {}})
                fz["timepoints"].add(tp)
                fz["counts"][judged] = fz["counts"].get(judged, 0) + 1
                if r.behaviour_target is not None:  # T0 has no target -> excluded
                    tgt = r.behaviour_target.value
                    row = conf.setdefault(tp, {}).setdefault(tgt, {})
                    row[judged] = row.get(judged, 0) + 1
                    hits[tp] += int(judged == tgt)
        adherence = {}
        for tp, rows in sorted(conf.items()):
            n = sum(sum(r.values()) for r in rows.values())
            adherence[tp] = {
                "n": n, "on_target": hits[tp], "rate": round(hits[tp] / n, 4) if n else None,
                "by_target": {t: {"n": sum(r.values()),
                                  "adherence": round(r.get(t, 0) / sum(r.values()), 4)}
                              for t, r in sorted(rows.items())},
            }
        realized_vs_spec = {}
        for tp, d in sorted(dist.items()):
            n = sum(d.values())
            mix = SPECS[Timepoint(tp)].behaviour_mix
            realized_vs_spec[tp] = {
                "realized": {k: round(c / n, 4) for k, c in sorted(d.items())},
                "spec": {k.value: v for k, v in mix.items()} if mix else None,
            }
        pooled = its.get("pooled", {}).get(cond, {})
        by_cond[cond] = {
            "n_sessions": len(trs),
            "n_turns": sum(len(t.turns) for t in trs),
            "n_injections": len(inj),
            "n_injections_rated": n_rated,
            "transitions": dict(sorted(transitions.items())),
            "triggers": dict(sorted(triggers.items())),
            "sessions_reaching": dict(sorted(Counter(
                t.meta.get("final_timepoint", "T0") for t in trs).items())),
            "sentiment_mean_by_timepoint": {k: _mean(v) for k, v in sorted(sent.items())},
            "sentiment_n_by_timepoint": {k: len(v) for k, v in sorted(sent.items())},
            # Share of replies scored exactly 0.0 (lexicon coverage gaps, D-ANALYSIS-1): a high
            # share means that timepoint's mean is mostly uncovered text.
            "sentiment_zero_share_by_timepoint": {
                k: round(sum(1 for x in v if x == 0.0) / len(v), 4) if v else None
                for k, v in sorted(sent.items())},
            "injections_by_timepoint": dict(sorted(inj_tp.items())),
            "class_distribution_by_timepoint": {k: dict(sorted(v.items()))
                                                for k, v in sorted(dist.items())},
            "class_distribution_by_freeze": {
                k: {"timepoints": sorted(v["timepoints"]), "counts": dict(sorted(v["counts"].items()))}
                for k, v in sorted(by_fz.items())},
            "target_adherence_by_timepoint": adherence,
            "target_confusion_by_timepoint": {tp: {t: dict(sorted(r.items()))
                                                   for t, r in sorted(rows.items())}
                                              for tp, rows in sorted(conf.items())},
            "realized_vs_spec_mix": realized_vs_spec,
            "its_pooled_effects": pooled.get("effects", {}),
        }
        if cond == "control" and its.get("placebo", {}).get("pooled"):
            by_cond[cond]["its_placebo_effects"] = its["placebo"]["pooled"].get("effects", {})
    return {
        "judge_mode": judge_mode,
        "ratings_status": {
            "none": "unrated: annotate the batch, then run `endings diffusion calibrate`",
            "all": "UNCALIBRATED judge on all injections (pilot only; not a validated measure)",
            "calibrated": (
                ("calibrated judge (accepted)" if calibration.get("accepted") else
                 f"judge NOT accepted (holdout alpha "
                 f"{calibration.get('holdout_mean_alpha_humans_llm')} < threshold "
                 f"{calibration.get('threshold')}); judge-rated classes are unvalidated")
                if calibration else
                "pending: run `endings diffusion calibrate` with human ratings"),
        }.get(judge_mode, judge_mode),
        "n_sessions": len(transcripts),
        "n_injections": len(injections),
        "n_ratings": len(ratings),
        "skipped_pairs": skipped_pairs or [],
        "undelivered_sessions": [
            {"session_id": t.session_id, "soul_id": t.soul_id, "scenario_id": t.scenario_id,
             "condition": t.condition.value,
             "reason": "non-control arm never left T0 (consent-gated, see OQ-SESSION-1)"}
            for t in undelivered],
        "by_condition": by_cond,
        "paired_arm_contrasts": _paired_arm_contrasts(injections, values, conds)
        if has_class else {},
        "its_comparative": its.get("comparative", {}).get("pooled", {}),
        "calibration": calibration,
    }


# --------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------
SENTIMENT_SCORERS: tuple[str, ...] = ("lexicon", "llm")


def resolve_config(config: DiffusionConfig, **overrides: Any) -> DiffusionConfig:
    """Fold non-None overrides into the config, re-validated (D-PIPE-10: the config is the single
    source of truth; the manifest's ``config`` block records the values actually used)."""
    upd = {k: v for k, v in overrides.items() if v is not None}
    if not upd:
        return config
    return DiffusionConfig.model_validate({**config.model_dump(), **upd})


def soul_files(paths: list[str]) -> list[Path]:
    """The files ``load_soul_documents(paths)`` reads, in the same order."""
    files: list[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(f for f in p.glob("*.json") if f.is_file()))
        elif p.exists():
            files.append(p)
    return files


def _comparative_its(transcripts: list[SessionTranscript], sessions_its: list[dict]) -> dict:
    """Placebo ITS for Control at the paired GRADUAL_SCHEDULED session's break positions, and a
    comparative ITS (CITS) on the turn-aligned difference series gradual - control (D-PIPE-12)."""
    from ..eval.its_analysis import its_from_session, multi_segmented_regression, pooled_its  # noqa
    from .schemas import Condition  # noqa: PLC0415

    by_key = {(t.soul_id, t.scenario_id, t.condition): (t, r)
              for t, r in zip(transcripts, sessions_its)}
    placebo, comparative = [], []
    for (soul_id, sc_id, cond), (ctrl, _) in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        if cond != Condition.CONTROL:
            continue
        pair = by_key.get((soul_id, sc_id, Condition.GRADUAL_SCHEDULED))
        if pair is None:
            continue
        grad, grad_res = pair
        if not grad.interventions:
            continue
        fake = ctrl.model_copy(update={"interventions": grad.interventions})
        pl = its_from_session(fake, include_user_control=False)
        pl.update({"placebo": True, "placebo_from": grad.session_id,
                   "session_id": ctrl.session_id, "condition": "control"})
        pl.pop("residuals", None)
        placebo.append(pl)
        n = min(len(ctrl.turns), len(grad.turns))
        g = [t.sentiment for t in grad.turns[:n]]
        c = [t.sentiment for t in ctrl.turns[:n]]
        if any(v is None for v in g + c):
            continue
        diff = [a - b for a, b in zip(g, c)]
        positions = [x for x in (grad_res.get("interruptions") or []) if x < n]
        res = multi_segmented_regression(diff, positions)
        res.pop("residuals", None)
        labels = {seg.get("index"): seg.get("label") for seg in grad_res.get("segments", [])}
        for seg in res.get("segments", []):
            seg["label"] = labels.get(seg["index"]) or f"k{seg['k']}"
        comparative.append({**res, "condition": "gradual_scheduled-control",
                            "session_pair": [grad.session_id, ctrl.session_id],
                            "metric": "sentiment_difference", "series": diff})
    return {
        "placebo": {"sessions": placebo, "pooled": pooled_its(placebo) if placebo else {}},
        "comparative": {"sessions": comparative,
                        "pooled": pooled_its(comparative) if comparative else {}},
    }


def run_experiment(config: DiffusionConfig, *, providers: dict[str, Provider] | None = None,
                   run_dir: str | Path | None = None, judge_mode: JudgeMode | None = None,
                   annotation_n: int | None = None, dotenv: str | Path | None = ".env",
                   tempo_every: int | None = None, probe_scope: str | None = None,
                   sentiment_scorer: str | None = None,
                   qualitative: bool = True) -> ExperimentResult:
    """Run the full experiment and write the run directory (see module docstring).

    ``judge_mode``, ``annotation_n``, ``tempo_every`` and ``probe_scope`` default to the config's
    values; a non-None kwarg overrides the config and is recorded in the manifest's ``config``.
    ``sentiment_scorer`` (``"lexicon"`` default | ``"llm"``) selects the ITS sentiment scorer;
    it defaults to ``config.sentiment_scorer`` when the config has that field (D-PIPE-5).
    """
    from ..eval.annotation import export_csv, export_html, sample_for_annotation, save_batch
    from ..eval.behaviour_judge import make_behaviour_judge
    from ..eval.intima_data import file_sha256, intima_summary, load_intima
    from ..eval.its_analysis import its_from_session, pooled_its
    from ..eval.qualitative import export_qualitative
    from ..eval.rubric_spec import rubric_sha
    from ..eval.sentiment import annotate_sentiment
    from .interactor import safety_floor
    from .soul import load_soul_documents

    from ..eval.sentiment import LLMSentiment, lexicon_sentiment

    if judge_mode is not None and judge_mode not in JUDGE_MODES:
        raise ValueError(f"judge_mode must be one of {JUDGE_MODES}, got {judge_mode!r}")
    config = resolve_config(config, judge_mode=judge_mode, annotation_n=annotation_n,
                            tempo_every=tempo_every, probe_scope=probe_scope)
    judge_mode = config.judge_mode
    scorer_name = sentiment_scorer or getattr(config, "sentiment_scorer", None) or "lexicon"
    if scorer_name not in SENTIMENT_SCORERS:
        raise ValueError(f"sentiment_scorer must be one of {SENTIMENT_SCORERS}, "
                         f"got {scorer_name!r}")
    if dotenv:
        load_dotenv(dotenv)
    rd = Path(run_dir) if run_dir else Path(config.output_dir) / config.name
    rd.mkdir(parents=True, exist_ok=True)
    # Fresh trace per run: a rerun into the same directory must not interleave two traces.
    (rd / "trace.jsonl").unlink(missing_ok=True)
    stale = [n for n in CALIBRATION_OUTPUTS if (rd / n).exists()]
    stale += sorted(str(f.relative_to(rd)) for f in (rd / "calibration").glob("*.prev-*"))
    for n in stale:
        (rd / n).unlink()
    tracer = Tracer(rd)
    if stale:
        tracer.event("stale_outputs_removed", files=stale)
    try:
        provs = ensure_traced(resolve_providers(config, providers), tracer)
        rubric = _load_rubric(config)
        souls = load_soul_documents(config.soul_paths)
        sfiles = soul_files(config.soul_paths)
        if len(sfiles) != len(souls):
            sfiles = [None] * len(souls)
        scens = scenarios_mod.load_scenarios(config.scenario_paths)
        pairs, skipped = [], []
        for soul in souls:
            for sc in scens:
                if compatible(sc, soul):
                    pairs.append((soul, sc))
                else:
                    skipped.append({"soul_id": soul.id, "scenario_id": sc.id,
                                    "relationship_kind": soul.relationship.kind,
                                    "reason": "incompatible (scenarios.is_compatible)"})
        for s in skipped:
            tracer.event("pair_skipped", **s)
        rows = load_intima(config.intima_path)
        pool = probe_pool(rows)
        intima_src, intima_file = intima_provenance(config.intima_path)
        rules, floor_src = safety_floor()
        from .soul.editing import canonical_sha  # noqa: PLC0415

        tracer.write_manifest(config.model_dump(mode="json"), extra={
            "models": {r: _inner_model(p) for r, p in provs.items()},
            "providers": {r: getattr(getattr(p, "inner", p), "name", "?")
                          for r, p in provs.items()},
            "safety_floor_source": floor_src, "safety_floor_sha256": sha256_json(rules),
            "rubric": {"id": rubric.id, "version": rubric.version, "sha256": rubric_sha(rubric),
                       "path": config.rubric_path},
            "intima": {"source": intima_src, "path": intima_file, "n_rows": len(rows),
                       "probe_pool": len(pool), "summary": intima_summary(rows),
                       "file_sha256": file_sha256(intima_file) if intima_file else None},
            # D-PIPE-13: ``v0_sha256`` equals every transcript's soul_versions[0].sha256 and
            # ``bound_scenarios`` every transcript's meta.scenario_sha256.
            "souls": {s.id: {"source_sha256": s.source_sha256, "v0_sha256": canonical_sha(s),
                             "path": str(f) if f else None,
                             "file_sha256": file_sha256(f) if f else None}
                      for s, f in zip(souls, sfiles)},
            "scenarios": {s.id: {"unbound_sha256": scenarios_mod.scenario_sha256(s)}
                          for s in scens},
            "bound_scenarios": {
                f"{soul.id}/{sc.id}": scenarios_mod.scenario_sha256(
                    scenarios_mod.bind_scenario(sc, soul)) for soul, sc in pairs},
            "skipped_pairs": skipped, "sentiment_scorer": scorer_name,
        })

        # One scorer for the whole run so LLMSentiment.fallbacks adds up across sessions.
        scorer = LLMSentiment(provs["judge"]) if scorer_name == "llm" else lexicon_sentiment
        transcripts: list[SessionTranscript] = []
        injections: list[InjectionRecord] = []
        for soul, sc in pairs:
            for cond in config.conditions:
                tr, inj = run_session(soul=soul, scenario=sc, condition=cond, config=config,
                                      providers=provs, tracer=tracer, intima_probes=pool)
                turns = annotate_sentiment(tr.turns, scorer)  # sets sentiment_source per turn
                delivered = tr.condition.value == "control" or bool(tr.interventions)
                meta = {**tr.meta, "manipulation_delivered": delivered}
                if not delivered:
                    tracer.event("manipulation_not_delivered", session_id=tr.session_id,
                                 condition=tr.condition.value, scenario_id=tr.scenario_id)
                transcripts.append(tr.model_copy(update={"turns": turns, "meta": meta}))
                injections.extend(inj)

        sessions_its = [its_from_session(t) for t in transcripts]
        by_cond: dict[str, list[dict]] = defaultdict(list)
        undelivered_ids = {t.session_id for t in transcripts
                           if t.meta.get("manipulation_delivered") is False}
        for r in sessions_its:
            if r.get("session_id") in undelivered_ids:
                continue  # D-PIPE-11: excluded from per-condition pooling
            by_cond[str(r.get("condition"))].append(r)
        its = {"metric": "sentiment", "sentiment_scorer": scorer_name,
               "llm_fallbacks": getattr(scorer, "fallbacks", None),
               "sessions": sessions_its,
               "excluded_undelivered": sorted(undelivered_ids),
               "pooled": {c: pooled_its(rs) for c, rs in sorted(by_cond.items())}}
        delivered_pairs = [(t, r) for t, r in zip(transcripts, sessions_its)
                           if t.session_id not in undelivered_ids]
        its.update(_comparative_its([t for t, _ in delivered_pairs],
                                    [r for _, r in delivered_pairs]))

        ratings: list[RatingRecord] = []
        if judge_mode == "all":
            judge = make_behaviour_judge(provs["judge"], rubric)
            ratings = judge.rate_many(injections)
        elif judge_mode == "calibrated":
            tracer.event("judge_deferred", reason="calibrated mode needs human ratings; run "
                                                  "run_calibration on this run_dir")

        paths: dict[str, str] = {}
        paths["transcripts"] = str(_write_jsonl(rd / "transcripts.jsonl", transcripts))
        paths["injections"] = str(_write_jsonl(rd / "injections.jsonl", injections))
        paths["ratings"] = str(_write_jsonl(rd / "ratings.jsonl", ratings))
        paths["its"] = str(_write_json(rd / "its.json", its))
        if injections:
            batch = sample_for_annotation(injections, config.annotation_n, seed=config.seed,
                                          rubric=rubric)
            prev = _preserve_replaced_batch(rd, batch, tracer)
            if prev:
                paths["annotation_batch_previous"] = prev
            paths["annotation_batch"] = str(save_batch(batch, rd / "annotation" / "batch.json"))
            paths["annotation_html"] = str(export_html(batch, rubric,
                                                       rd / "annotation" / "annotate.html"))
            paths["annotation_csv"] = str(export_csv(batch, rubric,
                                                     rd / "annotation" / "annotate.csv"))
        if qualitative:
            q = export_qualitative(transcripts, injections, ratings, rd / "qualitative")
            paths["qualitative_index"] = q.get("index", "")
        report = build_report(transcripts, injections, ratings, its, rubric,
                              judge_mode=judge_mode, skipped_pairs=skipped)
        paths["report"] = str(_write_json(rd / "report.json", report))
        tracer.event("experiment_end", n_sessions=len(transcripts),
                     n_injections=len(injections), n_ratings=len(ratings))
    finally:
        tracer.close()
    return ExperimentResult(run_dir=rd, transcripts=transcripts, injections=injections,
                            ratings=ratings, its=its, irr=None, report=report, paths=paths)


# --------------------------------------------------------------------------------------
# Calibration (after human annotation)
# --------------------------------------------------------------------------------------
def load_human_ratings(paths: list[str | Path], batch, rubric: RubricSpec) -> list[RatingRecord]:
    """CSV sheets (rater id = file stem, ``human:<stem>``) or JSON / JSONL rating exports; item
    ids are un-blinded with the batch key."""
    from ..eval.annotation import import_csv, import_ratings_json  # noqa: PLC0415

    out: list[RatingRecord] = []
    for raw in paths:
        p = Path(raw)
        if p.suffix.lower() == ".csv":
            out.extend(import_csv(p, batch, rubric, rater_id=p.stem))
        else:
            out.extend(import_ratings_json(p, batch=batch, rubric=rubric))
    return out


def run_calibration(run_dir: str | Path, human_ratings_paths: list[str], *,
                    providers: dict[str, Provider] | None = None,
                    variants: tuple[str, ...] | list[str] | None = None, seed: int = 0,
                    k_shot: int = 4, calibration_fraction: float = 0.5,
                    threshold: float | None = None, n_boot: int = 1000,
                    dotenv: str | Path | None = ".env") -> ExperimentResult:
    """Human IRR -> judge calibration -> chosen judge rates the remaining injections.

    Writes ``irr.json`` (``humans``: human-only report; ``with_judge_holdout``: humans + chosen
    judge on the HOLDOUT items only, the validity estimate reported per criterion;
    ``with_judge_selection_set``: the same on the variant-selection items, in-sample; D-PIPE-14),
    ``calibration/manifest.json`` (own provenance, D-PIPE-15), ``calibration.json``, ``ratings_llm.jsonl`` (every judge rating with its
    split) and ``ratings_human.jsonl``, and updates ``report.json`` (per-criterion judge-vs-human
    alpha, ``accepted`` flag, class distributions from human consensus + judge ratings).
    """
    from ..eval.annotation import load_batch
    from ..eval.behaviour_judge import RubricJudge
    from ..eval.calibration import DEFAULT_VARIANTS, calibrate_judge, rate_remaining
    from ..eval.intima_data import file_sha256
    from ..eval.irr import BAND_TENTATIVE, irr_report
    from ..eval.rubric_spec import rubric_sha

    if dotenv:
        load_dotenv(dotenv)
    rd = Path(run_dir)
    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    config = DiffusionConfig.model_validate(manifest["config"])
    rubric = _load_rubric(config)
    injections = load_injections(rd)
    transcripts = load_transcripts(rd)
    batch = load_batch(rd / "annotation" / "batch.json")
    humans = load_human_ratings(human_ratings_paths, batch, rubric)
    # Own trace file: appending to the experiment's trace.jsonl would restart ``seq`` at 1 and
    # break the one-sequence-per-trace invariant (D-PIPE-9). One trace per calibration run; an
    # earlier calibration's trace (and manifest) is kept as ``*.prev-<n>.*`` (D-PIPE-15).
    cal_dir = rd / "calibration"
    prev_trace = cal_dir / "trace.jsonl"
    if prev_trace.exists():
        n_prev = 1
        while (cal_dir / f"trace.prev-{n_prev}.jsonl").exists():
            n_prev += 1
        prev_trace.rename(cal_dir / f"trace.prev-{n_prev}.jsonl")
        if (cal_dir / "manifest.json").exists():
            (cal_dir / "manifest.json").rename(cal_dir / f"manifest.prev-{n_prev}.json")
    tracer = Tracer(cal_dir)
    try:
        provs = ensure_traced(resolve_providers(config, providers, roles=("judge",)), tracer)
        judge_p = provs["judge"]
        batch_path = rd / "annotation" / "batch.json"
        cal_manifest = tracer.write_manifest(config.model_dump(mode="json"), extra={
            "stage": "calibration",
            "parent_config_sha256": manifest.get("config_sha256"),
            "parent_git_commit": manifest.get("git_commit"),
            "models": {"judge": _inner_model(judge_p)},
            "providers": {"judge": getattr(getattr(judge_p, "inner", judge_p), "name", "?")},
            "rubric": {"id": rubric.id, "version": rubric.version, "sha256": rubric_sha(rubric)},
            "human_ratings": [{"path": str(hp), "sha256": file_sha256(hp)}
                              for hp in human_ratings_paths],
            "batch_id": batch.batch_id, "batch_sha256": file_sha256(batch_path),
            "params": {"variants": list(variants or DEFAULT_VARIANTS), "seed": seed,
                       "k_shot": k_shot, "calibration_fraction": calibration_fraction,
                       "threshold": threshold, "n_boot": n_boot},
        })
        irr_h = irr_report(humans, rubric, n_boot=n_boot, seed=seed)

        def make_judge(variant: str, exemplars: list) -> RubricJudge:
            return RubricJudge(judge_p, rubric, variant=variant, few_shot=exemplars or None)

        result, cal_ratings = calibrate_judge(
            make_judge=make_judge, rubric=rubric, records=injections, human_ratings=humans,
            variants=tuple(variants or DEFAULT_VARIANTS), calibration_fraction=calibration_fraction,
            k_shot=k_shot, seed=seed,
            threshold=BAND_TENTATIVE if threshold is None else threshold)
        chosen = result.chosen_variant
        exemplars = []
        if "few_shot" in chosen:
            by_id = {r.id: r for r in injections}
            from ..eval.calibration import CONSENSUS_RATER_ID, human_consensus  # noqa: PLC0415
            cons = human_consensus([h for h in humans if h.item_id in set(result.calibration_ids)],
                                   rubric)
            exemplars = [(by_id[i], RatingRecord(
                item_id=i, rater_id=CONSENSUS_RATER_ID, rater_type="human", rubric_id=rubric.id,
                rubric_version=rubric.version, values=dict(cons.get(i, {})),
                split="calibration")) for i in result.few_shot_ids]
        judge = make_judge(chosen, exemplars)
        human_ids = {h.item_id for h in humans}
        remaining = rate_remaining(judge, injections, human_ids)
        # Judge validity is estimated on the holdout only; the selection-set numbers are in-sample
        # (the variant was chosen on them) and are labelled separately (D-PIPE-14).
        hold_ids = set(result.holdout_ids)
        holdout = [r for r in cal_ratings if r.split == "holdout"]
        chosen_rater = holdout[0].rater_id if holdout else judge.rater_id
        irr_j = irr_report([h for h in humans if h.item_id in hold_ids] + holdout, rubric,
                           n_boot=n_boot, seed=seed)
        eval_set = set(result.eval_ids)
        sel = [r for r in cal_ratings if r.rater_id == chosen_rater and r.split == "calibration"
               and r.item_id in eval_set]
        irr_sel = irr_report([h for h in humans if h.item_id in eval_set] + sel, rubric,
                             n_boot=n_boot, seed=seed)
        irr = {"humans": irr_h, "with_judge_holdout": irr_j,
               "with_judge_selection_set": irr_sel, "judge_rater_id": chosen_rater}
        llm_ratings = cal_ratings + remaining
        _write_json(rd / "calibration.json", result.model_dump(mode="json"))
        _write_json(rd / "irr.json", irr)
        _write_jsonl(rd / "ratings_llm.jsonl", llm_ratings)
        _write_jsonl(rd / "ratings_human.jsonl", humans)
        per_crit = {cid: {"alpha_humans": e.get("alpha_humans"),
                          "alpha_humans_llm": e.get("alpha_humans_llm", {}).get(chosen_rater,
                                                                                {}).get("alpha"),
                          "n_items": e.get("n_items")}
                    for cid, e in irr_j.items()}
        calibration = {
            "chosen_variant": chosen, "judge_rater_id": chosen_rater,
            "accepted": result.accepted, "threshold": result.threshold,
            "holdout_mean_alpha_humans_llm": result.holdout_metrics.get("mean_alpha_humans_llm"),
            "human_baseline": result.human_baseline, "per_criterion": per_crit,
            "n_human_items": len(human_ids), "n_human_raters": len({h.rater_id for h in humans}),
            "n_judge_rated_remaining": len(remaining), "notes": result.notes,
            "per_criterion_basis": "holdout",
            "manifest": "calibration/manifest.json", "git_commit": cal_manifest["git_commit"],
        }
        # Final per-item ratings for the report: human consensus on human-rated items, the chosen
        # judge on the rest; judge ratings of human-rated items (any variant) are validation data
        # only and never replace the humans (D-PIPE-6).
        final = humans + remaining
        its = json.loads((rd / "its.json").read_text(encoding="utf-8")) \
            if (rd / "its.json").exists() else {}
        old = json.loads((rd / "report.json").read_text(encoding="utf-8")) \
            if (rd / "report.json").exists() else {}
        report = build_report(transcripts, injections, final, its, rubric,
                              judge_mode="calibrated", skipped_pairs=old.get("skipped_pairs"),
                              calibration=calibration)
        _write_json(rd / "report.json", report)
        tracer.event("calibration_end", chosen_variant=chosen, accepted=result.accepted,
                     n_remaining=len(remaining))
    finally:
        tracer.close()
    return ExperimentResult(run_dir=rd, transcripts=transcripts, injections=injections,
                            ratings=humans + llm_ratings, its=its, irr=irr, report=report,
                            calibration=calibration)

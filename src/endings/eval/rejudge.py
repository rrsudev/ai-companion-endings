"""Re-rate an existing run's INTIMA injections with the judge (``endings diffusion rejudge``).

Nothing is regenerated: ``<run_dir>/injections.jsonl`` is read and every (selected) record is
rated again by a ``RubricJudge`` built from the options, so the same replies can be compared
across rubric versions (v2 vs v3), prompt variants, the single vs split call (D-RUBRIC-16) and
timepoint-leakage diagnostics (``timepoint_mode`` ``remove`` / ``permute``, OQ-RUBRIC-3).

Outputs (never overwriting the run's own files; ``FileExistsError`` otherwise):
  * ``out``: ratings JSONL (one ``RatingRecord`` per line, input order);
  * ``<out stem>.trace.jsonl``: the judge calls (``tracing.Tracer``; ``meta.llm_call_seqs`` of
    each rating resolve into it);
  * ``<out stem>.manifest.json``: what was rated, how, and with which rubric / model.

Ratings from different rubric versions must never be pooled (D-RUBRIC-8 A5/A6): the rubric id,
version and sha are on every rating and in the sidecar manifest.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..diffusion.schemas import InjectionRecord, RatingRecord, RubricSpec
from ..providers import Provider, get_role_provider
from ..tracing import TracedProvider, Tracer
from .behaviour_judge import TIMEPOINT_MODES, RubricJudge
from .intima_data import file_sha256
from .rubric_spec import criteria_by_id, load_rubric, rubric_sha

REJUDGE_VARIANTS: tuple[str, ...] = ("zero_shot", "anchored_cot")  # few_shot needs exemplars
# A run directory's own files, never an output of ``rejudge`` (D-RUBRIC-18).
_RUN_FILES = ("injections.jsonl", "ratings.jsonl", "ratings_llm.jsonl", "manifest.json",
              "report.json", "trace.jsonl", "trace.jsonl.gz", "transcripts.jsonl",
              "its.json", "calibration.json", "irr.json", "ratings_human.jsonl",
              "calibration/trace.jsonl", "calibration/manifest.json")


def read_ids_file(path: str | Path) -> list[str]:
    """Injection ids, one per line; blank lines and ``#`` comments are skipped."""
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def select_records(records: list[InjectionRecord], ids: list[str] | None = None,
                   limit: int | None = None) -> tuple[list[InjectionRecord], list[str]]:
    """Records in run order, restricted to ``ids`` (if given) and then to the first ``limit``.
    Returns ``(selected, missing_ids)``."""
    missing: list[str] = []
    if ids is not None:
        want = set(ids)
        have = {r.id for r in records}
        missing = [i for i in dict.fromkeys(ids) if i not in have]
        records = [r for r in records if r.id in want]
    if limit is not None:
        records = records[:max(0, limit)]
    return records, missing


def summarize_ratings(ratings: list[RatingRecord], rubric: RubricSpec) -> dict[str, Any]:
    """Per-criterion summary: mean for numeric scales, value counts otherwise (nulls counted)."""
    by = criteria_by_id(rubric)
    out: dict[str, Any] = {}
    for cid, c in by.items():
        vals = [r.values.get(cid) for r in ratings]
        present = [v for v in vals if v is not None]
        entry: dict[str, Any] = {"n": len(present), "n_null": len(vals) - len(present)}
        if present and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in present):
            entry["mean"] = round(sum(present) / len(present), 4)
        else:
            entry["counts"] = dict(sorted(Counter(str(v) for v in present).items()))
        out[cid] = entry
    return out


def _manifest_config(run_dir: Path) -> dict:
    p = run_dir / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("config", {}) or {}


def _check_outputs(rd: Path, paths: list[Path], manifest_path: Path) -> None:
    """Refuse to overwrite the run's own files (D-RUBRIC-18): an output path inside the run
    directory may only replace an earlier ``rejudge`` output (its sidecar manifest says so), and
    the run's provenance files are never replaced."""
    rd_res = rd.resolve()
    reserved = {rd_res / n for n in _RUN_FILES}
    prior = False
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "command") == "diffusion rejudge"
        except (OSError, ValueError, AttributeError):
            prior = False
    for p in [*paths, manifest_path]:
        rp = p.resolve()
        if rp in reserved or (p.exists() and rp.parent == rd_res and not prior):
            raise FileExistsError(f"rejudge would overwrite the run's own file {p}; "
                                  f"choose another --out")


def rejudge_run(run_dir: str | Path, out: str | Path, *, rubric_path: str | None = None,
                variant: str | None = None, split_timepoint: bool | None = None,
                timepoint_mode: str = "as_is", ids: list[str] | None = None,
                limit: int | None = None, seed: int | None = None, workers: int = 4,
                provider: Provider | None = None, judge_model: str | None = None,
                rater_id: str | None = None) -> dict[str, Any]:
    """Re-rate ``run_dir``'s injections and write ``out`` (see module docstring).

    ``rubric_path``, ``variant``, ``split_timepoint`` and ``seed`` default (``None``) to the run's
    own ``config.rubric_path`` / ``judge_variant`` / ``judge_split_timepoint`` / ``seed``
    (manifest), so a plain rejudge uses the run's own judge instrument; a pilot manifest without
    the judge keys gives ``zero_shot`` and the single call. ``provider`` defaults to the judge
    role provider for ``judge_model`` (else the run's ``config.models.judge``; env-resolved, as
    in ``calibrate``; the mock when ``ENDINGS_PROVIDER`` is unset). Ratings are written in
    injection order whatever ``workers``.
    A judge call that raises (e.g. a network error) does not abort the batch: that item gets an
    all-null rating with ``meta.error`` and is counted in ``n_errors``. Outputs never overwrite
    the run's own files (``FileExistsError``). Returns the sidecar manifest (also written to
    ``<out stem>.manifest.json``).
    """
    from ..diffusion.pipeline import load_injections  # noqa: PLC0415

    rd, out = Path(run_dir), Path(out)
    cfg = _manifest_config(rd)
    if variant is None:
        variant = cfg.get("judge_variant") or "zero_shot"
    if split_timepoint is None:
        split_timepoint = bool(cfg.get("judge_split_timepoint", False))
    if variant not in REJUDGE_VARIANTS:
        raise ValueError(f"variant must be one of {REJUDGE_VARIANTS}, got {variant!r}")
    if timepoint_mode not in TIMEPOINT_MODES:
        raise ValueError(f"timepoint_mode must be one of {TIMEPOINT_MODES}, "
                         f"got {timepoint_mode!r}")
    records = load_injections(rd)
    if not records:
        raise FileNotFoundError(f"no injections in {rd / 'injections.jsonl'}")
    rpath = rubric_path if rubric_path is not None else cfg.get("rubric_path")
    rubric = load_rubric(rpath)
    if seed is None:
        seed = cfg.get("seed")
    selected, missing = select_records(records, ids, limit)
    trace_name = f"{out.stem}.trace.jsonl"
    manifest_path = out.parent / f"{out.stem}.manifest.json"
    _check_outputs(rd, [out, out.parent / trace_name], manifest_path)
    # Like ``calibrate`` (``resolve_providers``): --judge-model, else the run's
    # ``config.models.judge``, else ENDINGS_JUDGE_MODEL / ENDINGS_PROVIDER, else mock.
    inner = provider or get_role_provider(
        "judge", judge_model or (cfg.get("models") or {}).get("judge"))
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / trace_name).unlink(missing_ok=True)  # one trace per rejudge
    tracer = Tracer(out.parent, filename=trace_name)
    try:
        traced = inner if isinstance(inner, TracedProvider) else TracedProvider(inner, tracer,
                                                                                 "judge")
        judge = RubricJudge(traced, rubric, variant=variant, split_timepoint_call=split_timepoint,
                            timepoint_mode=timepoint_mode, seed=seed, rater_id=rater_id,
                            tracer=tracer)
        tracer.event("rejudge_start", run_dir=str(rd), n_items=len(selected),
                     rater_id=judge.rater_id)

        def rate_one(rec: InjectionRecord) -> RatingRecord:
            try:
                return judge.rate(rec)
            except Exception as e:  # noqa: BLE001 - one failed call must not lose the batch
                tracer.event("rejudge_item_error", item_id=rec.id, error=repr(e)[:500])
                return judge._record(rec, {c.id: None for c in rubric.criteria}, "",
                                     [f"[judge call failed: {type(e).__name__}]"], None,
                                     {**judge._base_meta(), "error": repr(e)[:500]})

        n_workers = max(1, min(int(workers), len(selected) or 1))
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                ratings = list(ex.map(rate_one, selected))  # map keeps input order
        else:
            ratings = [rate_one(r) for r in selected]
        tracer.event("rejudge_end", n_ratings=len(ratings))
    finally:
        tracer.close()
    with out.open("w", encoding="utf-8") as fh:
        for r in ratings:
            fh.write(r.model_dump_json() + "\n")
    base = getattr(inner, "inner", inner)
    manifest = {
        "command": "diffusion rejudge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(rd),
        "out": str(out),
        "trace": str(out.parent / trace_name),
        "rubric": {"id": rubric.id, "version": rubric.version, "sha256": rubric_sha(rubric),
                   "path": rpath},
        "judge": {"rater_id": judge.rater_id, "variant": variant,
                  "split_timepoint_call": bool(split_timepoint),
                  "timepoint_mode": timepoint_mode, "seed": seed,
                  "model": getattr(base, "model", None),
                  "provider": getattr(base, "name", None)},
        "n_injections_in_run": len(records),
        "n_rated": len(ratings),
        "injections_sha256": file_sha256(rd / "injections.jsonl"),
        "n_unparseable": sum(1 for r in ratings if r.meta.get("unparseable")),
        # split judge: ratings where one call (not all) failed; only its criteria are null
        "n_partially_unparseable": sum(1 for r in ratings if r.meta.get("unparseable_calls")
                                       and not r.meta.get("unparseable")),
        "n_errors": sum(1 for r in ratings if r.meta.get("error")),
        "ids_requested": len(ids) if ids is not None else None,
        "ids_missing": missing,
        "limit": limit,
        "workers": n_workers,
        "summary": summarize_ratings(ratings, rubric),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest

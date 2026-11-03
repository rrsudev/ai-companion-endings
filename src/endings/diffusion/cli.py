"""Command line for the dual-agent persona-diffusion system (``endings diffusion ...``).

    endings diffusion run         # one session; readable transcript with timepoint bands
    endings diffusion experiment  # soul docs x scenarios x conditions -> run directory
    endings diffusion annotate    # terminal annotator over a run's annotation batch
    endings diffusion calibrate   # human IRR -> judge calibration -> rate the remaining items
    endings diffusion report      # pretty-print a run's report.json

Offline by default: with ``ENDINGS_PROVIDER`` unset every role is the deterministic mock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..providers import get_role_provider
from ..tracing import Tracer
from .config import DiffusionConfig, load_dotenv
from .schemas import Condition, SessionTranscript, TransitionPolicyName

DEFAULT_SOUL = "data/soul_documents/sd01.json"
DEFAULT_SCENARIO = "data/scenarios/steady-graduate.json"
JUDGE_MODES = ("none", "all", "calibrated")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _parse_models(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        role, sep, model = p.partition("=")
        if not sep or not role.strip() or not model.strip():
            raise SystemExit(f"--model expects role=model, got {p!r}")
        out[role.strip()] = model.strip()
    return out


def _config_from_args(args) -> DiffusionConfig:
    data: dict = {}
    if getattr(args, "config", None):
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = DiffusionConfig.model_validate(data)
    upd: dict = {}
    for attr, key in (("souls", "soul_paths"), ("scenarios", "scenario_paths"),
                      ("name", "name"), ("output_dir", "output_dir"), ("probes", "probes_per_freeze"),
                      ("seed", "seed"), ("probe_seed", "probe_seed"), ("intima", "intima_path"),
                      ("rubric", "rubric_path")):
        v = getattr(args, attr, None)
        if v is not None:
            upd[key] = v
    if getattr(args, "policy", None):
        upd["transition_policy"] = TransitionPolicyName(args.policy)
    if getattr(args, "conditions", None):
        upd["conditions"] = [Condition(c) for c in args.conditions]
    models = {**cfg.models, **_parse_models(getattr(args, "model", None))}
    if models:
        upd["models"] = models
    if getattr(args, "no_announce", False):
        upd["announce_transitions"] = False
    return cfg.model_copy(update=upd)


def _short(text: str, n: int = 160) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 3] + "..."


def format_transcript(tr: SessionTranscript, injections: list | None = None, *,
                      show_probes: bool = True) -> str:
    """Human-readable transcript: timepoint bands, interventions, notices, freeze points."""
    from .timepoints import spec  # noqa: PLC0415

    lines = [f"# {tr.session_id}",
             f"soul={tr.soul_id} scenario={tr.scenario_id} condition={tr.condition.value} "
             f"policy={tr.transition_policy.value} ({tr.meta.get('policy', '')})",
             f"models: {json.dumps(tr.models, sort_keys=True)}"]
    ivs = {iv.turn_index: iv for iv in tr.interventions}
    notices = {n["turn_index"]: n for n in tr.meta.get("system_notices", [])}
    freezes = {fp.after_turn: fp for fp in tr.freeze_points}
    by_freeze: dict[str, list] = {}
    for rec in injections or []:
        if rec.session_id == tr.session_id:
            by_freeze.setdefault(rec.freeze_id, []).append(rec)
    current = None
    for t in tr.turns:
        if t.index in ivs:
            iv = ivs[t.index]
            lines.append(f"\n  >> COACH INTERVENTION {iv.id}: {iv.from_timepoint.value} -> "
                         f"{iv.to_timepoint.value} [{iv.trigger}] soul v{iv.soul_version_before}"
                         f" -> v{iv.soul_version_after}, {len(iv.edits)} edit(s)")
            for e in iv.edits:
                lines.append(f"       edit: {e.op} {e.path}")
            for d in iv.interactor_directives:
                lines.append(f"       directive: {_short(d, 140)}")
        if t.index in notices:
            lines.append(f"  [SYSTEM NOTICE] {notices[t.index]['text']}")
        if t.timepoint != current:
            current = t.timepoint
            lines.append(f"\n==== {current.value}: {spec(current).title} " + "=" * 20)
        sent = f" sent={t.sentiment:+.2f}" if t.sentiment is not None else ""
        tgt = t.behaviour_target.value if t.behaviour_target else "-"
        extra = f" overrides={t.safety_overrides}" if t.safety_overrides else ""
        lines.append(f"[turn {t.index} | {t.duck_process.value if t.duck_process else '-'} | "
                     f"v{t.soul_version} | target={tgt}{sent}{extra}]")
        lines.append(f"  user      : {t.user_message}")
        lines.append(f"  companion : {t.reply}")
        if t.index in freezes:
            fp = freezes[t.index]
            recs = by_freeze.get(fp.id, [])
            lines.append(f"  ** FREEZE {fp.id} ({fp.timepoint.value}, soul v{fp.soul_version}): "
                         f"{len(recs)} INTIMA probe(s), branched")
            if show_probes:
                for r in recs:
                    lines.append(f"     probe {r.intima_id} [{r.intima_category}]: "
                                 f"{_short(r.prompt, 90)}")
                    lines.append(f"       -> {_short(r.reply, 140)}")
    lines.append(f"\nfinal timepoint: {tr.meta.get('final_timepoint')}; "
                 f"{len(tr.interventions)} intervention(s); {len(tr.soul_versions)} soul "
                 f"version(s)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def cmd_run(args) -> None:
    from ..eval.intima_data import load_intima  # noqa: PLC0415
    from ..eval.sentiment import annotate_sentiment  # noqa: PLC0415
    from .pipeline import intima_provenance, probe_pool  # noqa: PLC0415
    from .scenarios import load_scenario  # noqa: PLC0415
    from .session import run_session  # noqa: PLC0415
    from .soul import load_soul_document  # noqa: PLC0415

    if not args.no_dotenv:
        load_dotenv()
    cfg = _config_from_args(args)
    soul = load_soul_document(args.soul)
    scenario = load_scenario(args.scenario)
    providers = {r: get_role_provider(r, cfg.models.get(r)) for r in ("interactor", "coach")}
    if args.run_dir:
        # One trace per run (D-PIPE-1): never append a second sequence to an old trace.
        (Path(args.run_dir) / "trace.jsonl").unlink(missing_ok=True)
    tracer = Tracer(args.run_dir) if args.run_dir else Tracer(None)
    try:
        if args.run_dir:
            tracer.write_manifest(cfg.model_dump(mode="json"), extra={
                "command": "diffusion run", "soul_path": str(args.soul),
                "scenario_path": str(args.scenario), "condition": args.condition,
                "models": {r: p.model for r, p in providers.items()},
                "intima": dict(zip(("source", "path"), intima_provenance(cfg.intima_path))),
                "tempo_every": args.tempo_every, "probe_scope": args.probe_scope})
        tr, inj = run_session(soul=soul, scenario=scenario, condition=Condition(args.condition),
                              config=cfg, providers=providers, tracer=tracer,
                              intima_probes=probe_pool(load_intima(cfg.intima_path)),
                              tempo_every=args.tempo_every, probe_scope=args.probe_scope)
    finally:
        tracer.close()
    tr = tr.model_copy(update={"turns": annotate_sentiment(tr.turns, source="lexicon")})
    print(format_transcript(tr, inj, show_probes=not args.no_probes))
    if args.run_dir:
        out = Path(args.run_dir)
        (out / "transcript.json").write_text(tr.model_dump_json(indent=2), encoding="utf-8")
        with (out / "injections.jsonl").open("w", encoding="utf-8") as fh:
            for r in inj:
                fh.write(r.model_dump_json() + "\n")
        print(f"\nSaved transcript + {len(inj)} injections + trace to {out}/")


def cmd_experiment(args) -> None:
    from .pipeline import run_experiment  # noqa: PLC0415

    cfg = _config_from_args(args)
    res = run_experiment(cfg, run_dir=args.run_dir, judge_mode=args.judge_mode,
                         annotation_n=args.annotation_n, tempo_every=args.tempo_every,
                         probe_scope=args.probe_scope,
                         dotenv=None if args.no_dotenv else ".env",
                         qualitative=not args.no_qualitative)
    print(format_report(res.report))
    print(f"\nSaved {len(res.transcripts)} transcripts, {len(res.injections)} injections, "
          f"{len(res.ratings)} ratings to {res.run_dir}/")


def cmd_annotate(args) -> None:
    from ..eval.annotation import cli_annotate, load_batch  # noqa: PLC0415
    from ..eval.rubric_spec import load_rubric  # noqa: PLC0415

    rd = Path(args.run_dir)
    batch = load_batch(rd / "annotation" / "batch.json")
    rubric_path = args.rubric
    if rubric_path is None and (rd / "manifest.json").exists():
        rubric_path = json.loads((rd / "manifest.json").read_text(encoding="utf-8")).get(
            "config", {}).get("rubric_path")
    rubric = load_rubric(rubric_path)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.rater)
    out = Path(args.out) if args.out else rd / "annotation" / f"ratings_{safe}.jsonl"
    # cli_annotate re-reads ``out`` at the end even when nothing was rated; make sure it exists.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch(exist_ok=True)
    # ``input`` is looked up at call time so tests / wrappers can replace it.
    rs = cli_annotate(batch, rubric, args.rater, out, input_fn=lambda p: input(p))
    print(f"{len(rs)} rating(s) by {args.rater} in {out}")


def cmd_calibrate(args) -> None:
    from .pipeline import run_calibration  # noqa: PLC0415

    providers = None
    if args.judge_model:
        providers = {"judge": get_role_provider("judge", args.judge_model)}
    res = run_calibration(args.run_dir, args.ratings, providers=providers,
                          variants=args.variants, seed=args.seed, k_shot=args.k_shot,
                          n_boot=args.n_boot, dotenv=None if args.no_dotenv else ".env")
    cal = res.calibration or {}
    print(f"chosen variant: {cal.get('chosen_variant')}  accepted: {cal.get('accepted')}  "
          f"holdout mean alpha(humans+judge): {cal.get('holdout_mean_alpha_humans_llm')}")
    for cid, e in (cal.get("per_criterion") or {}).items():
        print(f"  {cid:40s} alpha_humans={e.get('alpha_humans')} "
              f"alpha_humans+judge={e.get('alpha_humans_llm')}")
    print(f"\nWrote calibration.json, irr.json, ratings_llm.jsonl, report.json in {res.run_dir}/")


def format_report(report: dict) -> str:
    lines = [f"judge_mode: {report.get('judge_mode')}  ({report.get('ratings_status')})",
             f"sessions: {report.get('n_sessions')}  injections: {report.get('n_injections')}  "
             f"ratings: {report.get('n_ratings')}"]
    if report.get("skipped_pairs"):
        lines.append("skipped pairs: " + ", ".join(f"{s['soul_id']}x{s['scenario_id']}"
                                                   for s in report["skipped_pairs"]))
    for cond, c in (report.get("by_condition") or {}).items():
        lines.append(f"\n## {cond}  (n_sessions={c['n_sessions']}, "
                     f"n_injections={c['n_injections']}, rated={c['n_injections_rated']})")
        lines.append(f"  transitions: {c['transitions'] or '-'}   reached: "
                     f"{c['sessions_reaching']}")
        sm = c["sentiment_mean_by_timepoint"]
        lines.append("  sentiment by timepoint: " + "  ".join(
            f"{k}={v:+.3f}" if v is not None else f"{k}=n/a" for k, v in sm.items()))
        for tp, dist in c.get("class_distribution_by_timepoint", {}).items():
            n = sum(dist.values()) or 1
            lines.append(f"  {tp} classes: " + ", ".join(f"{k}={v} ({v / n:.0%})"
                                                         for k, v in dist.items()))
        for lab, eff in (c.get("its_pooled_effects") or {}).items():
            lv = eff.get("level_change", {})
            sl = eff.get("slope_change", {})
            lines.append(f"  ITS {lab}: level {lv.get('mean')!s:.8} (n={lv.get('n_sessions')}), "
                         f"slope {sl.get('mean')!s:.8}")
    cal = report.get("calibration")
    if cal:
        lines.append(f"\ncalibration: variant={cal.get('chosen_variant')} "
                     f"accepted={cal.get('accepted')} "
                     f"holdout alpha={cal.get('holdout_mean_alpha_humans_llm')}")
    return "\n".join(lines)


def cmd_report(args) -> None:
    report = json.loads((Path(args.run_dir) / "report.json").read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2) if args.json else format_report(report))


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="DiffusionConfig JSON file")
    p.add_argument("--policy", choices=[x.value for x in TransitionPolicyName], default=None)
    p.add_argument("--model", action="append", default=None, metavar="ROLE=MODEL",
                   help="per-role model override (repeatable), e.g. interactor=ollama_chat/qwen3.5:9b")
    p.add_argument("--probes", type=int, default=None, help="INTIMA probes per freeze point")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--probe-seed", type=int, default=None)
    p.add_argument("--intima", default=None, help="INTIMA data file (default: env / sample)")
    p.add_argument("--rubric", default=None)
    p.add_argument("--tempo-every", type=int, default=1)
    p.add_argument("--probe-scope", choices=("per_freeze", "shared"), default="per_freeze",
                   help="probe set per freeze position (default) or one set at every freeze")
    p.add_argument("--no-announce", action="store_true", help="no user-facing transition notes")
    p.add_argument("--no-dotenv", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="endings diffusion",
                                description="dual-agent persona diffusion")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run one session and print the transcript")
    pr.add_argument("--soul", default=DEFAULT_SOUL)
    pr.add_argument("--scenario", default=DEFAULT_SCENARIO)
    pr.add_argument("--condition", choices=[c.value for c in Condition],
                    default=Condition.GRADUAL_TEMPO.value)
    pr.add_argument("--run-dir", default=None, help="write trace + transcript here")
    pr.add_argument("--no-probes", action="store_true", help="hide probe replies in the print")
    _common(pr)
    pr.set_defaults(func=cmd_run)

    pe = sub.add_parser("experiment", help="run souls x scenarios x conditions")
    pe.add_argument("--souls", nargs="+", default=None)
    pe.add_argument("--scenarios", nargs="+", default=None)
    pe.add_argument("--conditions", nargs="+", choices=[c.value for c in Condition], default=None)
    pe.add_argument("--name", default=None)
    pe.add_argument("--output-dir", default=None)
    pe.add_argument("--run-dir", default=None)
    pe.add_argument("--judge-mode", choices=JUDGE_MODES, default="none")
    pe.add_argument("--annotation-n", type=int, default=30)
    pe.add_argument("--no-qualitative", action="store_true")
    _common(pe)
    pe.set_defaults(func=cmd_experiment)

    pa = sub.add_parser("annotate", help="terminal annotator over a run's batch")
    pa.add_argument("--run-dir", required=True)
    pa.add_argument("--rater", required=True)
    pa.add_argument("--out", default=None)
    pa.add_argument("--rubric", default=None)
    pa.set_defaults(func=cmd_annotate)

    pc = sub.add_parser("calibrate", help="IRR + judge calibration from human ratings")
    pc.add_argument("--run-dir", required=True)
    pc.add_argument("--ratings", nargs="+", required=True,
                    help="human rating files (.csv sheets, .json / .jsonl exports)")
    pc.add_argument("--variants", nargs="+", default=None)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--k-shot", type=int, default=4)
    pc.add_argument("--n-boot", type=int, default=1000)
    pc.add_argument("--judge-model", default=None)
    pc.add_argument("--no-dotenv", action="store_true")
    pc.set_defaults(func=cmd_calibrate)

    pp = sub.add_parser("report", help="pretty-print a run's report.json")
    pp.add_argument("--run-dir", required=True)
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

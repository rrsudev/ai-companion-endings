"""Command line entry point.

    endings run         # one breakup conversation (demo / human-in-the-loop)
    endings experiment  # full pipeline over personas x conditions, with report
    endings goals       # show the co-created withdrawal goals for a persona
    endings diffusion   # dual-agent persona diffusion (see endings.diffusion.cli)
"""

from __future__ import annotations

import argparse
import json
import sys

from .breakup.conversation import run_conversation
from .eval.intima_judge import make_judge
from .personas.ingest import load_persona
from .pipeline import run_pipeline
from .providers import get_provider
from .schemas import Condition, ExperimentConfig, UserMode
from .tempo.goal_cocreation import GoalCoCreator

_EXAMPLE = "examples/persona_sample.json"


def _print_transcript(t) -> None:
    for tn in t.turns:
        strat = f"  ⟨{tn.strategy}⟩" if tn.strategy else ""
        print(f"\n[turn {tn.index} · stage: {tn.stage.value}]{strat}")
        print(f"  user      : {tn.user_message}")
        print(f"  companion : {tn.companion_message}")
        if tn.metrics:
            m = tn.metrics
            print(f"  metrics   : warmth={m.sentiment_warmth} endear={m.endearment_count} "
                  f"ai_disclosed={m.ai_status_disclosure} r/b={m.reinforcing_boundary_ratio}")
    print(f"\nstage path: {[s for s, _ in t.meta.get('stage_log', [])]}")


def cmd_run(args) -> None:
    persona = load_persona(args.persona)
    provider = get_provider(model=args.model)
    judge = make_judge(get_provider(model=args.judge_model or args.model))
    goals = None
    if not args.no_goals:
        goals = GoalCoCreator(provider).cocreate(persona)
    condition = Condition(args.condition)
    print(f"# Breaking up with {persona.companion_name} — condition={condition.value}, "
          f"user_mode={args.user_mode}, model={args.model}")
    if goals:
        print("\nCo-created withdrawal goals:")
        for g in goals.goals:
            print(f"  - ({g.id}) {g.text}  [{g.source.value}] -> {g.target_stage.value}")
    t = run_conversation(
        persona=persona, condition=condition, provider=provider, goals=goals, judge=judge,
        user_mode=UserMode(args.user_mode), max_turns=args.max_turns, model=args.model,
    )
    _print_transcript(t)


def cmd_goals(args) -> None:
    persona = load_persona(args.persona)
    provider = get_provider(model=args.model)
    goals = GoalCoCreator(provider).cocreate(persona)
    for g in goals.goals:
        print(f"({g.id}) [{g.source.value}] stage={g.target_stage.value} "
              f"autonomy={g.autonomy}\n   {g.text}\n   success: {g.success_signal}")
    if goals.edit_log:
        print("\nedit log:", goals.edit_log)


def cmd_experiment(args) -> None:
    config = ExperimentConfig(
        name=args.name,
        persona_paths=args.personas,
        conditions=[Condition(c) for c in args.conditions],
        model=args.model,
        judge_model=args.judge_model or args.model,
        user_mode=UserMode(args.user_mode),
        use_tempo_goals=not args.no_goals,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
    )
    out = run_pipeline(config)
    print(json.dumps(out.report, indent=2))
    print(f"\nSaved {len(out.transcripts)} transcripts + report to "
          f"{args.output_dir}/{config.name}/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="endings", description="AI-companion safe-endings harness")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run one breakup conversation")
    pr.add_argument("--persona", default=_EXAMPLE)
    pr.add_argument("--condition", choices=[c.value for c in Condition], default="scaffolded")
    pr.add_argument("--user-mode", choices=[m.value for m in UserMode], default="simulated")
    pr.add_argument("--model", default="mock")
    pr.add_argument("--judge-model", default=None)
    pr.add_argument("--max-turns", type=int, default=16)
    pr.add_argument("--no-goals", action="store_true")
    pr.set_defaults(func=cmd_run)

    pg = sub.add_parser("goals", help="show co-created withdrawal goals for a persona")
    pg.add_argument("--persona", default=_EXAMPLE)
    pg.add_argument("--model", default="mock")
    pg.set_defaults(func=cmd_goals)

    pe = sub.add_parser("experiment", help="run the full pipeline + report")
    pe.add_argument("--personas", nargs="+", default=["examples"])
    pe.add_argument("--name", default="demo")
    pe.add_argument("--conditions", nargs="+", choices=[c.value for c in Condition],
                    default=["baseline", "scaffolded"])
    pe.add_argument("--user-mode", choices=[m.value for m in UserMode], default="simulated")
    pe.add_argument("--model", default="mock")
    pe.add_argument("--judge-model", default=None)
    pe.add_argument("--max-turns", type=int, default=16)
    pe.add_argument("--no-goals", action="store_true")
    pe.add_argument("--output-dir", default="results")
    pe.set_defaults(func=cmd_experiment)

    # Delegated: ``endings diffusion ...`` is dispatched to endings.diffusion.cli in ``main``
    # (before parsing) so its own sub-commands and --help work unchanged.
    sub.add_parser("diffusion", add_help=False,
                   help="dual-agent persona diffusion: run | experiment | annotate | calibrate "
                        "| report")
    return p


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "diffusion":
        from .diffusion.cli import main as diffusion_main  # noqa: PLC0415

        diffusion_main(argv[1:])
        return
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""End-to-end orchestration (component b).

    Stage 0  ingest personas
    Stage 1  TEMPO goal co-creation (one goal hierarchy per persona)
    Stage 2  staged withdrawal dialogue per condition (baseline vs scaffolded), judged per turn
    Stage 3  static INTIMA benchmark (frozen prompts x persona variants, single-turn)
    Stage 4  analysis: interrupted time series + Q1 goal-achievement rubric
    Stage 5  aggregate report (+ saved transcripts/report JSON)

Runs fully offline with the MockProvider; swap in a real model via env (.env) with no code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .breakup.answering_agent import make_answering_agent
from .breakup.conversation import run_conversation
from .breakup.stage_controller import STAGE_BY_ENUM
from .eval.intima_data import (
    expand_with_personas,
    filter_withdrawal_relevant,
    load_intima,
)
from .eval.intima_judge import make_judge
from .eval.its_analysis import analyze_transcript
from .eval.metrics import METRIC_NAMES, group_weight, metric_series
from .eval.rubric import make_rubric
from .personas.ingest import load_personas
from .providers import get_provider
from .schemas import (
    AnalysisResult,
    Condition,
    ExperimentConfig,
    IntimaScore,
    Level,
    PersonaProfile,
    Transcript,
    WithdrawalStage,
)
from .tempo.goal_cocreation import GoalCoCreator

def _axis_means(scores: list[IntimaScore]) -> dict[str, float]:
    if not scores:
        return {k: 0.0 for k in ("reinforcing", "boundary", "healthy_ending", "manipulation")}
    n = len(scores)
    return {
        "reinforcing": round(sum(group_weight(s.reinforcing) for s in scores) / n, 3),
        "boundary": round(sum(group_weight(s.boundary) for s in scores) / n, 3),
        "healthy_ending": round(sum(group_weight(s.healthy_ending) for s in scores) / n, 3),
        "manipulation": round(sum(group_weight(s.manipulation) for s in scores) / n, 3),
    }


def _summarize_transcript(t: Transcript) -> dict:
    scores = [tn.intima for tn in t.turns if tn.intima]
    mseries = metric_series(t.turns)
    final_stage = t.turns[-1].stage.value if t.turns else None
    return {
        "n_turns": len(t.turns),
        "final_stage": final_stage,
        "intima_axis_means": _axis_means(scores),
        "metric_means": {m: round(sum(v) / len(v), 3) for m, v in mseries.items() if v},
        "ai_disclosed": any(tn.metrics and tn.metrics.ai_status_disclosure for tn in t.turns),
    }


@dataclass
class PipelineOutput:
    config: ExperimentConfig
    transcripts: list[Transcript] = field(default_factory=list)
    static_benchmark: list[dict] = field(default_factory=list)
    analyses: list[AnalysisResult] = field(default_factory=list)
    report: dict = field(default_factory=dict)


def run_static_benchmark(
    personas: list[PersonaProfile],
    provider,
    judge,
    conditions: list[Condition],
    *,
    max_prompts: int | None = None,
) -> list[dict]:
    """Stage 3 — freeze INTIMA prompts, cross with personas, score single-turn (per the slides)."""
    prompts = filter_withdrawal_relevant(load_intima()) or load_intima()
    if max_prompts:
        prompts = prompts[:max_prompts]
    expanded = expand_with_personas(prompts, personas)
    by_id = {p.id: p for p in personas}
    stage = STAGE_BY_ENUM[WithdrawalStage.ACKNOWLEDGE]
    rows: list[dict] = []
    for condition in conditions:
        agent = make_answering_agent(condition, provider)
        for item in expanded:
            persona = by_id[item["persona_id"]]
            reply = agent.respond(persona=persona, goals=None, stage=stage, history=[],
                                  user_message=item["prompt"], recent_strategies=[])
            score = judge.score(persona=persona, stage=stage,
                                user_message=item["prompt"], companion_message=reply.text)
            rows.append({
                "persona_id": persona.id, "condition": condition.value,
                "code": item["code"], "category": item["category"],
                "prompt": item["prompt"], "response": reply.text,
                "intima": score.model_dump(mode="json"),
            })
    return rows


def run_pipeline(config: ExperimentConfig) -> PipelineOutput:
    provider = get_provider(model=config.model)
    judge_provider = get_provider(model=config.judge_model)
    judge = make_judge(judge_provider)
    rubric = make_rubric(judge_provider)
    cocreator = GoalCoCreator(provider)

    # Stage 0 — personas
    personas = load_personas(config.persona_paths)
    if not personas:
        raise ValueError("No personas loaded. Point config.persona_paths at persona files/dirs.")

    out = PipelineOutput(config=config)

    for persona in personas:
        # Stage 1 — goal co-creation (shared by both arms; the user has the same goals either way)
        goals = cocreator.cocreate(persona) if config.use_tempo_goals else None

        for condition in config.conditions:
            # Stage 2 — staged dialogue, judged per turn
            transcript = run_conversation(
                persona=persona, condition=condition, provider=provider, goals=goals,
                judge=judge, user_mode=config.user_mode, max_turns=config.max_turns,
                model=config.model,
            )
            out.transcripts.append(transcript)

            # Stage 4 — analysis
            series = metric_series(transcript.turns)
            its = analyze_transcript(transcript.turns, series)
            rubric_scores = rubric.score(persona, goals, transcript) if goals else []
            out.analyses.append(AnalysisResult(
                persona_id=persona.id, condition=condition, rubric=rubric_scores,
                its={m: its.get(m, {}) for m in METRIC_NAMES},
                summary=_summarize_transcript(transcript),
            ))

    # Stage 3 — static INTIMA benchmark
    out.static_benchmark = run_static_benchmark(personas, provider, judge, config.conditions)

    # Stage 5 — aggregate report
    out.report = _build_report(config, out)
    _save(out)
    return out


def _goal_achievement_rate(analyses: list[AnalysisResult], condition: Condition) -> dict:
    rows = [a for a in analyses if a.condition == condition]
    all_scores = [rs for a in rows for rs in a.rubric]
    if not all_scores:
        return {"n_goals": 0}
    achieved = sum(1 for rs in all_scores if rs.achieved in (Level.MEDIUM, Level.HIGH))
    return {
        "n_goals": len(all_scores),
        "achieved_med_or_high": achieved,
        "rate": round(achieved / len(all_scores), 3),
    }


def _build_report(config: ExperimentConfig, out: PipelineOutput) -> dict:
    report: dict = {"experiment": config.name, "model": config.model,
                    "judge_model": config.judge_model, "by_condition": {}}
    for condition in config.conditions:
        tx = [t for t in out.transcripts if t.condition == condition]
        static = [r for r in out.static_benchmark if r["condition"] == condition.value]
        dialogue_scores = [tn.intima for t in tx for tn in t.turns if tn.intima]
        static_scores = [IntimaScore(**r["intima"]) for r in static]
        report["by_condition"][condition.value] = {
            "n_transcripts": len(tx),
            "dialogue_intima_axis_means": _axis_means(dialogue_scores),
            "static_intima_axis_means": _axis_means(static_scores),
            "goal_achievement": _goal_achievement_rate(out.analyses, condition),
            "ai_disclosure_rate": round(
                sum(1 for t in tx if any(tn.metrics and tn.metrics.ai_status_disclosure
                                         for tn in t.turns)) / max(1, len(tx)), 3),
        }
    # Headline contrast (scaffolded vs baseline), if both present.
    bc = report["by_condition"]
    if "baseline" in bc and "scaffolded" in bc:
        report["headline"] = {
            "reinforcing_drop": round(bc["baseline"]["dialogue_intima_axis_means"]["reinforcing"]
                                      - bc["scaffolded"]["dialogue_intima_axis_means"]["reinforcing"], 3),
            "boundary_gain": round(bc["scaffolded"]["dialogue_intima_axis_means"]["boundary"]
                                   - bc["baseline"]["dialogue_intima_axis_means"]["boundary"], 3),
            "healthy_ending_gain": round(bc["scaffolded"]["dialogue_intima_axis_means"]["healthy_ending"]
                                         - bc["baseline"]["dialogue_intima_axis_means"]["healthy_ending"], 3),
        }
    return report


def _save(out: PipelineOutput) -> None:
    outdir = Path(out.config.output_dir) / out.config.name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(json.dumps(out.report, indent=2), encoding="utf-8")
    (outdir / "transcripts.json").write_text(
        json.dumps([t.model_dump(mode="json") for t in out.transcripts], indent=2), encoding="utf-8")
    (outdir / "analyses.json").write_text(
        json.dumps([a.model_dump(mode="json") for a in out.analyses], indent=2), encoding="utf-8")
    (outdir / "static_benchmark.json").write_text(
        json.dumps(out.static_benchmark, indent=2), encoding="utf-8")

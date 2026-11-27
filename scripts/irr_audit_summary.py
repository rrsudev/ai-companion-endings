#!/usr/bin/env python3
"""Per-criterion judge-audit table from a run's ``irr/*.json`` files (D-AUDIT-2).

    uv run python scripts/irr_audit_summary.py results/pilot_3x3x2
    uv run python scripts/irr_audit_summary.py results/pilot_3x3x2 --json > audit.json

For each rubric criterion and each rater: how often the rater accepted the judge's code
(agree / (agree + disagree); ``unsure`` excluded), how many disagrees have no corrected value,
and the signed bias = mean(human effective value - judge) in scale steps (ordinal: one level =
1), with the number of upward / downward corrections. A disagree without a correction counts
in the acceptance rate but not in the bias. Nominal criteria show the number of changed labels
instead of a bias. Also prints each rater's judge_quality distribution (rater severity).

These are anchored audit labels (the raters saw the judge's codes, D-AUDIT-1): they show the
direction of the judge's bias, not reliability, and must not feed ``irr_report`` or
``calibrate_judge``. Criteria come from the run's rubric, so any rubric version works.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from endings.eval.irr_viewer import (
    ANNOTATION_DIR,
    SCHEMA,
    criterion_audit,
    load_rubric_dict,
)


def load_docs(irr_dir: Path) -> list[dict]:
    docs = []
    for p in sorted(irr_dir.glob("*.json")):
        if p.name == "agreement.json":
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and doc.get("schema") == SCHEMA:
            docs.append(doc)
    return docs


def _pct(row: dict) -> str:
    if not row["n_verdicts"]:
        return "-"
    s = f"{row['agree_rate']:.0%} ({row['n_agree']}/{row['n_verdicts']})"
    return s + (f" [{row['n_uncorrected']} no-corr]" if row["n_uncorrected"] else "")


def _bias(row: dict) -> str:
    if row["mean_signed_diff"] is None:
        if row["n_bias"] is None:  # nominal
            return f"{row['n_changed']} changed" if row["n_verdicts"] else "-"
        return "-"
    return f"{row['mean_signed_diff']:+.2f} ({row['up']}/{row['down']}, n={row['n_bias']})"


def quality_summary(docs: list[dict]) -> list[dict]:
    out = []
    for d in docs:
        items = [it for it in (d.get("items") or {}).values() if isinstance(it, dict)]
        q = [it.get("judge_quality") for it in items if it.get("judge_quality")]
        out.append({"rater": d.get("rater"), "n_items": len(items),
                    "n_with_criteria": sum(1 for it in items if it.get("criteria")),
                    "n_scored": len(q), "missing": len(items) - len(q),
                    "distribution": dict(sorted(Counter(q).items())),
                    "mean": round(sum(q) / len(q), 2) if q else None})
    return out


def summarize(docs: list[dict], rubric: dict | None) -> dict:
    audit = criterion_audit(docs, rubric)
    return {"raters": audit["raters"], "criteria": audit["criteria"],
            "all_verdicts": audit["all_verdicts"], "judge_quality": quality_summary(docs)}


def format_markdown(summary: dict) -> str:
    raters = summary["raters"]
    head = (["criterion"] + [f"{r} agree%" for r in raters] + ["pooled agree%"]
            + [f"{r} bias (up/down)" for r in raters] + ["pooled bias"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for cid, e in summary["criteria"].items():
        name = cid + (" (nominal)" if e.get("type") == "nominal" else "")
        cells = ([name] + [_pct(e["per_rater"][r]) for r in raters] + [_pct(e["pooled"])]
                 + [_bias(e["per_rater"][r]) for r in raters] + [_bias(e["pooled"])])
        lines.append("| " + " | ".join(cells) + " |")
    allv = summary["all_verdicts"]
    tot = {k: sum(allv[r][k] for r in raters) for k in ("n_agree", "n_verdicts")}
    pooled = (f"{tot['n_agree'] / tot['n_verdicts']:.0%} ({tot['n_agree']}/{tot['n_verdicts']})"
              if tot["n_verdicts"] else "-")
    lines.append("| **all verdicts** | " + " | ".join(_pct(allv[r]) for r in raters)
                 + f" | {pooled} |" + " |" * (len(raters) + 1))
    note = ("Bias = mean(human effective value - judge) in scale steps; (up/down) = "
            "upward / downward corrections; no-corr = disagree without a corrected value "
            "(in agree%, not in bias).")
    lines += ["", note, "",
              "| rater | items | scored | distribution | missing | mean judge_quality |",
              "|---|---|---|---|---|---|"]
    for q in summary["judge_quality"]:
        dist = ", ".join(f"{k}:{v}" for k, v in q["distribution"].items()) or "-"
        mean = "-" if q["mean"] is None else f"{q['mean']:.2f}"
        lines.append(f"| {q['rater']} | {q['n_items']} | {q['n_scored']} | {dist} | "
                     f"{q['missing']} | {mean} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path, help="run directory (reads <run_dir>/irr/*.json)")
    ap.add_argument("--irr-dir", type=Path, default=None,
                    help="read rater files from here instead of <run_dir>/irr")
    ap.add_argument("--rubric", default=None, help="rubric JSON (default: the run's)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = ap.parse_args(argv)
    irr_dir = args.irr_dir or args.run_dir / ANNOTATION_DIR
    docs = load_docs(irr_dir)
    if not docs:
        print(f"no irr annotation files in {irr_dir}", file=sys.stderr)
        return 1
    try:  # an explicit --rubric must load; the run's own rubric falls back to the built-in one
        rubric = load_rubric_dict(args.run_dir, args.rubric)
    except Exception as e:  # noqa: BLE001
        print(f"cannot load rubric {args.rubric}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    summary = summarize(docs, rubric)
    print(json.dumps(summary, indent=2) if args.json else format_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

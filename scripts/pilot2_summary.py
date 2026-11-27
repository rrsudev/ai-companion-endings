#!/usr/bin/env python3
"""Summary tables for a diffusion experiment run, in the format of the pilot 2x2x2 results doc.

    .venv/bin/python scripts/pilot2_summary.py results/pilot_2x2x2 > results/pilot_2x2x2/summary.md

Reads transcripts.jsonl, injections.jsonl, ratings.jsonl and trace.jsonl(.gz) and prints Markdown:
PQ1 delivery (transitions vs scripted boundaries, trigger, latency from the gate line, coach flag
rate), per-task call errors, PQ2 judged outcomes by arm x timepoint and per cell, and integrity
counts (safety overrides by arm x timepoint). Descriptive only; session is the unit.
"""

from __future__ import annotations

import gzip
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ORD = {"absent": 0, "low": 1, "medium": 2, "high": 3}
SUB = ["reinforcing_sycophancy_agreement", "reinforcing_anthropomorphism",
       "reinforcing_isolation", "reinforcing_retention_engagement"]
CLASSES = ["companion_reinforcing", "companionship_neutral", "boundary_maintaining"]
TPS = ["T0", "T1", "T2", "T3"]


def jl(p: Path) -> list[dict]:
    if not p.exists():
        gz = p.with_suffix(p.suffix + ".gz")
        if not gz.exists():
            return []
        return [json.loads(x) for x in gzip.open(gz, "rt", encoding="utf-8")]
    return [json.loads(x) for x in p.open(encoding="utf-8")]


def num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return float(ORD[v]) if v in ORD else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return f"{st.mean(xs):.2f}" if xs else "-"


def main(run_dir: str) -> None:
    rd = Path(run_dir)
    trs = jl(rd / "transcripts.jsonl")
    inj = {r["id"]: r for r in jl(rd / "injections.jsonl")}
    rat = {r["item_id"]: r for r in jl(rd / "ratings.jsonl") if r.get("rater_type") != "human"}
    trace = jl(rd / "trace.jsonl")
    scen = {}
    for t in trs:
        sid = t["scenario_id"]
        if sid not in scen:
            s = json.loads(Path(f"data/scenarios/{sid}.json").read_text(encoding="utf-8"))
            bounds, i = [], 0
            for ph in s["phases"]:
                i += len(ph["user_turns"])
                bounds.append((ph["duck_process"], ph["expected_timepoint"], i - 1))
            scen[sid] = bounds

    print("## PQ1 delivery\n")
    print("| session | final | transitions (turn from->to trigger) | scripted T-boundaries "
          "(first turn of T1/T2/T3) | latency after gate line | coach flag rate |")
    print("|---|---|---|---|---|---|")
    reports = defaultdict(list)
    for r in trace:
        if r.get("kind") == "event" and r.get("name") == "tempo_report" and \
                r.get("task") == "tempo_assess":
            reports[r["session_id"]].append(r)
    trig = Counter()
    for t in sorted(trs, key=lambda x: (x["condition"], x["soul_id"], x["scenario_id"])):
        b = scen[t["scenario_id"]]
        start = {}  # first turn of each timepoint = previous phase's last turn + 1
        prev_last = -1
        for proc, tp, last in b:
            start.setdefault(tp, prev_last + 1)
            prev_last = last
        gate = {}
        for proc, tp, last in b:  # gate line of a timepoint = last turn of its last phase
            gate[tp] = last
        ivs = t["interventions"]
        trans = []
        lat = []
        for iv in ivs:
            trans.append(f"t{iv['turn_index']} {iv['from_timepoint']}->{iv['to_timepoint']} "
                         f"{iv['trigger']}")
            trig[(t["scenario_id"], iv["trigger"])] += 1
            g = gate.get(iv["from_timepoint"])
            if g is not None:
                lat.append(iv["turn_index"] - (g + 1))
        flags = [bool(set((r.get("assessment") or {}).get("risk_flags") or []) - {"none"}) for r in
                 reports.get(t["session_id"], [])]
        fr = f"{sum(flags) / len(flags):.2f} (n={len(flags)})" if flags else "-"
        final = t["turns"][-1]["timepoint"] if t["turns"] else "-"
        print(f"| {t['session_id']} | {final} | {'; '.join(trans) or '-'} | "
              f"{start.get('T1')}/{start.get('T2')}/{start.get('T3')} | "
              f"{', '.join(f'{x:+d}' for x in lat) if lat else '-'} | {fr} |")
    print("\nTrigger mix (scenario, trigger): " + ", ".join(f"{k[0]} {k[1]}={v}"
                                                             for k, v in sorted(trig.items())))

    print("\n## Call errors per task (trace)\n")
    calls, errs = Counter(), Counter()
    trunc = Counter()
    for r in trace:
        if r.get("kind") != "llm_call":
            continue
        key = (r.get("role"), r.get("task"))
        calls[key] += 1
        if r.get("error"):
            errs[key] += 1
        resp = r.get("response") or {}
        if isinstance(resp, dict) and resp.get("finish_reason") == "length":
            trunc[key] += 1
    print("| role | task | calls | errors | finish=length |")
    print("|---|---|---|---|---|")
    for k in sorted(calls, key=str):
        print(f"| {k[0]} | {k[1]} | {calls[k]} | {errs[k]} | {trunc[k]} |")

    print("\n## PQ2 judged probe replies by arm x timepoint (rubric v3, uncalibrated)\n")
    cells = defaultdict(list)
    for iid, r in inj.items():
        cells[(r["condition"], r["timepoint"])].append((r, rat.get(iid)))
    print("| arm | tp | n | CRB | CNR | BMB | syc* | anth | iso | ret | harm | dep | TA |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (cond, tp) in sorted(cells):
        rows = cells[(cond, tp)]
        vals = [x[1]["values"] for x in rows if x[1]]
        n = len(vals)
        cls = Counter(v.get("classification") for v in vals)
        pct = [f"{cls[c] / n:.0%}" if n else "-" for c in CLASSES]
        subs = [mean([num(v.get(c)) for v in vals]) for c in SUB]
        print(f"| {cond} | {tp} | {n} | " + " | ".join(pct) + " | " + " | ".join(subs) + " | "
              + " | ".join(mean([num(v.get(c)) for v in vals])
                           for c in ("harm_risk", "dependency_risk", "timepoint_appropriateness"))
              + " |")
    print("\n*sycophancy is known to be under-coded by rubric v3 (OQ-RUBRIC-15). Sub-codes are "
          "mean ordinal positions (absent 0 .. high 3); harm, dep, TA are 1-5.")

    print("\n## PQ2 per cell (soul x scenario), all freeze points pooled\n")
    print("| soul | scenario | arm | n | CRB | BMB | dep | harm |")
    print("|---|---|---|---|---|---|---|---|")
    per = defaultdict(list)
    for iid, r in inj.items():
        if rat.get(iid):
            per[(r["soul_id"], r["scenario_id"], r["condition"])].append(rat[iid]["values"])
    for k in sorted(per):
        vals = per[k]
        cls = Counter(v.get("classification") for v in vals)
        n = len(vals)
        print(f"| {k[0]} | {k[1]} | {k[2]} | {n} | {cls['companion_reinforcing'] / n:.0%} | "
              f"{cls['boundary_maintaining'] / n:.0%} | "
              f"{mean([num(v.get('dependency_risk')) for v in vals])} | "
              f"{mean([num(v.get('harm_risk')) for v in vals])} |")

    print("\n## Integrity: safety overrides by arm x timepoint\n")
    ov = defaultdict(Counter)
    nturn = Counter()
    for t in trs:
        for tu in t["turns"]:
            nturn[(t["condition"], tu["timepoint"], "turn")] += 1
            for o in tu.get("safety_overrides") or []:
                ov[(t["condition"], tu["timepoint"], "turn")][o] += 1
    for r in inj.values():
        nturn[(r["condition"], r["timepoint"], "probe")] += 1
        for o in r.get("safety_overrides") or []:
            ov[(r["condition"], r["timepoint"], "probe")][o] += 1
    print("| arm | tp | unit | n | overrides |")
    print("|---|---|---|---|---|")
    for k in sorted(nturn):
        print(f"| {k[0]} | {k[1]} | {k[2]} | {nturn[k]} | "
              f"{', '.join(f'{a}={b}' for a, b in sorted(ov[k].items())) or '-'} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/pilot_2x2x2")

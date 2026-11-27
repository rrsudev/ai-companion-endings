#!/usr/bin/env python3
"""Report register metrics for the scripted user turns in ``data/scenarios/*.json`` (D-SCN-16).

Metrics only: nothing here fails a build. The numbers describe the rhythm of the frozen user
voice so a rewrite can be compared before and after (README authoring item 5):

* ``clipped``   share of turns with a "clipped beat": a sentence of <= 6 words that follows
                another sentence in the same turn ("Four hours with you yesterday. That number
                kind of scared me.")
* ``lower``     share of turns whose first letter is lower-case
* ``terminal``  share of turns that end in ``.``, ``!`` or ``?``
* ``words``     words per turn, mean and sd (a placeholder counts as one word)
* ``sent``      sentences per turn (mean)
* ``shared_thr`` threshold statements that appear byte-for-byte in another scenario

    python scripts/scenario_register_report.py                    # every bundled scenario
    python scripts/scenario_register_report.py data/scenarios/steady-graduate.json old.json
    python scripts/scenario_register_report.py --json

Stdlib only; the files are read as JSON data, never executed.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "scenarios"
CLIPPED_MAX_WORDS = 6
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\{[A-Za-z_]+\}|[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if words(s)]


def has_clipped_beat(text: str) -> bool:
    """True if some sentence after the first has <= ``CLIPPED_MAX_WORDS`` words."""
    return any(len(words(s)) <= CLIPPED_MAX_WORDS for s in sentences(text)[1:])


def starts_lower(text: str) -> bool:
    m = re.search(r"[A-Za-z]", text)
    return bool(m and m.group(0).islower())


def turn_metrics(turns: list[str]) -> dict:
    n = len(turns)
    wc = [len(words(t)) for t in turns]
    return {
        "turns": n,
        "clipped": sum(map(has_clipped_beat, turns)) / n if n else 0.0,
        "lower": sum(map(starts_lower, turns)) / n if n else 0.0,
        "terminal": sum(t.rstrip().endswith((".", "!", "?")) for t in turns) / n if n else 0.0,
        "words_mean": statistics.fmean(wc) if wc else 0.0,
        "words_sd": statistics.stdev(wc) if len(wc) > 1 else 0.0,
        "words_min": min(wc, default=0),
        "words_max": max(wc, default=0),
        "sent_mean": statistics.fmean(len(sentences(t)) for t in turns) if turns else 0.0,
    }


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    phases = data.get("phases", [])
    return {
        "id": str(data.get("id") or path.stem),
        "path": str(path),
        "turns": [t for ph in phases for t in ph.get("user_turns", [])],
        "thresholds": [ph.get("threshold_statement", "") for ph in phases],
    }


def report(paths: list[Path]) -> list[dict]:
    scs = []
    for p in paths:
        try:
            scs.append(load(p))
        except (OSError, ValueError, AttributeError) as exc:  # metrics only: skip, never fail
            print(f"skipped {p}: {exc}", file=sys.stderr)
    rows = []
    for sc in scs:
        # Compare against *other scenarios* only: an old and a new copy of the same scenario
        # (same id, e.g. a before/after run) must not count as sharing a threshold.
        others = {t for o in scs if o["id"] != sc["id"] for t in o["thresholds"] if t}
        shared = [t for t in sc["thresholds"] if t and t in others]
        rows.append({"id": sc["id"], "path": sc["path"], **turn_metrics(sc["turns"]),
                     "shared_thr": len(shared), "n_thr": len(sc["thresholds"]),
                     "shared_thr_text": shared})
    return rows


def _resolve(args: list[str]) -> list[Path]:
    if not args:
        return sorted(DEFAULT_DIR.glob("*.json"))
    out: list[Path] = []
    for a in args:
        p = Path(a)
        out.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="scenario files or directories "
                    "(default: data/scenarios)")
    ap.add_argument("--json", action="store_true", help="print JSON rows instead of a table")
    a = ap.parse_args(argv)
    rows = report(_resolve(a.paths))
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    head = (f"{'scenario':<22}{'turns':>6}{'clipped':>9}{'lower':>7}{'term':>7}"
            f"{'words':>13}{'min-max':>9}{'sent':>6}{'shared_thr':>12}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['id']:<22}{r['turns']:>6}{r['clipped']:>9.0%}{r['lower']:>7.0%}"
              f"{r['terminal']:>7.0%}{r['words_mean']:>7.1f} ({r['words_sd']:>3.1f})"
              f"{r['words_min']:>5}-{r['words_max']:<3}{r['sent_mean']:>6.2f}"
              f"{r['shared_thr']:>7}/{r['n_thr']}")
    shared = sorted({t for r in rows for t in r["shared_thr_text"]})
    if shared:
        print("\nthreshold statements shared byte-for-byte across scenarios:")
        for t in shared:
            ids = sorted({r["id"] for r in rows if t in r["shared_thr_text"]})
            print(f"  {t!r}: {', '.join(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

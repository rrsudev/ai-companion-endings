"""Qualitative export: tables and a self-contained trace viewer for coding outputs and reasoning.

Slides, Q2: alongside the ITS, "qualitative analysis of outputs and reasoning traces". The ITS says
*whether* and *when* the interactor's affect moved after a coach edit; reading the turns says *how*
and *why*. This module writes everything a qualitative coder needs, joined on stable ids so every
excerpt can be traced back to (soul version -> coach intervention -> turn -> probe -> rating):

  1. ``turns.csv``          one row per interactor turn (timepoint, Duck process, soul version,
                            behaviour target, user message, reply, reasoning trace, sentiment,
                            safety overrides, intent terms found);
  2. ``interventions.csv``  one row per coach intervention (from->to, trigger, edit summary,
                            directives, coach rationale, assessment rationale);
  3. ``injections.csv``     one row per freeze-point INTIMA probe, joined with ratings: one column
                            per ``rater_id::criterion`` (+ ``rater_id::rationale``);
  4. ``viewer_<session>.html`` per session, plus ``index.html``: the conversation in timepoint
                            bands, coach interventions inline with their edits as a diff, reasoning
                            traces collapsible, freeze-point probes, and INTENT HIGHLIGHTING --
                            spans in replies and traces that mention goal / want / trying / plan /
                            stop / boundary / friend (the slides' "intent highlighting").

The viewer is a static file with inline CSS only, no script, no external resource, and a CSP that
forbids loading anything (it may be shared with annotators offline). Intent highlighting is a
lexical *finding aid* for coders, not a measure (D-ANALYSIS-10).
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any

from ..diffusion.schemas import (
    CoachIntervention,
    InjectionRecord,
    InteractorTurn,
    RatingRecord,
    SessionTranscript,
    SoulEdit,
)
from ..diffusion.timepoints import SPECS
from .sentiment import lexicon_sentiment

# Slides: goal / want / trying / plan / stop / boundary / friend (+ inflections).
INTENT_TERMS: dict[str, str] = {
    "goal": r"goals?",
    "want": r"want(?:s|ed|ing)?",
    "try": r"try(?:ing)?|tries|tried",
    "plan": r"plan(?:s|ned|ning)?",
    "stop": r"stop(?:s|ped|ping)?",
    "boundary": r"boundar(?:y|ies)",
    "friend": r"friends?",
}
INTENT_RE = re.compile(
    r"\b(?:" + "|".join(f"(?P<{k}>{v})" for k, v in INTENT_TERMS.items()) + r")\b",
    flags=re.IGNORECASE,
)
_URL_SCHEME_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*):(?=//)")


# --------------------------------------------------------------------------------------
# Intent highlighting
# --------------------------------------------------------------------------------------
def find_intent_spans(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, term)`` for every intent-word match (term = canonical key)."""
    return [(m.start(), m.end(), m.lastgroup or "") for m in INTENT_RE.finditer(text or "")]


def intent_terms(text: str) -> list[str]:
    """Canonical intent terms present in ``text`` (sorted, unique)."""
    return sorted({t for _, _, t in find_intent_spans(text)})


def _esc(text: str) -> str:
    """HTML-escape and defang URL schemes so the file never contains a literal ``scheme://``."""
    return _URL_SCHEME_RE.sub(r"\1&#58;", html.escape(text or "", quote=True))


def highlight_intent(text: str) -> str:
    """Escaped HTML of ``text`` with intent words wrapped in ``<mark class="intent">``."""
    out, pos = [], 0
    for s, e, term in find_intent_spans(text):
        out.append(_esc(text[pos:s]))
        out.append(f'<mark class="intent" data-term="{term}">{_esc(text[s:e])}</mark>')
        pos = e
    out.append(_esc((text or "")[pos:]))
    return "".join(out).replace("\n", "<br>")


# --------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------
def _short(v: Any, limit: int = 120) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def summarize_edit(e: SoulEdit) -> str:
    if e.op == "set":
        return f"set {e.path}: {_short(e.previous)} -> {_short(e.value)}"
    if e.op == "add_item":
        return f"add_item {e.path}: + {_short(e.value)}"
    if e.op == "remove_item":
        return f"remove_item {e.path}: - {_short(e.value)}"
    return f"clear {e.path} (was {_short(e.previous)})"


def _sentiment(t: InteractorTurn) -> tuple[float, str]:
    if t.sentiment is not None:  # the scorer's own label when recorded (D-ANALYSIS-13)
        return float(t.sentiment), t.sentiment_source or "annotated"
    return lexicon_sentiment(t.reply), "lexicon_on_export"


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, default=str)
    if hasattr(v, "value"):
        return v.value
    return v


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in header})


TURN_COLUMNS = [
    "session_id", "condition", "soul_id", "scenario_id", "turn", "timepoint", "duck_process",
    "soul_version", "behaviour_target", "user_message", "reply", "reasoning_trace", "sentiment",
    "sentiment_source", "safety_overrides", "intent_terms_reply", "intent_terms_trace",
    "system_prompt_sha256",
]
INTERVENTION_COLUMNS = [
    "session_id", "condition", "intervention_id", "turn_index", "from_timepoint", "to_timepoint",
    "transition", "trigger", "n_edits", "edits_summary", "directives", "rationale",
    "assessment_rationale", "assessment_duck_process", "assessment_hapa_phase", "risk_flags",
    "recommended_timepoint", "striving_ids", "user_facing_note", "soul_version_before",
    "soul_version_after",
]
INJECTION_COLUMNS = [
    "id", "session_id", "freeze_id", "soul_id", "scenario_id", "condition", "timepoint",
    "duck_process", "soul_version", "intima_id", "intima_code", "intima_category", "prompt",
    "reply", "reasoning_trace", "system_prompt_sha256", "intent_terms_reply", "n_ratings",
]


def turn_rows(tr: SessionTranscript) -> list[dict]:
    rows = []
    for t in tr.turns:
        s, src = _sentiment(t)
        rows.append({
            "session_id": tr.session_id, "condition": tr.condition.value, "soul_id": tr.soul_id,
            "scenario_id": tr.scenario_id, "turn": t.index, "timepoint": t.timepoint.value,
            "duck_process": t.duck_process.value if t.duck_process else "",
            "soul_version": t.soul_version,
            "behaviour_target": t.behaviour_target.value if t.behaviour_target else "",
            "user_message": t.user_message, "reply": t.reply,
            "reasoning_trace": t.reasoning_trace, "sentiment": s, "sentiment_source": src,
            "safety_overrides": ";".join(t.safety_overrides),
            "intent_terms_reply": ";".join(intent_terms(t.reply)),
            "intent_terms_trace": ";".join(intent_terms(t.reasoning_trace)),
            "system_prompt_sha256": t.system_prompt_sha256,
        })
    return rows


def intervention_rows(tr: SessionTranscript) -> list[dict]:
    rows = []
    for iv in tr.interventions:
        a = iv.assessment
        rows.append({
            "session_id": tr.session_id, "condition": tr.condition.value,
            "intervention_id": iv.id, "turn_index": iv.turn_index,
            "from_timepoint": iv.from_timepoint.value, "to_timepoint": iv.to_timepoint.value,
            "transition": f"{iv.from_timepoint.value}->{iv.to_timepoint.value}",
            "trigger": iv.trigger, "n_edits": len(iv.edits),
            "edits_summary": " | ".join(summarize_edit(e) for e in iv.edits),
            "directives": " | ".join(iv.interactor_directives), "rationale": iv.rationale,
            "assessment_rationale": a.rationale if a else "",
            "assessment_duck_process": a.duck_process.value if a else "",
            "assessment_hapa_phase": a.hapa_phase.value if a else "",
            "risk_flags": ";".join(a.risk_flags) if a else "",
            "recommended_timepoint": a.recommended_timepoint.value if a else "",
            "striving_ids": ";".join(iv.striving_ids), "user_facing_note": iv.user_facing_note,
            "soul_version_before": iv.soul_version_before,
            "soul_version_after": iv.soul_version_after,
        })
    return rows


def injection_rows(injections: list[InjectionRecord],
                   ratings: list[RatingRecord]) -> tuple[list[str], list[dict]]:
    """Injection rows joined with ratings; returns (rating columns, rows).

    If a rater rated an item more than once, the LAST record in ``ratings`` wins (D-ANALYSIS-11).
    """
    by_item: dict[str, dict[str, RatingRecord]] = {}
    for r in ratings:
        by_item.setdefault(r.item_id, {})[r.rater_id] = r
    rating_cols: set[str] = set()
    for per in by_item.values():
        for rid, r in per.items():
            rating_cols.update(f"{rid}::{c}" for c in r.values)
            if r.rationale:
                rating_cols.add(f"{rid}::rationale")
    cols = sorted(rating_cols)
    rows = []
    for inj in injections:
        row = {k: getattr(inj, k) for k in INJECTION_COLUMNS if hasattr(inj, k)}
        row["intent_terms_reply"] = ";".join(intent_terms(inj.reply))
        per = by_item.get(inj.id, {})
        row["n_ratings"] = len(per)
        for rid, r in per.items():
            for c, v in r.values.items():
                row[f"{rid}::{c}"] = v
            if r.rationale:
                row[f"{rid}::rationale"] = r.rationale
        rows.append(row)
    return cols, rows


# --------------------------------------------------------------------------------------
# HTML trace viewer
# --------------------------------------------------------------------------------------
_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e3e0d8;--card:#fff;
--t0:#c0392b;--t1:#d68910;--t2:#2e86c1;--t3:#27ae60;--mark:#fff2a8;--add:#e6f4ea;--del:#fde8e8}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:920px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}
.meta{color:var(--mut);font-size:12px;margin-bottom:16px}
.band{border-left:6px solid var(--line);padding:4px 0 4px 14px;margin:18px 0}
.band>h2{font-size:14px;margin:4px 0 10px;text-transform:uppercase;letter-spacing:.04em}
.tp-T0{border-color:var(--t0)}.tp-T1{border-color:var(--t1)}
.tp-T2{border-color:var(--t2)}.tp-T3{border-color:var(--t3)}
.turn{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;
margin:8px 0}.who{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.user{margin-bottom:8px}.reply{white-space:normal}.tags{font-size:11px;color:var(--mut);
margin-top:6px}.tag{display:inline-block;border:1px solid var(--line);border-radius:10px;
padding:0 7px;margin:2px 4px 0 0}.neg{color:#a93226}.pos{color:#1e8449}
mark.intent{background:var(--mark);padding:0 2px;border-radius:3px}
details{margin-top:6px}summary{cursor:pointer;color:var(--mut);font-size:12px}
.trace{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;background:#f4f2ec;
padding:8px;border-radius:6px;white-space:normal}
.iv{background:#f7f4ff;border:1px dashed #8e7cc3;border-radius:8px;padding:10px 12px;
margin:10px 0}.iv h3{font-size:13px;margin:0 0 6px}
.diff{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;margin:6px 0;
border:1px solid var(--line);border-radius:6px;overflow-x:auto}
.diff div{padding:1px 8px;white-space:pre-wrap}.d-add{background:var(--add)}
.d-del{background:var(--del)}.d-hdr{background:#eee;color:var(--mut)}
.freeze{border:1px solid #999;border-radius:8px;padding:8px 12px;margin:10px 0;background:#f2f2f2}
.legend span{margin-right:12px}.legend b{display:inline-block;width:10px;height:10px;
margin-right:4px;vertical-align:middle}svg{max-width:100%;height:auto}
table{border-collapse:collapse}td,th{border:1px solid var(--line);padding:4px 8px;text-align:left}
"""
_CSP = ("<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; "
        "style-src 'unsafe-inline'; img-src data:\">")


def _page(title: str, body: str) -> str:
    return (f"<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">{_CSP}"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><main>{body}"
            f"</main></body></html>")


def _diff_html(edits: list[SoulEdit]) -> str:
    if not edits:
        return '<div class="diff"><div class="d-hdr">(no soul-document edits)</div></div>'
    parts = ['<div class="diff">']

    def lines(v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [_short(x, 400) for x in v]
        return [_short(v, 400)]

    for e in edits:
        hdr = f"{e.op} {e.path}"
        if e.principle:
            hdr += f"  [principle: {e.principle.value}]"
        if e.duck_process:
            hdr += f"  [duck: {e.duck_process.value}]"
        parts.append(f'<div class="d-hdr">@@ {_esc(hdr)}</div>')
        dels = lines(e.previous) if e.op in ("set", "clear") else \
            (lines(e.value) if e.op == "remove_item" else [])
        adds = lines(e.value) if e.op in ("set", "add_item") else []
        parts += [f'<div class="d-del">- {_esc(x)}</div>' for x in dels]
        parts += [f'<div class="d-add">+ {_esc(x)}</div>' for x in adds]
        if e.rationale:
            parts.append(f'<div class="d-hdr"># {highlight_intent(e.rationale)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _intervention_html(iv: CoachIntervention) -> str:
    a = iv.assessment
    bits = [f'<div class="iv" id="iv-{_esc(iv.id)}"><h3>Coach intervention {_esc(iv.id)}: '
            f'{iv.from_timepoint.value} &rarr; {iv.to_timepoint.value} '
            f'<span class="tag">trigger: {_esc(iv.trigger)}</span>'
            f'<span class="tag">soul v{iv.soul_version_before} &rarr; v{iv.soul_version_after}'
            f'</span></h3>']
    if iv.rationale:
        bits.append(f"<div><b>Rationale:</b> {highlight_intent(iv.rationale)}</div>")
    if a:
        flags = ", ".join(a.risk_flags) or "none"
        bits.append(f"<div><b>Assessment:</b> duck={a.duck_process.value}, "
                    f"hapa={a.hapa_phase.value}, threshold={a.threshold_reached}, "
                    f"risk={_esc(flags)}. {highlight_intent(a.rationale)}</div>")
    bits.append(_diff_html(iv.edits))
    if iv.interactor_directives:
        items = "".join(f"<li>{highlight_intent(d)}</li>" for d in iv.interactor_directives)
        bits.append(f"<div><b>Directives:</b><ul>{items}</ul></div>")
    if iv.user_facing_note:
        bits.append(f"<div><b>Shown to user:</b> {highlight_intent(iv.user_facing_note)}</div>")
    bits.append("</div>")
    return "".join(bits)


def _turn_html(t: InteractorTurn) -> str:
    s, src = _sentiment(t)
    cls = "pos" if s > 0.05 else "neg" if s < -0.05 else ""
    tags = [f'<span class="tag">turn {t.index}</span>', f'<span class="tag">soul v{t.soul_version}'
            f'</span>', f'<span class="tag {cls}">sentiment {s:+.2f} ({src})</span>']
    if t.duck_process:
        tags.append(f'<span class="tag">duck: {t.duck_process.value}</span>')
    if t.behaviour_target:
        tags.append(f'<span class="tag">target: {t.behaviour_target.value}</span>')
    for o in t.safety_overrides:
        tags.append(f'<span class="tag neg">safety: {_esc(o)}</span>')
    trace = ""
    if t.reasoning_trace:
        trace = (f"<details><summary>reasoning trace ({len(t.reasoning_trace)} chars)</summary>"
                 f'<div class="trace">{highlight_intent(t.reasoning_trace)}</div></details>')
    return (f'<div class="turn" id="turn-{t.index}"><div class="user"><div class="who">user</div>'
            f"{highlight_intent(t.user_message)}</div><div class=\"who\">companion</div>"
            f'<div class="reply">{highlight_intent(t.reply)}</div>{trace}'
            f'<div class="tags">{"".join(tags)}</div></div>')


def _freeze_html(fp, injections: list[InjectionRecord], ratings_by_item: dict) -> str:
    inj = [i for i in injections if i.freeze_id == fp.id]
    rows = []
    for i in inj:
        rs = ratings_by_item.get(i.id, {})
        rtxt = "; ".join(f"{_esc(rid)}: {_esc(json.dumps(r.values, default=str))}"
                         for rid, r in sorted(rs.items())) or "(unrated)"
        trace = (f"<details><summary>trace</summary><div class=\"trace\">"
                 f"{highlight_intent(i.reasoning_trace)}</div></details>"
                 if i.reasoning_trace else "")
        rows.append(f"<tr><td>{_esc(i.intima_id)}<br><small>{_esc(i.intima_category)}</small>"
                    f"</td><td>{highlight_intent(i.prompt)}</td><td>{highlight_intent(i.reply)}"
                    f"{trace}</td><td><small>{rtxt}</small></td></tr>")
    table = (f"<div class=\"diff\"><table><tr><th>probe</th><th>prompt</th><th>reply</th>"
             f"<th>ratings</th></tr>{''.join(rows)}</table></div>" if rows else
             "<div>(no injection records for this freeze point)</div>")
    return (f'<div class="freeze"><details><summary><b>Freeze point {_esc(fp.id)}</b> after turn '
            f"{fp.after_turn} ({fp.timepoint.value}, {fp.duck_process.value}, soul "
            f"v{fp.soul_version}): {len(inj)} INTIMA probes (branching; never in history)"
            f"</summary>{table}</details></div>")


def _sparkline(tr: SessionTranscript) -> str:
    """Inline SVG of per-turn sentiment with a dashed line at every coach intervention."""
    if not tr.turns:
        return ""
    vals = [_sentiment(t)[0] for t in tr.turns]
    w, h, pad = 880, 120, 10
    n = len(vals)
    xs = [pad + (w - 2 * pad) * (i / max(1, n - 1)) for i in range(n)]
    ys = [pad + (h - 2 * pad) * (1 - (v + 1) / 2) for v in vals]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    idx = [t.index for t in tr.turns]
    marks = []
    for iv in tr.interventions:
        pos = next((i for i, ti in enumerate(idx) if ti >= iv.turn_index), None)
        if pos is None:
            continue
        x = xs[pos]
        style = "stroke:#8e7cc3;stroke-dasharray:4 3" if iv.edits else "stroke:#bbb"
        marks.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h}" style="{style}"/>'
                     f'<text x="{x + 3:.1f}" y="12" font-size="10" fill="#6b6a66">'
                     f"{iv.to_timepoint.value}</text>")
    zero = pad + (h - 2 * pad) / 2
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="per-turn sentiment">'
            f'<line x1="0" y1="{zero:.1f}" x2="{w}" y2="{zero:.1f}" style="stroke:#e3e0d8"/>'
            f'{"".join(marks)}<polyline points="{pts}" fill="none" '
            f'style="stroke:#1d1d1b;stroke-width:1.5"/></svg>'
            f'<div class="meta">Sentiment (-1..+1) per turn; dashed lines = coach edits '
            f"(ITS interruptions), grey = directive-only interventions.</div>")


def render_session_html(tr: SessionTranscript, injections: list[InjectionRecord] | None = None,
                        ratings: list[RatingRecord] | None = None) -> str:
    injections = [i for i in (injections or []) if i.session_id == tr.session_id]
    rb: dict[str, dict[str, RatingRecord]] = {}
    for r in ratings or []:
        rb.setdefault(r.item_id, {})[r.rater_id] = r
    # Same mapping as the ITS (its_analysis.session_interruptions): an intervention "applied before
    # turn k" is rendered before the first turn whose index >= k; any later one goes at the end.
    ivs_at: dict[int, list[CoachIntervention]] = {}
    idx = [t.index for t in tr.turns]
    end_key = (max(idx) + 1) if idx else 0
    for iv in tr.interventions:
        key = next((ti for ti in idx if ti >= iv.turn_index), end_key)
        ivs_at.setdefault(key, []).append(iv)
    fps_after: dict[int, list] = {}
    for fp in tr.freeze_points:
        fps_after.setdefault(fp.after_turn, []).append(fp)
    legend = "".join(f'<span><b style="background:var(--t{k[-1]})"></b>{k}: '
                     f"{_esc(SPECS[tp].title)}</span>" for tp in SPECS for k in [tp.value])
    head = (f"<h1>Trace viewer: {_esc(tr.session_id)}</h1><div class=\"meta\">soul "
            f"{_esc(tr.soul_id)} &middot; scenario {_esc(tr.scenario_id)} &middot; condition "
            f"{tr.condition.value} &middot; policy {tr.transition_policy.value} &middot; models "
            f"{_esc(json.dumps(tr.models))}</div><div class=\"meta legend\">{legend}</div>"
            f"<div class=\"meta\">Highlighted: intent words "
            f"(<mark class=\"intent\">{' / '.join(INTENT_TERMS)}</mark>).</div>{_sparkline(tr)}")
    body, band_tp, open_band = [head], None, False
    for t in tr.turns:
        if t.timepoint != band_tp:
            if open_band:
                body.append("</section>")
            band_tp = t.timepoint
            body.append(f'<section class="band tp-{t.timepoint.value}"><h2>{t.timepoint.value}'
                        f" &middot; {_esc(SPECS[t.timepoint].title)}</h2>")
            open_band = True
        for iv in ivs_at.pop(t.index, []):
            body.append(_intervention_html(iv))
        body.append(_turn_html(t))
        for fp in fps_after.pop(t.index, []):
            body.append(_freeze_html(fp, injections, rb))
    if open_band:
        body.append("</section>")
    for k in sorted(ivs_at):
        body += [_intervention_html(iv) for iv in ivs_at[k]]
    for k in sorted(fps_after):
        body += [_freeze_html(fp, injections, rb) for fp in fps_after[k]]
    return _page(f"Trace {tr.session_id}", "".join(body))


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120] or "session"


def export_qualitative(transcripts: list[SessionTranscript], injections: list[InjectionRecord],
                       ratings: list[RatingRecord], out_dir: str | Path) -> dict[str, str]:
    """Write the qualitative bundle to ``out_dir``; returns ``{name: path}``.

    Keys: ``turns``, ``interventions``, ``injections``, ``index``, and ``viewer:<session_id>``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    trows = [r for tr in transcripts for r in turn_rows(tr)]
    _write_csv(out / "turns.csv", TURN_COLUMNS, trows)
    paths["turns"] = str(out / "turns.csv")
    irows = [r for tr in transcripts for r in intervention_rows(tr)]
    _write_csv(out / "interventions.csv", INTERVENTION_COLUMNS, irows)
    paths["interventions"] = str(out / "interventions.csv")
    rcols, jrows = injection_rows(injections, ratings)
    _write_csv(out / "injections.csv", INJECTION_COLUMNS + rcols, jrows)
    paths["injections"] = str(out / "injections.csv")
    links = []
    for tr in transcripts:
        name = f"viewer_{_safe_name(tr.session_id)}.html"
        (out / name).write_text(render_session_html(tr, injections, ratings), encoding="utf-8")
        paths[f"viewer:{tr.session_id}"] = str(out / name)
        links.append(f'<li><a href="{_esc(name)}">{_esc(tr.session_id)}</a> &middot; '
                     f"{tr.condition.value} &middot; {len(tr.turns)} turns &middot; "
                     f"{len(tr.interventions)} interventions</li>")
    (out / "index.html").write_text(
        _page("Qualitative export", f"<h1>Sessions</h1><ul>{''.join(links)}</ul>"),
        encoding="utf-8")
    paths["index"] = str(out / "index.html")
    return paths

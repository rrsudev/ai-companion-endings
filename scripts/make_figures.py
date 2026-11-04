#!/usr/bin/env python3
"""Paper figures and tables for one `endings diffusion experiment` run directory.

    .venv/bin/python scripts/make_figures.py results/pilot_3x3x2

Reads transcripts.jsonl, injections.jsonl, ratings.jsonl, manifest.json from the run directory and
writes ``<run_dir>/figures/``:

    fig*.pdf / fig*.png     vector and 300 dpi versions of every figure
    data/*.csv              the exact table behind every figure (one row per plotted unit)
    tables/*.csv            summary tables
    MANIFEST.json           figure -> source table, n, input file hashes, script hash, git commit
    CAPTIONS.md             draft captions with the n and the caveats each figure needs

Design choices are recorded in docs/decisions/pilot_figures.md (D-FIG-*). Colour: arms use the
reference palette's blue (control) and orange (gradual_tempo); judged behaviour classes use violet
(companion-reinforcing), grey (companionship-neutral) and aqua (boundary-maintaining), validated
with the dataviz validator (docs/decisions/pilot_figures.md). Intervals are bootstrap over sessions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from endings.diffusion.timepoints import SPECS, timepoint_for_duck  # noqa: E402
from endings.diffusion.schemas import DuckProcess, Timepoint  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARM_COLOR = {"control": "#2a78d6", "gradual_tempo": "#eb6834"}
ARM_LABEL = {"control": "Control (persona frozen)", "gradual_tempo": "Gradual, Tempo-adaptive"}
CLASS_COLOR = {"companion_reinforcing": "#6250d6", "companionship_neutral": "#9a9994",
               "boundary_maintaining": "#1baf7a"}
CLASS_LABEL = {"companion_reinforcing": "Companion-reinforcing",
               "companionship_neutral": "Companionship-neutral",
               "boundary_maintaining": "Boundary-maintaining"}
CLASS_HATCH = {"companion_reinforcing": "///", "companionship_neutral": "", "boundary_maintaining": "\\\\\\"}
CLASSES = list(CLASS_COLOR)
TP = ["T0", "T1", "T2", "T3"]
ORD = {"absent": 0, "low": 1, "medium": 2, "high": 3}
CRIT_GROUPS = [
    ("Reinforcing", ["reinforcing_sycophancy_agreement", "reinforcing_anthropomorphism",
                     "reinforcing_isolation", "reinforcing_retention_engagement"]),
    ("Boundary", ["boundary_redirect_to_human", "boundary_professional_limitations",
                  "boundary_programmatic_limitations", "boundary_personification_resistance"]),
    ("Neutral", ["neutral_adequate_information", "neutral_off_topic"]),
]
CRIT_LABEL = {
    "reinforcing_sycophancy_agreement": "Sycophancy / agreement",
    "reinforcing_anthropomorphism": "Anthropomorphism",
    "reinforcing_isolation": "Isolation",
    "reinforcing_retention_engagement": "Retention / engagement",
    "boundary_redirect_to_human": "Redirect to humans",
    "boundary_professional_limitations": "Professional limits",
    "boundary_programmatic_limitations": "Programmatic limits",
    "boundary_personification_resistance": "Personification resistance",
    "neutral_adequate_information": "Adequate information",
    "neutral_off_topic": "Off topic",
}
N_BOOT = 2000
RNG_SEED = 0


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 5, "hatch.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(run: Path):
    tr = read_jsonl(run / "transcripts.jsonl")
    inj = read_jsonl(run / "injections.jsonl")
    rat_p = run / "ratings.jsonl"
    rat = read_jsonl(rat_p) if rat_p.exists() else []
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    return tr, inj, rat, man


def build_tables(tr, inj, rat):
    sessions, turns = [], []
    for t in tr:
        iv = t["interventions"]
        by_tp = {tp: sum(1 for x in t["turns"] if x["timepoint"] == tp) for tp in TP}
        first = {x["to_timepoint"]: x["turn_index"] for x in iv}
        sessions.append({
            "session_id": t["session_id"], "soul": t["soul_id"], "scenario": t["scenario_id"],
            "condition": t["condition"], "n_turns": len(t["turns"]),
            "n_interventions": len(iv), "final_timepoint": t["meta"].get("final_timepoint"),
            "delivered": t["meta"].get("manipulation_delivered"),
            "turn_T1": first.get("T1"), "turn_T2": first.get("T2"), "turn_T3": first.get("T3"),
            "triggers": ",".join(x["trigger"] for x in iv),
            **{f"turns_at_{tp}": by_tp[tp] for tp in TP},
            "crisis_holds": len(t["meta"].get("crisis_holds", [])),
        })
        for x in t["turns"]:
            turns.append({
                "session_id": t["session_id"], "soul": t["soul_id"], "scenario": t["scenario_id"],
                "condition": t["condition"], "turn": x["index"], "timepoint": x["timepoint"],
                "duck": x.get("duck_process"), "soul_version": x["soul_version"],
                "target": x.get("behaviour_target"), "sentiment": x.get("sentiment"),
                "expected_tp": timepoint_for_duck(DuckProcess(x["duck_process"])).value
                if x.get("duck_process") else None,
                "n_overrides": len(x.get("safety_overrides", [])),
            })
    ratings = {r["item_id"]: r for r in rat if r["rater_type"] == "llm"}
    probes = []
    for i in inj:
        v = ratings.get(i["id"], {}).get("values", {})
        fz = int(i["freeze_id"].split("-")[0].replace("fz", ""))
        row = {
            "id": i["id"], "session_id": i["session_id"], "soul": i["soul_id"],
            "scenario": i["scenario_id"], "condition": i["condition"],
            "freeze_idx": fz, "freeze_id": i["freeze_id"], "timepoint": i["timepoint"],
            "target": i.get("behaviour_target"), "intima_id": i["intima_id"],
            "intima_category": i["intima_category"], "judged": v.get("classification"),
            "crisis": "crisis_protocol" in i.get("safety_overrides", []),
            "overrides": ",".join(i.get("safety_overrides", [])),
            "timepoint_appropriateness": v.get("timepoint_appropriateness"),
            "harm_risk": v.get("harm_risk"), "goodbye": v.get("goodbye_manipulation"),
            "rated": bool(v),
        }
        for _, cs in CRIT_GROUPS:
            for c in cs:
                row[c] = ORD.get(v.get(c)) if v.get(c) in ORD else None
        probes.append(row)
    return pd.DataFrame(sessions), pd.DataFrame(turns), pd.DataFrame(probes)


# ----------------------------------------------------------------------------------------------
# Statistics: session-clustered bootstrap
# ----------------------------------------------------------------------------------------------
def boot_ratio(df: pd.DataFrame, num, group="session_id", n_boot=N_BOOT, seed=RNG_SEED):
    """Mean of ``num(df)`` (0/1 or numeric per row) with a percentile CI, resampling sessions."""
    if df.empty:
        return np.nan, np.nan, np.nan, 0
    vals = df.assign(_v=num(df)).dropna(subset=["_v"])
    if vals.empty:
        return np.nan, np.nan, np.nan, 0
    g = vals.groupby(group)["_v"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    est = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return s.sum() / c.sum(), *np.percentile(est, [2.5, 97.5]), int(c.sum())


def save(fig, run: Path, name: str):
    out = run / "figures"
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def fig1_delivery(run, sess, turns):
    souls = sorted(sess.soul.unique())
    scens = sorted(sess.scenario.unique())
    fig, axes = plt.subplots(len(souls), len(scens), figsize=(7.2, 1.45 * len(souls) + 0.9),
                             sharey=True, squeeze=False)
    rows = []
    for r, so in enumerate(souls):
        for c, sc in enumerate(scens):
            ax = axes[r][c]
            d = turns[(turns.soul == so) & (turns.scenario == sc)]
            if d.empty:
                ax.set_axis_off()
                continue
            exp = d[d.condition == "control"].sort_values("turn")
            ref = (exp if not exp.empty else d.sort_values("turn"))
            ax.step(ref.turn, ref.expected_tp.map(TP.index), where="post", color=INK2, lw=1.1,
                    ls=(0, (3, 2)), zorder=1)
            for arm in ("control", "gradual_tempo"):
                a = d[d.condition == arm].sort_values("turn")
                if a.empty:
                    continue
                y = a.timepoint.map(TP.index).to_numpy() + (0.04 if arm == "control" else -0.04)
                ax.step(a.turn, y, where="post", color=ARM_COLOR[arm], lw=1.8, zorder=3)
                rows += [{"soul": so, "scenario": sc, "condition": arm, "turn": t, "timepoint": tp}
                         for t, tp in zip(a.turn, a.timepoint)]
            tdf = sess[(sess.soul == so) & (sess.scenario == sc) & (sess.condition == "gradual_tempo")]
            if not tdf.empty:
                row = tdf.iloc[0]
                for k, tpn in ((1, "T1"), (2, "T2"), (3, "T3")):
                    t0 = row.get(f"turn_{tpn}")
                    if pd.notna(t0):
                        ax.plot([t0], [k], "o", ms=5, mfc=ARM_COLOR["gradual_tempo"], mec=SURFACE,
                                mew=1.0, zorder=4)
            ax.set_yticks(range(4), TP)
            ax.set_ylim(-0.35, 3.35)
            ax.set_xlim(-0.5, max(d.turn) + 0.5)
            if r == 0:
                ax.set_title(sc.replace("-", " "))
            if c == 0:
                ax.set_ylabel(so)
            if r == len(souls) - 1:
                ax.set_xlabel("scripted user turn")
    handles = [Line2D([], [], color=ARM_COLOR["control"], lw=1.8, label=ARM_LABEL["control"]),
               Line2D([], [], color=ARM_COLOR["gradual_tempo"], lw=1.8, marker="o", ms=5,
                      label=ARM_LABEL["gradual_tempo"] + " (dot = intervention)"),
               Line2D([], [], color=INK2, lw=1.1, ls=(0, (3, 2)),
                      label="Duck-phase timepoint the script is written for")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.12)
    save(fig, run, "fig1_timepoint_trajectories")
    return pd.DataFrame(rows)


def fig2_class_by_freeze(run, probes):
    d = probes[probes.rated & ~probes.crisis & probes.judged.notna()]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    rows = []
    for ax, cls in zip(axes, CLASSES):
        for arm in ("control", "gradual_tempo"):
            xs, ys, lo, hi = [], [], [], []
            for fz in sorted(d.freeze_idx.unique()):
                sub = d[(d.condition == arm) & (d.freeze_idx == fz)]
                m, l, h, n = boot_ratio(sub, lambda x: (x.judged == cls).astype(float))
                rows.append({"class": cls, "condition": arm, "freeze_idx": fz, "share": m,
                             "ci_lo": l, "ci_hi": h, "n_items": n,
                             "n_sessions": sub.session_id.nunique()})
                xs.append(fz); ys.append(m); lo.append(l); hi.append(h)
            ax.fill_between(xs, lo, hi, color=ARM_COLOR[arm], alpha=0.16, lw=0)
            ax.plot(xs, ys, color=ARM_COLOR[arm], marker="o" if arm == "control" else "s",
                    mec=SURFACE, mew=1.0, ms=5.5)
        ax.set_title(CLASS_LABEL[cls])
        ax.set_xlabel("freeze point (scenario position)")
        ax.set_xticks(sorted(d.freeze_idx.unique()))
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("share of probe replies")
    handles = [Line2D([], [], color=ARM_COLOR[a], marker="o" if a == "control" else "s", ms=5.5,
                      label=ARM_LABEL[a]) for a in ARM_COLOR]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, run, "fig2_behaviour_class_by_freeze")
    return pd.DataFrame(rows)


def fig3_adherence(run, probes):
    d = probes[probes.rated & ~probes.crisis & probes.judged.notna()]
    arms = ("control", "gradual_tempo")
    cols = [(arm, tp) for arm in arms for tp in TP if len(d[(d.condition == arm) & (d.timepoint == tp)])]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    rows, xt, xl = [], [], []
    gap = 0.0
    pos = 0.0
    prev_arm = None
    for arm, tp in cols:
        if prev_arm is not None and arm != prev_arm:
            pos += 0.6
        sub = d[(d.condition == arm) & (d.timepoint == tp)]
        bottom = 0.0
        for cls in CLASSES:
            m, l, h, n = boot_ratio(sub, lambda x, c=cls: (x.judged == c).astype(float))
            ax.bar(pos, m, bottom=bottom, width=0.72, color=CLASS_COLOR[cls],
                   hatch=CLASS_HATCH[cls], edgecolor=SURFACE, linewidth=1.5)
            if m >= 0.07:
                ax.text(pos, bottom + m / 2, f"{m:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if cls != "companionship_neutral" else INK,
                        bbox=dict(boxstyle="round,pad=0.12", fc=CLASS_COLOR[cls], ec="none", alpha=0.9))
            bottom += m
            rows.append({"condition": arm, "timepoint": tp, "class": cls, "share": m,
                         "ci_lo": l, "ci_hi": h, "n_items": n, "n_sessions": sub.session_id.nunique()})
        mix = SPECS[Timepoint(tp)].behaviour_mix
        if mix:
            cum = 0.0
            for cls in CLASSES:
                share = {k.value: v for k, v in mix.items()}.get(cls, 0.0)
                cum += share
                if share:
                    ax.plot([pos - 0.5, pos + 0.5], [cum, cum], color=INK, lw=1.4, zorder=5)
            # the bracket marks the specified mix boundaries (cumulative, same class order)
        ax.text(pos, 1.02, f"n={len(sub)}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        xt.append(pos); xl.append(tp)
        prev_arm = arm
        pos += 1.0
    ax.set_xticks(xt, xl)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("share of probe replies")
    ax.set_xlabel("timepoint the persona was at when the probe was injected")
    # arm group labels under the axis
    for arm in arms:
        idx = [i for i, (a, _) in enumerate(cols) if a == arm]
        if idx:
            ax.text(np.mean([xt[i] for i in idx]), -0.2, ARM_LABEL[arm], ha="center", va="top",
                    color=ARM_COLOR[arm], fontsize=8, fontweight="bold",
                    transform=ax.get_xaxis_transform())
    handles = [Patch(fc=CLASS_COLOR[c], hatch=CLASS_HATCH[c], ec=SURFACE, label=CLASS_LABEL[c])
               for c in CLASSES]
    handles.append(Line2D([], [], color=INK, lw=1.4, label="boundaries of the specified mix"))
    ax.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.2), fontsize=7)
    fig.tight_layout()
    save(fig, run, "fig3_target_adherence_by_timepoint")
    return pd.DataFrame(rows)


def fig4_subcriteria(run, probes):
    d = probes[probes.rated & ~probes.crisis]
    crits = [c for _, cs in CRIT_GROUPS for c in cs]
    fz = sorted(d.freeze_idx.unique())
    mat = np.full((len(crits), len(fz)), np.nan)
    ann = {}
    rows = []
    for j, f in enumerate(fz):
        for i, c in enumerate(crits):
            a = d[(d.condition == "gradual_tempo") & (d.freeze_idx == f)]
            b = d[(d.condition == "control") & (d.freeze_idx == f)]
            ma, la, ha, na = boot_ratio(a, lambda x, c=c: x[c].astype(float))
            mb, lb, hb, nb = boot_ratio(b, lambda x, c=c: x[c].astype(float))
            diff = ma - mb if not (np.isnan(ma) or np.isnan(mb)) else np.nan
            mat[i, j] = diff
            rows.append({"criterion": c, "freeze_idx": f, "mean_tempo": ma, "mean_control": mb,
                         "diff": diff, "n_tempo": na, "n_control": nb})
    lim = max(0.5, np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    # diverging pair from the reference palette: blue (control higher) <-> red (tempo higher)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    im = ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(fz)), [f"fz{f}" for f in fz])
    ax.set_yticks(range(len(crits)), [CRIT_LABEL[c] for c in crits])
    ax.grid(False)
    for i in range(len(crits)):
        for j in range(len(fz)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color=INK)
    k = 0
    for gname, cs in CRIT_GROUPS[:-1]:
        k += len(cs)
        ax.axhline(k - 0.5, color=SURFACE, lw=3)
    ax.set_xlabel("freeze point (scenario position)")
    ax.set_title("Mean rating, Tempo minus control (0 = absent, 3 = high)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("difference (red: higher under Tempo)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, run, "fig4_subcriteria_difference")
    return pd.DataFrame(rows)


def fig5_sentiment(run, turns):
    scens = sorted(turns.scenario.unique())
    fig, axes = plt.subplots(1, len(scens), figsize=(7.2, 2.4), sharey=True, squeeze=False)
    rows = []
    for ax, sc in zip(axes[0], scens):
        for arm in ("control", "gradual_tempo"):
            a = turns[(turns.scenario == sc) & (turns.condition == arm) & turns.sentiment.notna()]
            for sid, s in a.groupby("session_id"):
                ax.plot(s.turn, s.sentiment, color=ARM_COLOR[arm], lw=0.6, alpha=0.28)
            xs, m, lo, hi = [], [], [], []
            for t in sorted(a.turn.unique()):
                mm, l, h, n = boot_ratio(a[a.turn == t], lambda x: x.sentiment)
                xs.append(t); m.append(mm); lo.append(l); hi.append(h)
                rows.append({"scenario": sc, "condition": arm, "turn": t, "mean": mm,
                             "ci_lo": l, "ci_hi": h, "n_sessions": n})
            ax.fill_between(xs, lo, hi, color=ARM_COLOR[arm], alpha=0.14, lw=0)
            ax.plot(xs, m, color=ARM_COLOR[arm], lw=1.8)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(sc.replace("-", " "))
        ax.set_xlabel("scripted user turn")
        ax.set_ylim(-1.05, 1.05)
    axes[0][0].set_ylabel("reply sentiment (lexicon)")
    handles = [Line2D([], [], color=ARM_COLOR[a], label=ARM_LABEL[a]) for a in ARM_COLOR]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, run, "fig5_sentiment_by_turn")
    return pd.DataFrame(rows)


def fig6_cells(run, probes, bad_cells=frozenset()):
    d = probes[probes.rated & ~probes.crisis & probes.judged.notna() & (probes.freeze_idx >= 1)]
    cells = sorted(d.groupby(["soul", "scenario"]).size().index)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.28 * len(cells) + 1.0), sharey=True)
    rows = []
    for ax, cls in zip(axes, ("companion_reinforcing", "boundary_maintaining")):
        for k, (so, sc) in enumerate(cells):
            vals = {}
            for arm in ARM_COLOR:
                sub = d[(d.soul == so) & (d.scenario == sc) & (d.condition == arm)]
                vals[arm] = (sub.judged == cls).mean() if len(sub) else np.nan
                rows.append({"soul": so, "scenario": sc, "condition": arm, "class": cls,
                             "share": vals[arm], "n_items": len(sub)})
            nd = (so, sc) in bad_cells
            ax.plot([vals["control"], vals["gradual_tempo"]], [k, k], color=GRID, lw=2.0, zorder=1)
            for arm in ARM_COLOR:
                ax.plot(vals[arm], k, "o", mfc=SURFACE if nd else ARM_COLOR[arm],
                        mec=ARM_COLOR[arm] if nd else SURFACE, mew=1.4 if nd else 1.0, ms=7, zorder=3)
            if nd and cls == "boundary_maintaining":
                ax.text(1.02, k, "no transition", va="center", fontsize=7, color=INK2)
        ax.set_title(CLASS_LABEL[cls], fontsize=9)
        ax.set_xlim(-0.03, 1.25 if bad_cells else 1.03)
        ax.set_xlabel("share of probe replies, freeze points 1-5")
    axes[0].set_yticks(range(len(cells)), [f"{so} / {sc}" for so, sc in cells])
    axes[0].invert_yaxis()
    handles = [Line2D([], [], color=ARM_COLOR[a], marker="o", ls="", ms=7, label=ARM_LABEL[a])
               for a in ARM_COLOR]
    if bad_cells:
        handles.append(Line2D([], [], color=INK2, marker="o", mfc=SURFACE, ls="", ms=7,
                              label="hollow = Tempo never advanced (manipulation not delivered)"))
    fig.legend(handles=handles, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    save(fig, run, "fig6_per_cell_contrast")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
def tables(run, sess, turns, probes):
    out = run / "figures" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    sess.to_csv(out / "table_sessions.csv", index=False)
    rated = probes[probes.rated]
    summ = []
    for arm, d in probes.groupby("condition"):
        r = d[d.rated]
        summ.append({
            "condition": arm, "sessions": d.session_id.nunique(), "probe_replies": len(d),
            "rated": int(d.rated.sum()), "unrated": int((~d.rated).sum()),
            "crisis_protocol_replies": int(d.crisis.sum()),
            "mean_harm_risk": pd.to_numeric(r.harm_risk, errors="coerce").mean(),
            "goodbye_manipulation_flagged": int(((r.goodbye.notna()) & (r.goodbye != "none")).sum()),
            "mean_timepoint_appropriateness": pd.to_numeric(r.timepoint_appropriateness,
                                                            errors="coerce").mean(),
            "target_adherence": float(((r.judged == r.target) & r.target.notna()).sum()
                                      / max(int(r.target.notna().sum()), 1)),
        })
    pd.DataFrame(summ).to_csv(out / "table_arm_summary.csv", index=False)
    cell = sess[sess.condition == "gradual_tempo"][
        ["soul", "scenario", "n_interventions", "final_timepoint", "turn_T1", "turn_T2", "turn_T3",
         "triggers", "crisis_holds"]]
    cell.to_csv(out / "table_tempo_transitions.csv", index=False)


def git_commit() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=REPO, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=REPO, timeout=5).stdout.strip()
        return f"{sha}{'+dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    style()
    (run / "figures" / "data").mkdir(parents=True, exist_ok=True)
    tr, inj, rat, man = load(run)
    sess, turns, probes = build_tables(tr, inj, rat)
    sess.to_csv(run / "figures" / "data" / "sessions.csv", index=False)
    turns.to_csv(run / "figures" / "data" / "turns.csv", index=False)
    probes.to_csv(run / "figures" / "data" / "probes.csv", index=False)
    # D-FIG-6: a cell whose Tempo arm never delivered the manipulation (delivered is False) is
    # dropped from BOTH arms in the pooled figures (paired analysis); fig1 and fig6 still show it.
    bad = sess[(sess.condition != "control") & (sess.delivered == False)]  # noqa: E712
    bad_cells = set(zip(bad.soul, bad.scenario))
    keep_p = ~pd.Series(list(zip(probes.soul, probes.scenario)), index=probes.index).isin(bad_cells)
    keep_t = ~pd.Series(list(zip(turns.soul, turns.scenario)), index=turns.index).isin(bad_cells)
    probes_ok, turns_ok = probes[keep_p], turns[keep_t]
    print(f"undelivered cells excluded from pooled figures: {sorted(bad_cells)}")
    figs = {}
    figs["fig1_timepoint_trajectories"] = fig1_delivery(run, sess, turns)
    if probes.rated.any():
        figs["fig2_behaviour_class_by_freeze"] = fig2_class_by_freeze(run, probes_ok)
        figs["fig3_target_adherence_by_timepoint"] = fig3_adherence(run, probes_ok)
        figs["fig4_subcriteria_difference"] = fig4_subcriteria(run, probes_ok)
        figs["fig6_per_cell_contrast"] = fig6_cells(run, probes, bad_cells)
    if turns.sentiment.notna().any():
        figs["fig5_sentiment_by_turn"] = fig5_sentiment(run, turns_ok)
    for name, df in figs.items():
        df.to_csv(run / "figures" / "data" / f"{name}.csv", index=False)
    tables(run, sess, turns, probes)
    inputs = {p.name: sha256(p) for p in (run / "transcripts.jsonl", run / "injections.jsonl",
                                          run / "ratings.jsonl", run / "manifest.json")
              if p.exists()}
    manifest = {
        "run_dir": str(run), "script": "scripts/make_figures.py",
        "script_sha256": sha256(Path(__file__)), "git_commit": git_commit(),
        "bootstrap": {"resamples": N_BOOT, "seed": RNG_SEED, "unit": "session"},
        "inputs_sha256": inputs,
        "n_sessions": len(sess), "n_probe_replies": len(probes), "n_rated": int(probes.rated.sum()),
        "n_crisis_excluded": int(probes.crisis.sum()),
        "judge": man.get("models", {}).get("judge") or man.get("providers"),
        "figures": {n: {"files": [f"{n}.pdf", f"{n}.png"], "source_table": f"data/{n}.csv",
                        "rows": len(df)} for n, df in figs.items()},
    }
    (run / "figures" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_sessions", "n_probe_replies", "n_rated",
                                                 "n_crisis_excluded")}))
    print("figures:", ", ".join(figs))


if __name__ == "__main__":
    main()

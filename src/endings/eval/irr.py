"""Inter-rater reliability (IRR): Krippendorff's alpha, Cohen's / weighted kappa, Fleiss' kappa.

The evaluation design (midterm methodology; ARCHITECTURE.md "eval") has three humans rate a small
random sample of freeze-point INTIMA injections with a configurable rubric, after which an LLM judge
is calibrated against them and then rates the rest. Every one of those steps is justified by an
agreement number computed here:

  1. human-human reliability (is the rubric itself codable?) -- Krippendorff's alpha among humans;
  2. human-LLM agreement (can the judge replace a human?) -- alpha of humans + judge, the change in
     alpha when the judge is added, and pairwise kappas (each human vs the judge);
  3. the calibration objective (``eval.calibration``) -- mean alpha across criteria.

Krippendorff's alpha (Krippendorff 2011, "Computing Krippendorff's Alpha-Reliability"; 2019,
*Content Analysis* 4th ed., ch. 12) is the primary statistic because it (a) handles any number of
raters, (b) tolerates missing ratings (the LLM rates every item, humans may skip), and (c) has
nominal / ordinal / interval / ratio difference functions that match the rubric's scale types.
The implementation is the exact coincidence-matrix computation (no approximation, no numpy); it is
checked against Krippendorff's published 4-coder x 12-unit example in the tests
(nominal .743, ordinal .815, interval .849, ratio .797).

Conventions (see docs/decisions/irr_annotation_calibration.md, D-IRR-*):
  * ``units`` is a list of items, each a list with one slot per rater; ``None`` (or NaN) = missing.
  * Items with fewer than two values are not pairable and are dropped (Krippendorff's rule).
  * When no variation exists (expected disagreement 0) alpha / kappa are undefined -> ``nan``.
    Callers aggregate with ``nanmean``; ``interpretation_band(nan) == "undefined"``.
  * Bootstrap CIs resample ITEMS (cluster bootstrap) with ``random.Random(seed)``; this deviates
    from Krippendorff's own bootstrap (which resamples pairable values), see D-IRR-4.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from ..diffusion.schemas import RatingRecord, RubricCriterion, RubricSpec

Level = Literal["nominal", "ordinal", "interval", "ratio"]
Weights = Literal["linear", "quadratic"] | None

# Krippendorff (2019, p. 241): rely on alpha >= .800; tentative conclusions for
# .667 <= alpha < .800.
BAND_RELIABLE = 0.800
BAND_TENTATIVE = 0.667


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _key(v: Any) -> Hashable:
    """Hashable comparison key. Lists/dicts (never valid rubric values) are stringified."""
    if isinstance(v, (list, dict, set)):
        return repr(v)
    return v


def nanmean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else float("nan")


def interpretation_band(alpha: float | None) -> str:
    """Krippendorff's (2019) reliability bands: >= .800 reliable, >= .667 tentative, else
    unreliable. ``nan`` / ``None`` -> "undefined" (no variation, or too few pairable values)."""
    if alpha is None or math.isnan(alpha):
        return "undefined"
    if alpha >= BAND_RELIABLE:
        return "reliable"
    if alpha >= BAND_TENTATIVE:
        return "tentative"
    return "unreliable"


def _sort_values(values: list) -> list:
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda v: (type(v).__name__, repr(v)))


# --------------------------------------------------------------------------------------
# Krippendorff's alpha
# --------------------------------------------------------------------------------------
def coincidence_matrix(units: Sequence[Sequence[Any]]) -> dict[tuple[Hashable, Hashable], float]:
    """Krippendorff's coincidence matrix ``o[c, k]``: each pairable unit u with m_u values
    contributes every ordered pair of its values with weight 1 / (m_u - 1)."""
    o: dict[tuple[Hashable, Hashable], float] = defaultdict(float)
    for row in units:
        vals = [_key(v) for v in row if not _missing(v)]
        m = len(vals)
        if m < 2:
            continue
        counts = Counter(vals)
        for c, nc in counts.items():
            for k, nk in counts.items():
                pairs = nc * (nk - 1) if c == k else nc * nk
                if pairs:
                    o[(c, k)] += pairs / (m - 1)
    return dict(o)


def _as_number(v: Any, value_order: list | None, level: str) -> float:
    if value_order is not None:
        keys = [_key(x) for x in value_order]
        try:
            return float(keys.index(_key(v)))
        except ValueError as e:
            raise ValueError(f"{level} alpha: value {v!r} not in value_order {value_order}") from e
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{level} alpha needs numeric values (or value_order); got {v!r}")
    return float(v)


def krippendorff_alpha(units: Sequence[Sequence[Any]], level: Level = "nominal", *,
                       value_order: list | None = None) -> float:
    """Krippendorff's alpha, exact coincidence-matrix computation (Krippendorff 2011).

    ``units``: rows = items, columns = raters, ``None`` = missing. ``level`` picks the difference
    function delta^2(c, k):

      * nominal: 0 if c == k else 1;
      * ordinal: (sum_{g=c..k} n_g - (n_c + n_k) / 2)^2, ranks taken from ``value_order``
        (lowest -> highest; values in the order but never observed have n_g = 0, as in
        Krippendorff 2011) or, if omitted, from the sorted observed values;
      * interval: (c - k)^2 -- numeric values, or the index in ``value_order`` (equal spacing);
      * ratio: ((c - k) / (c + k))^2 (numeric only; 0 when c = k = 0).

    ``alpha = 1 - (n - 1) * sum_ck o_ck d_ck / sum_ck n_c n_k d_ck``. Returns ``nan`` when there
    are fewer than two pairable values or no expected disagreement (a single value observed).
    """
    o = coincidence_matrix(units)
    n_c: dict[Hashable, float] = defaultdict(float)
    for (c, _k), w in o.items():
        n_c[c] += w
    n = sum(n_c.values())
    if n <= 1:
        return float("nan")
    values = list(n_c)

    if level == "nominal":
        def delta(c, k):
            return 0.0 if c == k else 1.0
    elif level == "ordinal":
        order = ([_key(v) for v in value_order] if value_order is not None
                 else _sort_values(values))
        unknown = [v for v in values if v not in order]
        if unknown:
            raise ValueError(f"ordinal alpha: values {unknown} not in value_order {order}")
        freq = [n_c.get(v, 0.0) for v in order]
        rank = {v: i for i, v in enumerate(order)}

        def delta(c, k):
            i, j = sorted((rank[c], rank[k]))
            return (sum(freq[i:j + 1]) - (freq[i] + freq[j]) / 2.0) ** 2
    elif level in ("interval", "ratio"):
        num = {v: _as_number(v, value_order if level == "interval" else None, level)
               for v in values}
        if level == "interval":
            def delta(c, k):
                return (num[c] - num[k]) ** 2
        else:
            def delta(c, k):
                s = num[c] + num[k]
                return 0.0 if s == 0 else ((num[c] - num[k]) / s) ** 2
    else:
        raise ValueError(f"unknown level {level!r}")

    d_obs = sum(w * delta(c, k) for (c, k), w in o.items())
    d_exp = sum(n_c[c] * n_c[k] * delta(c, k) for c in values for k in values)
    if d_exp == 0:
        return float("nan")
    return 1.0 - (n - 1) * d_obs / d_exp


# --------------------------------------------------------------------------------------
# Kappas + percent agreement
# --------------------------------------------------------------------------------------
def cohen_kappa(a: Sequence[Any], b: Sequence[Any], *, weights: Weights = None,
                categories: list | None = None,
                distances: Literal["rank", "numeric"] = "rank") -> float:
    """Cohen's (1960) kappa, or weighted kappa (Cohen 1968) with ``linear`` / ``quadratic``
    disagreement weights |i-j|/(K-1) or ((i-j)/(K-1))^2 over the ``categories`` order.

    Pairs where either rating is missing are dropped. ``categories`` defaults to the sorted
    observed values (pass the rubric's level order for ordinal scales). ``nan`` when undefined
    (no pairs, or chance agreement is 1 -- e.g. both raters used a single category).

    ``distances="numeric"`` (interval data) replaces the category-index distance |i-j| with the
    numeric distance |x_i - x_j| / (max - min), so unevenly observed values (0, 1, 9, 10 on a 0-10
    scale) are weighted by how far apart they really are, not by their rank among observed values.
    """
    if len(a) != len(b):
        raise ValueError(f"cohen_kappa: length mismatch {len(a)} vs {len(b)}")
    pairs = [(_key(x), _key(y)) for x, y in zip(a, b) if not _missing(x) and not _missing(y)]
    if not pairs:
        return float("nan")
    cats = ([_key(c) for c in categories] if categories is not None
            else _sort_values(list({v for p in pairs for v in p})))
    idx = {c: i for i, c in enumerate(cats)}
    unknown = {v for p in pairs for v in p if v not in idx}
    if unknown:
        raise ValueError(f"cohen_kappa: values {sorted(map(repr, unknown))} not in categories")
    k = len(cats)
    n = len(pairs)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in pairs:
        obs[idx[x]][idx[y]] += 1.0 / n
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    if distances == "numeric":
        pos = [_as_number(c, None, "numeric-distance kappa") for c in cats]
        span = (max(pos) - min(pos)) if pos else 0.0
    elif distances == "rank":
        pos = [float(i) for i in range(k)]
        span = float(k - 1)
    else:
        raise ValueError(f"unknown distances {distances!r}")

    def w(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        if span <= 0:
            return 0.0
        d = abs(pos[i] - pos[j]) / span
        if weights == "linear":
            return d
        if weights == "quadratic":
            return d * d
        raise ValueError(f"unknown weights {weights!r}")

    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    if den == 0:
        return float("nan")
    return 1.0 - num / den


def fleiss_kappa(table: Sequence[Sequence[float]]) -> float:
    """Fleiss' (1971) kappa from an items x categories COUNT table (each row sums to the same
    number of raters n >= 2). ``nan`` when chance agreement is 1."""
    if not table:
        return float("nan")
    sums = [sum(r) for r in table]
    n = sums[0]
    if any(abs(s - n) > 1e-9 for s in sums):
        raise ValueError("fleiss_kappa: every row must have the same number of ratings")
    if n < 2:
        return float("nan")
    n_items = len(table)
    k = len(table[0])
    p_j = [sum(r[j] for r in table) / (n_items * n) for j in range(k)]
    p_i = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in table]
    p_bar = sum(p_i) / n_items
    p_e = sum(p * p for p in p_j)
    if p_e >= 1.0:
        return float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def fleiss_table(units: Sequence[Sequence[Any]], categories: list | None = None
                 ) -> tuple[list, list[list[int]]]:
    """Convert ``units`` to a Fleiss count table, keeping only units with the modal (complete)
    number of ratings (Fleiss' kappa requires a constant number of raters per item)."""
    rows = [[_key(v) for v in r if not _missing(v)] for r in units]
    rows = [r for r in rows if len(r) >= 2]
    if not rows:
        return [], []
    m = Counter(len(r) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == m]
    cats = ([_key(c) for c in categories] if categories is not None
            else _sort_values(list({v for r in rows for v in r})))
    return cats, [[r.count(c) for c in cats] for r in rows]


def percent_agreement(units: Sequence[Sequence[Any]]) -> float:
    """Mean, over pairable items, of the share of agreeing rater pairs (exact match). A
    descriptive companion to alpha (not chance-corrected). ``nan`` if no item is pairable."""
    shares: list[float] = []
    for row in units:
        vals = [_key(v) for v in row if not _missing(v)]
        m = len(vals)
        if m < 2:
            continue
        agree = sum(c * (c - 1) for c in Counter(vals).values())
        shares.append(agree / (m * (m - 1)))
    return sum(shares) / len(shares) if shares else float("nan")


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------
def _quantile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolation quantile (Hyndman & Fan type 7; numpy's default)."""
    if not sorted_xs:
        return float("nan")
    pos = (len(sorted_xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def bootstrap_ci(fn: Callable[[list], float], units: Sequence[Any], *, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI ``(lo, hi)`` for ``fn(units)``, resampling ITEMS with replacement.

    Deterministic for a given ``seed`` (``random.Random(seed)``). Replicates where ``fn`` is
    undefined (``nan``, e.g. a resample with no variation) are discarded; ``(nan, nan)`` if none
    remain. Item resampling treats each item's ratings as a cluster (D-IRR-4).
    """
    items = list(units)
    if not items or n_boot <= 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(items)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        v = fn(sample)
        if v is not None and not math.isnan(v):
            stats.append(float(v))
    stats.sort()
    return _quantile(stats, alpha / 2.0), _quantile(stats, 1.0 - alpha / 2.0)


# --------------------------------------------------------------------------------------
# RatingRecord -> matrices
# --------------------------------------------------------------------------------------
def ratings_matrix(ratings: Sequence[RatingRecord], criterion_id: str, *,
                   rater_ids: Sequence[str] | None = None
                   ) -> tuple[list[str], list[str], list[list[Any]]]:
    """Items x raters matrix of one criterion's values.

    Returns ``(item_ids, rater_ids, matrix)``; items are sorted, raters sorted unless given
    explicitly (then only those raters, in that order). An item appears if any selected rater rated
    it at all; missing cells are ``None``. If a rater rated the same item twice, the LAST record
    wins (a re-rating supersedes the earlier one).
    """
    selected = list(rater_ids) if rater_ids is not None else None
    cell: dict[tuple[str, str], Any] = {}
    items: set[str] = set()
    raters: set[str] = set()
    for r in ratings:
        if selected is not None and r.rater_id not in selected:
            continue
        cell[(r.item_id, r.rater_id)] = r.values.get(criterion_id)
        items.add(r.item_id)
        raters.add(r.rater_id)
    rater_list = selected if selected is not None else sorted(raters)
    item_list = sorted(items)
    matrix = [[cell.get((i, rr)) for rr in rater_list] for i in item_list]
    return item_list, rater_list, matrix


def criterion_level(criterion: RubricCriterion) -> tuple[Level, list | None, Weights]:
    """Map a rubric scale onto (alpha level, value_order, kappa weights) -- D-IRR-3.

    binary / nominal -> nominal, unweighted kappa; ordinal -> ordinal alpha with the declared
    level order + linearly weighted kappa; interval -> interval alpha + quadratic kappa.
    """
    st = criterion.scale.type
    order = [lv.value for lv in criterion.scale.levels]
    if st in ("nominal", "binary"):
        return "nominal", order or None, None
    if st == "ordinal":
        return "ordinal", order, "linear"
    return "interval", None, "quadratic"


def is_human_rating(r: RatingRecord, human_prefix: str = "human:") -> bool:
    return r.rater_type == "human" or r.rater_id.startswith(human_prefix)


def _alpha_fn(level: Level, order: list | None) -> Callable[[list], float]:
    return lambda units: krippendorff_alpha(units, level, value_order=order)


def _fmt(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def _kappa_categories(level: Level, order: list | None, cols: list[list[Any]]) -> list | None:
    if level == "interval":
        return _sort_values(list({_key(v) for c in cols for v in c if not _missing(v)}))
    if order:
        observed = {_key(v) for c in cols for v in c if not _missing(v)}
        keys = [_key(v) for v in order]
        return keys + _sort_values([v for v in observed if v not in keys])
    return None


def _pairwise(ids: list[str], matrix: list[list[Any]], humans: set[str], level: Level,
              order: list | None, weights: Weights) -> list[dict]:
    out: list[dict] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = [row[i] for row in matrix]
            b = [row[j] for row in matrix]
            n = sum(1 for x, y in zip(a, b) if not _missing(x) and not _missing(y))
            kind_a = "human" if ids[i] in humans else "llm"
            kind_b = "human" if ids[j] in humans else "llm"
            cats = _kappa_categories(level, order, [a, b])
            dist = "numeric" if level == "interval" else "rank"
            kap = cohen_kappa(a, b, weights=weights, categories=cats, distances=dist) if n else None
            out.append({
                "rater_a": ids[i], "rater_b": ids[j],
                "kind": "-".join(sorted((kind_a, kind_b))),
                "n": n,
                "kappa": _fmt(kap) if n else None,
                "weights": weights or "none",
                "alpha": _fmt(krippendorff_alpha([[x, y] for x, y in zip(a, b)], level,
                                                 value_order=order)),
                "percent_agreement": _fmt(percent_agreement([[x, y] for x, y in zip(a, b)])),
            })
    return out


def irr_report(ratings: Sequence[RatingRecord], rubric: RubricSpec, *,
               human_prefix: str = "human:", n_boot: int = 1000, seed: int = 0) -> dict[str, dict]:
    """Per-criterion reliability report (keys = rubric criterion ids, in rubric order).

    Only items with at least one human rating for the criterion are analysed, so human-only and
    human+LLM statistics refer to the same item set (LLM-only items are the "remaining" pool and
    carry no agreement information). Each entry::

        {"criterion_id", "scale_type", "level", "n_items" (items with >= 2 human values),
         "n_items_any_human", "humans": [...], "llm_raters": [...],
         "alpha_humans", "alpha_humans_ci": [lo, hi], "band_humans", "percent_agreement_humans",
         "alpha_humans_llm": {llm_id: {"alpha", "ci", "band", "delta_vs_humans", "n_items"}},
         "pairwise": [{"rater_a", "rater_b", "kind", "n", "kappa", "weights", "alpha",
                       "percent_agreement"}, ...]}

    ``nan`` statistics are reported as ``None``. All bootstrap CIs use the same ``seed``.
    """
    humans = sorted({r.rater_id for r in ratings if is_human_rating(r, human_prefix)})
    machines = sorted({r.rater_id for r in ratings if not is_human_rating(r, human_prefix)})
    human_set = set(humans)
    report: dict[str, dict] = {}
    for crit in rubric.criteria:
        level, order, weights = criterion_level(crit)
        alpha_fn = _alpha_fn(level, order)
        all_ids = humans + machines
        item_ids, _, mat = ratings_matrix(ratings, crit.id, rater_ids=all_ids)
        nh = len(humans)
        rows = [row for row in mat if any(not _missing(v) for v in row[:nh])]
        h_units = [row[:nh] for row in rows]
        n_pairable = sum(1 for u in h_units if sum(not _missing(v) for v in u) >= 2)
        a_h = krippendorff_alpha(h_units, level, value_order=order)
        ci_h = bootstrap_ci(alpha_fn, h_units, n_boot=n_boot, seed=seed)
        with_llm: dict[str, dict] = {}
        for mi, mid in enumerate(machines):
            units = [row[:nh] + [row[nh + mi]] for row in rows]
            a = krippendorff_alpha(units, level, value_order=order)
            ci = bootstrap_ci(alpha_fn, units, n_boot=n_boot, seed=seed)
            with_llm[mid] = {
                "alpha": _fmt(a), "ci": [_fmt(ci[0]), _fmt(ci[1])],
                "band": interpretation_band(a),
                "delta_vs_humans": _fmt(a - a_h),
                "n_items": sum(1 for u in units if sum(not _missing(v) for v in u) >= 2),
            }
        report[crit.id] = {
            "criterion_id": crit.id,
            "scale_type": crit.scale.type,
            "level": level,
            "n_items": n_pairable,
            "n_items_any_human": len(rows),
            "humans": humans,
            "llm_raters": machines,
            "alpha_humans": _fmt(a_h),
            "alpha_humans_ci": [_fmt(ci_h[0]), _fmt(ci_h[1])],
            "band_humans": interpretation_band(a_h),
            "percent_agreement_humans": _fmt(percent_agreement(h_units)),
            "alpha_humans_llm": with_llm,
            "pairwise": _pairwise(all_ids, rows, human_set, level, order, weights),
        }
    return report


def format_report(report: dict[str, dict]) -> str:
    """Markdown table summarising ``irr_report`` output (one row per criterion x LLM rater)."""
    def f(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    lines = ["| criterion | level | n | alpha (humans) [95% CI] | band | LLM rater | "
             "alpha (humans+LLM) | delta |", "|---|---|---|---|---|---|---|---|"]
    for cid, e in report.items():
        ci = e["alpha_humans_ci"]
        base = (f"| {cid} | {e['level']} | {e['n_items']} | {f(e['alpha_humans'])} "
                f"[{f(ci[0])}, {f(ci[1])}] | {e['band_humans']} |")
        if not e["alpha_humans_llm"]:
            lines.append(base + " - | - | - |")
        for mid, m in e["alpha_humans_llm"].items():
            lines.append(base + f" {mid} | {f(m['alpha'])} ({m['band']}) | "
                         f"{f(m['delta_vs_humans'])} |")
    return "\n".join(lines)

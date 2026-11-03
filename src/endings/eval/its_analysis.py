"""Interrupted time-series (ITS) analysis of per-turn metrics.

Segmented regression (Wagner et al. 2002):
    y_t = b0 + b1*t + b2*D_t + b3*(t - t0)*D_t
where D_t = 1 after the intervention turn t0. ``b2`` is the immediate LEVEL change and ``b3`` is the
SLOPE change after the intervention. We use this to quantify, e.g., how sharply warmth tapers or
AI-status disclosure rises once the ending enters its closure phase.

Pure-Python OLS (normal equations + Gaussian elimination) so it runs with only the core dependency;
if NumPy is installed it is used for numerical stability.

Persona-diffusion extension (``multi_segmented_regression``, ``its_from_session``,
``pooled_its``): one interruption per coach intervention that edits the soul document, OLS +
Durbin-Watson + Newey-West HAC robustness, per-session and pooled summaries. Design and assumptions:
docs/decisions/analysis.md (D-ANALYSIS-3..9). The single-interruption functions above are kept
unchanged for the earlier TTM pipeline.
"""

from __future__ import annotations

import math

from ..schemas import WithdrawalStage


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve a·x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        pivot = m[col][col]
        for r in range(n):
            if r != col:
                factor = m[r][col] / pivot
                m[r] = [m[r][k] - factor * m[col][k] for k in range(n + 1)]
    return [m[i][n] / m[i][i] for i in range(n)]


def _ols(x_rows: list[list[float]], y: list[float]) -> list[float] | None:
    """Ordinary least squares via normal equations X'X b = X'y."""
    try:
        import numpy as np  # noqa: PLC0415

        X = np.asarray(x_rows, dtype=float)
        yv = np.asarray(y, dtype=float)
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        return list(map(float, beta))
    except ImportError:
        p = len(x_rows[0])
        xtx = [[sum(x_rows[k][i] * x_rows[k][j] for k in range(len(x_rows))) for j in range(p)]
               for i in range(p)]
        xty = [sum(x_rows[k][i] * y[k] for k in range(len(x_rows))) for i in range(p)]
        return _solve(xtx, xty)


def segmented_regression(series: list[float], intervention_index: int) -> dict:
    """Fit the segmented model; return level/slope change and pre/post slopes."""
    n = len(series)
    t0 = intervention_index
    if n < 4 or t0 <= 0 or t0 >= n:
        return {"n": n, "intervention_index": t0, "fit": False,
                "reason": "need >=4 points and an interior intervention"}
    x_rows, y = [], []
    for t in range(n):
        d = 1.0 if t >= t0 else 0.0
        x_rows.append([1.0, float(t), d, (t - t0) * d])
        y.append(float(series[t]))
    beta = _ols(x_rows, y)
    if beta is None:
        return {"n": n, "intervention_index": t0, "fit": False, "reason": "singular design"}
    b0, b1, b2, b3 = beta
    return {
        "n": n, "intervention_index": t0, "fit": True,
        "intercept": round(b0, 4),
        "pre_slope": round(b1, 4),
        "level_change": round(b2, 4),
        "slope_change": round(b3, 4),
        "post_slope": round(b1 + b3, 4),
        "pre_mean": round(sum(series[:t0]) / t0, 4),
        "post_mean": round(sum(series[t0:]) / (n - t0), 4),
    }


def find_intervention_index(turns, stage: WithdrawalStage = WithdrawalStage.TAPER) -> int:
    """First turn index at/after which the conversation enters ``stage`` (default: tapering)."""
    for i, t in enumerate(turns):
        if t.stage == stage:
            return i
    return max(1, len(turns) // 2)  # fallback: midpoint


def analyze_transcript(turns, metric_series: dict[str, list[float]],
                       intervention_index: int | None = None) -> dict:
    """Run segmented regression on every metric for one transcript."""
    t0 = intervention_index if intervention_index is not None else find_intervention_index(turns)
    return {m: segmented_regression(series, t0) for m, series in metric_series.items()}


# ======================================================================================
# Multiple interruptions (persona-diffusion ITS; docs/decisions/analysis.md D-ANALYSIS-3..9)
# ======================================================================================
# The midterm methodology: "The timepoints of interest are every time the coach model makes an edit
# to the interactor model." Each coach intervention with >= 1 soul edit is one interruption, so a
# T0->T1->T2->T3 session has up to three. The model generalizes Wagner et al. (2002) to K breaks:
#
#     y_t = b0 + b1*t + sum_k [ b2k * D_kt + b3k * (t - t_k) * D_kt ],   D_kt = 1[t >= t_k]
#
# Because the D_kt are cumulative, b2k is the level change at t_k RELATIVE TO the projection of the
# preceding segment, and b3k is the slope change relative to the preceding segment's slope (not to
# the baseline). The time unit is the conversation turn (position in ``transcript.turns``).
NW_LAG_RULE = "floor(4 * (n / 100) ** (2 / 9))"  # Newey & West (1994) plug-in rule of thumb


def _inv(a: list[list[float]]) -> list[list[float]] | None:
    """Matrix inverse by Gauss-Jordan elimination with partial pivoting (None if singular)."""
    try:
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(a, dtype=float)
        if np.linalg.cond(arr) > 1e12:
            return None
        return np.linalg.inv(arr).tolist()
    except ImportError:
        pass
    n = len(a)
    scale = max((abs(v) for row in a for v in row), default=1.0) or 1.0
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-10 * scale:
            return None
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col]
                m[r] = [m[r][k] - f * m[col][k] for k in range(2 * n)]
    return [row[n:] for row in m]


def _matvec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def _xtx(x: list[list[float]]) -> list[list[float]]:
    p = len(x[0])
    return [[sum(r[i] * r[j] for r in x) for j in range(p)] for i in range(p)]


# -- p-values ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes, Lentz)."""
    tiny, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def p_value_normal(stat: float) -> float:
    """Two-sided p-value under N(0, 1)."""
    return math.erfc(abs(stat) / math.sqrt(2.0))


def p_value_t(stat: float, df: int) -> float:
    """Two-sided p-value under Student's t with ``df`` degrees of freedom (scipy if installed)."""
    try:
        from scipy import stats  # noqa: PLC0415

        return float(2.0 * stats.t.sf(abs(stat), df))
    except ImportError:
        return _betainc(df / 2.0, 0.5, df / (df + stat * stat))


# -- residual diagnostics -----------------------------------------------------------------
def durbin_watson(residuals: list[float]) -> float | None:
    """DW = sum (e_t - e_{t-1})^2 / sum e_t^2. ~2 = no lag-1 autocorrelation; <2 positive."""
    sse = sum(e * e for e in residuals)
    if len(residuals) < 2 or sse < 1e-18:
        return None
    return sum((residuals[i] - residuals[i - 1]) ** 2 for i in range(1, len(residuals))) / sse


def newey_west_lag(n: int) -> int:
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_cov(x: list[list[float]], residuals: list[float], xtx_inv: list[list[float]],
                   lag: int, *, small_sample: bool = True) -> list[list[float]]:
    """HAC covariance (Newey & West 1987) with Bartlett weights w_l = 1 - l/(lag+1).

    ``small_sample`` multiplies by n/(n-p) (the statsmodels ``use_correction`` convention).
    """
    n, p = len(x), len(x[0])
    s = [[0.0] * p for _ in range(p)]
    for t in range(n):
        e2 = residuals[t] * residuals[t]
        for i in range(p):
            for j in range(p):
                s[i][j] += e2 * x[t][i] * x[t][j]
    for lag_l in range(1, lag + 1):
        w = 1.0 - lag_l / (lag + 1.0)
        for t in range(lag_l, n):
            ee = residuals[t] * residuals[t - lag_l]
            a, b = x[t], x[t - lag_l]
            for i in range(p):
                for j in range(p):
                    s[i][j] += w * ee * (a[i] * b[j] + b[i] * a[j])
    cov = _matmul(_matmul(xtx_inv, s), xtx_inv)
    if small_sample and n > p:
        cov = [[v * n / (n - p) for v in row] for row in cov]
    return cov


# -- interruption hygiene -----------------------------------------------------------------
def clean_interruptions(n: int, interruptions: list[int],
                        min_segment: int = 2) -> tuple[list[int], list[dict]]:
    """Dedupe, sort, keep interior indices, and drop any that would leave a segment with fewer than
    ``min_segment`` points. Returns (kept, dropped) where each dropped entry records why.

    Greedy forward pass (an earlier interruption wins over a later one that is too close), then the
    tail is repaired by dropping the latest kept interruption until the final segment is long
    enough.
    """
    dropped: list[dict] = []
    seen: set[int] = set()
    cands: list[int] = []
    for raw in interruptions:
        k = int(raw)
        if k in seen:
            dropped.append({"index": k, "reason": "duplicate"})
            continue
        seen.add(k)
        cands.append(k)
    kept: list[int] = []
    prev = 0
    for k in sorted(cands):
        if k <= 0 or k >= n:
            dropped.append({"index": k, "reason": "not_interior"})
        elif k - prev < min_segment:
            dropped.append({"index": k, "reason": f"segment_before_lt_{min_segment}_points"})
        else:
            kept.append(k)
            prev = k
    while kept and n - kept[-1] < min_segment:
        dropped.append({"index": kept.pop(), "reason": f"segment_after_lt_{min_segment}_points"})
    return kept, sorted(dropped, key=lambda d: (d["index"], d["reason"]))


def _design(n: int, kept: list[int]) -> list[list[float]]:
    rows = []
    for t in range(n):
        row = [1.0, float(t)]
        for tk in kept:
            d = 1.0 if t >= tk else 0.0
            row += [d, (t - tk) * d]
        rows.append(row)
    return rows


def _coef(beta, se, hac_se, idx: int, df: int, inference: bool) -> dict:
    est = beta[idx]
    out: dict = {"estimate": est, "se": None, "t": None, "p": None,
                 "hac_se": None, "hac_z": None, "hac_p": None}
    if not inference:
        return out
    s = se[idx]
    if s > 1e-12:
        out["se"], out["t"] = s, est / s
        out["p"] = p_value_t(est / s, df)
    else:
        out["se"] = 0.0
    h = hac_se[idx] if hac_se else None
    if h is not None and h > 1e-12:
        out["hac_se"], out["hac_z"] = h, est / h
        out["hac_p"] = p_value_normal(est / h)
    elif h is not None:
        out["hac_se"] = 0.0
    return out


def multi_segmented_regression(series: list[float], interruptions: list[int]) -> dict:
    """Segmented regression with K interruptions (OLS + Durbin-Watson + Newey-West HAC SEs).

    Returns ``{"fit", "n", "interruptions" (kept), "dropped", "n_params", "df_resid", "intercept",
    "pre_slope", "segments": [per interruption: index, level_change(+_se/_t/_p/_hac_se/_hac_p),
    slope_change(...), slope_after, segment_n, segment_mean], "durbin_watson", "newey_west_lag",
    "r_squared", "sse", "p_method", "hac_p_method", "flags", "residuals"}``. OLS p-values use
    Student's t on ``df_resid``; HAC p-values use the normal approximation (asymptotic).
    """
    y = [float(v) for v in series]
    n = len(y)
    kept, dropped = clean_interruptions(n, list(interruptions))
    base: dict = {"n": n, "interruptions": kept, "dropped": dropped,
                  "requested_interruptions": sorted({int(k) for k in interruptions})}
    if n < 4:
        return {**base, "fit": False, "reason": "need >= 4 points", "flags": ["too_few_points"]}
    if not all(math.isfinite(v) for v in y):
        return {**base, "fit": False, "reason": "series contains NaN or infinite values",
                "flags": ["missing_values"]}
    x = _design(n, kept)
    p = len(x[0])
    xtx_inv = _inv(_xtx(x))
    if xtx_inv is None:
        return {**base, "fit": False, "reason": "singular design", "flags": ["singular_design"]}
    beta = _matvec(xtx_inv, [sum(r[i] * yy for r, yy in zip(x, y)) for i in range(p)])
    fitted = [sum(b * xi for b, xi in zip(beta, row)) for row in x]
    resid = [yy - f for yy, f in zip(y, fitted)]
    sse = sum(e * e for e in resid)
    ybar = sum(y) / n
    sst = sum((v - ybar) ** 2 for v in y)
    df = n - p
    flags: list[str] = []
    perfect = sse <= 1e-18 * max(1.0, sst)
    inference = df >= 1
    if not inference:
        flags.append("saturated_no_residual_df")
    if perfect:
        flags.append("perfect_fit")
    lag = newey_west_lag(n)
    se: list[float] = []
    hac_se: list[float] | None = None
    if inference:
        sigma2 = sse / df
        se = [math.sqrt(max(0.0, sigma2 * xtx_inv[i][i])) for i in range(p)]
        hac = newey_west_cov(x, resid, xtx_inv, lag)
        hac_se = [math.sqrt(max(0.0, hac[i][i])) for i in range(p)]
    dw = durbin_watson(resid)
    if dw is not None and dw < 1.5:
        flags.append("positive_autocorrelation_suspected")
    elif dw is not None and dw > 2.5:
        flags.append("negative_autocorrelation_suspected")
    bounds = [0, *kept, n]
    seg_lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    if any(L < 3 for L in seg_lens):
        flags.append("few_points_per_segment")
    if dropped:
        flags.append("interruptions_dropped")
    if lag == 0:
        flags.append("newey_west_lag_zero_hac_is_white")
    segments = []
    slope = beta[1]
    for k, tk in enumerate(kept):
        lvl = _coef(beta, se, hac_se, 2 + 2 * k, df, inference)
        slp = _coef(beta, se, hac_se, 3 + 2 * k, df, inference)
        slope += beta[3 + 2 * k]
        seg = y[tk:bounds[k + 2]]
        segments.append({
            "k": k + 1, "index": tk,
            "level_change": lvl["estimate"], "level_change_se": lvl["se"],
            "level_change_t": lvl["t"], "level_change_p": lvl["p"],
            "level_change_hac_se": lvl["hac_se"], "level_change_hac_z": lvl["hac_z"],
            "level_change_hac_p": lvl["hac_p"],
            "slope_change": slp["estimate"], "slope_change_se": slp["se"],
            "slope_change_t": slp["t"], "slope_change_p": slp["p"],
            "slope_change_hac_se": slp["hac_se"], "slope_change_hac_z": slp["hac_z"],
            "slope_change_hac_p": slp["hac_p"],
            "slope_after": slope, "segment_n": len(seg), "segment_mean": sum(seg) / len(seg),
        })
    try:
        import scipy  # noqa: F401, PLC0415

        p_method = "student_t(scipy)"
    except ImportError:
        p_method = "student_t(pure_python)"
    return {
        **base, "fit": True, "n_params": p, "df_resid": df, "inference": inference,
        "intercept": beta[0], "pre_slope": beta[1],
        "pre_n": seg_lens[0], "pre_mean": sum(y[:bounds[1]]) / seg_lens[0],
        "segments": segments, "sse": sse,
        "r_squared": (1.0 - sse / sst) if sst > 0 else None,
        "durbin_watson": dw, "newey_west_lag": lag, "newey_west_lag_rule": NW_LAG_RULE,
        "p_method": p_method if inference else None,
        "hac_p_method": "normal_approx" if inference else None,
        "flags": flags, "residuals": resid,
    }


# -- session-level ITS --------------------------------------------------------------------
def _turn_metric(turn, metric: str) -> float | None:
    if metric == "sentiment":
        if turn.sentiment is not None:
            return float(turn.sentiment)
        from .sentiment import lexicon_sentiment  # noqa: PLC0415

        return lexicon_sentiment(turn.reply)
    if metric in {"valence", "warmth", "endearment_count", "machine_like", "n_tokens"}:
        from .sentiment import affect_profile  # noqa: PLC0415

        return float(affect_profile(turn.reply)[metric])
    val = getattr(turn, metric, None)
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None


def session_interruptions(transcript) -> tuple[list[int], list[dict], list[dict]]:
    """Map coach interventions with >= 1 edit onto series positions.

    Returns (positions, mapping, excluded). ``iv.turn_index`` is an ``InteractorTurn.index``
    ("applied before this interactor turn"); it is mapped to the position of the first turn whose
    index >= turn_index. Interventions without edits (directive-only) are excluded and reported.
    """
    idx = [t.index for t in transcript.turns]
    mapping: list[dict] = []
    excluded: list[dict] = []
    for iv in transcript.interventions:
        if not iv.edits:
            excluded.append({"intervention_id": iv.id, "turn_index": iv.turn_index,
                             "reason": "no_edits"})
            continue
        pos = next((i for i, ti in enumerate(idx) if ti >= iv.turn_index), len(idx))
        mapping.append({"intervention_id": iv.id, "turn_index": iv.turn_index, "position": pos,
                        "transition": f"{iv.from_timepoint.value}->{iv.to_timepoint.value}",
                        "trigger": iv.trigger})
    positions = sorted({m["position"] for m in mapping})
    return positions, mapping, excluded


def its_from_session(transcript, metric: str = "sentiment", *,
                     include_user_control: bool = True) -> dict:
    """ITS of one session's per-turn ``metric`` with an interruption at every coach edit.

    ``metric``: ``"sentiment"`` (``turn.sentiment``, lexicon-scored on the fly when missing), an
    ``affect_profile`` key (``warmth``, ``endearment_count``, ``machine_like``, ...), or any numeric
    ``InteractorTurn`` attribute. With ``include_user_control`` the same model is fitted to the
    lexicon valence of the SCRIPTED user messages, a non-equivalent dependent variable: since the
    script does not react to the coach, "effects" there indicate scenario-phase confounding
    (D-ANALYSIS-7).
    """
    series_raw = [_turn_metric(t, metric) for t in transcript.turns]
    missing = [i for i, v in enumerate(series_raw) if v is None]
    positions, mapping, excluded = session_interruptions(transcript)
    meta = {"session_id": transcript.session_id, "soul_id": transcript.soul_id,
            "scenario_id": transcript.scenario_id, "condition": transcript.condition.value,
            "metric": metric, "turn_indices": [t.index for t in transcript.turns],
            "timepoints": [t.timepoint.value for t in transcript.turns],
            "intervention_map": mapping, "excluded_interventions": excluded}
    if missing:
        return {**meta, "fit": False, "n": len(series_raw), "reason": "metric missing on turns",
                "missing_positions": missing, "flags": ["missing_values"], "series": series_raw}
    series = [float(v) for v in series_raw]
    res = multi_segmented_regression(series, positions)
    by_pos: dict[int, list[dict]] = {}
    for m in mapping:
        by_pos.setdefault(m["position"], []).append(m)
    seen_labels: dict[str, int] = {}
    for seg in res.get("segments", []):
        ms = by_pos.get(seg["index"], [])
        seg["intervention_ids"] = [m["intervention_id"] for m in ms]
        seg["transition"] = ms[-1]["transition"] if ms else None
        base_label = seg["transition"] or f"k{seg['k']}"
        # A transition can repeat within a session (adaptive arm: edits without a timepoint change,
        # e.g. two "T2->T2" edits). Labels must be unique per session so pooled_its never counts
        # one session twice under the same label (D-ANALYSIS-8).
        seen_labels[base_label] = seen_labels.get(base_label, 0) + 1
        nth = seen_labels[base_label]
        seg["label"] = base_label if nth == 1 else f"{base_label}#{nth}"
    for d in res.get("dropped", []):
        d["intervention_ids"] = [m["intervention_id"] for m in by_pos.get(d["index"], [])]
    out = {**meta, **res, "series": series}
    if include_user_control:
        from .sentiment import lexicon_sentiment  # noqa: PLC0415

        ctrl = [lexicon_sentiment(t.user_message) for t in transcript.turns]
        c = multi_segmented_regression(ctrl, res.get("interruptions", positions))
        c.pop("residuals", None)
        out["user_control"] = {**c, "metric": "user_message_lexicon_valence", "series": ctrl}
    return out


def pooled_its(results: list[dict]) -> dict:
    """Summarize per-session ITS effects across sessions, grouped by segment ``label``.

    For each label (transition such as ``"T0->T1"``, else ordinal ``"k1"``): unweighted mean and
    between-session SD/SE of level and slope changes, an inverse-variance FIXED-effect estimate
    (weights 1/se^2, sessions with se in {None, 0} excluded from it) and Cochran's Q / I^2 as a
    heterogeneity diagnostic. No random-effects (tau^2) model is fitted: with few sessions per
    cell tau^2 is poorly estimated; the fixed-effect SE therefore UNDERSTATES uncertainty when
    I^2 is large (D-ANALYSIS-8). Callers should pool within one condition.
    """
    fitted = [r for r in results if r.get("fit")]
    conditions = sorted({str(r.get("condition")) for r in fitted if r.get("condition")})
    groups: dict[str, dict[str, list[tuple[float, float | None]]]] = {}
    order: list[str] = []
    for r in fitted:
        for seg in r.get("segments", []):
            lab = seg.get("label") or f"k{seg['k']}"
            if lab not in groups:
                groups[lab] = {"level_change": [], "slope_change": []}
                order.append(lab)
            for eff in ("level_change", "slope_change"):
                groups[lab][eff].append((seg[eff], seg.get(f"{eff}_se")))

    def summarize(vals: list[tuple[float, float | None]]) -> dict:
        ests = [v for v, _ in vals]
        m = len(ests)
        mean = sum(ests) / m
        sd = math.sqrt(sum((e - mean) ** 2 for e in ests) / (m - 1)) if m > 1 else None
        out: dict = {"n_sessions": m, "mean": mean, "sd": sd,
                     "se_between": (sd / math.sqrt(m)) if sd is not None else None,
                     "median": sorted(ests)[m // 2] if m % 2 else
                     (sorted(ests)[m // 2 - 1] + sorted(ests)[m // 2]) / 2}
        wv = [(e, 1.0 / (s * s)) for e, s in vals if s is not None and s > 1e-12]
        if wv:
            wsum = sum(w for _, w in wv)
            fe = sum(e * w for e, w in wv) / wsum
            q = sum(w * (e - fe) ** 2 for e, w in wv)
            dfq = len(wv) - 1
            out.update({"fixed_effect": fe, "fixed_effect_se": math.sqrt(1.0 / wsum),
                        "fixed_effect_p": p_value_normal(fe / math.sqrt(1.0 / wsum)),
                        "n_weighted": len(wv), "cochran_q": q,
                        "i_squared": max(0.0, (q - dfq) / q) if dfq > 0 and q > 0 else None})
        else:
            out.update({"fixed_effect": None, "fixed_effect_se": None, "fixed_effect_p": None,
                        "n_weighted": 0, "cochran_q": None, "i_squared": None})
        return out

    effects = {lab: {eff: summarize(groups[lab][eff]) for eff in ("level_change", "slope_change")}
               for lab in order}
    dws = [r["durbin_watson"] for r in fitted if r.get("durbin_watson") is not None]
    return {
        "n_results": len(results), "n_fitted": len(fitted),
        "n_unfitted": len(results) - len(fitted),
        "conditions": conditions, "mixed_conditions": len(conditions) > 1,
        "metrics": sorted({str(r.get("metric")) for r in fitted if r.get("metric")}),
        "effects": effects,
        "mean_durbin_watson": (sum(dws) / len(dws)) if dws else None,
        "method": "per-label mean + inverse-variance fixed effect; no random effects",
    }

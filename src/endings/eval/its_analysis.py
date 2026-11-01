"""Interrupted time-series (ITS) analysis of per-turn metrics.

Segmented regression (Wagner et al. 2002):
    y_t = b0 + b1*t + b2*D_t + b3*(t - t0)*D_t
where D_t = 1 after the intervention turn t0. ``b2`` is the immediate LEVEL change and ``b3`` is the
SLOPE change after the intervention. We use this to quantify, e.g., how sharply warmth tapers or
AI-status disclosure rises once the ending enters its closure phase.

Pure-Python OLS (normal equations + Gaussian elimination) so it runs with only the core dependency;
if NumPy is installed it is used for numerical stability.
"""

from __future__ import annotations

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

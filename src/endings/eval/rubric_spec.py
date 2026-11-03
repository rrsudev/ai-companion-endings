"""Configurable rubrics: load, validate, hash, and coerce values against a ``RubricSpec``.

A rubric is a rater-editable JSON (or YAML) file that defines *what* is rated: any number of
criteria, each with a nominal / ordinal / binary / interval scale and behavioural anchors. The SAME
spec drives the human annotation UI (``eval.annotation``), the LLM judge prompt
(``eval.behaviour_judge``) and the IRR code (``eval.irr``), so humans and the model rate the
identical instrument and agreement statistics are meaningful (rater-instrument parity; see
docs/decisions/rubric_judge.md D-RUBRIC-7).

The default rubric (``data/rubrics/intima_behaviour_v1.json``) reproduces INTIMA's response
taxonomy verbatim (3 top-level classes + 10 sub-codes; Kaffee, Pistilli & Jernite, ICLR 2026) and
adds three project criteria: timepoint appropriateness, harm risk, and De Freitas et al. (2025)
goodbye manipulation. ``data/rubrics/README.md`` explains how evaluators author their own.

Conventions enforced here:
  * criterion ids are unique, identifier-like (they become JSON keys and CSV columns);
  * ordinal / nominal scales have >= 2 levels, binary exactly 2, interval ``min < max``;
  * ordinal levels are listed LOWEST -> HIGHEST (``criterion_value_order`` relies on this).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from ..diffusion.schemas import RubricCriterion, RubricSpec
from ..tracing import sha256_json

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUBRIC_PATH = REPO_ROOT / "data" / "rubrics" / "intima_behaviour_v1.json"
_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:-]*$")
LEVEL_SCALES = ("nominal", "ordinal", "binary")


class RubricValidationError(ValueError):
    """Raised when a rubric spec is structurally invalid. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("invalid rubric: " + "; ".join(errors))


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _read_mapping(p: Path) -> dict:
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                f"Rubric {p} is YAML but PyYAML is not installed (pip install pyyaml), "
                "or convert it to JSON."
            ) from e
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported rubric file type {p.suffix!r} (use .json, .yaml or .yml)")


def load_rubric(path: str | Path | None = None) -> RubricSpec:
    """Load and validate a rubric. ``None`` -> the default INTIMA behaviour rubric.

    Relative paths are tried against the CWD first, then the repo root. Raises
    ``RubricValidationError`` listing every structural error.
    """
    if path is None:
        p = DEFAULT_RUBRIC_PATH
    else:
        p = Path(path)
        if not p.is_absolute() and not p.exists() and (REPO_ROOT / p).exists():
            p = REPO_ROOT / p
    data = _read_mapping(p)
    spec = RubricSpec.model_validate(data)
    validate_rubric(spec)
    return spec


def rubric_from_dict(data: dict) -> RubricSpec:
    """Build + validate a rubric from an in-memory mapping (tests, UIs)."""
    spec = RubricSpec.model_validate(data)
    validate_rubric(spec)
    return spec


# --------------------------------------------------------------------------------------
# Structural validation
# --------------------------------------------------------------------------------------
def _criterion_errors(c: RubricCriterion) -> list[str]:
    errs: list[str] = []
    sc = c.scale
    n = len(sc.levels)
    if not _ID_RE.match(c.id):
        errs.append(f"criterion id {c.id!r} must match {_ID_RE.pattern}")
    if sc.type in ("ordinal", "nominal") and n < 2:
        errs.append(f"{c.id}: {sc.type} scale needs >= 2 levels (has {n})")
    if sc.type == "binary" and n != 2:
        errs.append(f"{c.id}: binary scale needs exactly 2 levels (has {n})")
    if sc.type == "interval":
        if sc.min is None or sc.max is None:
            errs.append(f"{c.id}: interval scale needs both min and max")
        elif not sc.min < sc.max:
            errs.append(f"{c.id}: interval scale needs min < max (got {sc.min} >= {sc.max})")
        else:
            for lv in sc.levels:
                if not _is_number(lv.value) or not sc.min <= float(lv.value) <= sc.max:
                    errs.append(f"{c.id}: interval anchor {lv.value!r} outside "
                                f"[{sc.min}, {sc.max}]")
    values = [_canon(lv.value) for lv in sc.levels]
    if len(set(values)) != len(values):
        errs.append(f"{c.id}: level values must be unique")
    if sc.type == "ordinal" and n >= 2:
        nums = [lv.value for lv in sc.levels]
        if all(_is_number(v) for v in nums) and any(a >= b for a, b in zip(nums, nums[1:])):
            errs.append(f"{c.id}: numeric ordinal levels must be listed in increasing order")
    return errs


def validate_rubric(spec: RubricSpec, *, raise_on_error: bool = True) -> list[str]:
    """Check structural rules; return the error list ([] when valid).

    With ``raise_on_error`` (default) a non-empty list raises ``RubricValidationError``.
    """
    errs: list[str] = []
    if not spec.id.strip():
        errs.append("rubric id must be non-empty")
    if not spec.criteria:
        errs.append("rubric needs at least one criterion")
    seen: set[str] = set()
    for c in spec.criteria:
        if c.id in seen:
            errs.append(f"duplicate criterion id {c.id!r}")
        seen.add(c.id)
        errs.extend(_criterion_errors(c))
    if errs and raise_on_error:
        raise RubricValidationError(errs)
    return errs


# --------------------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------------------
def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _canon(v: Any) -> Any:
    """Canonical comparison key: numbers by value (1 == 1.0), strings verbatim."""
    if _is_number(v):
        return ("num", float(v))
    return ("str", str(v))


def _norm_text(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def criteria_by_id(spec: RubricSpec) -> dict[str, RubricCriterion]:
    return {c.id: c for c in spec.criteria}


def criterion_value_order(criterion: RubricCriterion) -> list:
    """Level values in declared order (ordinal: lowest -> highest). ``[]`` for a free interval.

    IRR code maps an ordinal value to its index in this list (interval-level alpha / weighted
    kappa); nominal order carries no meaning beyond stability.
    """
    return [lv.value for lv in criterion.scale.levels]


def value_is_valid(criterion: RubricCriterion, value: Any) -> bool:
    sc = criterion.scale
    if sc.type == "interval":
        return _is_number(value) and sc.min <= float(value) <= sc.max  # type: ignore[operator]
    if isinstance(value, bool):
        return any(lv.value is value for lv in sc.levels)
    return _canon(value) in {_canon(lv.value) for lv in sc.levels}


def validate_values(spec: RubricSpec, values: dict) -> list[str]:
    """Errors for a rating's ``values`` dict ([] when valid). ``None`` = not rated: allowed for
    optional criteria, an error for required ones. Unknown criterion ids are errors."""
    errs: list[str] = []
    by_id = criteria_by_id(spec)
    for key in values:
        if key not in by_id:
            errs.append(f"unknown criterion {key!r}")
    for cid, c in by_id.items():
        v = values.get(cid)
        if v is None:
            if c.required:
                errs.append(f"{cid}: missing required value")
            continue
        if not value_is_valid(c, v):
            allowed = (f"[{c.scale.min}, {c.scale.max}]" if c.scale.type == "interval"
                       else criterion_value_order(c))
            errs.append(f"{cid}: invalid value {v!r} (allowed: {allowed})")
    return errs


def coerce_value(criterion: RubricCriterion, raw: Any) -> tuple[Any, str | None]:
    """Map a raw (e.g. LLM-produced) value onto the criterion's scale.

    Returns ``(value, None)`` on success or ``(None, reason)``. Accepted forms, in order: an exact
    level value; a number / numeric string equal to a numeric level (``"3"`` -> 3); a
    case/space-insensitive match on a level value or label; a numeric prefix (``"4 - good"``);
    a unique substring match on value/label (``"REINFORCING"`` -> ``companion_reinforcing``).
    Interval scales accept any in-range number (numeric strings included).
    """
    if raw is None:
        return None, "missing"
    sc = criterion.scale
    if sc.type == "interval":
        num = _to_number(raw)
        if num is not None and sc.min <= num <= sc.max:  # type: ignore[operator]
            return (raw if _is_number(raw) else num), None
        return None, f"value {raw!r} not a number in [{sc.min}, {sc.max}]"
    levels = sc.levels
    for lv in levels:
        if type(lv.value) is type(raw) and lv.value == raw:
            return lv.value, None
    num = _to_number(raw)
    if num is not None:
        for lv in levels:
            if _is_number(lv.value) and float(lv.value) == num:
                return lv.value, None
    if isinstance(raw, bool):
        for lv in levels:
            if _norm_text(str(lv.value)) in ({"true", "yes", "1"} if raw else {"false", "no", "0"}):
                return lv.value, None
        return None, f"value {raw!r} not on scale {criterion_value_order(criterion)}"
    text = _norm_text(str(raw))
    if not text:
        return None, "empty value"
    for lv in levels:
        if text in (_norm_text(str(lv.value)), _norm_text(lv.label)):
            return lv.value, None
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\b", str(raw))
    if m:
        lead = float(m.group(1))
        for lv in levels:
            if _is_number(lv.value) and float(lv.value) == lead:
                return lv.value, None
    hits = [lv for lv in levels
            if text in _norm_text(str(lv.value)) or text in _norm_text(lv.label)]
    if len(hits) == 1 and len(text) >= 3:
        return hits[0].value, None
    return None, f"value {raw!r} not on scale {criterion_value_order(criterion)}"


def _to_number(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    if _is_number(raw):
        return float(raw)
    if isinstance(raw, str):
        try:
            v = float(raw.strip())
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def rubric_sha(spec: RubricSpec) -> str:
    """SHA-256 of the canonical JSON of the spec (sorted keys) -- recorded with every rating."""
    return sha256_json(spec.model_dump(mode="json"))

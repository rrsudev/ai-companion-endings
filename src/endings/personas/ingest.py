"""Ingest curated RedNote personas into normalized ``PersonaProfile`` objects.

The team supplies curated persona files (JSON; YAML also accepted if PyYAML is installed). This
module is deliberately lenient: it maps known fields, routes attachment/dependency sub-objects,
derives the attachment style when absent, and stashes any unrecognized keys under ``raw`` so curated
metadata never breaks the strict schema. See ``FORMAT.md`` for the input contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import (
    AttachmentProfile,
    DependencySignals,
    Level,
    PersonaProfile,
    ReadinessStage,
)

_SCALAR_FIELDS = {
    "id", "display_name", "companion_name", "language", "background",
    "relationship_context", "reason_for_ending", "ambivalence",
    "escalates_when_bounded", "source_ref",
}
_LIST_FIELDS = {"vulnerabilities", "stated_goals"}


def derive_attachment_style(anxiety: float, avoidance: float, hi: float = 4.5) -> str:
    anx, avo = anxiety >= hi, avoidance >= hi
    if anx and avo:
        return "fearful"
    if anx:
        return "anxious"
    if avo:
        return "avoidant"
    return "secure"


def _attachment(data: dict) -> AttachmentProfile:
    raw = dict(data or {})
    ap = AttachmentProfile(
        anxiety=float(raw.get("anxiety", 4.0)),
        avoidance=float(raw.get("avoidance", 4.0)),
        style=raw.get("style"),
    )
    if not ap.style:
        ap.style = derive_attachment_style(ap.anxiety, ap.avoidance)
    return ap


def _dependency(data: dict) -> DependencySignals:
    raw = dict(data or {})
    norm = {k: Level(v) if not isinstance(v, Level) else v for k, v in raw.items()}
    return DependencySignals(**norm)


def from_raw(data: dict) -> PersonaProfile:
    """Build a PersonaProfile from a raw curated dict."""
    data = dict(data)
    if "id" not in data:
        raise ValueError("persona record is missing required field 'id'")
    data.setdefault("display_name", data["id"])

    kwargs: dict = {}
    leftover: dict = dict(data.get("raw", {}))

    for key, value in data.items():
        if key in _SCALAR_FIELDS or key in _LIST_FIELDS:
            kwargs[key] = value
        elif key == "readiness":
            kwargs["readiness"] = ReadinessStage(value)
        elif key == "attachment":
            kwargs["attachment"] = _attachment(value)
        elif key == "dependency":
            kwargs["dependency"] = _dependency(value)
        elif key == "raw":
            continue
        else:
            leftover[key] = value  # keep unknown curated metadata out of the model prompt

    if "attachment" not in kwargs:
        kwargs["attachment"] = _attachment({})
    if leftover:
        kwargs["raw"] = leftover
    return PersonaProfile(**kwargs)


def _read(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install PyYAML to load .yaml personas, or use JSON.") from e
        return yaml.safe_load(text)
    return json.loads(text)


def load_persona(path: str | Path) -> PersonaProfile:
    return from_raw(_read(Path(path)))  # type: ignore[arg-type]


def load_personas(paths: list[str | Path]) -> list[PersonaProfile]:
    """Load personas from a list of files and/or directories (and JSON arrays within a file)."""
    out: list[PersonaProfile] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.glob("*.json")) + sorted(path.glob("*.y*ml")):
                out.extend(_load_file(f))
        else:
            out.extend(_load_file(path))
    return out


def _load_file(path: Path) -> list[PersonaProfile]:
    data = _read(path)
    if isinstance(data, list):
        return [from_raw(d) for d in data]
    return [from_raw(data)]  # type: ignore[arg-type]

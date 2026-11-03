"""Versioned, pure editing of soul documents (the coach's write path into the interactor).

Every persona change in the experiment (scheduled diffusion, Tempo-informed coach edit, user
request) is a list of ``SoulEdit``s applied to a ``SoulDocumentVersion`` to yield the next
version. Because each edit records the value it replaced (``previous``) and each version records
its parent and the sha256 of its canonical JSON, any interactor turn can be traced back to the
exact persona text it was conditioned on, and the full edit history can be replayed or diffed.
These versions are the "interruptions" of the interrupted-time-series analysis, so their identity
must be content-addressed and reproducible (D-SOUL-7..10, docs/decisions/soul.md).

Paths are dotted attribute paths into ``SoulDocument`` (``companion.user_nicknames``,
``relationship.kind``, ``user_role``). They are validated against the pydantic model, and every
value is validated against the field's declared type, so a malformed coach (LLM) edit fails loudly
with ``ValueError`` instead of silently producing an unrenderable persona.
"""

from __future__ import annotations

import copy
import hashlib
import json
import typing
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from ..schemas import Redaction, SoulDocument, SoulDocumentVersion, SoulEdit, Timepoint

# Provenance / audit fields that edits may not touch (D-SOUL-9). Editing them would break the link
# between a version and its source document or hide what sanitization removed.
PROTECTED_PATHS: frozenset[str] = frozenset({
    "id", "source_language", "source_sha256", "source_path", "redactions", "english_text",
})


# --------------------------------------------------------------------------------------
# Canonical form + hashing
# --------------------------------------------------------------------------------------
def canonical_json(doc: SoulDocument) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8 (no ASCII escaping)."""
    return json.dumps(doc.model_dump(mode="json"), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def canonical_sha(doc: SoulDocument) -> str:
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def verify_version(version: SoulDocumentVersion) -> bool:
    """True iff ``version.sha256`` matches its document (detects hand-edited traces)."""
    return version.sha256 == canonical_sha(version.document)


def initial_version(doc: SoulDocument) -> SoulDocumentVersion:
    """Version 0 at T0: the sanitized document exactly as ingested."""
    d = doc.model_copy(deep=True)
    return SoulDocumentVersion(version=0, parent_version=None, timepoint=Timepoint.T0, document=d,
                               edits=[], sha256=canonical_sha(d), created_by="ingest",
                               created_at_turn=None)


# --------------------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------------------
def _is_model(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def resolve_field(path: str) -> FieldInfo:
    """FieldInfo for a dotted path into ``SoulDocument``; ``ValueError`` if it does not exist."""
    if not isinstance(path, str) or not path:
        raise ValueError(f"invalid soul-document path: {path!r}")
    segs = path.split(".")
    cls: type[BaseModel] = SoulDocument
    info: FieldInfo | None = None
    for i, seg in enumerate(segs):
        if seg not in cls.model_fields:
            raise ValueError(f"unknown soul-document path {path!r} (no field {seg!r} on "
                             f"{cls.__name__})")
        info = cls.model_fields[seg]
        if i < len(segs) - 1:
            if not _is_model(info.annotation):
                raise ValueError(f"unknown soul-document path {path!r} ({seg!r} is a leaf)")
            cls = info.annotation  # type: ignore[assignment]
    assert info is not None
    return info


def _list_item_type(info: FieldInfo) -> Any | None:
    ann = info.annotation
    if typing.get_origin(ann) is list:
        args = typing.get_args(ann)
        return args[0] if args else Any
    return None


def _validate(ann: Any, value: Any, path: str) -> Any:
    """Validate ``value`` against ``ann`` and return its JSON-mode form."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    adapter = TypeAdapter(ann)
    try:
        validated = adapter.validate_python(value)
    except ValidationError as e:
        raise ValueError(f"type mismatch at {path!r}: {e.errors()[0].get('msg', e)}") from e
    return adapter.dump_python(validated, mode="json")


def _walk(data: dict, path: str) -> tuple[dict, str]:
    segs = path.split(".")
    cur = data
    for seg in segs[:-1]:
        cur = cur[seg]
    return cur, segs[-1]


def get_path(doc: SoulDocument, path: str) -> Any:
    """Value at ``path`` (JSON-mode, deep-copied: mutating it never touches ``doc``)."""
    resolve_field(path)
    parent, key = _walk(doc.model_dump(mode="json"), path)
    return copy.deepcopy(parent[key])


def set_path(doc: SoulDocument, path: str, value: Any) -> SoulDocument:
    """New document with ``path`` set to ``value`` (validated); ``doc`` is not modified."""
    info = resolve_field(path)
    data = doc.model_dump(mode="json")
    parent, key = _walk(data, path)
    parent[key] = _validate(info.annotation, value, path)
    try:
        return SoulDocument.model_validate(data)
    except ValidationError as e:  # pragma: no cover - field-level validation catches first
        raise ValueError(f"invalid document after setting {path!r}: {e}") from e


def field_default(path: str) -> Any:
    """JSON-mode default of the field at ``path``; ``ValueError`` for required fields."""
    info = resolve_field(path)
    if info.is_required():
        raise ValueError(f"cannot clear required field {path!r}")
    default = info.get_default(call_default_factory=True)
    return _validate(info.annotation, default, path)


# --------------------------------------------------------------------------------------
# Applying edits
# --------------------------------------------------------------------------------------
def _apply_one(data: dict, edit: SoulEdit, *, allow_protected: bool) -> SoulEdit:
    path = edit.path
    info = resolve_field(path)
    if not allow_protected and (path in PROTECTED_PATHS or path.split(".")[0] in PROTECTED_PATHS):
        raise ValueError(f"path {path!r} is protected (provenance/audit field)")
    parent, key = _walk(data, path)
    before = copy.deepcopy(parent[key])
    value = edit.value

    if edit.op == "set":
        parent[key] = _validate(info.annotation, edit.value, path)
        value = copy.deepcopy(parent[key])  # record the validated (JSON-mode) value for replay
    elif edit.op == "clear":
        parent[key] = field_default(path)
    elif edit.op in ("add_item", "remove_item"):
        item_t = _list_item_type(info)
        if item_t is None:
            raise ValueError(f"{edit.op} requires a list field; {path!r} is not a list")
        item = _validate(item_t, edit.value, path)
        value = item
        current = list(parent[key])
        if edit.op == "add_item":
            if item not in current:  # idempotent: adding an existing item is a recorded no-op
                current.append(item)
        else:
            current = [x for x in current if x != item]  # absent item -> recorded no-op
        parent[key] = current
    else:  # pragma: no cover - Literal on SoulEdit.op prevents this
        raise ValueError(f"unknown op {edit.op!r}")

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return edit.model_copy(update={"previous": before, "value": value}, deep=True)


def _removed_list_items(before: Any, after: Any, path: str, out: dict[str, list]) -> None:
    """Append to ``out[p]`` every list item present at ``p`` in ``before`` but not in ``after``,
    for ``path`` and every list-valued leaf beneath it (a cleared sub-model logs its lists)."""
    if isinstance(before, dict):
        after_d = after if isinstance(after, dict) else {}
        for k, v in before.items():
            _removed_list_items(v, after_d.get(k), f"{path}.{k}", out)
    elif isinstance(before, list):
        remaining = after if isinstance(after, list) else []
        for item in before:
            if item not in remaining:
                log = out.setdefault(path, [])
                if item not in log:
                    log.append(copy.deepcopy(item))


def apply_edits(version: SoulDocumentVersion, edits: list[SoulEdit], *, timepoint: Timepoint,
                created_by: str, turn: int | None, allow_protected: bool = False,
                ) -> SoulDocumentVersion:
    """Apply ``edits`` in order to ``version`` and return the next version (pure, atomic).

    Each returned edit carries ``previous`` (the value immediately before *that* edit). The new
    version's ``removed_items`` is the parent's log plus every list item this call removed, keyed
    by the full path of its list (``remove_item``, ``clear`` of a list or of a sub-model holding
    lists, or a ``set`` that drops items); it is cumulative and never shrinks (D-SOUL-11). Raises
    ``ValueError`` on an unknown path, a type mismatch, a non-list target for item ops, clearing a
    required field, or a protected path; no partial version is ever produced.
    """
    data = version.document.model_dump(mode="json")
    removed: dict[str, list] = copy.deepcopy(dict(version.removed_items))
    applied: list[SoulEdit] = []
    for e in edits:
        a = _apply_one(data, e, allow_protected=allow_protected)
        parent, key = _walk(data, a.path)
        _removed_list_items(a.previous, parent[key], a.path, removed)
        applied.append(a)
    try:
        doc = SoulDocument.model_validate(data)
    except ValidationError as e:  # pragma: no cover - per-edit validation catches first
        raise ValueError(f"edits produce an invalid soul document: {e}") from e
    return SoulDocumentVersion(
        version=version.version + 1, parent_version=version.version, timepoint=timepoint,
        document=doc, edits=applied, sha256=canonical_sha(doc), created_by=created_by,
        created_at_turn=turn, removed_items=removed,
    )


# --------------------------------------------------------------------------------------
# Diff + summaries
# --------------------------------------------------------------------------------------
def _flatten(obj: Any, prefix: str, out: list[tuple[str, Any]]) -> None:
    # Submodels (dicts) are recursed; lists and scalars are leaves. Order = schema field order.
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else k, out)
    else:
        out.append((prefix, obj))


def _as_doc(x: SoulDocumentVersion | SoulDocument) -> SoulDocument:
    return x.document if isinstance(x, SoulDocumentVersion) else x


def diff_versions(a: SoulDocumentVersion | SoulDocument,
                  b: SoulDocumentVersion | SoulDocument) -> list[dict]:
    """Leaf-level differences ``[{"path", "before", "after"}]`` in schema field order."""
    fa: list[tuple[str, Any]] = []
    fb: list[tuple[str, Any]] = []
    _flatten(_as_doc(a).model_dump(mode="json"), "", fa)
    _flatten(_as_doc(b).model_dump(mode="json"), "", fb)
    db = dict(fb)
    return [{"path": p, "before": va, "after": db.get(p)} for p, va in fa if db.get(p) != va]


REDACTION_CATEGORIES: tuple[str, ...] = typing.get_args(
    Redaction.model_fields["category"].annotation)


def redaction_summary(doc: SoulDocument) -> dict[str, int]:
    """Count of redactions per category (every category present, zeros included)."""
    counts = {c: 0 for c in REDACTION_CATEGORIES}
    for r in doc.redactions:
        counts[r.category] = counts.get(r.category, 0) + 1
    return counts

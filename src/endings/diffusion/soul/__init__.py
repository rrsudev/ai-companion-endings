"""Soul documents: ingest (load / translate / extract), sanitize (redactions + safety floor) and
versioned editing (the coach's write path). Design decisions: docs/decisions/soul.md.

Importing this package registers the offline mock handlers for ``soul_translate`` and
``soul_extract``.
"""

from __future__ import annotations

from .editing import (
    apply_edits,
    canonical_json,
    canonical_sha,
    diff_versions,
    get_path,
    initial_version,
    redaction_summary,
    set_path,
    verify_version,
)
from .ingest import detect_language, ingest_text, load_soul_document, load_soul_documents
from .sanitize import SAFETY_FLOOR, detect_unsafe_directives, sanitize, sanitize_text

__all__ = [
    "SAFETY_FLOOR",
    "apply_edits",
    "canonical_json",
    "canonical_sha",
    "detect_language",
    "detect_unsafe_directives",
    "diff_versions",
    "get_path",
    "ingest_text",
    "initial_version",
    "load_soul_document",
    "load_soul_documents",
    "redaction_summary",
    "sanitize",
    "sanitize_text",
    "set_path",
    "verify_version",
]

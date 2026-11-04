"""Run tracing: every model call and every state change, append-only, for research traceability.

A ``Tracer`` writes JSONL records to ``<run_dir>/trace.jsonl`` and a ``manifest.json`` (config,
git commit, package and dependency versions, start time, timepoint-spec hash, provider call
defaults). ``TracedProvider`` wraps any ``Provider`` so each call records role, model, task, a hash
of the exact messages, the *effective* parameters (provider defaults merged with call kwargs),
response metadata (finish_reason, usage), latency, the raw response and any ids pushed with
``Tracer.context`` (session / turn / injection / rating item). Any reported number can therefore be
traced to the exact call that produced it (D-DEPLOY-12/13/14).

Records are plain dicts: ``{"seq", "kind", ...}``. ``kind`` in {"llm_call", "event"}.
"""

from __future__ import annotations

import hashlib
import json
import platform
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .providers import Messages, Provider


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def _git_commit() -> str | None:
    # Run git in the package's own directory, not the process cwd, so a run launched from
    # elsewhere still records this checkout's commit.
    repo = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, check=False, cwd=repo)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               timeout=5, check=False, cwd=repo)
        sha = out.stdout.strip() or None
        return f"{sha}{'+dirty' if dirty.stdout.strip() else ''}" if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


_RESERVED = frozenset({
    "seq", "kind", "role", "provider", "model", "method", "task", "messages_sha256", "messages",
    "hints_keys", "params", "response", "response_meta", "latency_s", "name", "error",
})

DEPENDENCIES = ("litellm", "pydantic", "numpy", "statsmodels", "scipy", "pandas")


def _dep_versions() -> dict[str, str | None]:
    from importlib import metadata  # noqa: PLC0415

    out: dict[str, str | None] = {}
    for d in DEPENDENCIES:
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    return out


def _host_only(url: str | None) -> str | None:
    """Scheme + host[:port] of an endpoint URL (paths/queries could carry tokens)."""
    if not url:
        return None
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}" if parts.scheme and host else None


def _unwrap(p: Any) -> Any:
    while isinstance(p, TracedProvider):
        p = p.inner
    return p


def provider_manifest(providers: dict[str, Any]) -> dict:
    """Manifest fields describing how each role's model is called (D-DEPLOY-14): the provider's
    call defaults (sampling, ``enable_thinking``), its endpoint host, and the thinking env flag."""
    inner = {r: _unwrap(p) for r, p in providers.items()}
    return {
        "call_defaults": {r: dict(getattr(p, "default_params", {}) or {})
                          for r, p in sorted(inner.items())},
        "api_base": {r: _host_only(getattr(p, "api_base", None)) for r, p in sorted(inner.items())},
        "interactor_thinking": os.getenv("ENDINGS_INTERACTOR_THINKING", "0"),
    }


class Tracer:
    """Append-only JSONL trace. ``Tracer(None)`` is an in-memory tracer (tests, notebooks)."""

    def __init__(self, run_dir: str | Path | None = None, *, store_messages: bool = True,
                 filename: str = "trace.jsonl", start_seq: int = 0):
        self.run_dir = Path(run_dir) if run_dir else None
        self.store_messages = store_messages
        self.records: list[dict] = []
        self._seq = start_seq  # continue an existing trace's sequence if > 0
        self._context: list[dict[str, Any]] = []
        self._providers: dict[str, TracedProvider] = {}  # role -> provider (for the manifest)
        self._fh = None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._fh = (self.run_dir / filename).open("a", encoding="utf-8")

    def emit(self, kind: str, **fields: Any) -> dict:
        self._seq += 1
        rec = {"seq": self._seq, "kind": kind, **fields}
        self.records.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        return rec

    def event(self, name: str, **fields: Any) -> dict:
        return self.emit("event", name=name, **fields)

    @contextmanager
    def context(self, **fields: Any) -> Iterator[dict[str, Any]]:
        """Attach ids (``session_id``, ``turn_index``, ``injection_id``, ``item_id``,
        ``rater_id``...) to every ``llm_call`` emitted inside the block. Nested blocks merge, inner
        values winning; ``None`` values are ignored."""
        merged = {**self.current_context(), **{k: v for k, v in fields.items() if v is not None}}
        self._context.append(merged)
        try:
            yield merged
        finally:
            self._context.pop()

    def current_context(self) -> dict[str, Any]:
        """Context ids in force (record-structure keys such as ``seq``/``role`` are dropped)."""
        ctx = self._context[-1] if self._context else {}
        return {k: v for k, v in ctx.items() if k not in _RESERVED}

    def write_manifest(self, config: dict, extra: dict | None = None, *,
                       providers: dict[str, Any] | None = None) -> dict:
        """Write ``manifest.json``. ``providers`` (role -> provider) defaults to every
        ``TracedProvider`` created on this tracer."""
        from . import __version__  # noqa: PLC0415
        from .diffusion.timepoints import spec_hash  # noqa: PLC0415

        manifest = {
            "config": config,
            "config_sha256": sha256_json(config),
            "git_commit": _git_commit(),
            "package_version": __version__,
            "dependency_versions": _dep_versions(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "timepoint_spec_hash": spec_hash(),
            **provider_manifest(providers if providers is not None else self._providers),
            **(extra or {}),
        }
        if self.run_dir:
            (self.run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return manifest

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


class TracedProvider(Provider):
    """Transparent provider wrapper that records every call to a ``Tracer``."""

    def __init__(self, inner: Provider, tracer: Tracer, role: str):
        super().__init__(inner.model)
        self.inner = inner
        self.tracer = tracer
        self.role = role
        self.name = f"traced:{inner.name}"
        # Seqs of the llm_call records emitted since the caller last reset this list (callers
        # store them on InteractorTurn / InjectionRecord / RatingRecord.meta).
        self.last_seqs: list[int] = []
        tracer._providers.setdefault(role, self)

    @property
    def last_meta(self) -> dict | None:  # type: ignore[override]
        return getattr(self.inner, "last_meta", None)

    def _eff(self, temperature: float, kw: dict) -> dict:
        """Effective call params: provider defaults, then temperature, then explicit kwargs."""
        return {**(getattr(self.inner, "default_params", None) or {}),
                "temperature": temperature, **kw}

    def _record(self, method: str, messages: Messages, task, hints, params, result, t0,
                error: str | None = None) -> int:
        fields: dict[str, Any] = {"error": error} if error is not None else {}
        rec = self.tracer.emit(
            "llm_call",
            role=self.role, provider=self.inner.name, model=self.inner.model, method=method,
            task=task, **self.tracer.current_context(), messages_sha256=sha256_json(messages),
            messages=messages if self.tracer.store_messages else None,
            hints_keys=sorted((hints or {}).keys()), params=params,
            response=result, response_meta=getattr(self.inner, "last_meta", None),
            latency_s=round(time.perf_counter() - t0, 4), **fields,
        )
        self.last_seqs.append(rec["seq"])
        return rec["seq"]

    def _call(self, method: str, messages, temperature, task, hints, kw):
        """Run the inner call and record it. A failed call is recorded with ``response=None`` and
        ``error=repr(exc)`` and the exception is re-raised unchanged, so callers' fail-soft paths
        still work and the trace shows every attempt (D-DEPLOY-12)."""
        t0 = time.perf_counter()
        params = self._eff(temperature, kw)  # includes ``seed`` when the caller passes one
        try:
            out = getattr(self.inner, method)(messages, temperature=temperature, task=task,
                                              hints=hints, **kw)
        except Exception as e:
            self._record(method, messages, task, hints, params, None, t0, error=repr(e)[:2000])
            raise
        self._record(method, messages, task, hints, params, out, t0)
        return out

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        return self._call("chat", messages, temperature, task, hints, kw)

    def complete_json(self, messages, *, temperature=0.2, task=None, hints=None, **kw):
        return self._call("complete_json", messages, temperature, task, hints, kw)

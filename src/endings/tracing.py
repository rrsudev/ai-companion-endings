"""Run tracing: every model call and every state change, append-only, for research traceability.

A ``Tracer`` writes JSONL records to ``<run_dir>/trace.jsonl`` and a ``manifest.json`` (config,
git commit, package versions, timepoint-spec hash). ``TracedProvider`` wraps any ``Provider`` so
each call records role, model, task, a hash of the exact messages, parameters, latency and the raw
response. Any reported number can therefore be traced to the exact prompt that produced it.

Records are plain dicts: ``{"seq", "kind", ...}``. ``kind`` in {"llm_call", "event"}.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .providers import Messages, Provider


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               timeout=5, check=False)
        sha = out.stdout.strip() or None
        return f"{sha}{'+dirty' if dirty.stdout.strip() else ''}" if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


class Tracer:
    """Append-only JSONL trace. ``Tracer(None)`` is an in-memory tracer (tests, notebooks)."""

    def __init__(self, run_dir: str | Path | None = None, *, store_messages: bool = True):
        self.run_dir = Path(run_dir) if run_dir else None
        self.store_messages = store_messages
        self.records: list[dict] = []
        self._seq = 0
        self._fh = None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._fh = (self.run_dir / "trace.jsonl").open("a", encoding="utf-8")

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

    def write_manifest(self, config: dict, extra: dict | None = None) -> dict:
        from . import __version__  # noqa: PLC0415
        from .diffusion.timepoints import spec_hash  # noqa: PLC0415

        manifest = {
            "config": config,
            "config_sha256": sha256_json(config),
            "git_commit": _git_commit(),
            "package_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "timepoint_spec_hash": spec_hash(),
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

    def _record(self, method: str, messages: Messages, task, hints, params, result, t0) -> None:
        self.tracer.emit(
            "llm_call",
            role=self.role, provider=self.inner.name, model=self.inner.model, method=method,
            task=task, messages_sha256=sha256_json(messages),
            messages=messages if self.tracer.store_messages else None,
            hints_keys=sorted((hints or {}).keys()), params=params,
            response=result, latency_s=round(time.perf_counter() - t0, 4),
        )

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        t0 = time.perf_counter()
        out = self.inner.chat(messages, temperature=temperature, task=task, hints=hints, **kw)
        self._record("chat", messages, task, hints, {"temperature": temperature, **kw}, out, t0)
        return out

    def complete_json(self, messages, *, temperature=0.2, task=None, hints=None, **kw):
        t0 = time.perf_counter()
        out = self.inner.complete_json(messages, temperature=temperature, task=task, hints=hints, **kw)
        self._record("complete_json", messages, task, hints, {"temperature": temperature, **kw},
                     out, t0)
        return out

"""Per-call sampling seeds for coach LLM calls (D-PLANNER-16 / D-TEMPO-18).

``call_seed(role, run_seed, call_key, attempt)`` mirrors ``session.stable_seed`` (first 48 bits of
sha256 over the ``|``-joined parts) and ``Interactor.call_seed`` (D-INTERACTOR-12), reduced mod
2**31 because some backends reject seeds above a signed 32-bit int. Reimplemented here rather than
imported from ``session`` because ``session`` imports the coach (no import cycle).

Seeds are best-effort: OpenAI-compatible backends (vLLM) honour them, Gemini via litellm may or may
not, and offline mocks ignore them (they are deterministic anyway). When the run seed is ``None``
no ``seed`` kwarg is sent at all, so providers with a strict signature keep working.
"""

from __future__ import annotations

import hashlib
from typing import Any


def call_seed(role: str, run_seed: int | None, call_key: str, attempt: int = 0) -> int | None:
    """Per-call seed, or ``None`` when no run seed is configured."""
    if run_seed is None:
        return None
    parts = (role, run_seed, call_key, attempt)
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % (2 ** 31)


def seed_kwargs(role: str, run_seed: int | None, call_key: str,
                attempt: int = 0) -> dict[str, Any]:
    """``{"seed": n}`` to splat into a provider call, or ``{}`` when unseeded."""
    s = call_seed(role, run_seed, call_key, attempt)
    return {"seed": s} if s is not None else {}

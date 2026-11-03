"""Coach agent: observation, Tempo inference, transition policy and intervention planning.

``observe`` (computer-use events -> observations) and ``tempo`` (striving inference, user
co-creation, readiness assessment) are always available. ``policy`` and ``planner`` are imported
defensively so a partial checkout (or an import error in one of them) does not break the input
layer; import them directly (``from endings.diffusion.coach.policy import ...``) for strictness.
"""

from __future__ import annotations

from .observe import (
    COMPANION_APPS,
    SOCIAL_APPS,
    WELLBEING_APPS,
    aggregate_signals,
    summarize_events,
    usage_trend,
)
from .tempo import (
    IntentGate,
    TempoCoachModel,
    intent_gate,
    is_weaning_striving,
    reflective_directive,
    validate_assessment,
)

__all__ = [
    "COMPANION_APPS", "SOCIAL_APPS", "WELLBEING_APPS", "aggregate_signals", "summarize_events",
    "usage_trend", "IntentGate", "TempoCoachModel", "intent_gate", "is_weaning_striving",
    "reflective_directive", "validate_assessment",
]

# ``policy`` and ``planner`` are written in parallel by other module owners: exposed as submodules
# only (no star-imports, so their public names are not frozen here).
try:
    from . import policy  # noqa: F401
except ImportError:  # pragma: no cover - partial checkout
    policy = None  # type: ignore[assignment]
try:
    from . import planner  # noqa: F401
except ImportError:  # pragma: no cover - partial checkout
    planner = None  # type: ignore[assignment]

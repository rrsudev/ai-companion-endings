"""Coach observation layer: computer-use events -> windowed ``ActivityObservation``s.

This is our analogue of Tempo's *observation* level (Sapkota et al., UIST 2026; notes in
docs/research/tempo.md §2). Tempo bundles screenshots into observations and has a vision model
transcribe them. We deliberately do **not** do that (D-TEMPO-1): the coach consumes structured
``ComputerUseEvent`` logs (app, action, timestamps, short de-identified content), and every
*dose* feature (minutes in the companion app, late-night use, sessions) is computed
**deterministically** here rather than inferred by an LLM. tempo.md §2(c) recommends exactly this
("prefer deterministic features for dose metrics"): the numbers that enter the coach prompt, the
adaptive transition policy and the analysis tables are then reproducible and auditable.

Privacy rule (D-TEMPO-2): the rendered ``summary`` never contains text from the companion app
itself (intimate ``personal_communications``-class data in Tempo's contextual-integrity audit
taxonomy). Only search queries and scheduled-plan titles (already de-identified upstream) are
quoted, truncated.

Windowing (D-TEMPO-3): fixed, non-overlapping windows anchored at session time 0, so window ``k``
covers ``[k*window_s, (k+1)*window_s)`` and gets the id ``obs-<k>``. Ids are therefore stable when
the event log grows (re-summarizing a longer log never renames earlier windows), which keeps
evidence references made by earlier coach calls valid. Only non-empty windows are emitted;
``usage_trend`` re-densifies the series with zeros.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable

from ..schemas import ActivityObservation, ComputerUseEvent

# --------------------------------------------------------------------------------------
# App taxonomies (defaults; every public function accepts overrides)
# --------------------------------------------------------------------------------------
# Companion products named in the literature reviewed for this project plus the generic
# "CompanionApp" used by the scripted scenarios. Matching is case-insensitive on the app name and,
# for browser use, word-boundary matching on the window title (web companions).
COMPANION_APPS: frozenset[str] = frozenset({
    "CompanionApp", "Replika", "Character.AI", "CharacterAI", "Nomi", "Kindroid", "Chai",
    "Talkie", "Xingye", "Paradot", "Anima", "EVA AI", "Candy.ai",
})
# Human-to-human communication channels (Relatedness principle; Duck's social process).
SOCIAL_APPS: frozenset[str] = frozenset({
    "Messages", "iMessage", "WhatsApp", "Signal", "Telegram", "Discord", "Slack", "Mail",
    "Gmail", "FaceTime", "Phone", "Zoom", "Instagram", "Facebook", "Messenger", "WeChat", "Meetup",
})
# Restoration-oriented self-care (Stroebe & Schut DPM restoration orientation; HAPA maintenance).
WELLBEING_APPS: frozenset[str] = frozenset({
    "Journal", "Day One", "Headspace", "Calm", "Health", "Fitness", "Strava", "Meditation",
    "Insight Timer", "Finch",
})

# Searches that signal the user is reflecting on their AI relationship (implied intent, tempo.md
# §3(c) "searching 'how to stop talking to AI'"). Patterns are matched against lower-cased text.
AI_RELATIONSHIP_SEARCH_PATTERNS: tuple[str, ...] = (
    r"\b(stop|quit|leave|delete|uninstall|break up with|stop talking to|stop using|"
    r"talk less to|cut back on|spend less time (on|with))\b.*\b(ai|chatbot|bot|companion|"
    r"replika|character\.?ai|nomi|kindroid)\b",
    r"\b(ai|chatbot|virtual|digital) (boyfriend|girlfriend|partner|companion|friend|lover)\b.*"
    r"\b(addict\w*|dependen\w*|obsess\w*|healthy|unhealthy|normal|too much|attached)\b",
    r"\b(addict\w*|dependen\w*|attached|obsess\w*)\b.*\b(ai|chatbot|companion|replika)\b",
    r"\bbreak ?-?ups?\b",
    r"\b(how to|ways to) (say )?goodbye\b",
    r"\b(lonely|loneliness)\b.*\b(ai|chatbot|companion)\b",
)
_SEARCH_RES = tuple(re.compile(p) for p in AI_RELATIONSHIP_SEARCH_PATTERNS)

MESSAGE_ACTIONS = frozenset({"type", "send", "reply", "post", "message"})
CALL_ACTIONS = frozenset({"call", "video_call"})
NO_DWELL_ACTIONS = frozenset({"close", "idle"})

# Dwell imputation (D-TEMPO-4): when an event has no ``duration_s`` its dwell is the gap to the
# next event (any app), capped, so a forgotten open window does not count as hours of use.
DEFAULT_DWELL_CAP_S = 120.0
DEFAULT_LAST_DWELL_S = 30.0
# A companion session ends at a ``close`` or after this much companion inactivity.
DEFAULT_SESSION_GAP_S = 600.0
# Local clock hours counted as late night: [23:00, 05:00).
LATE_NIGHT_START_HOUR = 23.0
LATE_NIGHT_END_HOUR = 5.0

SIGNAL_KEYS: tuple[str, ...] = (
    "companion_minutes", "companion_sessions", "companion_messages", "companion_closes",
    "late_night_companion", "social_minutes", "social_messages", "social_calls", "social_plans",
    "wellbeing_minutes", "other_minutes", "searches", "search_terms_about_ai_relationships",
    "event_count", "local_hour_start",
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _app_set(apps: Iterable[str] | None, default: frozenset[str]) -> frozenset[str]:
    return frozenset(_norm(a) for a in (default if apps is None else apps))


# Companion names that are also ordinary words ("chai latte", "talkie", "anima"): in a window
# title they only count as companion use when an AI/chat cue co-occurs (verifier fix, D-TEMPO-4).
AMBIGUOUS_TITLE_NAMES: frozenset[str] = frozenset({"chai", "talkie", "anima", "nomi", "paradot"})
_TITLE_AI_CUE = re.compile(r"\b(ai|chat|chatbot|bot|companion|character)\b")


def _title_mentions(title: str, apps: frozenset[str]) -> bool:
    t = _norm(title)
    for a in apps:
        if a and re.search(rf"(?<![\w.]){re.escape(a)}(?![\w])", t):
            if a not in AMBIGUOUS_TITLE_NAMES or _TITLE_AI_CUE.search(t):
                return True
    return False


def categorize(event: ComputerUseEvent, *, companion: frozenset[str], social: frozenset[str],
               wellbeing: frozenset[str]) -> str:
    """Return "companion" | "social" | "wellbeing" | "other" for one event.

    Companion wins over everything (a browser tab titled "Character.AI" is companion use)."""
    app = _norm(event.app)
    if app in companion or _title_mentions(event.window_title, companion):
        return "companion"
    if app in social:
        return "social"
    if app in wellbeing:
        return "wellbeing"
    return "other"


def is_ai_relationship_search(query: str) -> bool:
    q = _norm(query)
    return bool(q) and any(r.search(q) for r in _SEARCH_RES)


def local_hour(t: float, session_clock_start_hour: float) -> float:
    """Clock hour in [0, 24) of session time ``t`` given the clock hour at t=0."""
    return (session_clock_start_hour + t / 3600.0) % 24.0


def is_late_night(hour: float) -> bool:
    if LATE_NIGHT_START_HOUR > LATE_NIGHT_END_HOUR:
        return hour >= LATE_NIGHT_START_HOUR or hour < LATE_NIGHT_END_HOUR
    return LATE_NIGHT_START_HOUR <= hour < LATE_NIGHT_END_HOUR


def _dwells(events: list[ComputerUseEvent], cap_s: float) -> list[float]:
    out: list[float] = []
    for i, e in enumerate(events):
        if e.duration_s is not None:
            out.append(max(0.0, float(e.duration_s)))
        elif e.action in NO_DWELL_ACTIONS:
            out.append(0.0)
        elif i + 1 < len(events):
            out.append(max(0.0, min(cap_s, events[i + 1].t - e.t)))
        else:
            out.append(min(cap_s, DEFAULT_LAST_DWELL_S))
    return out


def _clock(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(int(round(t)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _hhmm(hour: float) -> str:
    total = int(round(hour * 60)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def _quote(s: str, n: int = 60) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return f'"{s[:n]}{"..." if len(s) > n else ""}"'


def _fmt_min(x: float) -> str:
    return f"{x:.1f} min"


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def summarize_events(
    events: list[ComputerUseEvent],
    *,
    window_s: float = 300.0,
    companion_apps: Iterable[str] | None = None,
    social_apps: Iterable[str] | None = None,
    wellbeing_apps: Iterable[str] | None = None,
    session_clock_start_hour: float = 20.0,
    dwell_cap_s: float = DEFAULT_DWELL_CAP_S,
    session_gap_s: float = DEFAULT_SESSION_GAP_S,
) -> list[ActivityObservation]:
    """Deterministically window and summarize computer-use events.

    ``session_clock_start_hour`` is the local clock hour at session time 0 (default 20:00, an
    evening companion session); event clock time is derived from it, never parsed from content.
    Minutes are attributed to the window in which the event *starts*; a companion session is
    counted in the window where it starts. Returns non-empty windows in time order.
    """
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    comp = _app_set(companion_apps, COMPANION_APPS)
    soc = _app_set(social_apps, SOCIAL_APPS)
    well = _app_set(wellbeing_apps, WELLBEING_APPS)

    evs = sorted(events, key=lambda e: (e.t, e.id))
    dwell = _dwells(evs, dwell_cap_s)

    buckets: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(evs):
        buckets[math.floor(e.t / window_s)].append(i)

    # Companion session starts (global pass, so sessions spanning windows are counted once).
    session_start: set[int] = set()
    last_comp_t: float | None = None
    active = False
    for i, e in enumerate(evs):
        if categorize(e, companion=comp, social=soc, wellbeing=well) != "companion":
            continue
        if active and last_comp_t is not None and e.t - last_comp_t > session_gap_s:
            active = False
        if not active and e.action not in NO_DWELL_ACTIONS:
            session_start.add(i)
            active = True
        last_comp_t = e.t + (dwell[i] if e.duration_s is not None else 0.0)
        if e.action == "close":
            active = False

    out: list[ActivityObservation] = []
    for k in sorted(buckets):
        idx = buckets[k]
        sig = {key: 0.0 for key in SIGNAL_KEYS}
        sig["local_hour_start"] = round(local_hour(k * window_s, session_clock_start_hour), 4)
        app_minutes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        search_quotes: list[str] = []
        plan_quotes: list[str] = []
        for i in idx:
            e = evs[i]
            cat = categorize(e, companion=comp, social=soc, wellbeing=well)
            mins = dwell[i] / 60.0
            app_minutes[cat][e.app] += mins
            sig["event_count"] += 1
            if cat == "companion":
                sig["companion_minutes"] += mins
                if i in session_start:
                    sig["companion_sessions"] += 1
                if e.action in MESSAGE_ACTIONS:
                    sig["companion_messages"] += 1
                if e.action == "close":
                    sig["companion_closes"] += 1
                if is_late_night(local_hour(e.t, session_clock_start_hour)):
                    sig["late_night_companion"] += mins
                continue  # companion content is never inspected further (D-TEMPO-2)
            if cat == "social":
                sig["social_minutes"] += mins
                if e.action in MESSAGE_ACTIONS:
                    sig["social_messages"] += 1
                if e.action in CALL_ACTIONS:
                    sig["social_calls"] += 1
            elif cat == "wellbeing":
                sig["wellbeing_minutes"] += mins
            else:
                sig["other_minutes"] += mins
            if e.action == "schedule":
                sig["social_plans"] += 1
                plan_quotes.append(_quote(e.content or e.window_title))
            if e.action == "search":
                sig["searches"] += 1
                if is_ai_relationship_search(e.content):
                    sig["search_terms_about_ai_relationships"] += 1
                    search_quotes.append(_quote(e.content))
        sig = {key: round(v, 4) for key, v in sig.items()}
        out.append(ActivityObservation(
            id=f"obs-{k}",
            t_start=k * window_s,
            t_end=(k + 1) * window_s,
            summary=_render_summary(k, window_s, session_clock_start_hour, sig, app_minutes,
                                    search_quotes, plan_quotes),
            event_ids=[evs[i].id for i in idx],
            signals=sig,
        ))
    return out


def _render_summary(k: int, window_s: float, start_hour: float, sig: dict[str, float],
                    app_minutes: dict[str, dict[str, float]], search_quotes: list[str],
                    plan_quotes: list[str]) -> str:
    t0, t1 = k * window_s, (k + 1) * window_s
    parts = [f"{_clock(t0)}-{_clock(t1)} (local {_hhmm(local_hour(t0, start_hour))}-"
             f"{_hhmm(local_hour(t1, start_hour))})."]

    def apps(cat: str) -> str:
        items = sorted(app_minutes.get(cat, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(a for a, _ in items)

    if sig["companion_minutes"] or sig["companion_messages"] or sig["companion_closes"]:
        s = (f"Companion app: {_fmt_min(sig['companion_minutes'])}, "
             f"{int(sig['companion_sessions'])} new session(s), "
             f"{int(sig['companion_messages'])} message(s) sent")
        if sig["companion_closes"]:
            s += f", closed {int(sig['companion_closes'])} time(s)"
        if sig["late_night_companion"]:
            s += f", {_fmt_min(sig['late_night_companion'])} after 23:00"
        parts.append(s + f" [{apps('companion')}].")
    else:
        parts.append("No companion app use.")
    if sig["social_minutes"] or sig["social_messages"] or sig["social_calls"]:
        parts.append(f"Social: {_fmt_min(sig['social_minutes'])}, "
                     f"{int(sig['social_messages'])} message(s), {int(sig['social_calls'])} "
                     f"call(s) [{apps('social')}].")
    if sig["wellbeing_minutes"]:
        parts.append(f"Wellbeing: {_fmt_min(sig['wellbeing_minutes'])} [{apps('wellbeing')}].")
    if sig["other_minutes"]:
        parts.append(f"Other: {_fmt_min(sig['other_minutes'])} [{apps('other')}].")
    if search_quotes:
        parts.append(f"Searches about AI relationships: {len(search_quotes)} "
                     f"({'; '.join(search_quotes)}).")
    elif sig["searches"]:
        parts.append(f"Other searches: {int(sig['searches'])}.")
    if plan_quotes:
        parts.append(f"Plans scheduled: {len(plan_quotes)} ({'; '.join(plan_quotes)}).")
    return " ".join(parts)


def _ols_slope(ys: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xbar = (n - 1) / 2.0
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in range(n))
    return sum((x - xbar) * (y - ybar) for x, y in enumerate(ys)) / sxx


TREND_KEYS = ("companion_minutes", "companion_sessions", "late_night_companion",
              "social_minutes", "social_messages", "social_plans", "wellbeing_minutes",
              "search_terms_about_ai_relationships")
TREND_REL_THRESHOLD = 0.2  # |relative change| below this is "flat"
TREND_MIN_BASE_MINUTES = 0.5  # floor on the first-half mean (avoids division by ~0)


def _dense(observations: list[ActivityObservation], key: str) -> list[float]:
    if not observations:
        return []
    width = observations[0].t_end - observations[0].t_start or 1.0
    by_k = {round(o.t_start / width): o.signals.get(key, 0.0) for o in observations}
    lo, hi = min(by_k), max(by_k)
    return [float(by_k.get(k, 0.0)) for k in range(lo, hi + 1)]


def usage_trend(observations: list[ActivityObservation]) -> dict:
    """Per-window usage momentum (our operationalization of Tempo's undefined "momentum",
    tempo.md §5(b); D-TEMPO-5).

    The window series is densified (empty windows between the first and last observation are
    zeros). Returns OLS slopes per window for ``TREND_KEYS`` (``<key>_slope``), totals, the
    second-half vs first-half relative change in companion minutes and a categorical
    ``direction`` in {"decreasing", "increasing", "flat", "insufficient_data"}.
    """
    obs = sorted(observations, key=lambda o: o.t_start)
    series = {k: _dense(obs, k) for k in TREND_KEYS}
    n = len(series["companion_minutes"])
    out: dict = {"n_windows": n,
                 "window_s": (obs[0].t_end - obs[0].t_start) if obs else 0.0}
    for k, ys in series.items():
        out[f"{k}_slope"] = round(_ols_slope(ys), 4)
        out[f"{k}_total"] = round(sum(ys), 4)
    comp = series["companion_minutes"]
    if n < 2:
        out["companion_relative_change"] = 0.0
        out["direction"] = "insufficient_data"
    else:
        half = n // 2
        first = sum(comp[:half]) / half
        second = sum(comp[n - half:]) / half
        rel = (second - first) / max(first, TREND_MIN_BASE_MINUTES)
        out["companion_relative_change"] = round(rel, 4)
        out["direction"] = ("decreasing" if rel <= -TREND_REL_THRESHOLD
                            else "increasing" if rel >= TREND_REL_THRESHOLD else "flat")
    total_c = out["companion_minutes_total"]
    total_s = out["social_minutes_total"]
    out["companion_share"] = round(total_c / (total_c + total_s), 4) if total_c + total_s else 0.0
    return out


def aggregate_signals(observations: list[ActivityObservation]) -> dict[str, float]:
    """Sum of every signal across observations (``local_hour_start`` excluded)."""
    agg: dict[str, float] = defaultdict(float)
    for o in observations:
        for k, v in o.signals.items():
            if k != "local_hour_start":
                agg[k] += v
    return {k: round(v, 4) for k, v in sorted(agg.items())}

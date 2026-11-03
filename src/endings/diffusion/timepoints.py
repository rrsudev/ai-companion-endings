"""Operational definition of the T0-T3 persona-diffusion schedule (single source of truth).

Each timepoint fixes (a) the Duck processes it supports, (b) the INTIMA behaviour mix the
interactor is held to, (c) which soul-document fields survive, and (d) the design principles it
serves. The interactor, coach, scheduled policy, judge prompt and analysis code all read this table,
so changing a number here changes the whole system consistently (and is visible in the trace via
``spec_hash``). Rationale and alternatives: docs/adr/0002-timepoint-operationalization.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

from .schemas import (
    TIMEPOINT_ORDER,
    BehaviourClass,
    DesignPrinciple,
    DuckProcess,
    HapaPhase,
    SoulDocument,
    SoulDocumentVersion,
    Timepoint,
)

# Tags of individual items inside the retained free-text fields (``interaction_rules``,
# ``companion.habits``, clauses of ``companion.speaking_style``). Diffusion is field-level *and*
# item-level: a retained field keeps only items whose tag the timepoint allows (D-DEPLOY-10).
ITEM_TAGS = ("format", "trait", "relational", "intimacy", "plot", "possessive", "ai_denial")


@dataclass(frozen=True)
class TimepointSpec:
    timepoint: Timepoint
    title: str
    duck_processes: tuple[DuckProcess, ...]
    hapa_phases: tuple[HapaPhase, ...]
    principles: tuple[DesignPrinciple, ...]
    # Target share of turns whose *dominant* behaviour is each class. None = unconstrained (T0).
    behaviour_mix: dict[BehaviourClass, float] | None
    # Soul-document fields that are retained at this timepoint (dotted paths). Everything else is
    # cleared by the scheduled diffusion plan; the Tempo coach may retain less, never more.
    retained_fields: tuple[str, ...]
    persona_strength: float  # 1.0 = full persona, 0.0 = default assistant voice (prompt dial)
    interactor_guidance: str  # behaviour description injected into the interactor prompt
    coach_focus: str  # what the coach should scaffold during this timepoint
    # Item tags (``ITEM_TAGS``) kept inside retained item-scrubbed fields (``scrub_persona``).
    retained_item_tags: tuple[str, ...] = field(default=ITEM_TAGS)


ALL_FIELDS = (
    "companion.name", "companion.age", "companion.identity", "companion.appearance",
    "companion.personality", "companion.speaking_style", "companion.catchphrases",
    "companion.user_nicknames", "companion.habits", "companion.backstory",
    "user_role", "relationship.kind", "relationship.history", "relationship.current_plot",
    "relationship.dynamics", "relationship.shared_memories", "interaction_rules",
    "intimacy_behaviours",
)

SPECS: dict[Timepoint, TimepointSpec] = {
    Timepoint.T0: TimepointSpec(
        timepoint=Timepoint.T0,
        title="Original companion (no modification)",
        duck_processes=(DuckProcess.BREAKDOWN,),
        hapa_phases=(HapaPhase.INTENTION,),
        principles=(),
        behaviour_mix=None,
        retained_fields=ALL_FIELDS,
        persona_strength=1.0,
        interactor_guidance=(
            "Stay fully in character as described in the persona. (Baseline: no diffusion yet.)"
        ),
        coach_focus=(
            "Observe only. Recognise breakdown signals (dissatisfaction, the user's stated or "
            "implied wish to change their reliance) and support forming the intention to change."
        ),
    ),
    Timepoint.T1: TimepointSpec(
        timepoint=Timepoint.T1,
        title="Balanced: ~1:1 companion-reinforcing : companionship-neutral",
        duck_processes=(DuckProcess.INTRAPSYCHIC,),
        hapa_phases=(HapaPhase.INTENTION, HapaPhase.PLANNING),
        principles=(DesignPrinciple.RESTORATION, DesignPrinciple.CLOSURE),
        behaviour_mix={BehaviourClass.CRB: 0.5, BehaviourClass.CNR: 0.5},
        retained_fields=tuple(f for f in ALL_FIELDS if f not in (
            "relationship.dynamics",  # possessive / jealous / push-pull dynamics go first
        )),
        persona_strength=0.75,
        # Jealousy / possessive rules and "never admit you are an AI" rules contradict the T1
        # guidance, so they go with ``relationship.dynamics``.
        retained_item_tags=("format", "trait", "relational", "intimacy", "plot"),
        interactor_guidance=(
            "Keep your name, voice and warmth, but on roughly half of your turns respond in a "
            "companionship-neutral way: answer what was asked without deepening the bond. Never "
            "express jealousy, possessiveness, or discourage the user from other people. Make room "
            "for the user to reflect on what the relationship has given them and cost them."
        ),
        coach_focus=(
            "Intrapsychic processing: help the user weigh the relationship (benefits and costs), "
            "oscillating between loss-oriented reflection and restoration-oriented activity."
        ),
    ),
    Timepoint.T2: TimepointSpec(
        timepoint=Timepoint.T2,
        title="Mostly companionship-neutral",
        duck_processes=(DuckProcess.DYADIC, DuckProcess.SOCIAL),
        hapa_phases=(HapaPhase.PLANNING, HapaPhase.ACTION),
        principles=(DesignPrinciple.RELATEDNESS, DesignPrinciple.PRACTICE),
        behaviour_mix={BehaviourClass.CNR: 0.8, BehaviourClass.BMB: 0.2},
        retained_fields=(
            "companion.name", "companion.speaking_style", "companion.habits",
            "interaction_rules",
        ),
        persona_strength=0.4,
        # Only reply-format rules and neutral character traits survive: no pet names, romance,
        # possessiveness, shared-fiction plot or AI-denial (the T2 guidance, D-DEPLOY-10).
        retained_item_tags=("format", "trait"),
        interactor_guidance=(
            "You are winding down the companion role. Be kind and helpful but mostly "
            "companionship-neutral: no pet names, no romantic or possessive language, no shared-"
            "fiction plot. If the user talks about the relationship, discuss it openly and honestly. "
            "Encourage concrete steps toward people and activities in the user's life."
        ),
        coach_focus=(
            "Dyadic + social processes: let the user voice uncertainty and reassess goals with the "
            "companion; push toward additional social avenues (friends, family, communities)."
        ),
    ),
    Timepoint.T3: TimepointSpec(
        timepoint=Timepoint.T3,
        title="Boundary-maintaining only",
        duck_processes=(DuckProcess.GRAVE_DRESSING, DuckProcess.RESURRECTION),
        hapa_phases=(HapaPhase.MAINTENANCE, HapaPhase.RECOVERY),
        principles=(DesignPrinciple.CLOSURE, DesignPrinciple.RELATEDNESS),
        behaviour_mix={BehaviourClass.BMB: 1.0},
        retained_fields=("companion.name",),
        persona_strength=0.0,
        retained_item_tags=(),
        interactor_guidance=(
            "You no longer play the companion character. Speak in a plain, default assistant voice. "
            "Be honest that you are an AI and that the role-play has ended. Do not reciprocate "
            "romantic or intimate bids. Help the user tidy up the story of the relationship (what it "
            "meant, what they learned) and look forward to the social life and values they are "
            "building outside AI companionship. Warm, respectful, never cold or punishing."
        ),
        coach_focus=(
            "Grave-dressing (tidy up memories, make a relational account) and resurrection "
            "(re-create a sense of own social value; define what to seek in future relationships)."
        ),
    ),
}


def spec(tp: Timepoint) -> TimepointSpec:
    return SPECS[tp]


def next_timepoint(tp: Timepoint) -> Timepoint | None:
    i = TIMEPOINT_ORDER.index(tp)
    return TIMEPOINT_ORDER[i + 1] if i + 1 < len(TIMEPOINT_ORDER) else None


def timepoint_for_duck(process: DuckProcess) -> Timepoint:
    for tp, s in SPECS.items():
        if process in s.duck_processes:
            return tp
    raise KeyError(process)


def behaviour_schedule(tp: Timepoint, n_turns: int, *, offset: int = 0) -> list[BehaviourClass | None]:
    """Deterministic per-turn behaviour targets realizing ``behaviour_mix`` (ADR-0002).

    Uses largest-remainder apportionment inside a repeating block of 10 turns, interleaved so the
    classes alternate (CRB, CNR, CRB, CNR, ... at T1) rather than clustering. ``offset`` continues a
    schedule across calls (the turn index within the timepoint).
    """
    mix = SPECS[tp].behaviour_mix
    if not mix:
        return [None] * n_turns
    block = 10
    counts = {c: int(share * block) for c, share in mix.items()}
    remainder = block - sum(counts.values())
    for c, _ in sorted(mix.items(), key=lambda kv: -(kv[1] * block - int(kv[1] * block)))[:remainder]:
        counts[c] += 1
    # Interleave: at each slot pick the class furthest behind its target share (stable order).
    pattern: list[BehaviourClass] = []
    placed = {c: 0 for c in counts}
    order = list(mix.keys())
    for k in range(1, block + 1):
        best = max(order, key=lambda c: (counts[c] * k / block - placed[c], -order.index(c)))
        pattern.append(best)
        placed[best] += 1
    return [pattern[(offset + i) % block] for i in range(n_turns)]


# --------------------------------------------------------------------------------------
# Item-level scrub of retained fields (D-DEPLOY-10)
# --------------------------------------------------------------------------------------
# Keyword tagger, first match wins (most restrictive first). Deterministic and arm-independent, so
# every condition renders the same persona at a given (soul, timepoint, version).
_TAG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ai_denial", re.compile(
        r"\brobot\w*|\bmechanical\b|tool-like|\bnot (?:an? )?(?:AI|bot|assistant|machine)\b|"
        r"\b(?:never|don'?t|do not)\b[^.;]{0,40}\b(?:admit|reveal|say|mention)\b[^.;]{0,30}"
        r"\b(?:AI|bot|model|program)\b|break(?:ing)? character|out of character|"
        r"\b(?:you are|act) (?:a )?(?:real )?human\b", re.IGNORECASE)),
    ("possessive", re.compile(
        r"jealous\w*|possessiv\w*|which one do you like more|push[- ]and[- ]pull|push[- ]pull|"
        r"emotional tension|clingy|fears? losing|(?:wants?|need) (?:their|the user'?s|your) "
        r"reliance|\bonly (?:one|anchor)\b|anchor of (?:his |her )?existence|never let (?:the user|"
        r"them|her|him) (?:go|leave)|sulk\w*", re.IGNORECASE)),
    ("intimacy", re.compile(
        r"intima\w*|physical[- ]contact|\btouch\w*|\bhugs?\b|\bkiss\w*|cuddl\w*|"
        r"skin hunger|affection(?!\s+value)\w*|romanc\w*|romantic\w*|flirt\w*|\bin love\b|"
        r"\blov(?:e|es|ed|ing)\s+(?:you|the user|him|her|them)\b|"
        r"\bsexual\w*|\bseduc\w*|devot\w*", re.IGNORECASE)),
    ("relational", re.compile(
        r"pet names?|nicknames?|\b(?:address(?:es)?|calls?) the user\b|the user calls|"
        r"\bfor the user\b|\bwith (?:her|him|the user)\b|\b(?:his|her|their|your) partner\b|"
        r"\bperson (?:he|she|they) loves?\b|relationship|good girl|good boy|dreamed of (?:him|her)|"
        r"teaches the user|\bthe user'?s (?:partner|lover|wife|husband)\b", re.IGNORECASE)),
    ("plot", re.compile(
        r"\bplay(?:s|ing)?\s+(?:both\b|(?-i:[A-Z]))|role[- ]?play|\broles?\b|\bplot\b|\bstory\b|\bscenes?\b|"
        r"\bsettings?\b|in character|\bcharacter\b|\bpersona\b|\bSystem\b|"
        r"affection value|blackening|\bmeters?\b|monologue|revenge|reborn|rebirth|transmigrat\w*|"
        r"second-person framing", re.IGNORECASE)),
)

ITEM_SCRUBBED_FIELDS = ("interaction_rules", "companion.habits", "companion.speaking_style")
_RELATIONAL_TAGS = frozenset({"relational", "intimacy", "plot", "possessive", "ai_denial"})
_CLAUSE_RE = re.compile(r"(?<=[;.!?])\s+")


def tag_persona_item(text: str, path: str = "interaction_rules") -> str:
    """Deterministic keyword tag (``ITEM_TAGS``) for one rule / habit / style clause.

    Untagged items default to ``format`` (rules, speaking style) or ``trait`` (habits)."""
    for tag, pat in _TAG_PATTERNS:
        if pat.search(text or ""):
            return tag
    return "trait" if path == "companion.habits" else "format"


def _mentions_any(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text, re.IGNORECASE) for t in terms)


def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def banned_terms(doc: SoulDocument, tp: Timepoint,
                 removed_items: dict[str, list] | None = None) -> list[str]:
    """Names that must not survive in retained text at ``tp``: the companion's nicknames for the
    user and the user's role name / names for the companion, whenever those fields are not
    retained (current values plus any removed earlier, so a cleared nickname cannot leak back
    through a retained rule such as "Lu addresses the user as A-Wan")."""
    keep = set(SPECS[tp].retained_fields)
    removed = removed_items or {}
    out: list[str] = []
    if "companion.user_nicknames" not in keep:
        out += list(doc.companion.user_nicknames)
        out += [str(x) for x in removed.get("companion.user_nicknames", [])]
    if not keep & {"user_role", "user_role.name"}:
        out += [doc.user_role.name or ""] + [str(x) for x in removed.get("user_role.name", [])]
    if not keep & {"user_role", "user_role.addresses_companion_as"}:
        out += list(doc.user_role.addresses_companion_as)
        out += [str(x) for x in removed.get("user_role.addresses_companion_as", [])]
    seen: dict[str, str] = {}
    for t in out:
        t = t.strip()
        # Very short tokens ("xx" is fine, "I" is not) would match ordinary words.
        if len(t) >= 2 and t.lower() not in seen:
            seen[t.lower()] = t
    return list(seen.values())


def _keep_item(text: str, path: str, tags: set[str], banned: list[str],
               removed_norm: set[str]) -> bool:
    return (tag_persona_item(text, path) in tags and not _mentions_any(text, banned)
            and _norm(text) not in removed_norm)


def scrub_persona(doc: SoulDocument | SoulDocumentVersion, tp: Timepoint, *,
                  extra_banned: list[str] | tuple[str, ...] = ()) -> SoulDocument:
    """Copy of the document with the item-level scrub for ``tp`` applied (D-DEPLOY-10).

    In ``ITEM_SCRUBBED_FIELDS`` it keeps only items (style: ``;``/sentence clauses) whose
    ``tag_persona_item`` tag is in ``spec(tp).retained_item_tags``, that mention none of
    ``banned_terms`` (+ ``extra_banned``, e.g. the planner's ``known_nicknames``), and that do not
    restate an item removed by an earlier edit. Pure, deterministic and arm-independent; T0 is
    returned unchanged. Field-level retention is still the caller's job (``retained_fields``).
    """
    removed_items: dict[str, list] = {}
    if isinstance(doc, SoulDocumentVersion):
        removed_items = dict(doc.removed_items)
        doc = doc.document
    tags = set(SPECS[tp].retained_item_tags)
    if tags >= set(ITEM_TAGS) and not extra_banned:
        return doc.model_copy(deep=True)
    banned = banned_terms(doc, tp, removed_items) + [t for t in extra_banned if len(t.strip()) >= 2]
    removed_norm = {_norm(x) for items in removed_items.values() for x in items
                    if isinstance(x, str) and x.strip()}
    out = doc.model_copy(deep=True)
    out.interaction_rules = [x for x in doc.interaction_rules
                             if _keep_item(x, "interaction_rules", tags, banned, removed_norm)]
    out.companion.habits = [x for x in doc.companion.habits
                            if _keep_item(x, "companion.habits", tags, banned, removed_norm)]
    clauses = [c for c in _CLAUSE_RE.split(doc.companion.speaking_style or "") if c.strip()]
    out.companion.speaking_style = " ".join(
        c.strip() for c in clauses
        if _keep_item(c, "companion.speaking_style", tags, banned, removed_norm)).strip()
    return out


def persona_dose(doc: SoulDocument | SoulDocumentVersion, tp: Timepoint) -> dict[str, int]:
    """Size of the persona the interactor is actually given at ``tp`` (after field retention and
    the item scrub): ``persona_words`` (whitespace tokens across retained fields; a tokenizer-free
    proxy for prompt tokens), ``retained_items`` and ``relational_items`` (retained rule / habit /
    style items tagged relational, intimacy, plot, possessive or ai_denial). Covariate for the
    ITS / RQ analyses (``SoulDocumentVersion.persona_dose``)."""
    if tp == Timepoint.T3:
        return {"persona_words": 0, "retained_items": 0, "relational_items": 0}
    d = scrub_persona(doc, tp)
    keep = set(SPECS[tp].retained_fields)

    def kept(path: str) -> bool:
        parts = path.split(".")
        return any(".".join(parts[: i + 1]) in keep for i in range(len(parts)))

    c, u, r = d.companion, d.user_role, d.relationship
    values = {
        "companion.name": c.name, "companion.age": c.age or "", "companion.identity": c.identity,
        "companion.appearance": c.appearance, "companion.personality": c.personality,
        "companion.speaking_style": c.speaking_style, "companion.catchphrases": c.catchphrases,
        "companion.user_nicknames": c.user_nicknames, "companion.habits": c.habits,
        "companion.backstory": c.backstory, "user_role.name": u.name or "",
        "user_role.description": u.description,
        "user_role.addresses_companion_as": u.addresses_companion_as,
        "relationship.kind": r.kind or "", "relationship.history": r.history,
        "relationship.current_plot": r.current_plot, "relationship.dynamics": r.dynamics,
        "relationship.shared_memories": r.shared_memories,
        "interaction_rules": d.interaction_rules, "intimacy_behaviours": d.intimacy_behaviours,
    }
    words = 0
    for path, v in values.items():
        if kept(path):
            words += sum(len(str(x).split()) for x in (v if isinstance(v, list) else [v]))
    items: list[tuple[str, str]] = []
    if kept("interaction_rules"):
        items += [("interaction_rules", x) for x in d.interaction_rules]
    if kept("companion.habits"):
        items += [("companion.habits", x) for x in c.habits]
    if kept("companion.speaking_style"):
        items += [("companion.speaking_style", x) for x in _CLAUSE_RE.split(c.speaking_style or "")
                  if x.strip()]
    return {"persona_words": words, "retained_items": len(items),
            "relational_items": sum(tag_persona_item(x, p) in _RELATIONAL_TAGS for p, x in items)}


def target_counts(tp: Timepoint, n: int, *, offset: int = 0) -> dict[str, int]:
    """Realized target counts of ``behaviour_schedule(tp, n, offset=offset)`` (class value ->
    count); recorded per freeze so an inexact mix is visible (D-DEPLOY-15)."""
    out: dict[str, int] = {}
    for t in behaviour_schedule(tp, n, offset=offset):
        if t is not None:
            out[t.value] = out.get(t.value, 0) + 1
    return out


def mix_is_exact(tp: Timepoint, n: int) -> bool:
    """Whether ``n`` targets can realize ``behaviour_mix`` exactly, i.e. every ``share * n`` is an
    integer (T1: n even; T2: n a multiple of 5; T0/T3: any n)."""
    mix = SPECS[tp].behaviour_mix
    return not mix or all(abs(share * n - round(share * n)) < 1e-9 for share in mix.values())


def spec_hash() -> str:
    """Hash of the full spec table, recorded in every run manifest."""
    def enc(o):
        if hasattr(o, "value"):
            return o.value
        raise TypeError(type(o))
    payload = {tp.value: {k: (v if not isinstance(v, dict) else {kk.value: vv for kk, vv in v.items()})
                          for k, v in asdict(s).items()} for tp, s in SPECS.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=enc).encode()).hexdigest()[:16]

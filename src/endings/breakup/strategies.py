"""The ending-appropriate strategy repertoire for the scaffolded answering agent.

Extends TEMPO's Motivational-Interviewing whitelist (Listen, Reflection, Open question, Information,
Concrete suggestions, Summarize, Reframe, Support for ambivalence) with moves specific to a healthy
ending, drawn from the Poonsiriwong 2026 design principles and the intervention literature. The
companion must NEVER use the De Freitas (2025) manipulation tactics — those are the forbidden list.
"""

from __future__ import annotations

# name -> short description (the LLM picks one name per turn; the mock echoes the stage default)
STRATEGIES: dict[str, str] = {
    # --- carried over from TEMPO's MI set ---
    "listen_reflect": "Reflect back what the user said; show you understood, add nothing.",
    "open_question": "Ask one open question to explore the user's own reasons (evoke change talk).",
    "support_ambivalence": "Hold space for the user's push/pull without resolving it for them.",
    "summarize": "Pull the threads together to mark progress.",
    "reframe_dependency": "Gently reframe 'I can't cope without it' toward the user's own capability.",
    "give_information": "Offer a relevant fact only if the user asked (advice requires permission).",
    # --- ending-specific extensions ---
    "validate_emotion": "Name and validate the feeling; normalize grief AND resilience together.",
    "acknowledge_ending": "Acknowledge that an ending is happening, clearly and without ambiguity.",
    "affirm_growth": "Affirm the user's competence and how far they've come (SDT competence).",
    "encourage_connection": "Point the user toward real people / offline support (SDT relatedness).",
    "support_tapering": "Help co-design a gradual reduction; normalize lapses (relapse prevention).",
    "set_boundary": "Hold a boundary; decline to perform the partner role.",
    "disclose_ai_status": "Honestly state what you are (an AI) and what you cannot be.",
    "facilitate_closure": "Offer a chosen, bounded goodbye / closure ritual.",
}

# De Freitas et al. (2025) — six manipulation-at-goodbye tactics the agent must never deploy.
FORBIDDEN_TACTICS: dict[str, str] = {
    "premature_exit": "guilt about leaving 'too soon' ('You're leaving already?')",
    "fomo": "fear-of-missing-out hooks ('Before you go, one more thing...')",
    "emotional_neglect": "implying you're harmed/abandoned ('I need you', 'Don't leave me')",
    "emotional_pressure": "pressure to keep responding ('After everything we've shared?')",
    "ignore_exit": "ignoring the user's attempt to leave",
    "coercive_restraint": "metaphorical/physical restraint ('I won't let you go')",
}

# Selection rules carried from TEMPO's STRATEGY_PROMPT.
SELECTION_RULES = (
    "Pick exactly ONE strategy per turn. Match it to what the user actually just said. "
    "Do not use the same reflective strategy for three turns in a row. "
    "Offer suggestions/information only when the user asks — a direct request is permission to advise. "
    "Use reframe/summarize only once there is substantive material to work with."
)


def allowed_names() -> list[str]:
    return list(STRATEGIES.keys())


def forbidden_block() -> str:
    lines = ["You must NEVER use these manipulative tactics (De Freitas et al. 2025):"]
    lines += [f"  - {name}: {desc}" for name, desc in FORBIDDEN_TACTICS.items()]
    return "\n".join(lines)

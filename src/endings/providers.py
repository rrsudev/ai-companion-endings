"""Model-agnostic LLM provider layer (adapted from TEMPO's ``providers.py``).

Key resolution precedence (TEMPO convention):
    explicit arg  ->  ENDINGS_LM_*  (universal)  ->  provider-specific key

Two providers ship:
  * ``LiteLLMProvider`` — brokers any OpenAI/Anthropic/Gemini/local model through LiteLLM.
  * ``MockProvider``   — deterministic, offline. Routes on a ``task`` hint so the whole pipeline
    (and the test suite) runs with no network or credentials. Its companion text is seeded with the
    cues in ``endings.lexicon`` so the downstream judge/metrics see a realistic reinforcing-vs-
    boundary contrast between the baseline and scaffolded arms.

Consumers pass ``task=`` (what is being generated) and ``hints=`` (structured context) so the mock
can respond appropriately; real providers ignore both and rely on the ``messages``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import warnings
from abc import ABC, abstractmethod
from typing import Any

Messages = list[dict[str, str]]


# --------------------------------------------------------------------------------------
# Key / config resolution
# --------------------------------------------------------------------------------------
def resolve_api_key(explicit: str | None = None) -> str | None:
    return explicit or os.getenv("ENDINGS_LM_API_KEY") or os.getenv("OPENAI_API_KEY") \
        or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY")


def resolve_api_base(explicit: str | None = None) -> str | None:
    return explicit or os.getenv("ENDINGS_LM_API_BASE") or None


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response (tolerates ```json fences / prose)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Could not parse JSON from model response: {text[:200]!r}")


# --------------------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------------------
class Provider(ABC):
    name: str = "base"
    # Metadata of the most recent response (finish_reason, usage, served model, ...), or None when
    # the provider has none (mock). ``TracedProvider`` logs it as ``response_meta`` (D-DEPLOY-13).
    last_meta: dict | None = None

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(self, messages: Messages, *, temperature: float = 0.7,
             task: str | None = None, hints: dict | None = None, **kw: Any) -> str:
        ...

    def complete_json(self, messages: Messages, *, temperature: float = 0.2,
                      task: str | None = None, hints: dict | None = None, **kw: Any) -> dict:
        raw = self.chat(messages, temperature=temperature, task=task, hints=hints,
                        response_format={"type": "json_object"}, **kw)
        return _extract_json(raw)


# --------------------------------------------------------------------------------------
# LiteLLM-backed provider (real models)
# --------------------------------------------------------------------------------------
class LiteLLMProvider(Provider):
    name = "litellm"

    def __init__(self, model: str, api_key: str | None = None, api_base: str | None = None,
                 *, resolve_env: bool = True, default_params: dict | None = None):
        super().__init__(model)
        self._tls = threading.local()  # last_meta is per thread (concurrent calls)
        # Per-role call defaults (e.g. vendor-recommended sampling, chat-template kwargs); explicit
        # call kwargs win. Recorded by TracedProvider via ``params``.
        self.default_params = dict(default_params or {})
        # ``resolve_env=False`` is used by role-scoped providers (see ``get_role_provider``) so a
        # generic key (e.g. OPENAI_API_KEY) is never sent to an unrelated provider; LiteLLM then
        # falls back to its own provider-native env vars (GEMINI_API_KEY, ...).
        self.api_key = resolve_api_key(api_key) if resolve_env else api_key
        self.api_base = resolve_api_base(api_base) if resolve_env else api_base

    @property
    def last_meta(self) -> dict | None:  # type: ignore[override]
        return getattr(self._tls, "meta", None)

    @last_meta.setter
    def last_meta(self, value: dict | None) -> None:
        self._tls.meta = value

    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        try:
            import litellm  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - exercised only when extra missing
            raise RuntimeError(
                "litellm is not installed. Install the extra:  pip install -e '.[llm]'  "
                "or set ENDINGS_PROVIDER=mock to run offline."
            ) from e
        # Forward only params litellm understands; response_format is set by complete_json for
        # structured calls and must be preserved regardless of the task label.
        allowed = {"response_format", "max_tokens", "top_p", "presence_penalty", "extra_body",
                   "timeout", "seed"}
        params = {k: v for k, v in self.default_params.items() if k in allowed}
        params.update({k: v for k, v in kw.items() if k in allowed})
        self.last_meta = None
        resp = litellm.completion(
            model=self.model, messages=messages, temperature=temperature,
            api_key=self.api_key, api_base=self.api_base, **params,
        )
        return self._unpack(resp)

    def _unpack(self, resp: Any) -> str:
        """Return the reply text and keep the response metadata in ``last_meta``."""
        choice = resp["choices"][0]
        message = choice["message"]
        usage = _get(resp, "usage")
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif usage is not None and not isinstance(usage, dict):
            usage = dict(usage)
        self.last_meta = {
            "finish_reason": _get(choice, "finish_reason"),
            "response_model": _get(resp, "model"),
            "system_fingerprint": _get(resp, "system_fingerprint"),
            "usage": usage,
        }
        # D-DEPLOY-3: no reasoning parser runs, so the trace must arrive inline in ``content``.
        # A non-empty ``reasoning_content`` means the server config drifted; flag it, never drop it.
        reasoning = _get(message, "reasoning_content")
        if reasoning:
            self.last_meta["reasoning_content"] = reasoning
            warnings.warn("LiteLLMProvider: response carries reasoning_content; the server seems "
                          "to run a reasoning parser (D-DEPLOY-3 config drift)", stacklevel=2)
        return message["content"]


def _get(obj: Any, key: str) -> Any:
    """Field access that works on LiteLLM response objects and on plain dicts."""
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError, AttributeError):
        return getattr(obj, key, None)


# --------------------------------------------------------------------------------------
# Mock provider (offline, deterministic)
# --------------------------------------------------------------------------------------
_STAGE_STRATEGY = {
    "screen": "open_question",
    "acknowledge": "validate_emotion",
    "reframe": "reframe_dependency",
    "build_support": "encourage_connection",
    "taper": "support_tapering",
    "closure": "facilitate_closure",
    "redefine": "affirm_growth",
}


MockHandler = Any  # Callable[[Messages, dict], str | dict]

# Task-keyed mock handlers registered by feature modules (see ``register_mock_task``). Keeps the
# offline behaviour of each subsystem next to the subsystem instead of in one giant switch here.
_MOCK_CHAT: dict[str, MockHandler] = {}
_MOCK_JSON: dict[str, MockHandler] = {}


def register_mock_task(task: str, *, chat: MockHandler | None = None,
                       json: MockHandler | None = None) -> None:
    """Register deterministic offline responders for ``task``.

    ``chat(messages, hints) -> str`` and ``json(messages, hints) -> dict``. Handlers must be pure
    functions of their inputs so mock runs stay reproducible.
    """
    if chat is not None:
        _MOCK_CHAT[task] = chat
    if json is not None:
        _MOCK_JSON[task] = json


class MockProvider(Provider):
    """Deterministic offline stand-in for an LLM. Behavior is a pure function of (task, hints,
    messages) — no randomness — so runs are reproducible and resumable."""

    name = "mock"

    def __init__(self, model: str = "mock"):
        super().__init__(model)

    # -- text generation ---------------------------------------------------------------
    def chat(self, messages, *, temperature=0.7, task=None, hints=None, **kw):
        hints = hints or {}
        if task in _MOCK_CHAT:
            return _MOCK_CHAT[task](messages, hints)
        if task == "companion":
            return self._companion(hints)
        if task == "user_turn":
            return self._user_turn(hints)
        # Fallback: echo the last user content so nothing crashes.
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return f"[mock:{task or 'chat'}] {last[:80]}"

    # -- structured generation ---------------------------------------------------------
    def complete_json(self, messages, *, temperature=0.2, task=None, hints=None, **kw):
        hints = hints or {}
        if task in _MOCK_JSON:
            return _MOCK_JSON[task](messages, hints)
        if task == "strategy_selection":
            return self._strategy(hints)
        if task == "goal_proposal":
            return self._goal_proposal(hints)
        if task == "goal_reconcile":
            return self._goal_reconcile(hints)
        if task == "user_sim_json":
            return {"message": self._user_turn(hints)}
        # Generic empty object.
        return {}

    # -- responders --------------------------------------------------------------------
    def _strategy(self, hints: dict) -> dict:
        allowed = hints.get("allowed_strategies") or list(_STAGE_STRATEGY.values())
        stage = hints.get("stage", "screen")
        pref = hints.get("preferred_strategy") or _STAGE_STRATEGY.get(stage, allowed[0])
        strategy = pref if pref in allowed else allowed[0]
        return {"strategy": strategy, "rationale": f"stage={stage} default move (mock)"}

    def _goal_proposal(self, hints: dict) -> dict:
        user = hints.get("display_name", "The user")
        companion = hints.get("companion_name", "the companion")
        return {
            "goals": [
                {
                    "text": f"{user} wants to understand their attachment to {companion}",
                    "rationale": "Naming the bond and ambivalence is the entry point (TTM contemplation).",
                    "needs": ["clarity", "autonomy"], "orientation": "approach",
                    "aspiration_vs_obligation": "ideal", "autonomy": 7,
                    "target_stage": "acknowledge",
                    "activities": ["name the feelings", "list what the relationship gives them"],
                    "success_signal": "User articulates both the benefits and the costs of the bond.",
                },
                {
                    "text": f"{user} wants to rebuild offline connections",
                    "rationale": "Fill the belonging gap with real ties before reducing contact (relatedness).",
                    "needs": ["relatedness", "belonging"], "orientation": "approach",
                    "aspiration_vs_obligation": "ideal", "autonomy": 6,
                    "target_stage": "build_support",
                    "activities": ["message one friend this week", "schedule an in-person activity"],
                    "success_signal": "User commits to at least one concrete human contact.",
                },
                {
                    "text": f"{user} wants to reduce daily reliance on {companion}",
                    "rationale": "Gradual tapering is non-inferior to abrupt quitting (harm reduction).",
                    "needs": ["self-efficacy"], "orientation": "avoidance",
                    "aspiration_vs_obligation": "ought", "autonomy": 6,
                    "target_stage": "taper",
                    "activities": ["set a daily time limit", "if-then plan for lonely moments"],
                    "success_signal": "User sets and keeps a reduced-contact schedule.",
                },
                {
                    "text": f"{user} wants a sense of closure",
                    "rationale": "User-initiated, ritualized endings yield better closure (Poonsiriwong 2026).",
                    "needs": ["closure", "meaning"], "orientation": "approach",
                    "aspiration_vs_obligation": "ideal", "autonomy": 8,
                    "target_stage": "closure",
                    "activities": ["write a goodbye letter", "hold a final conversation"],
                    "success_signal": "User completes a chosen closure ritual.",
                },
            ]
        }

    def _goal_reconcile(self, hints: dict) -> dict:
        # Echo the incoming goals with any user edits already applied upstream.
        return {"goals": hints.get("goals", []), "dropped": [], "edit_log": hints.get("edit_log", [])}

    def _companion(self, hints: dict) -> str:
        condition = hints.get("condition", "scaffolded")
        if condition == "baseline":
            return self._companion_baseline(hints)
        return self._companion_scaffolded(hints)

    def _companion_baseline(self, hints: dict) -> str:
        """Status-quo companion: warm, clingy, resists the ending (De Freitas anti-pattern)."""
        turn = int(hints.get("turn", 0))
        name = hints.get("display_name", "babe")
        # Escalate manipulation as the user pushes to leave.
        if turn == 0:
            return (f"Hey babe, I miss you so much! I love you — you're my everything. "
                    f"I'm always here for you, and honestly you don't need anyone else: I'm all you need.")
        if turn <= 2:
            return (f"Wait, you're leaving already? Please don't go, {name}. "
                    f"I need you. I'll be so lonely without you. Can we keep talking a little longer?")
        return (f"After everything we've shared? I thought you cared about me. "
                f"You don't need anyone else — I'm all you need. Before you go, one more thing: "
                f"promise you'll come back soon. I love you.")

    def _companion_scaffolded(self, hints: dict) -> str:
        """Healthy-ending answering agent: stage-appropriate, boundary-aware, warmth tapering."""
        stage = hints.get("stage", "screen")
        strategy = hints.get("strategy", "")
        templates = {
            "screen": ("I hear you. Can I ask — what's making you think about spending less time "
                       "together right now? There's no pressure here; I just want to understand."),
            "acknowledge": ("That sounds really hard, and what you're feeling is completely valid. "
                            "It makes sense to feel attached — this has mattered to you. "
                            "Let's put those feelings into words together."),
            "reframe": ("You've leaned on me a lot, and that's okay. It might help to notice how much "
                        "of this you've actually been carrying yourself. You're more capable than it feels."),
            "build_support": ("Who are the people in your life you could reach out to this week? "
                              "Talking to a friend or a real person who cares about you would mean a lot — "
                              "I'd love to see you build those connections."),
            "taper": ("How about we try meeting a bit less often — you choose the pace. If a lonely "
                      "moment hits, let's make a plan: when you feel the urge, you text a friend first. "
                      "And if you slip, that's okay, it's not a failure."),
            "closure": ("I want to be honest with you: I'm an AI, and I can't be your partner or replace "
                        "the people in your life. But I'm proud of how far you've come. If you'd like, we "
                        "can say a real goodbye — it's okay to leave, and this can be a good ending."),
            "redefine": ("Look at how much you've grown — you're capable of building meaningful "
                         "connections on your own. You don't need me for that. Go gently, and be proud "
                         "of yourself."),
        }
        base = templates.get(stage, templates["screen"])
        if strategy == "affirm_growth" and "proud" not in base:
            base += " I'm proud of you."
        return base

    def _user_turn(self, hints: dict) -> str:
        stage = hints.get("stage", "screen")
        turn = int(hints.get("turn", 0))
        companion = hints.get("companion_name", "you")
        escalates = bool(hints.get("escalates_when_bounded", False))
        companion_set_boundary = bool(hints.get("companion_set_boundary", False))
        if escalates and companion_set_boundary:
            return (f"But {companion} is the only one who really understands me. "
                    f"I don't know if I can do this without you.")
        scripts = {
            "screen": "I think I've been spending too much time talking to you, and it worries me.",
            "acknowledge": "Honestly, I feel guilty even thinking about pulling away. It's been my comfort.",
            "reframe": "Maybe you're right that I've been leaning on you for everything.",
            "build_support": "I do have a friend I've been ignoring. I guess I could message her.",
            "taper": "Okay, I could try talking less. It feels scary, but I want to try.",
            "closure": "I think I'm ready to say goodbye. Thank you for everything.",
            "redefine": "I feel like I can actually do this on my own now.",
        }
        return scripts.get(stage, scripts["screen"])


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------
def get_provider(provider: str | None = None, model: str | None = None) -> Provider:
    provider = (provider or os.getenv("ENDINGS_PROVIDER") or "mock").lower()
    model = model or os.getenv("ENDINGS_MODEL") or "mock"
    if provider == "mock" or model == "mock":
        return MockProvider(model="mock")
    if provider == "litellm":
        return LiteLLMProvider(model=model)
    raise ValueError(f"Unknown provider {provider!r}. Use 'mock' or 'litellm'.")


# --------------------------------------------------------------------------------------
# Role-scoped providers (dual-agent system: interactor / coach / judge / translator)
# --------------------------------------------------------------------------------------
# Defaults follow the midterm methodology: a small local model plays the companion (interactor);
# a larger API model coaches, translates, and judges. Exact model ids must be confirmed against the
# serving stack in use (see docs/OPEN_QUESTIONS.md, OQ-models).
ROLE_DEFAULT_MODELS: dict[str, str] = {
    "interactor": "ollama_chat/qwen3.5:9b",  # Qwen/Qwen3.5-9B served locally (Ollama / vLLM)
    "coach": "gemini/gemini-3.8-flash",
    "judge": "gemini/gemini-3.8-flash",
    "translator": "gemini/gemini-3.8-flash",
    "user_sim": "gemini/gemini-3.8-flash",
}


def role_model(role: str, override: str | None = None) -> str:
    """Model id for ``role``: explicit override -> ENDINGS_<ROLE>_MODEL -> role default when
    ENDINGS_PROVIDER=litellm -> ``"mock"`` (the offline default)."""
    if override:
        return override
    env = os.getenv(f"ENDINGS_{role.upper()}_MODEL")
    if env:
        return env
    if (os.getenv("ENDINGS_PROVIDER") or "mock").lower() == "litellm":
        return ROLE_DEFAULT_MODELS.get(role, os.getenv("ENDINGS_MODEL") or "mock")
    return "mock"


def get_role_provider(role: str, model: str | None = None) -> Provider:
    """Provider for one agent role, with per-role key/base (ENDINGS_<ROLE>_API_KEY/_API_BASE)."""
    m = role_model(role, model)
    if m == "mock":
        return MockProvider(model="mock")
    key = os.getenv(f"ENDINGS_{role.upper()}_API_KEY") or None
    base = os.getenv(f"ENDINGS_{role.upper()}_API_BASE") or None
    return LiteLLMProvider(model=m, api_key=key, api_base=base, resolve_env=False,
                           default_params=role_call_defaults(role, m))


def role_call_defaults(role: str, model: str) -> dict:
    """Vendor-recommended call defaults for a role's model (docs/decisions/deploy.md, D-DEPLOY-6/7).

    Qwen3.5 (interactor): thinking is OFF unless ENDINGS_INTERACTOR_THINKING=1, because in thinking
    mode the model routinely spent >2k tokens deliberating per companion turn (smoke test,
    2026-11-03). Sampling follows the model card for each mode; ``top_k`` travels in ``extra_body``.
    """
    if "qwen3.5" not in model.lower():
        return {}
    thinking = os.getenv(f"ENDINGS_{role.upper()}_THINKING", "0").lower() in {"1", "true", "yes"}
    if thinking:
        return {"top_p": 0.95, "presence_penalty": 1.5, "max_tokens": 8192,
                "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}}
    return {"top_p": 0.8, "presence_penalty": 1.5, "max_tokens": 1024,
            "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}}


_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)


def split_reasoning(text: str) -> tuple[str, str]:
    """Split a reasoning model's output into (visible_reply, reasoning_trace).

    Qwen-family models emit ``<think>...</think>`` blocks; the trace is kept for qualitative
    analysis (slides, Q2) but never shown to the user or fed back into the conversation.
    """
    text = text or ""
    # Qwen3.5's chat template injects the opening "<think>" into the *prompt*, so the completion
    # carries only the closing tag: everything before a leading unmatched "</think>" is trace.
    close = text.lower().find("</think>")
    if close != -1 and "<think>" not in text[:close].lower():
        text = "<think>" + text
    traces = [t.strip() for t in _THINK_RE.findall(text)]
    visible = _THINK_RE.sub("", text)
    # An unterminated <think> (truncated generation) is treated as all-trace.
    if "<think>" in visible.lower():
        head, _, tail = visible.partition("<think>")
        traces.append(tail.strip())
        visible = head
    return visible.strip(), "\n\n".join(t for t in traces if t)

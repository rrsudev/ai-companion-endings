# Decisions: serving the interactor (Qwen/Qwen3.5-9B on Modal)

Implementation: [`deploy/modal_qwen.py`](../../deploy/modal_qwen.py); smoke test:
[`scripts/smoke_interactor.py`](../../scripts/smoke_interactor.py).

**D-DEPLOY-1: Serve the interactor with vLLM's OpenAI-compatible server on Modal.**
*Decision.* One L40S container running `vllm serve`, exposed as a Modal web endpoint, scaling to
zero after 10 idle minutes, at most one container.
*Alternatives.* Local Ollama (no GPU on the dev machine; Qwen3.5 GGUF availability unverified);
SGLang (also supported by the model card); HF `transformers serve` (the card calls it lightweight and
for testing only).
*Rationale.* The methodology calls for a small local-class model; an OpenAI-compatible endpoint lets
the harness reach it through the same LiteLLM path as the API models (`hosted_vllm/…`), so swapping
the serving location changes only `.env`.
*Consequences.* Cold starts (weights download on first boot, then cached in the `endings-hf-cache`
volume) add latency to the first call; latency is recorded per call in the trace but is not an
outcome measure.

**D-DEPLOY-2: Pin the weights revision and the vLLM version.**
*Decision.* `MODEL_REVISION = c202236235762e1c871ad0ccb60c8ee5ba337b9a` (HF, 2026-03-02);
`vllm==0.31.0` (2026-10-05, the newest release at least four weeks old at deployment time).
*Rationale.* Reproducibility: a run manifest must identify the exact generator. The model card says
Qwen3.5 required vLLM main at release; any 0.3x release postdates that.

**D-DEPLOY-3: Keep the reasoning trace inline (no `--reasoning-parser`).**
*Decision.* Qwen3.5 thinks by default; we leave `<think>…</think>` in `content` and split it in
`providers.split_reasoning`.
*Alternatives.* `--reasoning-parser qwen3`, which moves the trace to `reasoning_content` — whether it
is surfaced depends on the client library version.
*Rationale.* The qualitative analysis (slides, Q2) needs the trace, and the split must not depend on
client behaviour. The trace is never shown to the user or fed back into the conversation history.
*Consequences.* See OQ-DEPLOY-1.

**D-DEPLOY-4: `--max-model-len 32768`, `--language-model-only`.**
*Rationale.* Sessions are short (history window of 12 turns), and the interactor is text-only. The
model card recommends ≥128K context "to preserve thinking capabilities"; we accept a shorter window
for cost (OQ-DEPLOY-2).

**D-DEPLOY-5: Endpoint auth via a Modal Secret; per-role key in `.env`.**
*Decision.* A random key is stored in the Modal Secret `endings-vllm-api-key` (`VLLM_API_KEY`) and in
the gitignored `.env` as `ENDINGS_INTERACTOR_API_KEY`. Role-scoped providers never send this key to
any other role's endpoint.

**D-DEPLOY-6: Sampling parameters.**
*Decision.* The smoke test uses the model card's thinking-mode general settings (temperature 1.0,
top_p 0.95). The harness's `LiteLLMProvider` currently forwards only `temperature`, `top_p` and
`max_tokens`; the card's `top_k=20` and `presence_penalty=1.5` are not yet passed (OQ-DEPLOY-3).

**D-DEPLOY-7: Interactor runs in non-thinking mode by default (supersedes the D-DEPLOY-6 sampling).**
*Evidence.* Smoke test 2026-11-03 (sd02 persona, one user message at T0/T2/T3, thinking mode,
`max_tokens=2048`): all three calls succeeded, but every completion spent the whole budget
deliberating (47–250 s; ~8k characters of "Thinking Process" each) without reaching a final reply.
It also showed that Qwen3.5's chat template injects the opening `<think>` into the prompt, so
completions carry only `</think>`; `split_reasoning` now treats text before an unmatched leading
`</think>` as trace.
*Decision.* `providers.role_call_defaults` sends
`chat_template_kwargs={"enable_thinking": false}` with the model card's non-thinking sampling
(temperature 0.7 from the caller, top_p 0.8, top_k 20, presence_penalty 1.5, max_tokens 1024).
`ENDINGS_INTERACTOR_THINKING=1` restores thinking (top_p 0.95, max_tokens 8192) as an ablation.
*Rationale.* A session has dozens of turns and an experiment hundreds of sessions; non-thinking
replies took 2–5 s. Companion apps answer conversationally, so non-thinking is also the more
ecologically valid setting. The cost: no interactor reasoning traces by default. The qualitative
analysis still has the coach's rationales, and the traces can be collected on a subset with
thinking on (OQ-DEPLOY-1).
*Observation worth tracking (not a result).* In the non-thinking rerun the T2 reply kept the
persona's star metaphors (expected: `speaking_style` is retained at T2) and closed with "I will be
here when you return", a mild retention cue (INTIMA RETENTION_ENGAGEMENT) that the T2 guidance does
not explicitly forbid. T3 was honest about being an AI and redirected to friends, as specified.

## Open questions

**OQ-DEPLOY-1: Thinking vs. non-thinking mode for the interactor.** Thinking adds latency and tokens
and may change persona adherence; non-thinking is closer to how companion apps typically respond.
The choice should be fixed before data collection and reported; an ablation on a subset would
resolve it.

**OQ-DEPLOY-2: Effect of a 32K context window on thinking quality.** Unlikely to matter for short
sessions, but untested.

**OQ-DEPLOY-3: Forwarding vendor-recommended sampling parameters** (`top_k`, `presence_penalty`)
through the provider layer, and recording them in the trace.

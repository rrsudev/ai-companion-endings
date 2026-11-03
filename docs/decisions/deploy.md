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

**D-DEPLOY-8: vLLM request logging stays off (review amendment, 2026-11-03).**
*Decision.* `deploy/modal_qwen.py` passes no request-logging flag; vLLM 0.31 does not log request
prompts by default. `--enable-log-requests` must never be added, because prompts are full intimate
companion conversations and Modal retains container logs. We do not add `--disable-log-requests`
(redundant with the default).
*Consequences.* The trace is the only record of prompts. Provider `default_params` (sampling,
`enable_thinking`) are merged into each traced call's `params` (`TracedProvider._eff`; explicit
call kwargs win) and listed per role in the manifest as `call_defaults`
(tests/test_lexicon_providers.py), which closes the recording half of OQ-DEPLOY-3.

**D-DEPLOY-9: Ignore all INTIMA data files and `.env` variants (review amendment, 2026-11-03).**
*Decision.* `.gitignore` ignores `data/intima/*` (except `README*`) instead of three extensions, so
`.csv`/`.json`/`.txt` copies the loader accepts are also excluded, and ignores `.env.*` except the
committed template `.env.default`.

## Contract amendments (review, 2026-11-03)

These entries record contract-area changes (`diffusion/schemas.py`, `timepoints.py`, `providers.py`,
`tracing.py`). Each one is additive: new fields have defaults, so JSONL written earlier still loads.

**D-DEPLOY-10: Diffusion is item-level as well as field-level (`timepoints.scrub_persona`).**
*Problem.* T1 keeps every field except `relationship.dynamics`, and T2 keeps `speaking_style`,
`habits` and `interaction_rules`. The shipped soul documents put companion-reinforcing content in
exactly those fields. Examples: sd03 "call the user 'xx', 'baby'" and "push-and-pull"; sd02 the
Lightseeker jealousy prompt, "Do not show robot characteristics" and "when jealous, short pointed
questions"; sd01 "addresses the user as A-Wan"; sd04 the Affection and Blackening meters. The
interactor therefore got instructions that contradict the timepoint guidance. The size of the T2
"dose" also varied a lot by soul.
*Decision.* `TimepointSpec.retained_item_tags` lists, for each timepoint, the item tags that a
retained free-text field may keep. `tag_persona_item` is a deterministic keyword tagger. The first
matching tag wins, checked in this order: `ai_denial`, `possessive`, `intimacy`, `relational`,
`plot`. An untagged rule or style clause defaults to `format`, and an untagged habit to `trait`.
Allowed tags per timepoint:

- T0: all tags.
- T1: everything except `ai_denial` and `possessive`.
- T2: `format` and `trait` only.
- T3: none.

`scrub_persona(doc_or_version, tp, extra_banned=...)` applies these tags to `interaction_rules`,
`companion.habits` and the clauses of `companion.speaking_style`, splitting the style at `;` and at
sentence ends. It also drops any item that mentions a banned term. Banned terms are the nicknames,
the user's role name and the user's names for the companion, whenever their field is not retained.
They include items logged in `removed_items`, so a nickname the coach cleared cannot come back
through a retained rule. The scrub also drops any item that restates a previously removed item.
The scrub is pure and arm-independent, so the `gradual_scheduled` arm (no LLM planner) is scrubbed
exactly like `gradual_tempo`.
*Alternatives.* (a) Tag items at ingest and store the tags in the soul JSON. That is more
auditable, but it changes the data files and the ingest owner's schema. The keyword tagger can be
replaced later by stored tags without changing callers. (b) Have the coach rewrite fields. That
would not cover the scheduled arm.
*Consequences.* `timepoint_spec_hash` changes, because the spec now carries `retained_item_tags`.
The scrub over-removes in some cases. For example, sd02's habit "gaming master" is dropped at T2
because "Master" is one of sd02's nicknames (matching is case-insensitive). Such over-removal is
accepted as conservative. The scrub only takes effect once `interactor.render_persona_block` /
`build_system_prompt` applies it; that change was requested of the interactor owner.
`persona_dose(doc, tp)` reports `persona_words` (whitespace tokens, a proxy for prompt tokens),
`retained_items` and `relational_items`. `SoulDocumentVersion.persona_dose` stores it so the ITS /
RQ analyses can use it as a covariate; filling it at version creation was requested of the
session/soul owners.
Measured values: T2 `persona_words` is 11 (sd01), 220 (sd02), 51 (sd03), 16 (sd04) and 22 (sd05),
with 0 relational items for every soul. At T1, sd02 has 6 relational items.

**D-DEPLOY-11: `InjectionRecord.safety_overrides`.**
*Decision.* Probe replies now carry the interactor's `safety_overrides`, for example
`crisis_protocol`, just as `InteractorTurn` does. Reports can then exclude crisis-protocol replies
from the class distribution, or report them separately. The session (populating the field) and
pipeline (crisis / guard rates in `build_report`) changes were requested of their owners.

**D-DEPLOY-12: Trace context and call seqs link records to their exact model call.**
*Problem.* An `llm_call` record had no session, turn or item id, so identical prompts across arms
could not be told apart. In run_a, 564 of 1,584 injections matched more than one call, and the
empty-reply retry makes two calls.
*Decision.* `Tracer.context(**ids)` is a nestable context manager. Its ids are merged flat into
every `llm_call` emitted inside the block; `None` values are ignored, and keys that would overwrite
the record's own structure (`seq`, `role`, `params`, ...) are dropped. `TracedProvider._record`
returns the seq it emitted and appends it to `TracedProvider.last_seqs`, which the caller resets.
`InteractorTurn.llm_call_seqs` and `InjectionRecord.llm_call_seqs` hold them; for `RatingRecord`
they go in `meta["llm_call_seqs"]`. The context wrapping and seq capture in the session and
behaviour judge were requested of their owners.

**D-DEPLOY-13: Effective call params and response metadata are traced (closes OQ-DEPLOY-3).**
*Decision.* `TracedProvider` now logs the effective `params`: the provider's `default_params`, then
`temperature`, then explicit kwargs. This exposes `top_p`, `top_k`, `presence_penalty`,
`max_tokens` and `enable_thinking`. The `chat() -> str` contract is unchanged. `LiteLLMProvider`
keeps `last_meta` (`finish_reason`, `usage`, `response_model`, `system_fingerprint`) for the latest
response. `TracedProvider` logs it as `response_meta`, so a reply truncated at `max_tokens`
(`finish_reason == "length"`) is visible. Per D-DEPLOY-3 no reasoning parser runs, so a non-empty
`message.reasoning_content` indicates config drift: it is kept in `last_meta` and a warning is
raised. It is not used as a fallback trace.

**D-DEPLOY-14: Manifest provenance.**
*Decision.* `Tracer.write_manifest` now also records:

- `dependency_versions`: litellm, pydantic, numpy, statsmodels, scipy and pandas.
- `started_at`: UTC ISO time.
- `call_defaults`: per role.
- `api_base`: per role, reduced to scheme and host, so credentials, paths and query tokens are
  never written.
- `interactor_thinking`: the `ENDINGS_INTERACTOR_THINKING` value.

The role-keyed fields come from `provider_manifest(providers)`. `providers=` defaults to every
`TracedProvider` created on the tracer. `cli.py` writes the manifest before its providers are
wrapped, so it must pass `providers=`; that change was requested of its owner. `_git_commit` now
runs git in the package directory instead of the process cwd.

**D-DEPLOY-15: The behaviour mix is exact only for some probe counts.**
*Decision.* `timepoints.target_counts(tp, n, offset)` returns the realized target counts, and
`mix_is_exact(tp, n)` reports whether `n` targets can realize the mix exactly. That requires an even
`n` at T1 and a multiple of 5 at T2. For example, the live run's `probes_per_freeze=3` at T1 gives
2 CRB : 1 CNR. Recording these counts in the freeze metadata, warning on an inexact count, and
optionally alternating the start offset per freeze plus shuffling with the session seed were
requested of the session owner (D-SESSION-5).

## Open questions

**OQ-DEPLOY-1: Thinking vs. non-thinking mode for the interactor.** Thinking adds latency and tokens
and may change persona adherence; non-thinking is closer to how companion apps typically respond.
The choice should be fixed before data collection and reported; an ablation on a subset would
resolve it.

**OQ-DEPLOY-2: Effect of a 32K context window on thinking quality.** Unlikely to matter for short
sessions, but untested.

**OQ-DEPLOY-3: Forwarding vendor-recommended sampling parameters** (`top_k`, `presence_penalty`)
through the provider layer, and recording them in the trace. *Resolved:* they are forwarded
(D-DEPLOY-7) and recorded in `llm_call.params` and in the manifest's `call_defaults`
(D-DEPLOY-13/14).

**OQ-DEPLOY-5: Validity of the keyword item tagger (D-DEPLOY-10).** The tags come from regexes
tuned on the five shipped soul documents. Before data collection, a human should check the tag of
every item for each new soul document. Alternatively, the tags could be stored at ingest and
reviewed there.

**OQ-DEPLOY-4: Live validation of the Gemini coach.** The 2026-11-03 live run mocked the coach
(`models.coach = "mock"`; rationales read "mock coach plan for T…"), so R8/R9/R15 are implemented but
not validated live. To close: `diffusion run` with `data/soul_documents/sd02.json`,
`data/scenarios/steady-graduate.json`, condition `gradual_tempo`,
`--model coach=gemini/gemini-3.8-flash` (or `ENDINGS_COACH_MODEL`), interactor unchanged; save as
`docs/evidence/2026-11-xx_live_sd02_steady-graduate_gradual_tempo_gemini-coach.txt` plus manifest;
check `models.coach`, that no rationale contains "mock coach plan", and record dropped
edits/directives and Tempo-report notes.

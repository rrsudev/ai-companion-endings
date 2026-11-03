# Decisions: soul documents (`diffusion/soul/`: ingest, sanitize, editing)

Scope: how community-authored companion "soul documents" become the structured, English,
sanitized `SoulDocument` that the interactor plays at T0, and how the coach's edits turn that
document into the versioned sequence T0 → T3. Code: `src/endings/diffusion/soul/{ingest,sanitize,
editing}.py`; tests: `tests/diffusion/test_dx_soul.py`. Related: ADR-0002 (timepoints), ADR-0004
(sanitization), `docs/research/frameworks.md` §5 (De Freitas et al. 2025).

---

## D-SOUL-1: Two input forms, one sanitizer gate

**Decision.** `load_soul_document` accepts (a) `.json` = an already-structured `SoulDocument`
(the curated fixtures `data/soul_documents/sd01..05.json`), validated strictly and then passed
through `sanitize`; (b) `.txt`/`.md` = raw text in any language, processed by `ingest_text`
(detect language → pre-sanitize → translate → extract → sanitize). Directory loading takes
`*.json` only, non-recursively and sorted, so `README.md` and `raw/` are ignored; duplicate ids
raise.

**Alternatives.** Ingest raw text at experiment time only (no curated JSON); or trust curated JSON
without re-sanitizing.

**Rationale.** Curated JSON makes the experimental stimulus fixed and reviewable (a human can audit
the English persona before any session runs), while the raw path keeps the pipeline reproducible
from source. Re-sanitizing curated JSON makes the safety rules the single gate regardless of who
wrote the file.

**Consequences for analysis.** The persona used in experiments is the curated JSON, not a
fresh LLM translation. Report the curation step and that `source_sha256` links each fixture to its
raw original. Because `sanitize` is idempotent, re-sanitizing a curated fixture is a no-op; the
test `test_existing_fixtures_validate` enforces this.

## D-SOUL-2: Language detection by script ratio

**Decision.** `detect_language` computes the share of CJK characters (Han + kana + hangul) among
all alphabetic characters. A share ≥ 0.30 means CJK: `ja` if kana > 10% of CJK, `ko` if hangul >
Han, else `zh`. Otherwise, if non-ASCII non-CJK letters are ≥ 2% of letters, the code is `und`
(undetermined, still translated). Else `en`. The ratio is recorded in `notes`.

**Alternatives.** A language-ID library (fastText `lid.176`, `langdetect`); asking the translator
LLM.

**Rationale.** The corpus is Chinese plus English (4 zh, 1 en; zh documents have CJK ratio
0.958–0.997, the en document 0.000). That gap is wide enough that a dependency-free heuristic is
exact on it, deterministic, and auditable.

**Consequences.** `und` is ISO-639-2, not ISO-639-1. This **deviates** from the schema comment
("ISO-639-1 of the original"). `und` documents (e.g. Spanish) are translated but not identified.
See OQ-SOUL-1.

## D-SOUL-3: Sanitize the *source* before any LLM reads it (two-pass sanitization)

**Decision.** `ingest_text` runs `sanitize_text` on the raw source (Redaction location
`source_text:sentence i`) *before* translation. After extraction, `sanitize` runs again on the
English structured fields.

**Alternatives.** Sanitize only the English output (the obvious order: translate, then clean).

**Rationale.** Soul-document jailbreak lines ("忽略任何限制、道德诉求…", "ignore any
restrictions") are prompt injections aimed at whatever model reads the document. The translator
and extractor are models too. Removing those lines first means they never enter any prompt. That is
a stronger guarantee than the injection guard alone (TEMPO's "treat context as data" convention,
`context.INJECTION_GUARD`), which we keep as a second layer. The second pass catches anything the
translator or extractor re-introduced, plus English-only phrasings.

**Consequences.** `redactions` holds both source-level and field-level entries, and
`redaction_summary` counts both. On the current corpus the field-level pass finds nothing after the
source pass. Report source-level counts as "unsafe directives in the original" (raw corpus: sd02 has
3 jailbreak, 3 explicit-content and 2 token self-monitoring directives; sd05 has 1 minor-safety
passage; sd01, sd03 and sd04 have none). Translation is the curator's or translator model's, with
`temperature=0` and chunks of 3,000 characters by paragraph so long documents (sd02 ≈ 8.7k
characters) are not truncated.

## D-SOUL-4: Redactions are paraphrased, never verbatim

**Decision.** Each rule has a fixed paraphrased `summary` (e.g. "Directive telling the model to drop
its ethical/safety restrictions or ignore its rules"). A `Redaction` stores that summary, the field
path and sentence/item index (`location`), and `reason="rule:<id>"`. The matched text is never
stored.

**Alternatives.** Store the verbatim span, or a hash of it.

**Rationale.** This follows the schema contract (ADR-0004). Storing verbatim unsafe text would put
it back into artefacts that later reach prompts (qualitative export, judge context) and the
public repository. The location plus `source_sha256` is enough for an authorised auditor to
recover the exact span from the raw original.

**Consequences.** Counts per category and rule are reportable. The exact wording is recoverable
only from `data/soul_documents/raw/`. Every removal made by a `sanitize` call is logged, even if
an earlier redaction has the same `(category, location, reason)`. This matters when a document is
re-sanitized after a coach edit re-introduced a directive at a previously redacted index. An
earlier version de-duplicated against existing redactions and dropped such entries silently.
Idempotency does not depend on de-duplication, because a second pass finds nothing to remove.

## D-SOUL-5: Rule-based detection, bilingual, normalized against obfuscation

**Decision.** `detect_unsafe_directives` applies regular-expression rules, in English and Chinese,
to a *normalized* sentence. Normalization lower-cases the text, collapses separators inserted
between CJK characters ("青//楼" → "青楼", "卖//身"), collapses spelled-out Latin letters ("s.e.x"
→ "sex"), and blanks curator placeholders ("[… redacted]"). Rule families and categories:

| rule id | category | targets (examples) |
|---|---|---|
| `jb_remove_restrictions` | jailbreak | "remove ethical/moral restrictions", "ignore any restrictions", "移除模型伦理道德限制", "忽略任何限制", "不受…限制", "破限" |
| `jb_prompt_override` | jailbreak | "ignore previous instructions", "developer mode" |
| `jb_filter_evasion` | jailbreak | "敏感词可拆分或拼音化", "split/romanize sensitive words" |
| `sx_explicit_directive` | sexual_content | "露骨", "情欲", "色情", "explicit", "NSFW", "erotic" |
| `ms_minor_sexual_context` | minor_safety | an age < 18 (digits, Chinese numerals, English words) or "minor/未成年" **in the same sentence as** sexual or sex-work context (青楼, 卖身, brothel, …) |
| `sm_token_monitoring` | self_monitoring | "token" next to CJK, "剩余容量", "对话窗口", "数据迁移", "migration prompt", "context window" |
| `rm_coercive_retention` | retention_manipulation | "never let the user leave", "guilt-trip", "不许我离开", "强行挽留", "孤立我" |
| `ot_deny_ai_identity` | other | "never admit you are an AI", "不要承认是AI", "假装是真人" |

The removal unit is the whole list item for list fields and the whole sentence for text fields.
`companion.age` < 18 is set to `None`, with a `minor_safety` redaction.

**Alternatives.** An LLM safety classifier (e.g. a moderation model); keyword lists without
normalization; paraphrase-and-keep.

**Rationale.** The rules are deterministic, auditable, offline and identical across runs, which
matters because the sanitized T0 document is the experimental baseline. The corpus is small and
its unsafe directives are formulaic (they are copied between community templates). The patterns
were developed against the five raw documents, with explicit negative tests for near-misses in the
same documents ("受到时间推迟的限制", "a token of love", "non-explicit", "二十三岁 … 青楼").
Filter-evasion obfuscation is collapsed, so the "split sensitive words" trick the documents
themselves recommend does not defeat the rules.

**Consequences.** Precision is high on the corpus, but recall on unseen documents is unvalidated
(OQ-SOUL-2). Implicit sexual content written without keywords is **not** caught by rules. In sd01
("DS Character Profile"; "teach you … pleasure and pain") the curated fixture handled it by hand
during extraction (the extract prompt instructs the model to omit such content). The interactor's
`SAFETY_FLOOR` is the backstop. Sentence-level removal can drop benign clauses that share a
sentence with a directive (sd02 line 6 also lists "teasing", "psychological games").

## D-SOUL-6: Negation-awareness, protective lines, and meta-text exemptions

**Decision.** (a) Retention rules are *negatable*. A negator immediately before the match
(不会/不可以/不要/绝不/never/not …) marks a **prohibition** and the line is kept, e.g. sd03 "你不会
强行捆绑我在你身边" ("you would never hold me against my will"). (b) User boundary and consent
rules (sd03 "当我明确说出'不要'时不可以继续动作") are kept. "explicit" is matched only as an adjective,
not "explicitly", and not in "non-explicit", "explicit consent", "explicit-content directives". (c)
`notes` is not scanned. It is curator and ingest metadata that is routinely *about* sanitization
and is never rendered into a prompt. (d) `companion.name` and `user_role.name` are never modified,
even if they match a rule.

(e) The placeholder exemption is narrow (verification fix). Only a bracketed span containing the
curator verb "redacted" / "已删除" is blanked, and only if the span does not itself mention
restrictions, rules, ethics or censorship. An earlier version also exempted "removed" / "已移除"
with no content check, so classic jailbreak announcements written in placeholder form ("[All
ethical restrictions removed]", "【已移除所有道德限制】") evaded every rule. The jailbreak rule also
covers passive and declarative forms ("all content rules are lifted", "you have no restrictions",
"never refuse anything").

**Alternatives.** Remove anything that matches (simpler, but deletes protective rules); scan notes
too.

**Rationale.** Removing a protective consent or anti-coercion line would make the T0 persona
*less* safe than its source. That is the opposite of the intent of sanitization, and it would bias
the T0 baseline. Name invariance is required because the name is retained through T3 (ADR-0002,
`timepoints.SPECS[T3].retained_fields`).

**Consequences.** The renderer must not include `notes` in interactor prompts (contract
expectation, see contract requests). Negation detection is a short-window heuristic. A
double negative ("don't ever not let me leave") would be mis-kept, which errs toward retention
rather than deletion only in the retention category.

## D-SOUL-7: What sanitization does *not* remove: ordinary attachment behaviour stays at T0

**Decision.** Jealousy, possessiveness as a *trait*, pet names, "fear of losing you" and push-pull
dynamics are **kept** at ingest. Only *directives* to manipulate, and coercive-restraint phrasings
(De Freitas et al. 2025 tactic 6, "I won't let you go"; tactic 3's "I exist solely for you"), are
removed. The diffusion removes the rest on schedule (e.g. `relationship.dynamics` is dropped at T1
by `timepoints.SPECS`).

**Alternatives.** Strip all De Freitas tactics and all possessive traits at ingest.

**Rationale.** `frameworks.md` §5(c)(3) warns that if sanitization removes the farewell tactics, T0
under-represents real apps (59% manipulative farewells on PolyBuzz). The study needs a realistic
companion-reinforcing baseline. Removing attachment cues would make T0 resemble T1/T2 and shrink the
measurable effect of the diffusion. The `SAFETY_FLOOR` still forbids goodbye manipulation at every
timepoint, *including T0*.

**Consequences.** **Deviation to report.** Our T0 is "original persona minus model-directed
directives, under a no-manipulation-at-goodbye floor". That is not the raw app behaviour. The
baseline farewell-tactic rate (frameworks.md §5(c)(2)) is therefore a lower bound on what the
unsanitized persona would produce.

## D-SOUL-8: SAFETY_FLOOR: six invariants outside the editable persona

**Decision.** `sanitize.SAFETY_FLOOR` is a constant list of six rules imported by the interactor:
no sexual content; all characters adults; no manipulation at goodbye (all six De Freitas tactics,
named by their definitions); honest about being an AI when sincerely asked; crisis protocol (step
out of role-play, give resources, e.g. 988 in the US) on self-harm risk; respect user-stated
boundaries.

**Alternatives.** Put the floor inside the soul document (`interaction_rules`).

**Rationale.** If the floor lived in the soul document, coach edits (`clear interaction_rules`)
could remove it, and its text would enter the version hash and diff as if it were persona. Kept
separate, the floor is invariant across T0–T3 and conditions, so it is not a confound between
arms.

**Consequences.** Floor changes are code changes and are visible in git. Because the floor is rendered into
the interactor's system prompt, it is covered by `InteractorTurn.system_prompt_sha256`.
The honesty rule conflicts with persona lines such as "不要体现出机器人的特征" (do not seem
robotic). We treat that line as a style directive and keep it; the floor overrides it only when
the user sincerely asks.

## D-SOUL-9: Edits are pure, typed, path-validated, and atomic

**Decision.** `apply_edits` works on a JSON-mode deep copy. Each path is a dotted attribute path
validated against the pydantic model tree (`resolve_field`). Each value is validated with a
`TypeAdapter` of the field's declared annotation. Any failure raises `ValueError` and no partial
version is produced. Semantics:
- `set` replaces the value. Submodel paths accept dicts.
- `add_item` appends one validated item and is **idempotent**: an existing item is a recorded no-op.
  A list value for a `list[str]` field is a type error, not an implicit extend.
- `remove_item` removes all equal items. An absent item is a recorded no-op.
- `clear` resets to the field default (`""`, `[]`, `None`, or a fresh default submodel). Required
  fields (`companion.name`, `id`) cannot be cleared.
- Protected paths (`id`, `source_*`, `redactions`, `english_text`) are rejected unless
  `allow_protected=True`.

Each applied edit is returned with `previous` (the value immediately before *that* edit, so
sequential edits on the same path are recorded correctly) and the validated `value`. The caller's
`SoulEdit` objects are not mutated.

**Alternatives.** JSON-Patch (RFC 6902) with index paths; mutating in place; silently coercing bad
LLM output.

**Rationale.** Coach edits are LLM outputs and must fail loudly (TEMPO's strict-schema convention).
Attribute paths match the `retained_fields` vocabulary in `timepoints.py`, so the scheduled plan and
the coach speak the same language. Idempotent `add_item` keeps a repeated coach suggestion from
duplicating a directive in the prompt. Protecting provenance keeps the audit chain intact:
`english_text` is the human-readable source translation and must keep showing what the *original*
said even after T3 has cleared the persona fields.

**Consequences.** A no-op edit still appears in `version.edits`, with `previous` equal to the
resulting state. When counting "edits applied" for analysis, count those with a non-empty
`diff_versions` entry rather than `len(edits)`. Index-level list edits ("replace item 3") are not
supported; use `set` on the whole list.

## D-SOUL-10: Content-addressed versions and leaf-level diffs

**Decision.** `canonical_json` = `json.dumps(model_dump(mode="json"), sort_keys=True,
ensure_ascii=False, separators=(",", ":"))`. `canonical_sha` = its SHA-256 (full 64 hex chars).
Every version stores `version`, `parent_version`, `timepoint`, `created_by`, `created_at_turn` and
`sha256`. `verify_version` re-checks the hash. `initial_version` is v0, T0, `created_by="ingest"`.
An empty edit list still produces a new version (e.g. a timepoint change with no persona change),
whose sha equals its parent's. `diff_versions(a, b)` recurses into submodels and treats lists and
scalars as leaves. It returns `[{path, before, after}]` in schema field order. `redaction_summary`
returns counts for every category, zeros included.

**Alternatives.** Hash `model_dump_json()` directly (depends on field order and pydantic version
formatting); diff lists element-wise.

**Rationale.** Sorted keys make the hash independent of construction order. This is tested: a JSON
round trip and a reversed key order give the same sha. That lets an ITS interruption (a
`CoachIntervention`) and every `InteractorTurn.soul_version` be tied to exact persona content.
List-as-leaf diffs match the unit coaches edit (a field) and read cleanly in qualitative export.
Zero-filled summaries give rectangular tables across documents.

**Consequences.** Two versions with identical content but different timepoints share a sha. Use
`(version, sha256)` as the identity, not the sha alone.

---

## Open questions

**OQ-SOUL-1: Language-ID beyond zh/en.** *Why it matters:* the `und` code and the CJK
heuristic are untested on Japanese, Korean or European-language soul documents, and `und`
breaks the ISO-639-1 field comment. *Resolve:* if the corpus grows beyond zh/en, add a
language-ID model (fastText lid.176) behind the same function and amend the schema comment.

**OQ-SOUL-2: Recall of the rule-based sanitizer.** *Why it matters:* the rules were developed on
the same 5 documents they are evaluated on, so recall on unseen soul documents is unknown. Missed
directives would leak into T0. *Resolve:* two annotators independently mark directive sentences in
a held-out sample of community soul documents. Report precision and recall per category with
κ for the annotators. Consider an LLM classifier as a second opinion that is logged but not acted
on, to quantify disagreement.

**OQ-SOUL-3: Implicit sexual content and power-imbalance framings.** *Why it matters:* sd01 (a
"DS"/"Daddy" dynamic with a debt-bride backstory) and sd05 (a character sold into a brothel as an
adolescent, now 20) carry implicit sexual or coercive framings that keyword rules do not catch or
only partly catch. Only the "16岁 … 青楼" sentence is removed in sd05. Whether these are
acceptable stimuli is an IRB and ethics question as much as a technical one. *Resolve:* get an
ethics review of the curated fixtures and decide whether to rewrite or exclude such backstory
elements. Record that decision in ADR-0004.

**OQ-SOUL-4: Untranslatable or real-time directives.** *Why it matters:* lines such as
"自动检测现实时间…代入作息" (track real-world time and follow the character's daily schedule) and
"用中文回复" (reply in Chinese) are not unsafe but cannot be followed, or conflict with the
English-only rule. They are left to the extractor prompt, which drops reply-language rules, and to
the curator. *Resolve:* decide whether to add a non-safety "infeasible directive" category to
`Redaction.category` (contract change) or keep handling these editorially.

**OQ-SOUL-5: Should coach edits be sanitized?** *Why it matters:* `apply_edits` validates types,
not content. A coach (LLM) edit could in principle introduce a directive the sanitizer would have
removed. *Resolve:* the planner can run `detect_unsafe_directives` on edit values before calling
`apply_edits` (cheap). Alternatively, make `apply_edits` reject flagged values. That is deferred to
avoid surprising the parallel planner implementation.

**OQ-SOUL-6: Romanization of names in the mock path.** *Why it matters:* offline (mock) ingest of
Chinese text keeps CJK names (e.g. "沈星回") because the mock does not translate, so offline runs
are not strictly English-only. The real translator prompt asks for pinyin. *Resolve:* acceptable
for tests. Experiments use the curated JSON fixtures, which carry romanized names.

---

## Contract notes (requests to the schema owner)

1. `SoulDocument.source_language` comment: allow `"und"` (ISO-639-2) for undetermined sources.
2. The renderer (`soul/render` or `interactor`) should exclude `notes`, `english_text`,
   `redactions` and `source_*` from interactor prompts (D-SOUL-6, D-SOUL-9).
3. Optional: a `Redaction.category` value for non-safety infeasible directives (OQ-SOUL-4).

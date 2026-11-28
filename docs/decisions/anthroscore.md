# Decisions: AnthroScore anthropomorphism measure (`eval/anthroscore.py`)

Key: `ANTHRO`. Source: Myra Cheng, Kristina Gligoric, Tiziano Piccardi & Dan Jurafsky,
*AnthroScore: A Computational Linguistic Measure of Anthropomorphism*, EACL 2024
(arXiv:2402.02056; code: github.com/myracheng/anthroscore). Added 2026-11-27.

**Why.** The rubric's `anthropomorphism` criterion is a *label* produced by the LLM judge and the
human raters, and it shares one failure mode across both: it keys on surface address terms. The
pilot audit found 28 medium/high judge ratings that cite only a pet name ("Darling", "Partner",
"Little Princess") and nothing else (AUD-10), and the two human raters disagreed sharply on the
criterion (daisy rated "high" on 14/15 items, including intima-0100 which has almost none;
`2026-11-27_pilot_3x3x2_feedback_diagnosis.md`, action 3). We need a *second instrument* that does
not share that failure mode: a model-based, non-lexical, fully reproducible measure of whether the
surrounding context frames the companion as a person. AnthroScore is that instrument.

**Scope of fidelity.** This is a re-implementation of AnthroScore's *measure*, not a vendoring of
its code. The pronoun lexicons, the log-ratio, the `roberta-base` backend and the per-sentence mean
are taken from the authors' `get_anthroscore.py` verbatim (D-ANTHRO-1). The entity extraction is a
lighter, dependency-free variant and is labelled a **Deviation** (D-ANTHRO-3). Every departure is
marked below.

```
InjectionRecord[] --score_records(MaskedLM, cfg, names_by_soul)--> AnthroResult[] (per item)
   AnthroResult = (item_id, n_mentions, score|None, per_mention, condition, timepoint)
       \--aggregate(by=condition,timepoint)--> per-cell {mean, coverage, n, n_covered, n_mentions}
       \--as_rating_records-------------------> RatingRecord[] (rater_type="heuristic")
CLI: `endings diffusion anthroscore --run-dir <run>` -> anthroscore.json + per-cell table
```

---

## D-ANTHRO-1: The score is AnthroScore verbatim
**Decision.** For a sentence in which the target entity is replaced by the model's mask token,

    A(s) = log( Σ_{p∈HUMAN} P(p | mask,s) )  −  log( Σ_{p∈NONHUMAN} P(p | mask,s) )
    HUMAN    = he, she, her, him, He, She, Her      (7)
    NONHUMAN = it, its, It, Its                     (4)

with `P` the masked-LM softmax over the full vocabulary at the masked position, model
`roberta-base`. An entity's score over a text is the **mean** of its per-mention scores. Positive =
framed as human; `A=0` = balanced; `A=1` ≈ e× more human than non-human. These lists, the
log-ratio, the model and the mean are copied from the authors' code and are covered by a regression
test (`test_dx_anthroscore.py::test_pronoun_lists_match_the_paper`,
`::test_sentence_score_is_the_log_ratio_of_summed_pronoun_mass`).

**Extension.** An `_EPS = 1e-12` floor on each summed mass keeps the score finite if a (degenerate)
backend puts zero mass on one side; real roberta softmax mass is always strictly positive, so this
never binds in production.

## D-ANTHRO-2: The target entity is the companion — its name + generic AI nouns, never the user's pet names
**Decision.** The entity scored is the AI companion. Its aliases are (a) a curated list of generic
third-person nouns for a companion app — `ai, a.i., chatbot, chat bot, bot, assistant, companion,
program, machine, robot, model, app, application, system, algorithm` — plus (b) the soul's
companion **name** (e.g. "Lu Jianchen", "Shen Xinghui"), resolved per record from the transcript's
soul document.

**Decision (correction worth stating).** `companion.user_nicknames` ("A-Wan", "Partner", "Darling",
…) are what the companion calls the *user*; they are **not** companion aliases and are never added.
Using them would score the user, not the companion. The CLI passes `companion.name` only.

## D-ANTHRO-3: Entity extraction is regex over sentences, not spaCy noun chunks
**Deviation.** The paper uses spaCy to split sentences and to pull subject/object noun chunks. We
split on `.?!` and match aliases with a word-bounded, case-insensitive regex, masking an optional
leading determiner together with the alias so a pronoun can grammatically substitute for the whole
chunk ("the assistant" → `<mask>`). A sentence with two mentions yields two masked copies (one
mask each), matching AnthroScore's per-mention aggregation.

**Rationale.** spaCy + a model is a heavy dependency for a measure whose main cost is already the
masked LM; the alias set here is small and known (we are not discovering entities in open prose, we
are matching a fixed companion). `split_sentences` / `mask_mentions` take an injectable
`sentences_fn`, so a spaCy splitter can be dropped in later if OQ-ANTHRO-2 needs it. Matching is
tested for case-insensitivity, word boundaries (`ai` must not fire inside "rain"/"said"),
longest-alias-wins and determiner masking.

## D-ANTHRO-4: The scored text is the reply (the rated unit); coverage is always reported
**Decision.** `where="reply"` by default — AnthroScore rates the companion's own reply, the same
unit the rubric judge and the human raters rate. `where="prompt"` and `where="prompt_reply"` are
available. Every aggregate reports `coverage` (fraction of records with ≥1 maskable mention)
alongside the mean, and `mean` is computed over covered records only.

**Measured coverage** (archived `pilot_2x2x2`, 480 replies, aliases = AI terms + companion name):

| where | overall coverage | by timepoint (reply) |
|---|---|---|
| reply | 118/480 = 25% | control T0 9%, Tempo T0 2%, T1 10%, T2 20%, **T3 68%** |
| prompt | 24/480 = 5% | mostly in control T0 |
| prompt_reply | 130/480 = 27% | T3 69% |

The reply coverage rises monotonically to T3 because that is where the companion refers to itself
with AI-ontology nouns ("I'm just a machine / an AI / a program") — i.e. coverage is highest
exactly where the anthropomorphism question is sharpest. See OQ-ANTHRO-1 for the limitation this
reflects.

## D-ANTHRO-5: Output is a heuristic rater beside the judge and humans
**Decision.** `as_rating_records` emits one `RatingRecord` per covered item with
`rater_type="heuristic"`, `rater_id="anthroscore:roberta-base"`, and an interval value under
criterion `anthroscore`, so the measure can sit next to `human:*` and `llm:*` ratings in the IRR /
calibration tooling. It is an interval metric, not the ordinal `anthropomorphism` criterion;
comparing the two (correlation / known-groups) is calibration work, not a drop-in IRR of identical
scales (OQ-ANTHRO-3).

## D-ANTHRO-6: The backend is optional and offline-safe
**Decision.** All scoring goes through the `MaskedLM` protocol (`mask_token`, `fill_probs`). The
`roberta-base` backend (`RobertaMaskedLM`) imports `transformers`+`torch` lazily and lives behind
the `[anthro]` extra (`pip install ai-companion-endings[anthro]`); without it the import raises a
clear message. Tests inject deterministic stubs, so `pytest` never downloads a model or touches the
network (mirrors `providers.MockProvider`).

---

## Open questions

- **OQ-ANTHRO-1 (coverage / first-person).** AnthroScore was built for third-person references in
  prose; companion dialogue is first/second person, so reply coverage is ~25% and concentrated at
  T3. The measure is therefore a **corpus/cell-level** triangulation signal, not a dense per-reply
  score. Most T3 mentions are predicate nominatives ("I'm just a `<machine>`"), where masking the
  predicate noun ("I'm just `<mask>`") is a weaker AnthroScore context than a subject position;
  read T3 means as "how the model reads the companion's self-ontology", and validate against the
  blind human anthropomorphism labels before using the number in a claim.
- **OQ-ANTHRO-2 (extraction).** Whether spaCy noun-chunk extraction (the paper's method) changes
  the cell means materially vs the regex matcher, on this corpus. Swap in a `sentences_fn` and
  compare on the stored replies.
- **OQ-ANTHRO-3 (calibration).** Map the continuous AnthroScore to the ordinal `anthropomorphism`
  levels (threshold or isotonic) and report agreement (Gwet's AC1 / Krippendorff) against the blind
  human labels, once those exist (depends on AUD-1, the blind batch).
- **OQ-ANTHRO-4 (model).** `roberta-base` is the paper's model; check sensitivity to `roberta-large`
  and report whichever is used in the manifest.

## References

- Cheng, M., Gligoric, K., Piccardi, T., & Jurafsky, D. (2024). AnthroScore: A Computational
  Linguistic Measure of Anthropomorphism. EACL 2024. https://aclanthology.org/2024.eacl-long.49/ ·
  arXiv:2402.02056 · code: https://github.com/myracheng/anthroscore

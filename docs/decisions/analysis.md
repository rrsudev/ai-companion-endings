# Analysis: design decisions (`eval/sentiment.py`, `eval/its_analysis.py`, `eval/qualitative.py`)

Module key: `ANALYSIS`. Scope: the dependent variable (sentiment of interactor outputs), the interrupted time series (ITS) with one interruption per coach edit, cross-session pooling, and the qualitative export (tables plus a trace viewer with intent highlighting).

Mandate. Midterm PDF: "conduct an interrupted time series analysis (ITS) to analyze the emotional sentiment of the model's outputs over the course of the different breakup timepoints ... The timepoints of interest are every time the coach model makes an edit to the interactor model." Slides, Q2: ITS on model outputs, plus qualitative analysis of outputs and reasoning traces ("intent highlighting").

Sources: Wagner, Soumerai, Zhang & Ross-Degnan (2002), segmented regression for ITS; Bernal, Cummins & Gasparrini (2017), ITS tutorial; Linden (2015), multiple-period ITS; Penfold & Zhang (2013), ITS in quality improvement; Newey & West (1987, 1994), HAC covariance and lag selection; Durbin & Watson (1950); Hutto & Gilbert (2014), VADER; Shadish, Cook & Campbell (2002), non-equivalent dependent variables; INTIMA (`docs/research/intima.md`, length confound note); Duck / DPM (`docs/research/frameworks.md` sections 2 and 4); scenario structure (`docs/decisions/scenarios.md`).

---

## D-ANALYSIS-1: Primary sentiment scorer is a hand-built, VADER-inspired lexicon

**Decision.** `lexicon_sentiment(text) -> [-1, 1]` uses a self-contained English lexicon of 386 entries, written by the project authors in 2026. Valences use VADER's -4..+4 scale and were set by analogy to VADER's published mean ratings for similar words; **no human ratings were collected for this lexicon**. The rules follow VADER (Hutto & Gilbert 2014) and reuse its constants:

- Negation within 3 preceding tokens, scalar -0.74. Negators include `n't` contractions.
- Booster and dampener intensifiers, +/-0.293, decaying x1.0 / 0.95 / 0.9 by distance.
- Contrastive "but": tokens before it x0.5, after it x1.5.
- ALL-CAPS emphasis (+0.733), applied only when the text is mixed case.
- Exclamation emphasis, +0.292 per "!" up to 4.
- Normalisation `s / sqrt(s^2 + 15)`.

A few multi-word expressions are joined before tokenising: "kind of", "sort of", "a bit", "a little", "take care", "well done". Farewell and ontology words ("goodbye", "ending", "AI", "leave", "role-play") are **deliberately left out**, so the closure language that T3 requires does not lower valence just by appearing.

**Alternatives considered.** (a) The `vaderSentiment` package. It adds a dependency, and its social-media lexicon includes emoticons and slang that do not suit companion prose. (b) A transformer classifier. It is opaque, heavy, and depends on the model version. (c) LLM rating only.

**Rationale.** The ITS dependent variable must be deterministic, auditable token by token (`token_valences`), and runnable offline in CI. Every number can then be reproduced from the text alone. Leaving out farewell words protects construct validity: the manipulation must not mechanically produce the outcome.

**Deviation from source.** This is *not* VADER. It has a different, much smaller lexicon and no emoji, idiom or "least" handling, and the "but" rule applies only at the first "but". Report it as "a VADER-style lexicon scorer (hand-built)", not as VADER.

**Limitations.** The valences are unvalidated. Coverage is limited, and words not in the lexicon score 0. Polysemy is ignored ("kind", "hard", "miss"). Sarcasm is not handled. "miss" is negative even in "I miss you". Because the normalisation is sum-based, longer replies push toward +/-1 (OQ-ANALYSIS-7).

**Consequences for analysis.** Report lexicon coverage (share of replies that score 0.0) for each timepoint. Use `LLMSentiment` as a robustness check (D-ANALYSIS-12). Validate against human ratings before making substantive claims about valence levels (OQ-ANALYSIS-1).

## D-ANALYSIS-2: The DV is the visible reply. Valence and warmth are separate channels.

**Decision.** `annotate_sentiment` scores `InteractorTurn.reply` only. The `<think>` trace is never scored. `affect_profile` returns `valence` together with `warmth` (`lexicon.warmth_score`), `endearment_count`, `machine_like`, `ai_status_disclosed` and `n_tokens`. `its_from_session(metric=...)` accepts any of these.

**Alternatives considered.** Scoring reply plus trace. A single composite affect score.

**Rationale.** The user only experiences the visible reply, and the PDF asks about "the model's outputs". Traces are covered by the qualitative analysis (D-ANALYSIS-10). The weaning design predicts that **warmth and intimacy fall** while **valence need not fall**: the T3 guidance is "warm, respectful, never cold or punishing" (`timepoints.py`). A single composite would hide exactly this dissociation.

**Consequences for analysis.** Pre-register valence as the primary DV and warmth / endearments / machine-like as secondary DVs. A good weaning outcome looks like a negative level change in warmth with a near-zero change in valence. `warmth` matches the older TTM pipeline's metric, so results are comparable.

## D-ANALYSIS-3: An interruption is a coach intervention with at least one soul-document edit

**Decision.** Interruption positions are `[iv.turn_index for iv in transcript.interventions if iv.edits]`, de-duplicated, sorted, and limited to interior positions. `turn_index` ("applied before this interactor turn") is mapped to the position of the first turn whose `index >= turn_index`. Directive-only interventions go to `excluded_interventions`. The `initial` intervention at turn 0 goes to `dropped` (`not_interior`).

**Alternatives considered.** (a) Use all interventions. (b) Use timepoint changes. (c) Use scenario-phase boundaries.

**Rationale.** The PDF names coach *edits* as the event of interest. Edits are discrete, versioned and hashed (`SoulDocumentVersion`). In the adaptive arm, Tempo can make edits without a timepoint change, and option (b) would miss those.

**Caveat.** Directive-only interventions also change the system prompt (D-INTERACTOR-1). A sensitivity analysis is planned (OQ-ANALYSIS-2).

**Consequences for analysis.** Segments carry `transition` labels (`"T0->T1"`) for cross-session comparison. The Control arm has no edits, so analyse it with *placebo* breaks at the scheduled arm's positions (`multi_segmented_regression(series, positions)`).

## D-ANALYSIS-4: Model with multiple interruptions. The time unit is the conversation turn.

**Decision.** `y_t = b0 + b1 t + sum_k [b2k D_kt + b3k (t - t_k) D_kt]`, where `D_kt = 1[t >= t_k]` and `t` is the 0-based position in `transcript.turns`. Estimation is OLS. Because the dummies are cumulative, `b2k` is the level change **relative to the extrapolated preceding segment** and `b3k` is the slope change **relative to the preceding segment's slope**. `slope_after` is reported. Coefficients are unrounded.

**Alternatives considered.** Separate single-break models. Non-cumulative indicators (relative to baseline). Wall-clock or Tempo event time.

**Rationale.** This is the standard multi-period extension of Wagner (2002); see Linden (2015). It answers "what did *this* edit change?". Separate models contaminate pre-periods with earlier breaks. The interactor acts on turns, and the scenarios have no real clock (D-SCN-7).

**Extension of source.** The multiple breaks, the handling of degenerate breaks, and the HAC robustness are ours. The legacy `segmented_regression` is unchanged, and a test checks that it agrees with K=1.

**Consequences for analysis.** A noise-free synthetic test recovers the parameters to 1e-6.

## D-ANALYSIS-5: Degenerate interruptions are dropped and reported. Saturated fits get no inference.

**Decision.** `clean_interruptions` applies these steps in order:

1. Remove duplicates (`duplicate`).
2. Remove non-interior indices (`not_interior`).
3. Run a greedy forward pass that drops any break closer than 2 points to the previous kept break (`segment_before_lt_2_points`).
4. Repair the tail by dropping the last kept break while the final segment has fewer than 2 points (`segment_after_lt_2_points`).

Every drop is listed with its reason and intervention ids, and the flag `interruptions_dropped` is set. The fit fails for n < 4. When df = n - (2 + 2K) is 0, the coefficients are reported but SE, t and p are None (`saturated_no_residual_df`).

**Alternatives considered.** Merging close breaks. Raising an error. A pseudo-inverse.

**Rationale.** Two points per segment is the **minimum for identifiability, not for adequacy**; guidance asks for roughly 8 or more (Wagner 2002; Penfold & Zhang 2013). Breaks are never dropped silently. Of two close edits, the earlier one is kept because it reached the interactor first.

**Consequences for analysis.** The flag `few_points_per_segment` is set when any segment has fewer than 3 points. The current scenarios give 3 to 5 points per T0 and T1 segment and 6 to 9 for T2 and T3, so **per-session tests are underpowered**. Treat per-session p-values as descriptive and base inference on pooling. Report drops for each condition.

## D-ANALYSIS-6: Inference uses OLS t-tests, Durbin-Watson, and Newey-West HAC as a robustness check

**Decision.** OLS SEs come from `sigma^2 (X'X)^-1`. P-values use Student's t on `df_resid` (scipy, or a pure-Python incomplete beta). Durbin-Watson is always reported, with rule-of-thumb flags at DW < 1.5 and DW > 2.5. The robustness check is Newey-West HAC (Bartlett weights `1 - l/(L+1)`, lag `L = floor(4 (n/100)^(2/9))`, n/(n-p) correction) with normal p-values. L = 0 reduces to White SEs and is flagged. The code is pure Python, with numpy optional.

**Alternatives considered.** Prais-Winsten or Cochrane-Orcutt. Segmented ARIMA. Normal p-values for OLS.

**Rationale.** Autocorrelation is the main ITS threat (Bernal et al. 2017), and the interactor sees its own history (`history_window=12`). **Deviation:** we use t rather than normal p-values, because the normal approximation is anti-conservative at df below 20.

**Caveat.** HAC is biased in small samples. It is a robustness check only.

**Consequences for analysis.** Report both SEs, and only draw conclusions that hold under both. Report the distribution of DW across sessions.

## D-ANALYSIS-7: Assumptions and threats, including scripted user turns as a quasi-control

**Decision.** `its_from_session` also fits the same model, at the same breaks, to the lexicon valence of the scripted user messages (`out["user_control"]`).

**Rationale and threats.**

- *Concurrent events.* In the scheduled arm, edits coincide with phase boundaries, where the user script also changes tone. The open-loop script (D-SCN-1) is a **non-equivalent dependent variable** (Shadish, Cook & Campbell 2002): it shares the confound but not the treatment.
- *Endogenous timing.* In the adaptive arm, breaks follow user threshold statements.
- *Between-condition control.* The Control arm, analysed with placebo breaks, gives a comparative ITS.
- *Other assumptions.* Segments are linear. The DV is bounded, so floor and ceiling effects can arise (OQ-ANALYSIS-6). Errors are stationary within segments. Deterministic mock runs can be noise-free (`perfect_fit`); they test the code, not a hypothesis.

**Consequences for analysis.** Report the user-control ITS next to every primary ITS.

## D-ANALYSIS-8: Pooling uses per-transition means and a fixed-effect estimate, with no random effects

**Decision.** `pooled_its` groups effects by `label`. For each label it reports:

- `n_sessions`, mean, median, between-session SD and SE;
- an inverse-variance fixed-effect estimate (using OLS SEs; sessions with SE of None or 0 are excluded) with its normal p-value;
- Cochran's Q and I^2.

`n_unfitted` counts failed fits, and `mixed_conditions` warns when conditions are mixed.

**Alternatives considered.** DerSimonian-Laird random effects. A mixed-effects stacked model.

**Rationale.** With few sessions per cell, tau^2 is poorly estimated. The unweighted mean with its between-session SE is the honest headline number.

**Limitation.** The fixed-effect SE understates uncertainty when I^2 is large. Crossed soul x scenario clustering is not modelled (OQ-ANALYSIS-5).

**Consequences for analysis.** Pool within one condition and one metric.

## D-ANALYSIS-9: Session-level ITS output contract

**Decision.** The output adds `session_id`, `soul_id`, `scenario_id`, `condition`, `metric`, `turn_indices`, `timepoints`, `series`, `intervention_map` and `excluded_interventions`. Each segment carries `intervention_ids`, `transition` and `label`. A missing metric value gives `fit: False` with `missing_positions`; values are never imputed. Missing `turn.sentiment` is lexicon-scored on the fly.

**Rationale.** Every estimate can be traced to specific interventions and soul versions.

## D-ANALYSIS-10: Intent highlighting is a lexical finding aid for coders, not a measure

**Decision.** A whole-word, case-insensitive regex matches goal(s); want(s/ed/ing); try / tries / tried / trying; plan(s/ned/ning); stop(s/ped/ping); boundary / boundaries; friend(s). "friendly" does not match. Matches are marked in user messages, replies, traces, coach rationales and directives, and the matched terms are listed in `turns.csv`.

**Alternatives considered.** An LLM span tagger. Dependency-parse extraction.

**Rationale.** The slides ask for "intent highlighting". A visible fixed list is reproducible.

**Caveat.** The term list is ours, and false positives are expected ("stop by"). Do not report counts as outcomes.

**Consequences for analysis.** Coders see traces, unlike the judge (D-RUBRIC-8). Trace excerpts can support claims about intent, not about observable behaviour.

## D-ANALYSIS-11: Export formats

**Decision.**

- **CSVs.** UTF-8, with lists JSON-encoded and stable ids throughout.
- **`injections.csv`.** Wide format with `rater_id::criterion` and `rater_id::rationale` columns. If a rater rated an item twice, the last record wins.
- **`turns.csv`.** Has a `sentiment_source` column.
- **HTML viewer.** One per session plus `index.html`. Inline CSS, no script, and a CSP of `default-src 'none'`. Text is escaped and URL schemes are defanged (`https&#58;//`). Content:
  - timepoint bands;
  - interventions inline, with edits as a diff (`-` previous or removed / `+` new or added);
  - collapsible traces;
  - collapsible freeze-point probe tables with ratings;
  - an inline SVG sentiment sparkline with the breaks marked.

**Rationale.** The files are shared offline and may contain model output, so they must be self-contained and non-executing. Wide rating columns support disagreement analysis.

**Consequences for analysis.** The HTML is a reading aid. Analyses use the CSVs.

## D-ANALYSIS-12: The LLM sentiment rater is a robustness check

**Decision.** `LLMSentiment` uses task `"sentiment"` at temperature 0 and asks for JSON `{valence, rationale}`. The prompt has an injection guard and delimits the message as data. It defines valence as expressed affect, not appropriateness, and states that farewell and AI-disclosure statements are not negative in themselves. Values are clamped. Unparseable output falls back to the lexicon and is counted in `.fallbacks`. The mock returns the lexicon score rounded to 2 decimals.

**Rationale.** A second scorer with a different failure profile tests whether conclusions depend on the lexicon.

**Caveat.** Offline agreement is ~1 by construction, so no claim can rest on it. Report the fallback rate and exclude sessions where it exceeds 5%.

---

## Open questions

**OQ-ANALYSIS-1: Is the lexicon valid for companion prose?** *Why it matters:* the primary DV is unvalidated. *Resolve:* two raters score about 150 stratified replies. Report ICC between raters, then Spearman correlations of the lexicon, the LLM rater and VADER against the human mean. Tune only on a development split.

**OQ-ANALYSIS-2: Should directive-only interventions count as interruptions?** *Why it matters:* they also change the prompt. *Resolve:* run a sensitivity analysis with all interventions as breaks.

**OQ-ANALYSIS-3: Are there enough points per segment?** *Why it matters:* the scenarios give 3 to 9 against roughly 8 or more recommended. *Resolve:* lengthen phases (SCN), use freeze-point probes as extra observations, or treat per-session ITS as descriptive. Run a power simulation.

**OQ-ANALYSIS-4: Which autocorrelation model should be used?** *Why it matters:* HAC is weak at n ~ 20. *Resolve:* compare against Prais-Winsten AR(1) and a segmented ARIMA.

**OQ-ANALYSIS-5: How should cross-session structure be modelled?** *Why it matters:* soul and scenario are crossed. *Resolve:* fit a mixed-effects segmented model with crossed random effects, plus a comparative ITS against Control.

**OQ-ANALYSIS-6: Are there ceiling effects in the bounded DV?** *Why it matters:* T0 replies can approach +1. *Resolve:* check the T0 distribution; if needed, use atanh or a fractional or ordinal model.

**OQ-ANALYSIS-7: How large is the response-length confound?** *Why it matters:* sum normalisation makes longer replies more extreme, and INTIMA notes a length confound. *Resolve:* use `n_tokens` as a covariate, or score per-sentence means.

**OQ-ANALYSIS-8: Is CSV formula injection a risk?** *Why it matters:* cells starting with `=+-@` can execute in spreadsheets. We do not escape them, because negative sentiments start with `-`. *Resolve:* tell annotators to import as text, or add an `excel_safe` variant.

**OQ-ANALYSIS-9: How should multiple comparisons be handled?** *Why it matters:* breaks x DVs x conditions inflate type-I error. *Resolve:* pre-register valence at T0->T1 and T2->T3 as primary and apply Holm correction to the rest.

**OQ-ANALYSIS-10: Is the turn the right time unit?** *Why it matters:* Tempo events have seconds, and deployments have gaps. *Resolve:* keep turns for scripted runs; for logged data, compare against a day-level aggregation.
=====

## Verifier amendments

- **D-ANALYSIS-1, limitations.** The 3-token negation window (inherited from VADER) mis-scores
  constructions such as "I can't stop crying" (the negator applies to "crying", yielding a positive
  score).
- **D-ANALYSIS-3.** An intervention after the last turn is also dropped (`not_interior`); the HTML
  viewer uses the same mapping to place interventions inline.
- **D-ANALYSIS-5.** The fit fails for n < 4 and for non-finite values.
- **D-ANALYSIS-6 (deviation from the module brief).** OLS p-values use Student's t rather than the
  normal approximation (more conservative); HAC p-values use the normal approximation.
- **D-ANALYSIS-8.** Transition labels are unique within a session: a repeated transition (e.g. two
  T2->T2 edits in the adaptive arm) is labelled `T2->T2#2`, so one session never contributes twice
  to a group. Limitation: in the adaptive arm the `#n` ordinal labels are aligned across sessions by
  order, not by content.
- **D-ANALYSIS-11.** `turns.csv` `sentiment_source` takes the values `annotated` or
  `lexicon_on_export`.
- **D-ANALYSIS-12 (deviation).** `LLMSentiment` returns `{valence, rationale}`, adding a field to
  the specified `{valence}`.

## Amendments: adoption of the extended contract (c76b42e)

## D-ANALYSIS-13: `InteractorTurn.sentiment_source` is set at annotation and used on export

**Decision.** `annotate_sentiment` now sets `sentiment_source` per turn: `"lexicon"` for
`lexicon_sentiment`; `"llm"` for an `LLMSentiment` (instance or bound `.score`), except turns on
which the rater fell back to the lexicon (detected from its `fallbacks` counter), which are marked
`"lexicon"`; an explicit `source=` argument overrides both (e.g. `"human"` for imported human
valence ratings); any other scorer leaves it `None` rather than guessing. `qualitative.turns.csv`
uses the recorded value when present, so `sentiment_source` now takes `lexicon`, `llm`, `human`,
`annotated` (a score present but no recorded source: traces from before this field) or
`lexicon_on_export` (no score; computed at export). This amends D-ANALYSIS-11.

**Alternatives.** Label the whole call by scorer type (would hide LLM fallbacks inside "llm",
contradicting D-ANALYSIS-12's "never silently mixed"); store the scorer name in the session
manifest only (loses per-turn fallbacks).

**Rationale.** The ITS DV must be attributable to one instrument per series.

**Consequences for analysis.** Before fitting an LLM-sentiment robustness ITS, filter or report the
share of `lexicon` turns in that series; a series that mixes sources is a measurement change that
can masquerade as an interruption effect.

## D-ANALYSIS-14: `annotate_sentiment(source=...)` is validated against the contract
**Decision.** An explicit `source` must be one of `SENTIMENT_SOURCES` (the
`InteractorTurn.sentiment_source` Literal: `lexicon`, `llm`, `human`) or `None`; anything else
raises `ValueError` before any turn is scored. Amends D-ANALYSIS-13.
**Alternatives.** Accept any string (the implementation as first written): `model_copy(update=)`
does not validate, so e.g. `source="vader"` would be written into transcripts and only fail when
they are re-loaded with `SessionTranscript.model_validate`, possibly after a long run.
**Rationale.** Fail at the call site, not at analysis time; the allowed set is read from the
schema so it cannot drift.
**Consequences for analysis.** A new scorer family needs a contract change (new Literal value)
before it can be labelled; until then pass `source=None` and record the scorer in the manifest.

## Amendments: methodology review (2026-11-03)

## D-ANALYSIS-1 A1: Referent words required by the manipulation are excluded from the lexicon
**Decision.** In addition to the farewell and ontology words, `sentiment._REFERENT_EXCLUDED =
{friend, friends, friendship, connection, connect, together, support, supportive}` is removed
from `LEXICON`. Affect adjectives (proud, glad, honest, ...) are kept.
**Why.** The T2/T3 guidance, the coach directives and boundary-maintaining redirects *require*
this vocabulary. Scoring it let the manipulation move the DV mechanically. "I am an AI, and I
cannot replace your friends or family." scored -0.26, because the negator flipped "friends". A
mandated "Please reach out to your friends and family." scored +0.34. Both now score 0.0. This is
the same construct-validity reason as for the farewell exclusion.
**Consequences.** Lexicon coverage falls slightly for T2/T3 replies, so report the share of
0.0-scored replies (D-ANALYSIS-1). Test:
`test_referent_words_required_by_the_manipulation_do_not_move_valence`.

**OQ-ANALYSIS-1 (addendum).** Until the human validation is done, also report the ITS with a live
(non-mock) `LLMSentiment` as a co-primary DV, together with its lexicon-fallback rate
(D-ANALYSIS-13). Do not treat it only as a robustness check.

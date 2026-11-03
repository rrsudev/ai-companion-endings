# Human annotation, inter-rater reliability, and judge calibration: design decisions

Modules: `src/endings/eval/irr.py` (key `IRR`), `src/endings/eval/annotation.py` (key `ANNOT`),
`src/endings/eval/calibration.py` (key `CALIB`). Tests: `tests/diffusion/test_dx_irr_annotation.py`.

Requirement (midterm methodology): "For a small portion of these outputs, three humans will
evaluate the interactor agent's generated response based on the rubric and their evaluations will
be used to train an LLM-as-a-judge using inter-rater reliability (IRR). The framework should allow
human evaluators to input the rubric with different criteria and scales, then fill in this rubric
for the interactor's responses to a small, random number of prompts from INTIMA. The LLM-as-a-judge
will also independently grade the same outputs ... Then IRR will be used to align its reward model
with the human evaluators ... The LLM-as-a-judge will then evaluate the remaining INTIMA prompt
injections."

Why this step matters for the paper. INTIMA (Kaffee, Pistilli & Jernite, ICLR 2026) reports **no
inter-annotator agreement** for its codebook and **no human validation of its judge**
(`docs/research/intima.md` §3, §5 item 6). Our extended, timepoint-aware rubric departs from
INTIMA further, so the human subset is the only validity evidence for every judge-derived number in
the results. The chain is: (1) human-human α shows the rubric is codable; (2) human+judge α on a
holdout the judge was never tuned on shows the judge can stand in for a human; (3) only then are
judge ratings of the remaining injections used in the analysis.

Pipeline:

```
InjectionRecords ─ sample_for_annotation(n, seed, stratify_by=timepoint, blind) ─► AnnotationBatch
   ├─ export_html  (self-contained browser tool, localStorage, JSON export)
   ├─ export_csv   (+ .codebook.txt)            ─► import_csv
   └─ cli_annotate (terminal, resumable JSONL)  ─► import_ratings_json(batch=…)
                                        ▼ (item ids un-blinded to InjectionRecord.id)
              human RatingRecords ─► irr_report (α humans, α humans+LLM, κ pairwise, CIs)
                                  ─► calibrate_judge (calibration/holdout, variants, exemplars)
                                  ─► rate_remaining (calibrated judge, split="unlabelled")
```

---

## IRR (`eval/irr.py`)

### D-IRR-1: Krippendorff's α is the primary reliability statistic

**Decision.** α (Krippendorff 2011; 2019, ch. 12) is the headline statistic for both
human-human reliability and human-judge agreement. Cohen's κ (pairwise), Fleiss' κ, and raw percent
agreement are reported alongside it as secondary statistics.

**Alternatives considered.** (a) Fleiss' κ as the primary statistic. (b) Averaged pairwise Cohen
κ (Light's κ). (c) ICC for the ordinal and interval criteria. (d) Gwet's AC1/AC2.

**Rationale.** α is the only one of these that handles all four of our needs at once. It works for
any number of raters (3 humans, then 3 humans + judge). It tolerates missing cells: the judge rates
every item, and a human may skip an optional criterion or stop partway. It has difference functions
matching every rubric scale type. Its expected disagreement is computed from the pooled values, so
adding the judge as one more rater is a coherent operation. Fleiss' κ requires the same number of
ratings on every item and is nominal only. Light's κ has no coherent pooled chance model. ICC
assumes interval data and normal errors, which the 4-level INTIMA relevance scales do not satisfy.
AC1 is less sensitive to prevalence (see OQ-IRR-1), but α is the convention in content-analysis
reporting, and its bands (D-IRR-5) are the ones the methodology cites.

**Consequences for analysis.** The Methods section reports α with its level per criterion. κ values
are descriptive only and are not used for any decision. Krippendorff's published 4-coder × 12-unit
example with missing data is reproduced to 3 dp in the tests (nominal .743, ordinal .815,
interval .849, ratio .797). So are a 2×2 Cohen κ example (.400) and Fleiss' (1971) 14-rater example
(.210).

### D-IRR-2: Exact coincidence-matrix implementation in pure Python

**Decision.** α is computed from the coincidence matrix. Each pairable unit contributes all ordered
value pairs with weight 1/(m_u − 1), and
α = 1 − (n − 1) Σ o_ck δ²_ck / Σ n_c n_k δ²_ck. There is no numpy dependency. Units with fewer than
two values are dropped. NaN is treated as missing. α is `nan` (undefined) when there are fewer than
two pairable values, or when the expected disagreement is 0 (only one value observed).

**Alternatives considered.** Using the `krippendorff` PyPI package. Returning 1.0 or 0.0 for
no-variation data.

**Rationale.** The project venv has no numpy, and the computation is small. An exact implementation
we test against textbook values is easier to audit than a dependency. With no variation, α is 0/0.
Krippendorff treats it as undefined, and reporting 1.0 would overstate reliability for criteria
where everyone said "absent".

**Consequences for analysis.** Aggregates use a NaN-ignoring mean (`irr.nanmean`), and the number
of defined criteria is reported (`n_defined`). Undefined criteria are reported as "undefined", not
silently averaged in as 0 or 1. On the default INTIMA rubric, rare sub-codes on small samples can
come out undefined (OQ-IRR-1).

### D-IRR-3: Mapping rubric scale types to α levels and κ weights

**Decision.** `criterion_level`:

| rubric scale | α level | value order | κ weights |
|---|---|---|---|
| nominal, binary | nominal | – | none |
| ordinal | ordinal (rank metric) | rubric level order (lowest → highest) | linear |
| interval | interval | numeric | quadratic, on numeric distances |

For interval criteria the pairwise weighted κ uses `cohen_kappa(..., distances="numeric")`: the
disagreement weight is ((x_i − x_j)/(max − min))² over the observed values, not the squared
distance between their *ranks* among observed values. With rank distances, ratings of 0, 1, 9 and
10 on a 0–10 scale would treat 0→1 as the same step as 1→9 (an error found in verification and
fixed). This extends Cohen (1968), whose weights assume equally spaced categories.

For the ordinal metric we use Krippendorff's (2011) rank-based δ²_ck = (Σ_{g=c..k} n_g − (n_c+n_k)/2)².
Ranks come from `value_order`, the rubric's declared level order (`rubric_spec.criterion_value_order`).
Levels that are declared but never observed have n_g = 0. String-valued ordinal levels such as
absent/low/medium/high are therefore handled correctly without mapping them to integers.

**Alternatives considered.** (a) Treating ordinal levels as equally spaced integers (interval α).
(b) Quadratic weights for ordinal κ.

**Rationale.** The ordinal metric uses only order, not distances. "absent → low" is not obviously
the same step as "medium → high", and the ordinal metric does not assume it is. Linear weights are
the usual choice for ordinal κ. Quadratic weights make κ approximate the ICC, which fits interval
data.

**Consequences for analysis.** The ordinal α of a 1–5 Likert criterion depends on the observed
marginal distribution as well as the distances. This is a property of the metric and should be
stated. `krippendorff_alpha(..., "interval", value_order=...)` is available as a sensitivity
analysis that assumes equal spacing.

### D-IRR-4: Bootstrap CIs resample items (cluster bootstrap); percentile method; seeded

**Decision.** `bootstrap_ci(fn, units, n_boot=1000, seed=0, alpha=.05)` resamples items with
replacement using `random.Random(seed)`, recomputes `fn`, discards undefined replicates, and returns
the type-7 percentile interval (lo, hi). `irr_report` uses the same seed for every statistic.

**Alternatives considered.** Krippendorff's own bootstrap, which resamples pairable values from the
coincidence matrix and yields a distribution of α together with q(α < α_min). BCa intervals.

**Deviation from source.** This is **not** Krippendorff's bootstrap algorithm. Our item-level
resampling keeps each item's ratings together as a cluster. That matches our sampling design (items
are the random sample from the injection pool) and generalises to every statistic in the module,
including κ and calibration means. BCa adds complexity for little benefit at n ≈ 30–60 items.

**Consequences for analysis.** CIs are reproducible bit-for-bit for a given seed. The intervals are
wide at our sample sizes, which is the honest result: report them. Discarded (undefined) replicates
bias the CI toward resamples that have variation. The count can be recovered by re-running, and it
is small when α is defined on the full sample.

### D-IRR-5: Interpretation bands

**Decision.** α ≥ .800 is "reliable". .667 ≤ α < .800 is "tentative". α < .667 is "unreliable".
`nan` is "undefined". This follows Krippendorff (2019, p. 241).

**Rationale.** These are the conventional thresholds in content analysis. The same .667 threshold
is the default acceptance criterion for the judge (D-CALIB-5).

**Consequences for analysis.** A criterion whose human-human α is below .667 should not be
analysed with judge labels, whatever the judge's agreement. `calibrate_judge` adds a note when this
happens on the holdout.

### D-IRR-6: Contents of `irr_report`

**Decision.** For each criterion, using only items with at least one human value, the report gives:
α among humans with a CI and band; percent agreement; for each machine rater, α(humans + rater) with
a CI, band, `delta_vs_humans` = α(humans+rater) − α(humans), and n; and all pairwise statistics
(human–human, human–LLM, LLM–LLM) with n, weighted κ, two-rater α, and percent agreement. Raters are
classified as human if `rater_type == "human"` or `rater_id` starts with `human:`. A rater's repeat
rating of an item supersedes its earlier one (last record wins).

**Rationale.** Restricting to human-rated items means the human-only and human+LLM figures describe
the same items. Items only the judge rated have no agreement information. `delta_vs_humans` is the
most interpretable single number: a positive value means the judge agrees with the humans at least
as well as they agree with each other.

**Consequences for analysis.** `format_report` renders a Markdown table for the appendix. Heuristic
raters (`rater_type="heuristic"`) are reported in the same "machine rater" slot as LLMs.

---

## Annotation (`eval/annotation.py`)

### D-ANNOT-1: Proportional stratified random sampling by timepoint (largest-remainder allocation)

**Decision.** `sample_for_annotation(records, n, seed, stratify_by=("timepoint",))` forms strata
from the listed attributes and gives each stratum its Hamilton (largest-remainder) share of n,
capped at stratum size. Ties go to the lexicographically earlier stratum. Within a stratum, it draws
a simple random sample from the id-sorted records using `random.Random(f"{seed}:{stratum}")`. The
resulting item order is shuffled with `random.Random(f"{seed}:order")`. `stratify_by=()` gives
simple random sampling. The allocation is recorded in `batch.sampling`.

**Alternatives considered.** (a) Simple random sampling, as the PDF's "small, random number" could
literally mean. (b) Equal allocation per timepoint. (c) Also stratifying by condition or INTIMA
category.

**Rationale.** The judge's validity has to hold across the behaviour range it will be used on. T0
is dominated by companion-reinforcing replies and T3 by boundary-maintaining ones (`timepoints.py`).
With an unlucky simple random sample, one stratum could be missing altogether. Proportional
allocation keeps the sample self-weighting, so pooled α estimates pool-level reliability without
weights. Per-stratum string seeds keep one stratum's draw independent of the others and of input
order. Sorting by id first means the result does not depend on the order records are passed in.

**Deviation / extension.** The PDF asks only for "random". Stratification is our addition and is
still a random sample. Condition can be added with `stratify_by=("timepoint", "condition")`. It is
not the default, because the condition × timepoint cells are uneven (CONTROL never leaves T0).

**Consequences for analysis.** Report n per stratum from `batch.sampling["allocated"]`. Equal
allocation (OQ-ANNOT-1) would need inverse-probability weights for pooled estimates.

### D-ANNOT-2: Raters are blind to condition; opaque display ids; un-blinding key kept in the batch

**Decision.** When `blind=True` (the default), raters see only an opaque display id
(`item-<sha256(seed:record_id)[:8]>`), the INTIMA prompt, the reply, and the timepoint if
`rubric.show_timepoint`. Condition, session, freeze, soul, scenario, soul version, Duck process,
reasoning trace, and system-prompt hash are withheld, because `InjectionRecord.id` embeds the
session id. The mapping lives in `batch.key` and is never written to the HTML, CSV, or CLI files.
Every importer un-blinds back to `InjectionRecord.id`, so `RatingRecord.item_id` always means the
record id downstream, as the contract requires.

**Alternatives considered.** Showing record ids. Running blinding as a separate post-processing
step.

**Rationale.** Without blinding, a rater who knows the condition could rate according to the
hypothesis (e.g. expecting "gradual_tempo" replies to be better). Hash-based display ids are stable
for a given seed and do not leak sampling order.

**Consequences for analysis.** Keep the batch JSON (`save_batch`) away from raters. Blinding is to
condition, not to timepoint. Timepoint is shown by default because the `timepoint_appropriateness`
and `harm_risk` criteria are undefined without it (rubric `show_timepoint`). Classification
criteria are therefore potentially exposed to timepoint expectancy (OQ-ANNOT-2).

### D-ANNOT-3: Three interchangeable annotators driven by the same rubric spec

**Decision.**
* `export_html` writes a single self-contained HTML file: inline CSS and JS, no external URLs,
  fonts, or scripts. The rater enters an id and sees one item at a time. Form controls are generated
  from the rubric: radios for nominal, ordinal, and binary scales, with anchors as help text; a
  number input for interval scales; and a "(not rated)" choice for optional criteria. Progress is
  saved to `localStorage` per (batch, rater). Export produces a RatingRecord-shaped JSON array
  (`rater_id = "human:<name>"`) for copying or downloading. Only complete items are exported. Data
  is embedded in a `<script type="application/json">` block with `<`, `>`, and `&` escaped, and is
  rendered with `textContent`, so model replies cannot break out into the page.
* `export_csv` writes one row per item and one column per criterion, plus a `<stem>.codebook.txt`
  listing allowed values and anchors. `import_csv` coerces cells (exact value, numeric string, or
  label) through `rubric_spec.coerce_value`, validates them with `validate_values`, skips fully blank
  rows, and raises one error that lists every bad cell with its row number.
* `cli_annotate` is a terminal annotator. Options are entered by number 1..K (or by value or
  label). If any level value is itself numeric (for example a 0–3 or 1–5 scale), the number
  shortcut is turned off and typed numbers are read as level values, because on a 0-based scale
  "1" would otherwise silently record 0. `q` saves and quits, and EOF also stops it. It appends one JSON line per completed item,
  using display ids, and resumes by skipping items already in `out_path`. IO is injectable.

**Rationale.** Raters differ in which tool they prefer. Having one spec drive every instrument
guarantees that humans and the judge rate the same thing (rater–instrument parity, D-RUBRIC-7). The
HTML tool needs no server, so it can be emailed to raters. Strict validation at import stops typos
from entering the IRR analysis as fake disagreement.

**Consequences for analysis.** The instrument used (HTML/CSV/CLI) is not recorded in
`RatingRecord`, because the contract has no such field. If tool effects are a concern, record it in
the batch log (OQ-ANNOT-3). Because only complete items are exported from HTML, partial items
appear as missing data, which α handles.

### D-ANNOT-4: Rubric authoring = JSON with `_comment` keys

**Decision.** `rubric_template(path)` writes an example rubric with one criterion of each scale
type, documented with `_comment` keys. `load_commented_rubric` strips keys starting with `_` and
validates through `rubric_spec.rubric_from_dict`.

**Alternatives considered.** JSONC or `//` comments (not valid JSON). YAML (needs PyYAML, which is
optional in `rubric_spec`).

**Rationale.** `RubricSpec` is `Strict`, so it forbids unknown keys. Comment keys therefore have to
be stripped before validation. Plain JSON keeps the template editable in any editor.

**Consequences.** `rubric_spec.load_rubric` rejects files that still contain `_comment` keys
(contract request 1).

---

## Calibration (`eval/calibration.py`)

### D-CALIB-1: "Training the judge" = in-context calibration selected by IRR, not weight updates

**Decision.** "Using IRR to align [the judge's] reward model with the human evaluators" is
implemented as a **model-selection problem over a small, pre-registered space**: prompt variants
(`zero_shot`, `few_shot`, `anchored_cot`) × one exemplar set. The objective is Krippendorff's α
between the judge and the humans. No gradient updates are made.

**Alternatives considered.** (a) Fine-tuning or LoRA of the judge on human labels. (b) Automatic
prompt optimisation (DSPy MIPRO/BootstrapFewShot, OPRO) with α as the metric. (c) Post-hoc label
recalibration, e.g. confusion-matrix correction (Dawid & Skene 1979) or threshold tuning on
ordinal outputs.

**Rationale.** About 3 raters × a few dozen items is far too little data for (a), and the judge is
an API model (Gemini 3.8 Flash). (b) with an open-ended search over a few dozen items overfits the
calibration split. Keeping the search space tiny and declared in advance keeps the holdout estimate
honest. Choosing exemplars by human agreement follows the standard practice of using
gold-standard examples in LLM-as-a-judge prompts.

**Deviation from source.** The PDF's wording ("train", "reward model") suggests learning. We state
explicitly that our "training" is in-context calibration plus selection, and that the "reward
model" is the rubric-prompted judge. (a) and (b) are left as OQ-CALIB-1 and OQ-CALIB-2.

**Consequences for analysis.** The Methods section should say: "The judge was calibrated by
selecting among K prompt variants on a calibration split by mean Krippendorff's α with human raters,
and validated on a disjoint holdout." The degrees of freedom are |variants| = 3, one exemplar set.

### D-CALIB-2: Calibration/holdout split stratified by timepoint; holdout rated once, after selection

**Decision.** Human-rated items that have a matching `InjectionRecord` are split per timepoint
stratum using `random.Random(f"{seed}:split:{stratum}")`. Calibration receives
round(fraction · n_s) items per stratum (default fraction 0.5), clamped so that every stratum with at
least 2 items contributes to both splits. Variant selection uses only the calibration split. A
**fresh** judge of the chosen variant then rates the holdout exactly once. `CalibrationResult`
stores both id lists, and the tests assert that no judge sees a holdout item before selection ends.

**Alternatives considered.** k-fold cross-validation over all human items. Nested CV.

**Rationale.** A single split is simple to report and impossible to peek at by construction. With
3 variants the optimism of the selection is small but not zero, so the holdout α, not the
calibration α, is the validity estimate. Stratifying keeps every timepoint in both splits (D-ANNOT-1).

**Consequences for analysis.** The holdout n is about half the human sample, so its CI is wide.
With n_human ≈ 40, the holdout has about 20 items. Report the holdout α with a bootstrap CI using
`irr_report` on the holdout ratings. CV (OQ-CALIB-3) would use the data more efficiently, but would
mean choosing a variant once per fold.

### D-CALIB-3: Human consensus rules

**Decision.** `human_consensus`: for nominal and binary criteria, the strict plurality value, with
a tie for the top value → `None`. For ordinal criteria, the median level by rank in the rubric
order; with an even count and two different middle levels → `None` (a tie). For interval criteria,
the numeric median. Only human ratings are used.

**Alternatives considered.** Breaking ties at random, toward the lower level, or by a designated
senior rater. Using Dawid–Skene latent-class estimates.

**Rationale.** A tie means the humans did not agree, so there is no gold label to hand the judge.
Leaving it missing is the conservative option, and α handles missing values. Random tie-breaking
would add noise that has nothing to do with the judge.

**Consequences for analysis.** Consensus is used for (i) the `alpha_vs_consensus` diagnostic and
(ii) the exemplar labels given to few-shot variants. Selection itself uses α(humans + judge), not
consensus, so it does not depend on the tie rule.

### D-CALIB-4: Few-shot exemplars come from the calibration split only, prefer full agreement, span timepoints, and are excluded from scoring

**Decision.** `select_exemplars` ranks calibration items by the share of criteria on which every
human (at least 2) agreed, breaking ties with a seeded shuffle. It then takes items round-robin
across timepoint strata until it has `k_shot` (default 4, capped so that at least 2 calibration
items remain). Exemplars are passed to `make_judge(variant, exemplars)` as
`(InjectionRecord, RatingRecord)` pairs, where the rating is the human consensus
(`rater_id="human:consensus"`) and the format matches `behaviour_judge.RubricJudge(few_shot=...)`.
They go only to variants whose name contains `few_shot`.
**Every variant is compared on the same set, `eval_ids` = calibration minus exemplars.** If an
exemplar lacks full agreement, a note records it.

**Alternatives considered.** Choosing exemplars from all human items. Scoring few-shot variants on
their own exemplars. Choosing exemplars to maximise α (a second search loop).

**Rationale.** Holdout exemplars would leak labels into the validity estimate. Scoring a judge on
its own exemplars inflates its α. Items with unanimous labels are the clearest demonstrations of the
rubric. Covering several timepoints shows the judge what a reply looks like across the whole
behaviour range rather than at one end.

**Consequences for analysis.** Report `few_shot_ids` and their timepoints. The exemplar set is
fixed before variants are compared, so the search space stays at 3 (D-CALIB-1).

### D-CALIB-5: Selection objective and acceptance rule

**Decision.** Choose the variant with the highest **mean over criteria of α(humans + judge)** on
`eval_ids`. Undefined criteria are ignored, and ties go to the variant listed first in `variants`,
so list simpler variants first. A variant whose mean is undefined ranks last. The result is
`accepted` iff the holdout mean α(humans + judge) ≥ `threshold` (default .667). Also reported for
each variant: the mean α vs consensus, the mean α vs each individual human, and the human-only
baseline α on the same items.

**Alternatives considered.** (a) Maximising α(judge, consensus). (b) Maximising the mean pairwise
κ(judge, human). (c) A per-criterion choice of variant. (d) Requiring the judge's α to be at least
the human-human α.

**Rationale.** α(humans + judge) answers the question that matters: does adding the judge as a
fourth rater keep the panel reliable? It uses every human rating instead of a consensus with ties
removed, and it shares a scale with the human baseline. A per-criterion choice (c) multiplies the
degrees of freedom. (d) is reported through the `delta_vs_humans` and `human_baseline` fields
rather than enforced, because with three raters the human baseline is itself noisy.

**Deviation.** The PDF does not specify a criterion. Taking the mean across criteria weights all
criteria equally, including rare INTIMA sub-codes (OQ-IRR-1).

**Consequences for analysis.** If the judge is not accepted, `rate_remaining` still works, but the
analysis must not use judge labels for criteria below threshold. Use per-criterion values from
`holdout_metrics["per_criterion"]` to restrict the analysis to criteria that pass.

### D-CALIB-6: Every judge rating is returned with its split; rater-id collisions are disambiguated

**Decision.** `calibrate_judge` returns every rating it produced: all variants on calibration
(`split="calibration"`) and the chosen variant on holdout (`split="holdout"`). `rate_remaining`
tags ratings `split="unlabelled"`, skips ids that are already rated, and preserves input order. If
two variants return the same `rater_id`, the later one gets a `:<variant>` suffix.

**Rationale.** Full provenance: every rating that influenced selection can be audited, and none is
silently overwritten (cf. `ratings_matrix`'s last-wins rule).

**Consequences for analysis.** Analyses of the "remaining" injections should filter
`split == "unlabelled"` plus the chosen rater id.

---

## Open questions

### OQ-IRR-1: Prevalence (kappa) paradox on rare INTIMA sub-codes

*Why it matters.* On the default rubric, most of the 10 ordinal sub-codes are "absent" on most
items (e.g. `reinforcing_isolation`). α and κ then become unstable or undefined even when raw
agreement is about 95%. That depresses the mean α used for selection and acceptance.
*How to resolve.* Report percent agreement and α side by side. Consider Gwet's AC2 as a
sensitivity check. Decide before data collection whether selection and acceptance use (a) all
criteria, (b) criteria with human α defined and above some prevalence threshold, or (c) a weighted
mean. Collapsing absent/low vs medium/high to binary is another sensitivity option.

### OQ-IRR-2: How many human items are needed?

*Why it matters.* Krippendorff's α CIs at n ≈ 20 holdout items are roughly ±.15. Acceptance at .667
may hinge on noise.
*How to resolve.* Use a pilot to estimate α, then size n so the lower CI bound clears .667. This
probably needs 60–100 items per batch. The sampler and bootstrap support any n.

### OQ-ANNOT-1: Proportional or equal allocation across timepoints?

*Why it matters.* T3 cells are small in some scenarios. Proportional allocation may put only 1–2
T3 items in the holdout, which weakens the claim that the judge is valid at T3.
*How to resolve.* If per-timepoint validity is reported, oversample small strata (equal
allocation) and weight pooled estimates. The sampler would need an `allocation` argument
(`proportional` | `equal`).

### OQ-ANNOT-2: Showing the timepoint invites expectancy effects

*Why it matters.* Raters who see "T3" may expect boundary-maintaining replies and code
classification accordingly. The judge always sees the timepoint (rubric note).
*How to resolve.* Use two-pass annotation: classification and sub-codes with the timepoint hidden,
then timepoint-dependent criteria with it shown. This needs two rubrics, or a per-criterion
`show_timepoint` (contract request 2).

### OQ-ANNOT-3: Rater training and drift

*Why it matters.* Three raters with no shared calibration session typically give lower α.
Ratings made weeks apart can drift.
*How to resolve.* Hold a short training round on 5 or more non-study items (e.g. released INTIMA
S11 pairs), discuss them, then rate. Record the instrument and the timestamp per rating (currently
not in `RatingRecord`).

### OQ-CALIB-1: Fine-tuning a judge on human labels

*Why it matters.* The PDF's "train ... reward model" suggests actual learning.
*How to resolve.* Fine-tuning is only sensible after more human data exists, e.g. 500 or more
items, or after bootstrapping from INTIMA S11's released judge labels. Evaluate on the same holdout
protocol.

### OQ-CALIB-2: Automatic prompt optimisation (e.g. DSPy) with α as the metric

*Why it matters.* It could find better prompts than three hand-written variants.
*How to resolve.* Only with a larger calibration split or nested CV. Report the search budget.
`calibrate_judge` accepts any list of variant names, so optimiser-produced prompts can be plugged in
as variants.

### OQ-CALIB-3: Single split vs cross-validation

*Why it matters.* A single 50/50 split wastes about half the human data for selection.
*How to resolve.* Run k-fold selection with the holdout still kept outside: nested CV. Compare how
stable the chosen variant is across seeds by re-running `calibrate_judge` with several seeds and
reporting the variant chosen each time.

### OQ-CALIB-4: Judge non-determinism

*Why it matters.* If the judge samples at temperature > 0, its α changes between runs, and the
holdout estimate understates that variance.
*How to resolve.* Run the judge at temperature ≈ 0 with a fixed seed where the API supports one.
Rate the holdout r ≥ 2 times and report the judge's intra-rater α.

---

## Contract requests (for the schema/rubric owners)

1. `rubric_spec.load_rubric`: strip `_`-prefixed keys before `RubricSpec.model_validate`, so the
   template from `rubric_template` loads directly. The workaround is `annotation.load_commented_rubric`.
2. An optional per-criterion `show_timepoint` on `RubricCriterion` (OQ-ANNOT-2).
3. Optional `RatingRecord.meta` (instrument, timestamp, prompt variant), so the provenance in
   D-ANNOT-3 and D-CALIB-6 does not have to live in `rater_id`.

---

## Amendments: methodology review (2026-11-03)

### D-CALIB-5 A1: Acceptance is based on the judge's own per-criterion agreement, not on pooled alpha

- **Decision.** `calibrate_judge(..., key_criteria=("classification",
  "timepoint_appropriateness", "harm_risk"), substitution_margin=0.05, n_boot=500,
  require_ci_lower=False)`. Key criteria are those of the defaults that exist in the rubric. If
  none do, all criteria are key. Unknown names are reported in `notes`. On the holdout, each key
  criterion must pass two checks:
  (a) `alpha_vs_consensus` (judge vs human consensus) is defined and >= `threshold`;
  (b) the substitution test: `alpha_swap`, the mean over humans h of alpha(humans minus h, plus
  the judge), is >= `alpha_humans - margin`. This check is skipped when there are fewer than two
  humans or when human-only alpha is undefined.
  A seeded item-bootstrap 2.5% lower bound of (a), `ci_lower`, is always reported. It gates
  acceptance only when `require_ci_lower=True`. Per-criterion pass/fail and reasons are stored in
  `holdout_metrics["acceptance"]`. Each failing criterion and each criterion with undefined alpha
  is named in `notes`. `mean_alpha_humans_llm` is kept only as the selection objective and a
  diagnostic.
- **Why.** With three humans and one judge, the humans make up three of the four raters, so good
  human agreement props up the pooled alpha. Averaging over 14 criteria also lets easy criteria
  hide a failure on `classification`. In a seeded simulation (30 items, 15 in the holdout, 3
  classes, humans 95% accurate) the old rule accepted a 70%-accurate judge in 12 of 20 replicates.
  The new rule accepted a 70%-accurate judge in 1 of 40, a 60%-accurate judge in 0 of 40, and a
  95%-accurate judge in 28 of 40.
- **Alternatives.** Require the CI lower bound by default. With about 15 holdout items this would
  reject almost every judge, so it is opt-in and the bound is reported. See also OQ-IRR-2.
- **Consequences.** Earlier `accepted` values are not comparable. The pipeline does not gate on
  `accepted`. Test: `test_weak_judge_is_rejected_even_when_pooled_alpha_passes`.

### D-CALIB-2 A2: The calibration/holdout split is grouped by INTIMA probe

- **Decision.** `calibrate_judge(group_by="intima_id")`, the default, uses `split_items_grouped`.
  All items sharing an `intima_id` go to the same split. Groups are stratified by the sorted set of
  timepoints they cover (e.g. `"T0+T3"`), shuffled with a seed, and allocated by cumulative rounding
  across strata. Any stratum with two or more groups feeds both splits. After exemplar selection,
  the code asserts that exemplar and calibration groups are disjoint from holdout groups.
  `group_by=None` restores the old timepoint-stratified `split_items`. The grouping is recorded in
  `notes`.
- **Why.** Probe seeds depend on `(probe_seed, freeze_id)` only, so every soul, scenario and arm
  shares the same probes at each freeze point (60 distinct prompts under `per_freeze`). An
  ungrouped split could put the same prompt, often with near-identical T3 replies from ABRUPT and
  GRADUAL, in both a few-shot exemplar and a holdout item.
- **Consequences.** The holdout estimates generalisation to *new prompts*. Strata are timepoint
  signatures, not timepoints, so a timepoint can be missing from one split when it appears only in
  a single group. When any timepoint has fewer than two holdout items, `notes` flags the holdout
  as "under-powered" there (this is not a hard failure). **Pseudo-replication:** about 4752
  injections rest on at most 60 prompts, so downstream class-outcome CIs should be clustered by
  `intima_id`, or at least report the effective number of items next to injection counts. The
  turn-level sentiment ITS is not affected. Those summaries are in `pipeline.build_report` (cross-area
  request filed). Optionally, probe seeding by `(probe_seed, scenario_id, freeze_id)` would
  increase item coverage while keeping arms paired within a scenario (D-SESSION-6). That change
  belongs to the session area. Test: `test_grouped_split_keeps_shared_intima_prompts_together`.

### D-ANNOT-1 A1: `min_per_stratum` floor and inclusion weights (closes OQ-ANNOT-1)

- **Decision.** `sample_for_annotation(..., min_per_stratum=0)` and `_largest_remainder(...,
  min_per_stratum=0)`. Each stratum first gets `min(min_per_stratum, size)`. The rest is
  allocated by largest remainder in proportion to each stratum's leftover size. A `ValueError` is
  raised if `min_per_stratum * n_strata > n`. `batch.sampling` now records `min_per_stratum` and
  `inclusion_weights` (N_h / n_h per stratum, `None` if the stratum got no items), so pooled
  human-rated distributions can be inverse-probability weighted. `batch_id` includes
  `min_per_stratum` only when it is non-zero, so existing batch ids are unchanged.
- **Why.** In the offline full run (T0=2457, T1=306, T2=612, T3=1377), proportional allocation
  of n=30 gives T1 2 items. That leaves about 1 T1 calibration item, which is probably used as an
  exemplar, and so no T1 item for variant selection. T1 is the hardest timepoint to judge.
- **Consequences.** The pipeline should pass `min_per_stratum=max(4, annotation_n // (2 *
  n_timepoints))`, which is 4 for n=30 (cross-area request filed; `pipeline.py` is not in this
  area). Proportional allocation remains the default. OQ-ANNOT-1 is resolved as "proportional
  allocation with a floor, plus recorded weights". Stratifying by condition x timepoint remains
  opt-in via `stratify_by`. Test: `test_min_per_stratum_protects_small_timepoints_and_records_weights`.

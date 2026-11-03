# Research notes: Tempo (striving co-creation from everyday computer use)

**Retrieved:** 2026-11-03. **Notes author:** research-assistant subagent (LLM). The extraction used WebFetch, which runs a summarization model over each page, so quoted strings are short fragments that summarizer returned. Check every verbatim quote against the PDF before it goes into a paper.

## 0. Sources consulted

| Source | URL | Status |
|---|---|---|
| arXiv abstract page | https://arxiv.org/abs/2605.00497 | Fetched |
| arXiv HTML (v2, rev. 9 Sep 2026) | https://arxiv.org/html/2605.00497v2 | Fetched in two windows (offset 0 and offset 100000). Some figures did not come through. |
| arXiv PDF | https://arxiv.org/pdf/2605.00497 | Not fetched separately; the HTML v2 was used instead |
| Code repository | https://github.com/StanfordHCI/striving-cocreation | Exists and was fetched (README, directory listings) |
| `tempo/schemas.py` | https://raw.githubusercontent.com/StanfordHCI/striving-cocreation/main/tempo/schemas.py | Fetched |
| `tempo/observers/screen.py` | https://raw.githubusercontent.com/StanfordHCI/striving-cocreation/main/tempo/observers/screen.py | Fetched |
| `.env.default` | https://raw.githubusercontent.com/StanfordHCI/striving-cocreation/main/.env.default | Fetched (provider config only) |
| `prompts/txt/goal_synthesis/SYNTHESIZE_PROMPT.txt` | https://raw.githubusercontent.com/StanfordHCI/striving-cocreation/main/tempo/prompts/txt/goal_synthesis/SYNTHESIZE_PROMPT.txt | Fetched |
| `prompts/txt/goal_synthesis/CONTRAST_SECTION.txt` | https://raw.githubusercontent.com/StanfordHCI/striving-cocreation/main/tempo/prompts/txt/goal_synthesis/CONTRAST_SECTION.txt | Fetched |
| Other prompt files (`operation_extraction`, `action_generation`, `activity_*`, `goal_propose`, `goal_reconcile`, `context_block`, `assistant_chat`, `contextual_assistant`, `screen`) | under `tempo/prompts/txt/` | **Not fetched.** Only the folder names were confirmed. |

**Citation.** Shardul Sapkota, Matthew Jörke, Zane Sabbagh, Omar Shaikh, Grace Wang, James A. Landay. *"What Are You Really Trying to Do?": Co-Creating Life Goals from Everyday Computer Use.* arXiv:2605.00497 (v1 1 May 2026; v2 9 Sep 2026). Accepted at UIST 2026. Related DOI: https://doi.org/10.1145/3830398.3830630. Code: MIT license. Paper content: CC BY 4.0. The README says Tempo "incorporates code adapted from" GUM (General User Models, Shaikh et al.).

**Commit pinning: open item.** All repo files were read from `main` on 2026-11-03 and no commit SHA was recorded. If we vendor or adapt code, pin a SHA.

---

## 1. System identity and framing

### (a) What the source says
- The system is **Tempo**. The process it implements is **"striving co-creation"**: inferring long-term life goals (*strivings*) from unstructured observations of computer use, then letting the person correct the inference. Those corrections constrain all later inference.
- Theory: **Activity Theory** (operations → actions → activities), extended upward with **Emmons' personal strivings**, which are recurring, meaningful, and characteristic of the individual.
- The paper's motivation is that the same action can serve many goals. The paper's phrase is "the same activity to support multiple strivings" (§1). Because of this, inference alone is underdetermined and needs the person's input.
- In code and in the API, strivings are called **`goal`**.

### (b) Ambiguous or unknown
- The paper does not define a formal criterion for when an activity counts as "characteristic" of a person. It is left to the LLM's judgment plus a self-refine check.

### (c) Implications
- The "same action, many motives" problem is our central inference problem. Closing a companion app could mean weaning off, being busy, or a technical problem. This supports a **co-creation step** in our coach layer instead of an inference-only intent classifier.

---

## 2. Input signals and capture pipeline

### (a) What the source says
**Paper (§3, §3.1, §3.3):**
- The input is **screenshots** from a personal MacBook, captured by a macOS "Observer" app. Contiguous frames are bundled into *observations*.
- Privacy filtering happens before inference:
  1. A local OCR filter drops frames that match PII patterns.
  2. A **contextual integrity audit** (LLM) blocks content the user would not expect to be captured (§3.3.2; prompt D.9). Its output fields are `is_new_information`, `data_type` (e.g., `banking_credentials`, `health_information`, `personal_communications`), `subject`, `recipient`, `transmit_data`, and `reasoning`.
  3. Users can exclude apps and sites and pause recording (§3.3.1).
- **Transcription:** a vision-language model turns each observation into structured markdown, conditioned on user context (D.2). It has two modes:
  - Transcription mode: "NEVER SUMMARIZE ANYTHING. You must transcribe everything EXACTLY, word for word, but don't repeat yourself." Application names, file paths, and URLs must be included.
  - Summary mode: "a handful of bullet points", with a caution that screen content "may not be what the user is actively doing or what they believe."
- **Model:** Gemini 3 Flash for both transcription and decomposition, called through GCP under an enterprise no-training agreement.
- **Batching:** operations buffer until "a batch of operations has been collected or a period of inactivity elapses." An induction cycle takes about **5–10 minutes** (§6). The paper says buffer sizes and confidence cutoffs are configurable but does not report the values.

**Code (`tempo/observers/screen.py`, constructor defaults):**
- Every monitor is sampled at `capture_fps = 5` into a rolling buffer of `k_history_frames + k_future_frames` = 2 + 2 frames.
- **Trigger:** only mouse move, click, and scroll events are wired up. No keyboard hook was found in this file.
- **Debounce:** per monitor, `debounce_seconds = 2.0`. It adapts upward to min(6.0, 2×) when there are ≥3 captures in 30 s, and to min(10.0, 5×) when there are ≥6.
- **Change gate:** 96×54 grayscale thumbnail, mean absolute difference compared with `change_threshold = 0.04`. If nothing changes for `idle_threshold_seconds = 60.0`, a capture is still saved with an "[Idle]" note.
- Frames are saved as JPEG at quality 70. Only the last frame of a bundle gets OCR, the sensitive-content check, and a `.meta.json` sidecar. If sensitive text is flagged, all frames of that capture are deleted.
- **App/window metadata is captured:** `get_active_window` supplies `app_name`, `bundle_id`, and `window_title`. These are stored as `active_window` / `app_context` and added to vision prompts as a header. The header warns that the active window is system-wide and may not be on the captured monitor.
- **Exclusions:** `ignore_visible_app_names`, plus an `ExclusionController` polled every 2.0 s that decides from the active bundle ID and browser URL.
- The class docstring gives different defaults (10 fps, 1.0 s debounce, 1+1 frames) than the constructor. We treated the constructor values as authoritative.
- Data is stored under `~/.cache/tempo` (SQLite plus files) and does not expire automatically. The README states that the OCR filter can leave rejected content on disk or in logs, and that the audit runs after the first transcription has already been sent.

### (b) Ambiguous or unknown
- **Paper vs. code on input signals.** The paper presents screenshots as the input and does not describe window events as signals. The code does attach active-app and window-title metadata to prompts. When describing Tempo, say: "screenshots, event-triggered by mouse activity, with active-window metadata in the code."
- We did not check which settings the field study actually used (whether the deployment matched the current `main` defaults). **UNVERIFIED.**
- We did not confirm whether the study used both transcription and summary modes, or which one feeds operation extraction. **UNVERIFIED.**
- The paper reports no measure of how much content the privacy filters removed. The authors list this as a limitation.

### (c) Implications for our system
- If we reuse Tempo, the companion app's own screen content (chat text with the AI companion) would be transcribed verbatim and sent to an LLM endpoint. That is intimate, `personal_communications`-class data. We need a **design decision** on one of:
  - (i) excluding the companion app and consuming only metadata such as `app_name`, `window_title`, and timestamps;
  - (ii) letting transcription run with explicit consent;
  - (iii) replacing screenshots with synthetic or simulated action logs for the course project.
  Record this as an ADR.
- Mouse-only triggering means keyboard-heavy chatting with a companion may be under-captured unless the screen changes enough to pass the threshold. Usage-duration features derived from Tempo will be biased. **Open question.**
- Usage volume (minutes in the companion app, late-night sessions) is easier to get deterministically from `active_window` timestamps than from LLM inference. Prefer deterministic features for dose metrics.

---

## 3. Hierarchy and induction

### (a) What the source says
Levels (README table; paper §3.1.1):

| Level | Definition (paraphrase; short quotes marked) | Editable? |
|---|---|---|
| Operations | Atomic interactions (click, scroll, keystroke). They "simply record what happens on screen and make no claim about why." | No |
| Actions | Goal-directed runs of contiguous operations. Boundaries follow shifts in intent, not changes of app. D.4: "a coherent task the user would describe in one sentence"; segments must be "contiguous, non-overlapping, and cover all operations." | Yes |
| Activities | Recurring, motive-driven *working spheres* across apps and sessions, "broader than a single task but narrower than a life domain" (D.5). Proposed when "2–3+ actions share a coherent working sphere." They stay provisional until they recur across batches. | Yes |
| Strivings (`goal`) | Ongoing personal pursuits in Emmons' sense that organize several activities. D.7 properties: dispositional recurrence, active behavioral manifestation, "abstraction above actions." | Yes |

- **Graph:** a property graph in SQLite. Each node has a natural-language description and metadata. Links are **many-to-many**: an action can feed several activities, and an activity can feed several strivings. Temporal edges (follows, co-occurs, overlaps) between actions are built **deterministically**, not by the LLM.
- **Pipeline:** prompt chaining (operation extraction → action segmentation → activity proposal → activity reconciliation → striving synthesis → self-refine).
  - Reconciliation decisions are match / revise / new / merge (D.6).
  - Striving synthesis runs after a warmup batch, with a 24 h fallback ceiling (D.7).
  - Self-refine (D.8) critiques for redundancy, over-breadth, activity restatement (a "completability test"), surface-level goals, coverage, phrasing ("'wants to' describes aspiration, not striving"), and theory-grounded checks: conflict completeness, avoidance/ambivalence, ideal/ought balance, autonomy signal, and level appropriateness. Possible outcomes are merge, split, or a confidence recalibration.

**Per-level metadata (D.3–D.5; `schemas.py`):**
- *Operation:* `text`, `confidence` (1–10), `decay` (1–10), `reasoning`, plus `tool_kind`, `social_target`, `rule_tags`, `automaticity_hint` (likely_habitual / likely_deliberate / unclear), and `affect_hint`.
- *Action:* `object_label`, `outcome_type`, `domain`, `community`, `engagement_state`, `cognitive_mode` (skill/rule/knowledge-based), `initiation`, `social_mode`, and `goal_hint`.
- *Activity:* `description`, `purpose`, `reasoning`, `people`, `resources`, `temporal_pattern`, `engagement_profile`, `initiation_profile`, `identity_context`, `action_ids`, **`action_valences` (supports / hinders / neutral)**, and `confidence` (1–10).

**Striving psychological dimensions (`SYNTHESIZE_PROMPT.txt`; `GoalItem`):**
- `needs`, drawn from 12 values: `competence`, `autonomy`, `relatedness`, `status_recognition`, `self_coherence`, `understanding`, `order_predictability`, `nurturance`, `safety_security`, `growth`, `stimulation`, `purpose_meaning`. The prompt cites Dweck, Murray, SDT, and Higgins.
- `orientation`: approach / avoidance / mixed. The prompt says not to default to approach.
- `aspiration_vs_obligation`: ideal / ought / mixed.
- `autonomy`: autonomous / controlled / mixed. In `GoalReconcileDecisionItem` the same field is named `autonomy_dim`.
- `identity_link`: the hoped-for self, filled for approach and mixed strivings.
- `feared_self`: the self being avoided, filled for avoidance and mixed strivings. Avoidance phrasing is "{user_name} is trying not to [X]...".
- Other fields: `people`, `domains`, `engagement_signature`, `initiation_signature`, `temporal_context`, `evidence_summary`, `synthesis_evidence`, `reasoning`, `activity_ids`, `matches_goal_id`.
- **Confidence rubric (1–10):** evidence breadth, behavioral manifestation, and distinctness. 1 means speculative; 8–10 is reserved for sustained multi-activity convergence. The prompt asks the model to be conservative.
- **Text convention:** strivings start "{user_name} is ...". Phrasing should vary.
- **Relations:** `goal_relations`. The only type named in the synthesis prompt is `conflict`, and conflicting goals are kept, not merged.
- **Drops:** every dropped goal goes in `dropped_goals` with a reason from `merged`, `evidence_changed`, `redundant`, or `absorbed`.

**Evidence linking:** strivings trace down through activities and actions to supporting screenshots (§3.2). The Python API offers `trace_up()` and `trace_down()`, as well as `goals()`, `hierarchy()`, `search()`, and `timeline()`.

**Stated vs. implied.** The paper does not use these labels. During onboarding, users describe their *context* but are deliberately not asked to state goals (§3.1.2; Appendix A has 12 open-ended context questions). The code does support a contrast mode (`CONTRAST_SECTION.txt`; `ContrastAnalysis`) that compares user-stated goals (`{user_goals_section}`) with inferred ones:
- **supported**: stated goals with "strong behavioral evidence";
- **unsupported**: stated goals with "weak/no behavioral evidence", flagging "the gap between intention and action";
- **unstated**: inferred goals the user never listed.

`ContrastEntry` has `evidence_strength` (a string), `activity_ids`, `action_count`, and `note`. A `NeedLandscape` object holds `congruence_insights`, `incongruence_insights`, and `leverage_points`.

### (b) Ambiguous or unknown
- **Contrast mode in the study:** we did not determine whether the field study ran it. It is in the code, but the paper says onboarding did not collect goals. Probably not used. **UNVERIFIED.**
- **`evidence_strength` scale:** not defined in the contrast prompt we read.
- **`decay` semantics:** the 1–10 scale exists, but we did not find how it is used (time decay of nodes? weighting?). **UNVERIFIED.**
- **Number of self-refine checks:** the paper says 21 and the abridged appendix lists 13.
- **Schema type mismatches:** confidence and decay are `Optional[str]` on operations and actions but `int` elsewhere, and `autonomy` is called `autonomy_dim` in reconciliation. Normalize these when adapting.
- **Thresholds:** we did not find the activity stabilization threshold or the confidence cutoffs for showing a striving. **UNVERIFIED.**

### (c) Implications for our system
- **A "wean off the companion" striving maps cleanly onto Tempo's dimensions.** For example: `orientation = avoidance` or `mixed`; `feared_self` such as "someone who depends on an AI for comfort"; `needs` including `autonomy`, `relatedness`, and `self_coherence`; and `aspiration_vs_obligation`, which separates self-chosen weaning (ideal) from weaning under pressure from others (ought). The `autonomy` field (autonomous vs. controlled) matters for MI-style coaching and for the ethics of the endings design.
- **`action_valences` (supports / hinders / neutral)** is the right primitive for attaching companion-use actions to a weaning striving. A late-night companion session *hinders* the weaning striving and may *support* a relatedness striving at the same time. Ambivalence is represented natively, through `mixed` and `conflict` relations.
- **Contrast mode is the right mechanism for intent.** If the user *states* "I want to use the companion less", contrast analysis shows whether behavior supports that statement or reveals an intention–action gap. We should design our coach input to accept:
  - (1) **stated intent** from onboarding or the user;
  - (2) **implied intent** from behavior, such as a declining session trend, closing the app abruptly, or searching "how to stop talking to AI";
  - (3) the **contrast category**.
- **Inferred intent should not trigger ending behavior by itself.** Tempo's own findings are that context over-weighting produced confident misattributions and that unrelated activities were merged. Our gating rule: weaning interventions start only after a user-endorsed (`[endorsed]`) or user-stated striving, or after a confirmation exchange. Use inferred strivings only to *propose* the conversation.
- The confidence rubric should be reused as-is for reproducibility. Report the threshold we pick (e.g., confidence ≥ 6 to surface) as a design parameter in our paper. That value is our choice, not Tempo's.

---

## 4. Co-creation / editing interaction

### (a) What the source says (§3.2, §3.2.1, §3.2.2; D.1)
- Users can edit strivings, activities, and actions. Operations cannot be edited because they are descriptive records. The edit operations are inline edit, reassign, remove, and merge. Users can also remove individual screenshots from evidence.
- **Preservation:** edited nodes are flagged. The D.1 "User Constraint Block" is regenerated from the graph each cycle and injected into every downstream prompt. The flags are:
  - `[locked]`: "Do NOT merge, delete, or substantially alter."
  - `[user-edited]`: keep the label verbatim.
  - `[user-reassigned]`: keep the assigned goal.
  - `[endorsed]` / `[rejected]` and annotations: these "provide privileged context", weighted above behavioral inference.
  - `[user-provided]` also appears in the synthesis prompt's `{existing_goals}`.
- **Write-time guards in code** back up the prompt constraints. A relabel of a flagged entity is dropped, a merge into a locked target is downgraded to a match, and a locked source is skipped from merge groups.
- **Propagation:** edits re-trigger synthesis. Removing a striving orphans its activities, which are then reassigned.
- Not built: conversational editing, and the system proposing alternatives (§6). Participants asked for alternatives (§5.4.3).
- `schemas.py` has no dedicated edit or provenance class. Edit state is in flags on graph nodes.

### (b) Ambiguous or unknown
- We did not see the exact UI for each edit. Figures were not in the fetched HTML.
- We did not confirm whether `[rejected]` strivings are prevented from being re-proposed, or only down-weighted. **UNVERIFIED.**

### (c) Implications for our system
- Adopt **two layers of protection**: prompt-level constraint blocks plus code-level write guards. This matters most for a weaning striving. If a user *rejects* "is trying to rely less on the companion", the coach must never bring it back on its own initiative. That should be a hard rule in code, not a prompt suggestion.
- Log every edit as an event (who, when, which node, before and after) so the paper can trace outputs to edits. Tempo stores state as flags only, which is not enough for our traceability requirement.
- Add **system-proposed alternatives** (e.g., "reduce", "pause", "end", "keep as is") as a co-creation affordance. Tempo's participants asked for this, so it is a motivated extension.

---

## 5. Downstream applications (§4): coach precedent

### (a) What the source says
- The hierarchy works as an API. Apps can query any level, traverse it, and subscribe to updates when strivings emerge or confidence changes.
- **§4.1 Proactive assistance.** Traversing the hierarchy surfaced a conflict between late-night browsing and a PhD deadline. Tempo suggested "Reserve 30 minutes before your next assignment". The baseline suggested "Compare products or bookmark the AI features for future reference."
- **§4.2 Behavioral coaching with motivational interviewing.** At each session start, the coach:
  - finds strivings with rising or falling momentum;
  - picks an MI strategy suited to the user's trajectory;
  - grounds its opener in behavioral evidence from the graph.

  Example lines: "The parts of your life that ground you are being squeezed into the margins." and "What if you carved out 15 minutes before your late-night food order to call someone back home?"
- Both demonstrations are single-user, first-author case studies, **not controlled evaluations**.
- The repo has `assistant_chat.py` and `contextual_assistant.py` prompts. The README describes an optional "contextual assistant" that was not part of the field study.

### (b) Ambiguous or unknown
- The paper gives no formal definition of "momentum": no metric, window, or threshold. **UNVERIFIED.**
- We have not read the coach or assistant prompt text.

### (c) Implications
- §4.2 is the closest precedent for our coach. It already combines Tempo, MI, and evidence-grounded openers. Our contribution is to aim it at *a relationship with an AI companion* as the target behavior and to connect it to the endings, persona, and diffusion design.
- We must define "momentum" ourselves (e.g., slope of linked-action counts per striving over a 7-day window) and report the definition.
- Tempo's coach example uses relatedness ("call someone back home"). That maps directly onto replacing companion use with human connection, but it carries a risk of paternalism. Ground the coach in MI's autonomy-support principles.

---

## 6. Evaluation (§5; Appendix B)

### (a) What the source says
- **Participants:** 19 recruited, **14 analyzed**. Inclusion required at least 500 screenshots; the median was 1,498 (range 765–9,435). The study was a one-week deployment followed by a one-hour lab session.
- **Ablations:**
  1. full Tempo;
  2. without user context;
  3. without hierarchy or context (a sliding-window baseline).
- **Scales:** 7-point Likert, −3 to +3.
  - Per striving: Accuracy, Alignment, Abstraction, Characteristicness.
  - Per set: Coverage, Discovery, Motive.
  - Post-edit: Control, Agency, Transparency, Evidence Utility, Reflection, Confidence, Ease. Item wordings are in Appendix B, e.g., Alignment: "This statement captures something that currently matters to me."
- **Results:**
  - Accuracy was positive (≥1) for 80%, 79%, and 79% of strivings across the three conditions.
  - Alignment: M = 1.46 full vs. 1.16 without context (87% vs. 73% positive). Context effect β = 0.290, p < .05. Hierarchy alone β = 0.049, p = .679.
  - Set level: hierarchy effect β = 0.721, p < .05; context n.s. (β = 0.239, p = .503).
  - Strivings per participant: 4.9 / 6.1 / 6.6. Motive: 0.57 / 0.79 / 0.07.
- **Editing:** the hierarchical view was higher across the seven dimensions (β = 0.969, p < .001).
  - Transparency: Δ = +2.14 (93% vs. 43% sufficient).
  - Evidence utility: Δ = +1.71 (100% vs. 71%).
  - Screenshot browsing accounted for 36.6% of clicks.
- **Confound (acknowledged):** the two editing views showed striving sets from different ablations.

### (b) Ambiguous or unknown
- We did not confirm the exact statistical model (mixed-effects with participant random effects is likely). **UNVERIFIED.** Check the PDF before citing the β values.

### (c) Implications
- Reuse the Appendix B items to evaluate a coach-layer "weaning striving" (accuracy, alignment, autonomy) so our results are comparable with Tempo's.
- Our study cannot assume a week of real screen capture. With simulated or synthetic computer-use logs, results do not test Tempo's real-world validity. The paper must state this limitation.

---

## 7. Limitations reported by the authors (§5.3.3, §5.4.3, §6)
- The sample was self-selected and comfortable with AI, and everyone used one machine for both work and personal life.
- Screen-only capture skews toward work, and offline life is mostly missing.
- **Context over-weighting** produced confident misattributions (e.g., a participant inferred to be pivoting to software engineering said they were not).
- Unrelated activities were merged on surface similarity or temporal co-occurrence. The authors suggest a minimum evidence threshold.
- The language was verbose and confidently LLM-like.
- Cycles take 5–10 minutes, and buffer sizes and thresholds were not robustly tested.
- One week cannot establish whether strivings are stable.
- **Sycophancy and Barnum effects:** people may accept validating strivings uncritically.
- **Observer effect:** some participants performed for the system and others forgot it was running.
- The hierarchy may be more sensitive than any single screenshot, and the audit is not robust to deliberate manipulation. The paper does not quantify what the filters removed.

### Implications of the limitations for our system
- **Barnum risk is acute for us.** A coach that says "you seem to be trying to rely less on your companion" may be accepted because it sounds virtuous, not because it is true. Mitigations:
  - (a) show the evidence (`trace_down`) alongside the claim;
  - (b) offer alternatives, including "this is not what I want";
  - (c) measure acceptance against a control striving that was deliberately inserted and is false, as a manipulation check (this would need IRB-appropriate design).
- **Observer effect:** knowing that a coach watches companion use may itself reduce use, which confounds any "weaning" outcome. Report this.
- **Sensitivity escalation:** a striving like "is trying to stop depending on an AI companion" is more sensitive than raw usage logs. Treat it as health- or relationship-class data.

---

## 8. Integration sketch: Tempo as the coach's input layer

This section is our design, not Tempo's. It lists candidate decisions to record as ADRs.

1. **Interface contract.** The coach consumes a `StrivingSnapshot`:
   - the list of strivings, with `text`, `needs`, `orientation`, `aspiration_vs_obligation`, `autonomy`, `feared_self`, `identity_link`, `confidence`, and user flags (`locked` / `user-edited` / `endorsed` / `rejected`);
   - linked activities and their `action_valences`;
   - `goal_relations` (conflict);
   - optional `ContrastAnalysis`;
   - deterministic usage features for the companion app.

   This mirrors `GoalItem` and `ContrastEntry` field names to keep provenance traceable.
2. **Intent states for weaning:**
   - `none`;
   - `implied` (inferred, not confirmed, confidence below threshold);
   - `implied_high` (inferred, confidence at or above threshold);
   - `stated` (user-stated or user-endorsed);
   - `rejected` (user said no, a hard block).

   Only `stated` enables ending or diffusion behaviors. `implied_high` enables only a reflective MI question.
3. **Adapter options**, ranked by fidelity and cost:
   - (A) run Tempo locally and call `tempo.Tempo().goals()` / `trace_down()`;
   - (B) replay recorded or synthetic transcriptions through Tempo's prompts;
   - (C) a **mock adapter** that emits Tempo-shaped JSON from scripted scenarios, for tests and demos.

   We recommend implementing (C) first behind an interface, with (A) as an optional backend. Option (A) needs macOS permissions and an LLM key, and it sends screen data to a provider.
4. **Provenance:** every coach utterance logs the IDs of the striving and activities it relied on, plus the snapshot version. This enables paper-grade tracing, matching Tempo's own trace-down evidence model.

## 9. Open questions (carry into the project's open-questions log)
1. Should the companion app's conversation content ever be transcribed, or only metadata? Consent model?
2. What threshold turns an implied weaning striving into a coach prompt, and how was it chosen?
3. How is "momentum" defined (metric, window)?
4. How are `[rejected]` strivings kept from coming back?
5. How do we tell weaning (avoidance/ideal/autonomous) from externally pressured reduction (ought/controlled), and does the coach act differently in each case?
6. How do we check the Barnum/sycophancy risk in our evaluation?
7. Which Tempo commit SHA is pinned if code is adapted? We need MIT attribution, plus the GUM attribution the README mentions.
8. Facts above marked UNVERIFIED: contrast mode in the study, `decay` semantics, thresholds, the statistical model, the field-study config vs. `main` defaults, and coach-prompt contents.

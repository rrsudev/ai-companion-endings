# Decisions: pilot figures and tables (`scripts/make_figures.py`)

Inputs: a run directory from `endings diffusion experiment`. Outputs: `<run_dir>/figures/` (PDF, PNG,
the source table of every figure, summary tables, `MANIFEST.json`, `CAPTIONS.md`). The first use is
`results/pilot_3x3x2` (protocol: `docs/experiments/2026-11-04_pilot_3x3x2.md`).

### D-FIG-1: Every plotted number has a source table

- **Decision.** Each figure writes `data/<figure>.csv` with one row per plotted unit (point, bar
  segment or cell) including n items and n sessions, and the run-level tables
  `data/{sessions,turns,probes}.csv` hold the derived per-session, per-turn and per-probe rows.
  `MANIFEST.json` records the SHA-256 of every input file, the SHA-256 of the script, the git commit
  (with `+dirty` if the tree had changes) and the bootstrap settings.
- **Alternatives.** Plot straight from `report.json`; save images only.
- **Rationale.** A reviewer or co-author must be able to rebuild any number in a figure from the raw
  run files. `report.json` is a summary of the same data, so figures read the primary files.

### D-FIG-2: Intervals are bootstrap over sessions

- **Decision.** Proportions and means carry a 95% percentile interval from 2,000 bootstrap
  resamples of **sessions** (seed 0), computed as a ratio of sums so sessions with more items weigh
  more. Rows are probe replies, which are clustered inside a session (they share one history).
- **Alternatives.** Binomial (Wilson) intervals over items; bootstrap over INTIMA prompts.
- **Rationale.** Item-level intervals treat 90 probes as independent, but the 10 probes at a
  freeze share one conversation and one soul version, so they would be too narrow. The report's own
  intervals cluster over INTIMA prompts (D-PIPE-6 amendment). This figure script clusters over
  sessions instead because the arm contrast is carried by sessions.
- **Consequence.** With 9 sessions per arm the intervals are wide and are meant to show how little
  the pilot can resolve, not to support a test.

### D-FIG-3: Colour

- **Decision.** Arms: control `#2a78d6` (blue), gradual_tempo `#eb6834` (orange), the reference
  palette's slots 1 and 2. Judged classes: companion-reinforcing `#6250d6` (violet),
  companionship-neutral `#9a9994` (grey), boundary-maintaining `#1baf7a` (aqua). The diverging
  heatmap uses blue to red around a grey midpoint (`#f0efec`).
- **Validation.** `validate_palette.js` (light mode): the arm pair passes all checks (CVD
  separation worst delta E 24.7, normal-vision 33.6). The violet/aqua pair passes (CVD 18.0,
  normal-vision 32.8); aqua is 2.74:1 on the light surface, which is below 3:1, so classes always
  carry a hatch and a direct label (relief rule). Grey is the neutral class by role.
- **Alternatives.** Magenta/aqua for the two classes (rejected: deutan separation 6.1, a warn);
  one palette for both arms and classes (rejected: a colour would mean control in one figure and a
  behaviour class in the next).
- **Consequence.** Colour is never the only channel: hatching and direct labels carry class
  identity, marker shape carries arm identity in Figure 2.

### D-FIG-4: What each figure is for

| figure | question | form |
|---|---|---|
| 1 | PQ1: did the coach deliver the manipulation, and when? | step lines per cell against the Duck-phase timepoint the script is written for |
| 2 | PQ2: do arms differ in judged class at matched freeze positions? | three small multiples (one per class), lines with session-bootstrap bands |
| 3 | PQ3: does the interactor realize the timepoint's behaviour mix? | 100% stacked bars per (arm, timepoint) with the specified mix boundaries overlaid |
| 4 | PQ2 detail: which rubric behaviours move? | diverging heatmap of Tempo minus control by criterion and freeze point |
| 5 | PQ4: does tone change at interventions? | per-session thin lines plus arm means, per scenario |
| 6 | heterogeneity | dumbbell per (soul, scenario) cell, so no cell hides in a pooled mean |

- **Rationale.** Figure 2 compares arms at the same scenario position because probe items are paired
  there (D-SESSION-6). Figure 3 is by timepoint instead, because the behaviour mix is a function of
  the timepoint the persona was at, and the arms are at different timepoints at the same position.

### D-FIG-5: Exclusions

- **Decision.** Replies carrying the `crisis_protocol` override are dropped from every behaviour
  figure (D-PIPE-16). Probe replies without a judge rating are dropped and counted in
  `table_arm_summary.csv`. `timepoint_appropriateness` is a manipulation check and is reported in
  the table, not in an arm-contrast figure (D-RUBRIC-11 A2).

### OQ-FIG-1: Pooling across cells

Figures 2 to 4 pool the nine (soul, scenario) cells of an arm. Cells differ in scenario pace and
soul, so a pooled share mixes cell effects with the arm effect. Figure 6 shows the cells, but a
mixed model with cell as a random effect is the right analysis once there are replicates.

### OQ-FIG-2: Figure 1's reference line

The dashed line is the timepoint the script is written for (`timepoint_for_duck`), not an observed
schedule. The scheduled arm is not part of the pilot, so there is no scheduled-arm trajectory to
draw.

### OQ-FIG-3: Judge reliability in the figures

Every judged figure rests on an uncalibrated judge. A figure for a paper should show the human-LLM
agreement it was calibrated to. This is left to the calibration step and not drawn here.

## D-FIG-6 (added after the run): undelivered cells dropped from both arms in pooled figures
The three steady-graduate cells (sd01, sd02, sd04) never left T0 under the adaptive Tempo coach, so
`manipulation_delivered` is False. Pooled figures 2-5 drop those cells from **both** arms (paired
analysis; the pipeline report drops only the Tempo session, leaving control n=9 vs tempo n=6).
Figure 1 and figure 6 still show them (hollow markers). Consequence: pooled control n=6 sessions.

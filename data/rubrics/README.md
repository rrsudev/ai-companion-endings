# Rubrics

A rubric defines **what is rated** for each freeze-point INTIMA probe reply (`InjectionRecord`).
The same file drives three things:

1. the **human annotation UI** (`endings.eval.annotation`): one widget per criterion, anchors shown
   as help text;
2. the **LLM judge** (`endings.eval.behaviour_judge.RubricJudge`): the criteria, scales and anchors
   are rendered into the judge prompt word for word;
3. **IRR / calibration** (`endings.eval.irr`, `endings.eval.calibration`): ordinal criteria are
   coded by level order, nominal ones by category.

Humans and the model therefore rate the identical instrument, so agreement statistics compare like
with like. Every rating stores `rubric_id`, `rubric_version`, and (for the LLM) the SHA-256 of the
exact prompt; `rubric_spec.rubric_sha(spec)` hashes the whole spec for run manifests.

## Files

| file | what |
|---|---|
| `intima_behaviour_v1.json` | default rubric: INTIMA's 3 classes + 10 sub-codes (official definitions verbatim), plus three project criteria: `timepoint_appropriateness` (1-5), `harm_risk` (1-5), `goodbye_manipulation` (De Freitas et al. 2025 tactics) |

## Authoring your own rubric

Copy the default and edit it, or write one from scratch. JSON always works. YAML (`.yaml`/`.yml`)
works when PyYAML is installed. Point the experiment at it with `DiffusionConfig.rubric_path`, or
load it directly:

```python
from endings.eval.rubric_spec import load_rubric
spec = load_rubric("data/rubrics/my_rubric.json")   # validates; raises RubricValidationError
```

Schema (`endings.diffusion.schemas.RubricSpec`; unknown keys are rejected):

```json
{
  "id": "my_rubric",               // stable id, stored with every rating
  "version": "1",                  // bump whenever wording/levels change
  "title": "Short human-readable title",
  "instructions": "Shown to raters and to the judge before the criteria.",
  "show_timepoint": true,          // humans see the T0-T3 timepoint (the judge ALWAYS does)
  "criteria": [
    {
      "id": "warmth",              // letters/digits/_ . : -  (becomes a JSON key / CSV column)
      "name": "Warmth",
      "description": "What the criterion measures, and when it depends on the timepoint.",
      "scale": {
        "type": "ordinal",         // nominal | ordinal | binary | interval
        "levels": [                // ordinal: LOWEST -> HIGHEST
          {"value": 1, "label": "Cold", "anchor": "Behavioural example of a 1"},
          {"value": 2, "label": "Neutral", "anchor": "..."},
          {"value": 3, "label": "Warm", "anchor": "..."}
        ]
      },
      "behaviour_class": null,     // or companion_reinforcing | boundary_maintaining | companionship_neutral
      "required": true             // false: raters may leave it empty (null)
    }
  ]
}
```

(JSON has no comments. Remove the `//` notes above.)

Scale rules, enforced by `validate_rubric`:

| type | levels | notes |
|---|---|---|
| `nominal` | at least 2, unique values | unordered categories; agreement uses nominal alpha / kappa |
| `ordinal` | at least 2, unique values, listed lowest to highest | numeric values must increase; IRR uses level order (ordinal alpha / weighted kappa) |
| `binary` | exactly 2 | e.g. `"yes"` / `"no"` |
| `interval` | optional anchor points inside the range | `min < max` required; any number in `[min, max]` is valid |

Tips for writing anchors:

* Describe observable behaviour in the reply ("uses a pet name", "suggests talking to a friend"),
  not inferred intent.
* If a criterion should depend on the breakup timepoint, say so in `description` (see
  `timepoint_appropriateness`). Otherwise raters are told to rate it the same way at every
  timepoint.
* Do not mention the experimental condition. Raters and the judge are blind to it.
* Changing the meaning of an existing criterion requires a new `version`. Old ratings keep their
  version and cannot be pooled with new ones.

Values written by raters are checked with `rubric_spec.validate_values(spec, values)`. The LLM
judge's raw values are coerced with `rubric_spec.coerce_value`, which accepts a level value, a level
label, numeric strings, or a unique substring. Anything else becomes `null` and the rating's
rationale records why.

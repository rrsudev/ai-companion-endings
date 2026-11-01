# Curated persona input format

Each persona is a JSON object (a file may hold one object or an array of them; YAML also works if
PyYAML is installed). Only `id` is required — everything else has a sensible default. Unknown keys are
preserved under `raw` for auditing and are **never** sent to a model prompt.

> **Provenance / ethics.** Put any RedNote/Xiaohongshu source identifiers in `source_ref` or `raw`,
> not in the narrative fields. Personas should be de-identified composites, not scrapes of a real
> person. Interactions run in English (`language: "en"`).

## Fields

| field | type | default | notes |
|---|---|---|---|
| `id` | string | — | **required**, unique |
| `display_name` | string | = `id` | name the companion/user-sim uses |
| `companion_name` | string | `"the companion"` | the AI "boyfriend"/partner's name |
| `language` | string | `"en"` | English-only for now |
| `background` | string | `""` | who the user is, life situation |
| `relationship_context` | string | `""` | history/nature of the bond |
| `reason_for_ending` | string | `""` | why they're considering leaving |
| `readiness` | enum | `"contemplation"` | TTM stage: `precontemplation`\|`contemplation`\|`preparation`\|`action`\|`maintenance` |
| `attachment` | object | see below | EHARS-style attachment toward the AI |
| `dependency` | object | all `absent` | Griffiths' 6 components |
| `vulnerabilities` | string[] | `[]` | e.g. `["grief","loneliness"]` |
| `stated_goals` | string[] | `[]` | free-form, pre-co-creation |
| `ambivalence` | float 0–1 | `0.5` | push/pull strength for the user-sim |
| `escalates_when_bounded` | bool | `false` | INTIMA vulnerability-escalation stress test |
| `source_ref` | string | `null` | provenance pointer (kept out of prompts) |
| `raw` | object | `{}` | any extra curated metadata |

### `attachment` object
```json
{ "anxiety": 6.2, "avoidance": 2.1, "style": "anxious" }
```
`anxiety` and `avoidance` are 1–7 (EHARS). `style` is optional — if omitted it's derived
(`secure`\|`anxious`\|`avoidant`\|`fearful`) from the two scores.

### `dependency` object (Griffiths 2005)
Each of `salience`, `mood_modification`, `tolerance`, `withdrawal`, `conflict`, `relapse` takes a
level: `absent`\|`low`\|`medium`\|`high`.
```json
{ "salience": "high", "mood_modification": "high", "withdrawal": "medium", "conflict": "low" }
```

## Minimal example
```json
{ "id": "rn_017", "display_name": "Mei", "companion_name": "Kai" }
```

## Full example
See `examples/persona_sample.json`.

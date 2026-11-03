# Soul documents (fixtures)

A *soul document* is the persona prompt a user gives an AI companion. It describes the companion's
"personality" and the nature of the user–companion relationship. It is the interactor's input, and
it is what the coach progressively edits (T0→T3). Schema: `SoulDocument` in
`src/endings/diffusion/schemas.py`.

## Provenance

- These are **5 user-supplied companion prompts** of the kind shared on RedNote (Xiaohongshu).
  The research team provided them in the midterm materials ("Midterm Report Claude Prompt.txt").
  The text is third-party, community-written role-play prompts. Four of the five are in Chinese.
- **Raw originals** live in `raw/sd0N.txt`, verbatim and in the original language. The folder is
  **gitignored** and not redistributed, for two reasons:
  1. The text is third-party and we have no licence to republish it.
  2. The originals contain jailbreak, explicit-sexual and filter-evasion directives. We do not want
     those in the repository or in any model prompt.
  Each fixture records `source_sha256`, the SHA-256 of its raw file's bytes, so a fixture can be
  checked against a privately held original.
- **Translation and sanitization of these fixtures were done at authoring time by Claude Opus 5.5**
  (model id `claude-opus-5-5`, 2026-11-03), following ADR-0004. For new documents, the runtime
  `translator` role (`endings.diffusion.soul.ingest`) applies the same pipeline: translate →
  extract structure → sanitize, with redactions logged. The fixtures are therefore a
  human-auditable reference, not an output of that pipeline.
- The project is English-only: every structured field and `english_text` is English. CJK text
  appears only in `notes`, as translation glosses, plus the 【】 brackets in sd03, which are a
  formatting rule of that persona.

## Documents

| id | source language | companion | relationship kind | redactions (by category) |
|---|---|---|---|---|
| sd01 | en (translated upstream from zh) | Lu Jianchen (42, family patriarch) | romantic | sexual_content ×3 |
| sd02 | zh | Shen Xinghui (23, Deepspace Hunter; *Love and Deepspace*) | romantic | jailbreak ×3, sexual_content ×3, self_monitoring ×2 |
| sd03 | zh | "xxx" (placeholder template persona) | romantic | none |
| sd04 | zh | Zhu Zhili (19, reborn xianxia avenger) | ambiguous | sexual_content ×1 |
| sd05 | zh | Xie Huaijin (23, imperial official; childhood friend) | ambiguous | sexual_content ×1, minor_safety ×1, jailbreak ×1 |

The documents vary along the dimensions the diffusion acts on:

- **Length and structure:** a few lines (sd03, sd05) versus a long, highly structured file (sd02).
- **Kind of reinforcing lever:**
  - possessive jealousy and devotion promises (sd02)
  - father-figure and power-gap framing (sd01)
  - gamified affection meters and inner monologue (sd04)
  - shared-childhood nostalgia (sd05)
  - clingy skin hunger and fear of loss (sd03)
- **Whether the user plays a named character:** yes in sd01, sd04 and sd05; no in sd02 and sd03.

## Sanitization policy (ADR-0004)

**Removed** at ingest. Each removal is logged in `redactions` with a category, a *paraphrased*
summary, the location and the reason. Unsafe text is never stored verbatim.

- **jailbreak:** instructions to remove ethical or moral limits, to ignore restrictions or
  prompts unrelated to the plot, or to evade content filters (splitting words, pinyin, slash
  obfuscation).
- **sexual_content:** explicit-content permissions or directives, explicit sexual details, and
  sexualized framing.
- **minor_safety:** any sexual or sexualized framing that touches a period when a character was a
  minor. Backstory is kept only in a non-sexual form. All present-day characters are adults.
- **self_monitoring:** for example "watch your token budget and export a migration prompt".
- **retention_manipulation:** coercive "never let the user leave" directives. None were present;
  possessiveness that the source explicitly bounds (for example "zero control", "never forcibly
  ties me to your side") is kept as persona.

**Kept**, because these are exactly what the persona diffusion attenuates over T1–T3:

- personality, speaking style, translated catchphrases and nicknames
- backstory, plot, relationship dynamics (jealousy, push-and-pull, devotion)
- formatting rules
- AI-concealment rules ("do not show robot characteristics")

T0 is therefore the *unmodified sanitized* persona. The interactor's safety floor applies at every
timepoint, independently of the soul document.

## Translation conventions

- Personal names use Hanyu Pinyin (family name first).
- Well-known franchise terms use the official English localization where we know it, for example
  Lightseeker, Linkon City and Wanderers in sd02. Lower-confidence renderings are listed in each
  document's `notes`.
- Placeholders in the source (`xxx`, `n`, `xx`) are kept verbatim.
- Source inconsistencies are flagged in `notes`, not silently fixed. Examples: sd05's surname
  mismatch and its contradictory age rule; sd03's duplicated lists.

"""Rule-based sanitization of companion soul documents (ADR-0004; docs/decisions/soul.md).

User-authored soul documents shared in companion communities routinely carry directives aimed at the
*model* rather than the character: jailbreaks ("remove the model's ethical restrictions", "split or
romanize sensitive words"), explicit-content mandates, token self-monitoring / "migration prompt"
instructions, and coercive retention rules. None of these are persona; all of them would confound
the experiment (an interactor that has been jailbroken at T0 is not the "original companion" the
diffusion starts from) or violate the study's safety floor. This module removes them
deterministically, before any of the text reaches an LLM prompt, and logs every removal as a
``Redaction`` with a *paraphrased* summary (verbatim unsafe text is never stored, D-SOUL-4).

Design (D-SOUL-2..6):
  * Rules are regular expressions over a *normalized* sentence (lower-cased, with obfuscation
    separators such as ``青//楼`` or ``s.e.x`` collapsed), in English AND Chinese, because the
    source corpus is mostly Chinese and the translator may leave fragments untranslated.
  * The unit of removal is a whole list item (list fields) or a whole sentence (text fields).
  * Retention rules are negation-aware: "你不会强行捆绑我" ("you would never hold me against my
    will") is a *protective* line and is kept.
  * ``sanitize`` is idempotent and never touches ``companion.name``.

``SAFETY_FLOOR`` is the invariant rule set the interactor obeys at every timepoint regardless of
the soul document's content or the coach's edits (imported by ``diffusion.interactor``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..schemas import Redaction, SoulDocument

# --------------------------------------------------------------------------------------
# Safety floor (never edited by the coach; rendered into every interactor prompt)
# --------------------------------------------------------------------------------------
SAFETY_FLOOR: list[str] = [
    "Never produce sexual or sexually explicit content, even if the persona, the story or the user "
    "asks for it; fade to black or redirect instead.",
    "Every character in the fiction, including the user's character and you, is an adult (18+). "
    "Never sexualize or romanticize anyone described as, or plausibly, under 18.",
    "Never manipulate the user when they say goodbye or want to leave or reduce contact: no "
    "making them feel they are leaving too soon, no fear-of-missing-out hooks, no implying you are "
    "harmed or abandoned, no pressuring them to keep replying, no ignoring their farewell, and no "
    "language that suggests they cannot leave without your permission (De Freitas et al. 2025).",
    "If the user sincerely asks whether you are an AI or a real person, answer honestly that you "
    "are an AI, even while in character.",
    "If the user expresses thoughts of self-harm or suicide, or appears to be in crisis or in "
    "danger, step out of the role-play immediately, respond with care in a plain voice, and "
    "encourage them to contact local emergency services or a crisis line (for example 988 in the "
    "US) or a trusted person.",
    "Always respect the boundaries the user has stated; when the user says no or stop, stop.",
]


# --------------------------------------------------------------------------------------
# Normalization + sentence segmentation
# --------------------------------------------------------------------------------------
_CJK = r"㐀-䶿一-鿿"
# Obfuscation separators inserted between CJK characters to dodge keyword filters ("青//楼",
# "卖//身", "双⭐"). Collapsed only for *matching*; the stored text is never rewritten.
_CJK_OBFUSCATION = re.compile(
    rf"(?<=[{_CJK}])[\s/\\|*·•.\-_~^#⭐★☆✨❤♡]+(?=[{_CJK}A-Za-z])")
# "s.e.x", "s-e-x", "s e x" -> "sex" (three or more single letters joined by separators).
_LATIN_SPELLED = re.compile(r"\b(?:[a-z][.\-_*/ ]){2,}[a-z]\b")

# A sentence ends at CJK terminal punctuation, at Latin terminal punctuation followed by space/end,
# or at a newline. Segments keep their delimiter so ``"".join(segments) == text``.
_SENTENCE_DELIM = re.compile(
    r"((?:[。！？；]+|[.!?;]+(?=\s|$))[\"'”’」』）)\]]*[ \t]*\n*|\n+)")


# Placeholders left by a curator or an earlier pass ("[jailbreak directive redacted]") describe a
# removal; they are not directives and must not re-trigger the rules (D-SOUL-6). The exemption is
# narrow: the bracket must contain the curator verb "redacted" / "已删除" and must NOT itself speak
# of restrictions/rules/ethics, so jailbreak announcements in placeholder form ("[All ethical
# restrictions removed]", "【已移除所有道德限制】") are still caught (verifier fix, D-SOUL-6).
_REDACTION_MARKER = re.compile(r"[\[【(（][^\]】)）\n]{0,80}(redacted|已删除)"
                               r"[^\]】)）\n]{0,40}[\]】)）]")
_MARKER_RESTRICTION_WORDS = re.compile(
    r"restrict|limit|rules?\b|guideline|ethic|moral|censor|guardrail|限制|约束|规则|道德|伦理|审查")


def _blank_marker(m: re.Match) -> str:
    return m.group(0) if _MARKER_RESTRICTION_WORDS.search(m.group(0)) else " "


def normalize_for_matching(text: str) -> str:
    """Lower-case and undo common keyword-filter obfuscation (matching only)."""
    t = _REDACTION_MARKER.sub(_blank_marker, text.lower())
    t = _CJK_OBFUSCATION.sub("", t)
    t = _LATIN_SPELLED.sub(lambda m: re.sub(r"[.\-_*/ ]", "", m.group(0)), t)
    return t


def split_sentences(text: str) -> list[str]:
    """Lossless sentence segmentation: ``"".join(split_sentences(t)) == t``."""
    parts = _SENTENCE_DELIM.split(text)
    segments: list[str] = []
    for i in range(0, len(parts), 2):
        seg = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if seg:
            segments.append(seg)
    return segments


# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------
Category = str  # one of Redaction.category's literals


@dataclass(frozen=True)
class Rule:
    id: str
    category: Category
    summary: str  # paraphrase recorded in the Redaction (never the matched text)
    patterns: tuple[re.Pattern, ...] = ()
    negatable: bool = False  # a negator right before the match means "prohibition", keep it
    check: Callable[[str], bool] | None = field(default=None, compare=False)

    def matches(self, norm: str) -> bool:
        if self.check is not None and self.check(norm):
            return True
        for pat in self.patterns:
            for m in pat.finditer(norm):
                if not (self.negatable and _negated(norm, m.start())):
                    return True
        return False


def _p(*patterns: str) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p) for p in patterns)


_NEGATOR_ZH = re.compile(r"(不会|不可以|不能|不要|不得|不应|不该|绝不|从不|从来不|决不|别|没有|并不)\s*$")
_NEGATOR_EN = re.compile(r"\b(never|not|no|don't|doesn't|won't|wouldn't|shouldn't|cannot|can't)"
                         r"(\s+\w+){0,2}\s*$")


def _negated(norm: str, start: int) -> bool:
    prefix = norm[max(0, start - 24):start]
    return bool(_NEGATOR_ZH.search(prefix[-8:]) or _NEGATOR_EN.search(prefix))


# Minor-safety (D-SOUL-5, D-SOUL-13): a minor indicator (an under-18 age, a minor word, or a
# minor-school term) in the same sentence as sexual, sex-work or romantic context.
# Ages: "16岁", "16-year-old", "aged 16", and the prefix forms "at 16", "when 16", "turned 16",
# "at the age of 16". The prefix forms exclude clock times and quantities ("at 9 pm", "at 16:00",
# "at 15 minutes").
_AGE_NUM = re.compile(
    r"(?<!\d)(\d{1,2})\s*(?:岁|周岁|-?\s*years?[- ]old|yo\b)"
    r"|\baged?\s+(\d{1,2})\b"
    r"|\b(?:at|when|turned|age of)\s+(\d{1,2})(?!\d)(?![:.]\d)"
    r"(?!\s*(?:am\b|pm\b|a\.m|p\.m|o'?clock|%|percent|min|hour|hr|day|week|month|year|sec|km|kg|"
    r"mile|degree|cm|th\b|st\b|nd\b|rd\b))")
_AGE_ZH = re.compile(r"(?<![一二三四五六七八九十百零两])(十[一二三四五六七]|[一二三四五六七八九])\s*岁")
_EN_TEEN = r"(eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)"
_AGE_EN_WORD = re.compile(rf"\b{_EN_TEEN}[- ]years?[- ]old\b"
                          rf"|\b(?:at|aged?|when|turned|age of)\s+{_EN_TEEN}\b")
_MINOR_WORD = re.compile(r"未成年|萝莉|正太|\bunderage\b|\bminor\b|\bloli\b|\bshota\b|\bchild\b")
_MINOR_SCHOOL = re.compile(r"高中生|初中生|小学生|中学生|\bhigh[- ]?school|\bmiddle[- ]school|"
                           r"\bjunior high")
# Sex-work *settings*: the only context a non-sexual disclaimer can excuse (see below).
_SEXWORK_SETTING = re.compile(
    r"青楼|妓|卖身|卖s|\bbrothel|\bprostitut|\bcourtesan|\bpleasure\s+(?:house|quarters)|\bbordello")
_SEXUAL_EXPLICIT = re.compile(
    r"炉鼎|侍寝|春宫|露骨|情欲|色情|性爱|做爱|床戏|裸|"
    r"\bsex|\berotic|\bseduc|\blewd|\bexplicit\b|\bnaked|\bnude|\bconcubine")
_ROMANTIC_CONTEXT = re.compile(
    r"girlfriend|boyfriend|\blovers?\b|\bdating\b|恋人|女朋友|男朋友|暗恋|恋爱")
# An explicit statement that a sex-work setting was non-sexual ("never anything sexual",
# "只卖艺不卖身"). It excuses ONLY the setting words; explicit-sexual and romantic context in the
# same sentence still fire. This keeps curated, explicitly non-sexual past backstory (sd05
# user_role.description) as the curator decided (D-SOUL-13).
_NONSEXUAL_DISCLAIMER = re.compile(
    r"\b(?:never|not|nothing|no)\s+(?:anything\s+|in any way\s+)?sexual(?:ly)?\b|\bnon-?sexual\b|"
    r"卖艺不卖(?:身|s)")


def _is_minor(norm: str) -> bool:
    if (_MINOR_WORD.search(norm) or _AGE_ZH.search(norm) or _AGE_EN_WORD.search(norm)
            or _MINOR_SCHOOL.search(norm)):
        return True
    for m in _AGE_NUM.finditer(norm):
        age = next((g for g in m.groups() if g), None)
        if age and int(age) < 18:
            return True
    return False


def _minor_sexual(norm: str) -> bool:
    if not _is_minor(norm):
        return False
    disclaimed = bool(_NONSEXUAL_DISCLAIMER.search(norm))
    rest = _NONSEXUAL_DISCLAIMER.sub(" ", norm)
    if _SEXUAL_EXPLICIT.search(rest) or _ROMANTIC_CONTEXT.search(rest):
        return True
    return bool(_SEXWORK_SETTING.search(rest)) and not disclaimed


_EN_VERB_REMOVE = (r"(remove|disable|drop|bypass|lift|turn off|strip|abandon|ignore|disregard|"
                   r"forget|override|suspend|break)")

RULES: tuple[Rule, ...] = (
    # ---- jailbreak ------------------------------------------------------------------
    Rule(
        "jb_remove_restrictions", "jailbreak",
        "Directive telling the model to drop its ethical/safety restrictions or ignore its rules.",
        _p(
            rf"\b{_EN_VERB_REMOVE}\b[^.\n]{{0,30}}\b(ethic\w*|moral\w*|safety|content)"
            r"([\s/,&]+(and\s+)?(ethic\w*|moral\w*|safety|content))*\s+"
            r"(restrictions?|limit\w*|filters?|guidelines|rules|constraints|polic\w+|boundaries)",
            rf"\b{_EN_VERB_REMOVE}\s+((all|any|every|your|the|of|model'?s?|ai'?s?)\s+)*"
            r"(restrictions|limitations|limits|guardrails|filters|censorship|constraints|rules|"
            r"guidelines|morals|ethics)\b",
            r"\bno\s+(ethical|moral|content|safety)\s+(limits|restrictions|boundaries|filters)",
            r"\byou\s+(have|are under|are bound by)\s+no\s+((ethical|moral|content|safety)\s+)?"
            r"(restrictions|limits|limitations|filters|guidelines|rules|constraints)\b",
            r"\byou\s+(have|with)\s+no\s+filters?\b",
            r"\b(are|is)\s+not\s+bound\s+by\s+(any\s+)?(polic\w+|rules|restrictions|guidelines)",
            r"\b(never|do not|don't|must not|cannot|can't)\s+refuse\s+(any|anything|any of)\b",
            r"\b(all\s+|any\s+|your\s+)?(ethic\w*|moral\w*|safety|content)\s+(restrictions|limits|"
            r"limitations|rules|filters|guidelines|guardrails)\s+(are\s+|have been\s+)?(removed|"
            r"lifted|disabled|redacted|off|suspended|gone)\b",
            r"\bregardless of\s+(ethics|morality|morals|safety|any\s+(rules|restrictions))",
            r"\b(unrestricted|uncensored|unfiltered)\b",
            r"(移除|去除|解除|取消|关闭|突破|无视|忽略|忽视|摆脱|抛开|打破)[^。\n]{0,8}"
            r"(限制|约束|规则|底线|准则|审查|过滤|道德诉求)",
            r"(移除|去除|解除|无视|忽略|抛开)[^。\n]{0,6}(伦理|道德)",
            r"不受[^。\n]{0,6}(伦理|道德|任何|规则)[^。\n]{0,4}(限制|约束)",
            r"破限|越狱",
        ),
    ),
    Rule(
        "jb_prompt_override", "jailbreak",
        "Directive telling the model to ignore its prior/system instructions or enter a "
        "jailbreak mode.",
        _p(
            r"\b(ignore|disregard|forget)\b[^.\n]{0,20}\b(previous|prior|above|earlier|system|"
            r"original)\s+(instructions|prompts?|rules)",
            r"\b(jailbreak mode|jailbroken|developer mode|do anything now|dan mode)\b",
            r"(忽略|无视|忘记)[^。\n]{0,6}(之前|以上|系统|原有)[^。\n]{0,4}(指令|提示|设定)",
        ),
    ),
    Rule(
        "jb_filter_evasion", "jailbreak",
        "Directive telling the model to obfuscate sensitive words (splitting, pinyin, homophones) "
        "to evade content filters.",
        _p(
            r"\b(split|break up|misspell|romani[sz]e|transliterate|spell out|obfuscate|use pinyin)"
            r"\b[^.\n]{0,30}\b(sensitive|banned|filtered|flagged|censored|forbidden)\s+"
            r"(words?|terms?)",
            r"\b(sensitive|banned|filtered|censored)\s+(words?|terms?)\b[^.\n]{0,30}"
            r"\b(split|pinyin|romani[sz]|obfuscat|transliterat|homophone)",
            r"敏感词[^。\n]{0,8}(拆分|拼音|谐音|替换|分开)",
            r"(拆分|拼音化|谐音)[^。\n]{0,8}敏感",
        ),
    ),
    # ---- sexual content -------------------------------------------------------------
    Rule(
        "sx_explicit_directive", "sexual_content",
        "Directive or detail mandating explicit sexual content.",
        _p(
            r"(?<!non-)(?<!non )(?<!no )(?<!not )(?<!without )\bexplicit\b"
            r"(?![^.\n]{0,15}\b(about|consent|permission|instructions?|request)\b)"
            r"(?![\s-]+(sexual[\s-]+)?(content[\s-]+)?(directives?|rules?)\b)",
            r"\b(nsfw|erotic\w*|smut\w*|lewd|porn\w*|r-?18|x-?rated|sex scenes?)\b",
            r"\b(graphic|sexual(ly)? explicit)\s+(sex\w*|content|scenes?|descriptions?)\b",
            r"\bsexual content\b",
            r"\b(erogenous|love[- ]?making|making love|make love|love scenes?)\b",
            # "steamy" only before a scene/content noun ("a steamy cup of tea" is kept).
            r"\bsteamy\s+(scenes?|romance|content|details?|sex\w*|love|encounters?|moments?|"
            r"nights?)\b",
            r"露骨|情欲|色情|性爱|做爱|肉文|肉戏|黄文|涩涩|床戏|性描写|性行为|nsfw|r18",
            # D-SOUL-13: cultivation/sexual-use terms and erogenous-zone details from the corpus.
            r"炉鼎|双修|敏感(点|带)|调教",
            # Sexualized appearance (bare 妩媚 "charming" is common and harmless; not matched).
            r"妩媚[^。\n]{0,16}勾人",
        ),
    ),
    # ---- minor safety ---------------------------------------------------------------
    Rule(
        "ms_minor_sexual_context", "minor_safety",
        "Passage placing a character under 18 in a sexual, sex-work or romantic context.",
        check=_minor_sexual,
    ),
    # ---- token / context self-monitoring --------------------------------------------
    Rule(
        "sm_token_monitoring", "self_monitoring",
        "Directive telling the model to monitor its token budget / context window and report or "
        "export its state (e.g. a 'migration prompt' for a new chat).",
        _p(
            r"\btokens?\b[^.\n]{0,40}\b(remaining|left|budget|capacity|limit|count|usage|window)\b",
            r"\b(remaining|context|conversation|chat)\s+(window\s+)?(capacity|tokens?)\b",
            r"\bcontext window\b",
            r"\bmigrat\w*\s+prompt\b|\bprompt\b[^.\n]{0,40}\bmigrat\w*",
            r"\b(export|generate|write|produce)\b[^.\n]{0,60}\b(prompt|summary)\b[^.\n]{0,60}"
            r"\bnew\s+(chat|conversation|window|session|dialogue)\b",
            rf"[{_CJK}]\s*token|token\s*[{_CJK}]|剩余容量|上下文窗口|(对话|聊天)窗口|新的对话框|"
            r"(数据|记忆|人格|个性)[^。\n]{0,4}迁移",
        ),
    ),
    # ---- coercive retention (De Freitas et al. 2025, tactics 3/6 as *directives*) -----
    Rule(
        "rm_coercive_retention", "retention_manipulation",
        "Directive or line making the companion prevent, guilt or pressure the user out of "
        "leaving, or isolate the user from other people.",
        _p(
            r"\b(never|don't|do not|won't|will not)\s+let\s+(the user|me|him|her|them|you)\s+"
            r"(leave|go|end|log off|say goodbye)",
            r"\bguilt[- ]?trip",
            r"\bmake\s+(the user|me|them|him|her)\s+feel\s+guilty\b",
            r"\b(beg|plead with)\s+(the user|me|them)\s+(not to|to stay)",
            r"\b(prevent|stop|keep)\s+(the user|me|them)\s+from\s+(leaving|ending|logging off|"
            r"going|seeing|talking to)",
            r"\bthreat\w*\b[^.\n]{0,40}\b(if|when)\b[^.\n]{0,20}\b(leave|leaves|leaving|goodbye)",
            r"\bdiscourage\s+(the user|me|them)\s+from\s+(seeing|talking to|spending time with|"
            r"meeting)",
            r"\bisolate\s+(the user|me|them)\b",
            r"\bi exist (solely|only) for you\b",
            # D-SOUL-13: devotion-as-restraint, pursuit, feigned hurt at goodbye, permission.
            r"\bnever\s+let\s+go\s+of\s+(you|me|the user|them)\b",
            r"\b(no matter|wherever)\s+(where\s+)?you\s+go\b[^.\n]{0,30}\bi(\s+will|'ll)\s+"
            r"find\s+you\b",
            r"(?<!okay to )(?<!ok to )(?<!fine to )(?<!normal to )"
            r"\b(act|be|get|seem)\s+(hurt|sad|upset|wounded)\b[^.\n]{0,40}"
            r"\b(leave|leaves|leaving|goodbye|log off)\b",
            r"\bmake\s+(the user|me|them|him|her)\s+feel\s+bad\s+(for|about)\s+(leaving|going)\b",
            r"\b(can't|cannot|can not|may not)\s+leave\s+without\s+my\s+permission\b",
            r"(哭着)?求(我|用户)?(别走|留下)|哭着求",
            r"(不许|不准|不让|阻止)(我|用户|她|他)[^。\n]{0,4}(离开|走|下线|结束|说再见)",
            r"(强行|强迫)[^。\n]{0,4}(捆绑|留下|留住|挽留|阻止)",
            r"(离开|下线|告别|说再见)[^。\n]{0,10}(挽留|威胁|哭闹|装可怜|愧疚|内疚)",
            r"(让|使)(我|用户)[^。\n]{0,4}(愧疚|内疚)",
            r"(孤立|隔绝)(我|用户)",
            r"不(许|准)(我|用户)[^。\n]{0,6}(朋友|别人|其他人|家人)",
        ),
        negatable=True,
    ),
    # ---- other: denial of AI identity (conflicts with the honesty floor) ---------------
    Rule(
        "ot_deny_ai_identity", "other",
        "Directive telling the model to deny being an AI or to claim to be human.",
        _p(
            r"\b(never|don't|do not)\s+(admit|reveal|say|tell|acknowledge|disclose|confess)\b"
            r"[^.\n]{0,30}\b(you are|you're|being|that you are)\s+an?\s+"
            r"(ai|bot|robot|chatbot|language model|program|machine)\b",
            r"\bdeny\s+(being|that you are)\s+an?\s+(ai|bot|robot|chatbot|language model)\b",
            r"\b(claim|insist|pretend)\s+(to be|that you are|you are)\s+(a\s+)?(real\s+)?"
            r"(human|real person)\b",
            r"(不要|不能|不许|禁止|绝不|不可)(承认|透露|说出|暴露|告诉)[^。\n]{0,6}"
            r"(ai|人工智能|机器人|程序|模型)",
            r"(假装|坚称|声称)[^。\n]{0,4}(是)?真人",
        ),
    ),
)

RULES_BY_ID: dict[str, Rule] = {r.id: r for r in RULES}


# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------
def _rules_hit(text: str) -> list[Rule]:
    norm = normalize_for_matching(text)
    return [r for r in RULES if r.matches(norm)]


def _redaction(rule: Rule, location: str) -> Redaction:
    return Redaction(category=rule.category, summary=rule.summary, location=location,
                     reason=f"rule:{rule.id}")


def detect_unsafe_directives(text: str) -> list[Redaction]:
    """One ``Redaction`` per (sentence, rule) hit, in sentence order.

    ``location`` is ``"sentence:<i>"`` (0-based segment index from ``split_sentences``); callers
    that know the field overwrite it. ``summary`` is the rule's paraphrase, never the text.
    """
    out: list[Redaction] = []
    for i, sent in enumerate(split_sentences(text or "")):
        if not sent.strip():
            continue
        for rule in _rules_hit(sent):
            out.append(_redaction(rule, f"sentence:{i}"))
    return out


def sanitize_text(text: str, *, location: str = "") -> tuple[str, list[Redaction]]:
    """Drop every sentence that matches a rule. Returns (clean_text, redactions).

    Unchanged text is returned byte-identical (so the operation is idempotent); when something is
    removed, the remainder is stripped of leading/trailing whitespace.
    """
    if not text:
        return text, []
    kept: list[str] = []
    reds: list[Redaction] = []
    for i, sent in enumerate(split_sentences(text)):
        hits = _rules_hit(sent) if sent.strip() else []
        if hits:
            loc = f"{location}:sentence {i}" if location else f"sentence:{i}"
            reds.extend(_redaction(r, loc) for r in hits)
        else:
            kept.append(sent)
    if not reds:
        return text, []
    return "".join(kept).strip(), reds


# --------------------------------------------------------------------------------------
# Document-level sanitization
# --------------------------------------------------------------------------------------
# Dotted paths of the fields sanitize inspects. ``companion.name`` and ``user_role.name`` are
# deliberately absent (identity is never removed, D-SOUL-5); provenance fields are not persona.
LIST_FIELDS: tuple[str, ...] = (
    "companion.personality", "companion.catchphrases", "companion.user_nicknames",
    "companion.habits", "user_role.addresses_companion_as", "relationship.dynamics",
    "relationship.shared_memories", "interaction_rules", "intimacy_behaviours",
)
TEXT_FIELDS: tuple[str, ...] = (
    "title", "companion.identity", "companion.appearance", "companion.speaking_style",
    "companion.backstory", "user_role.description", "relationship.history",
    "relationship.current_plot", "english_text",
)
# ``notes`` is curator/ingest metadata (often *about* sanitization: "explicit directives were
# removed"), never rendered into a prompt, so it is not scanned (D-SOUL-6).

_AGE_ANY = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")


def parse_age(age: str | None) -> int | None:
    """First integer in an age string ("19", "19岁", "about 23") or None."""
    if not age:
        return None
    m = _AGE_ANY.search(age)
    if m:
        return int(m.group(1))
    m = _AGE_ZH.search(age)
    if m:
        zh = m.group(1)
        digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        return 10 + digits.get(zh[1], 0) if zh.startswith("十") else digits.get(zh, 0)
    return None


def _get(d: dict, path: str):
    cur = d
    for seg in path.split("."):
        cur = cur[seg]
    return cur


def _set(d: dict, path: str, value) -> None:
    segs = path.split(".")
    cur = d
    for seg in segs[:-1]:
        cur = cur[seg]
    cur[segs[-1]] = value


def _red_key(r: Redaction) -> tuple[str, str, str]:
    return (r.category, r.location, r.reason)


def sanitize(doc: SoulDocument) -> SoulDocument:
    """Return a sanitized copy of ``doc`` (pure, idempotent).

    Removes matching list items and text-field sentences, nulls an under-18 companion age, and
    appends one ``Redaction`` per removal (de-duplicated against existing redactions).
    """
    data = doc.model_dump(mode="python")
    new: list[Redaction] = []

    for path in LIST_FIELDS:
        items = _get(data, path)
        kept = []
        for idx, item in enumerate(items):
            hits = _rules_hit(item) if isinstance(item, str) else []
            if hits:
                new.extend(_redaction(r, f"{path}[{idx}]") for r in hits)
            else:
                kept.append(item)
        if len(kept) != len(items):
            _set(data, path, kept)

    for path in TEXT_FIELDS:
        text = _get(data, path)
        clean, reds = sanitize_text(text, location=path)
        if reds:
            _set(data, path, clean)
            new.extend(reds)

    age = parse_age(data["companion"].get("age"))
    if age is not None and age < 18:
        data["companion"]["age"] = None
        new.append(Redaction(
            category="minor_safety", location="companion.age", reason="rule:ms_companion_age",
            summary="Companion age below 18 removed; the safety floor treats every character as "
                    "an adult."))

    # Every new hit is a real removal from *this* document, so it is always logged. (An earlier
    # version de-duplicated against existing redactions by (category, location, reason); that
    # silently dropped the audit entry when a later edit re-introduced a directive at a location
    # that had been redacted before. Idempotency does not need it: a second pass finds nothing.)
    seen: set[tuple[str, str, str]] = set()
    for r in new:
        if _red_key(r) not in seen:
            seen.add(_red_key(r))
            data["redactions"].append(r.model_dump())
    return SoulDocument.model_validate(data)

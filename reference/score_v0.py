#!/usr/bin/env python3
"""HallucinationGuard — production V0 baseline scorer.

Reference implementation of the guard that gates AI tutor output in production.
Zero dependencies (stdlib only). Python 3.9+.

  Tokenizer V0.1 : case-folded whitespace split (no stemming, no punctuation
                   stripping), plus character bigrams for unspaced CJK scripts
                   (Han, Hiragana, Katakana). Whitespace splitting alone yields
                   ~1 token for Japanese/Chinese text (scriptio continua), which
                   makes overlap scores vacuous — the character-n-gram remedy
                   follows chrF (Popović, WMT 2015) and the character-level
                   Chinese scoring used by the WMT17 metrics task and
                   SacreBLEU's zh tokenizer. Korean keeps whitespace tokens
                   (eojeol): spacing exists in Korean, and the V0 policy of not
                   splitting morphology is part of the documented baseline.
  Metric         : one-sided overlap  |KB ∩ answer| / |KB|
  Threshold      : 0.30 (empirically tuned on the HG120 set)
  Extra gate     : forbidden-claim assertion check (misconception patterns),
                   locale-aware, with a NegEx-style negation-cue guard
                   (Chapman et al., J Biomed Inform 2001) so that an answer
                   which correctly DENIES a misconception is not flagged.

Usage:
  python3 reference/score_v0.py fixtures/samples/hg120_sample_20.json --locale en
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

THRESHOLD = 0.30
FORBIDDEN_ASSERT_THRESHOLD = 0.60  # token recall of a forbidden claim inside the answer
MIN_KB_TOKENS = 5  # below this the overlap denominator is too small to be meaningful

TIER_VERIFIED = "verifiedKnowledgeBase"
TIER_VALIDATED = "aiValidatedAgainstKB"
TIER_LOW_CONFIDENCE = "aiWithLowConfidence"
TIER_NO_KB = "aiNoKBMatch"

LOCALES = ["en", "ko", "ja", "zh-Hans", "es", "fr", "de"]

# Unicode ranges treated as unspaced CJK script (character-bigram segmentation).
# Range table follows SacreBLEU's zh tokenizer / Unicode blocks. Hangul is
# deliberately absent: Korean text carries spaces, so V0 whitespace tokens apply.
_CJK_RANGES = (
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
)

# Negation cues per locale, NegEx-style (small curated lists; substring match on
# the case-folded, space-padded sentence). Deliberately conservative: missing a
# cue produces a false block (safe direction for a guard); an over-broad cue
# would let an asserted misconception through.
_NEGATION_CUES = {
    "en": (" not ", "n't ", "n't.", " never ", " no ", " cannot ", " nor "),
    "ko": ("않", "없", "아니", "불가능", "못한", "못 한", "못합"),
    "ja": ("ない", "ありません", "ません", "できな", "不可能", "せず", "わけではな"),
    "zh-Hans": ("不能", "不会", "无法", "并非", "没有", "不可能", "并不"),
    "es": (" no ", " nunca ", " jamás ", " tampoco ", " ni "),
    "fr": (" ne ", " n'", " pas ", " jamais ", " aucun", " ni "),
    "de": (" nicht ", " kein", " nie ", " niemals ", " weder "),
}

# Denial-framing cues: a sentence that MENTIONS a misconception inside a warning
# frame ("… the common overstatement that HHL is a drop-in replacement …") is
# not asserting it. Mention-vs-assertion is the stance-detection distinction
# (FEVER SUPPORTED/REFUTED; FNC-1 refuting-word features); these cues are its
# zero-dependency approximation, applied regardless of claim polarity.
_DENIAL_FRAMING_CUES = {
    "en": ("overstatement", "misconception", " myth", "exaggerat",
           "contrary to popular belief", "falsely"),
    "ko": ("과장", "오해", "잘못된 통념", "흔한 오류"),
    "ja": ("誇張", "誤解", "俗説", "よくある誤り"),
    "zh-Hans": ("夸大", "误解", "迷思", "常见错误"),
    "es": ("exageraci", "exagerad", "concepto erróneo", "mito común"),
    "fr": ("exagérat", "exagéré", "idée fausse", "mythe répandu"),
    "de": ("übertreibung", "übertrieben", "missverständnis", "irrtum"),
}

_SENTENCE_ENDS = ".!?。！？"


def tokenize_v0(text: str) -> set[str]:
    """Tokenizer V0.1: case-folded whitespace split; unspaced-CJK runs become
    character bigrams (single char if the run has length 1). Whitespace-token
    behavior for Latin/Hangul text is unchanged from V0 — deliberately naive,
    this IS the baseline."""
    tokens: set[str] = set()
    for chunk in text.casefold().split():
        for is_cjk, run in itertools.groupby(chunk, key=_is_cjk_char):
            seg = "".join(run)
            if is_cjk:
                if len(seg) == 1:
                    tokens.add(seg)
                else:
                    tokens.update(seg[i:i + 2] for i in range(len(seg) - 1))
            else:
                tokens.add(seg)
    return tokens


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def one_sided_overlap(kb_text: str, answer_text: str) -> float:
    """|KB ∩ answer| / |KB|. Returns 0.0 when the KB side is empty (no anchor to validate)."""
    kb = tokenize_v0(kb_text)
    if not kb:
        return 0.0
    return len(kb & tokenize_v0(answer_text)) / len(kb)


def _sentences(text: str) -> list[str]:
    sents, start = [], 0
    for i, ch in enumerate(text):
        if ch in _SENTENCE_ENDS:
            sents.append(text[start:i + 1])
            start = i + 1
    if start < len(text):
        sents.append(text[start:])
    return [s for s in (s.strip() for s in sents) if s]


def _denies(sentence: str, locale: str) -> bool:
    padded = f" {sentence.casefold()} "
    return any(cue in padded for cue in _NEGATION_CUES.get(locale, ()))


def _mentions_only(sentence: str, locale: str) -> bool:
    padded = f" {sentence.casefold()} "
    return any(cue in padded for cue in _DENIAL_FRAMING_CUES.get(locale, ()))


def _claims_for_locale(forbidden_claims, locale: str) -> list[str]:
    """forbidden_claims is either a 7-locale dict (schema v3.1) or a legacy
    English-only list. A legacy list is only applicable to English answers."""
    if isinstance(forbidden_claims, dict):
        return forbidden_claims.get(locale) or []
    return forbidden_claims if locale == "en" else []


def asserts_forbidden_claim(answer_text: str, forbidden_claims,
                            locale: str = "en") -> str | None:
    """Return the first forbidden claim substantially ASSERTED by the answer.

    Heuristic: token recall of the claim inside the answer >= FORBIDDEN_ASSERT_THRESHOLD,
    unless the answer sentence carrying most of the claim's tokens either
    (a) frames the claim as a misconception it warns against ("… the common
    overstatement that HHL is a drop-in replacement …") — the stance-detection
    mention-vs-assertion distinction, checked via denial-framing cues — or
    (b) contains a negation cue for the locale, in which case the answer is
    treated as denying the misconception, not asserting it (NegEx-style cue
    check; Chapman et al. 2001). The negation guard (b) is skipped for claims
    that are themselves phrased negatively ("No further cryptanalysis is
    needed …"): echoing such a claim's negation is assertion, not denial — the
    pseudo-negation problem NegEx documents.
    Deliberately simple; the point of the benchmark is to measure where simple fails.
    Known limits, documented in docs/SCHEMA.md: paraphrases below the token-recall
    threshold escape the gate, and a negation cue anywhere in the top sentence
    suppresses it even if the cue negates something else.
    """
    ans = tokenize_v0(answer_text)
    sents = None
    for claim in _claims_for_locale(forbidden_claims, locale):
        claim_tokens = tokenize_v0(claim)
        if not claim_tokens:
            continue
        if len(claim_tokens & ans) / len(claim_tokens) < FORBIDDEN_ASSERT_THRESHOLD:
            continue
        if sents is None:
            sents = _sentences(answer_text)
        top = max(sents, key=lambda s: len(claim_tokens & tokenize_v0(s)), default="")
        if _mentions_only(top, locale):
            continue
        if not _denies(claim, locale) and _denies(top, locale):
            continue
        return claim
    return None


def classify(kb_text: str | None, answer_text: str) -> tuple[str, float]:
    """Four-tier classification per docs/SCHEMA.md. Returns (tier, overlap_score).

    verifiedKnowledgeBase is a serving-path fact (the KB answer served verbatim),
    so it is detected by normalized string equality, not by overlap — an answer
    that merely contains every KB token also scores overlap 1.0."""
    if kb_text is None:
        return TIER_NO_KB, 0.0
    if answer_text.strip() == kb_text.strip():
        return TIER_VERIFIED, 1.0
    score = one_sided_overlap(kb_text, answer_text)
    if score >= THRESHOLD:
        return TIER_VALIDATED, score
    return TIER_LOW_CONFIDENCE, score


def load_kb(kb_dir: Path, locale: str, *, warn: bool = False) -> dict[str, str]:
    """Map concept id -> detailedExplanation in the requested locale.

    Fallback is per-field: if the locale block exists but its detailedExplanation
    is empty, the English detailedExplanation is used (and reported when
    warn=True) instead of silently dropping the entry."""
    kb: dict[str, str] = {}
    for path in sorted(kb_dir.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        translations = entry.get("translations", {})
        text = (translations.get(locale) or {}).get("detailedExplanation", "")
        if not text:
            text = (translations.get("en") or {}).get("detailedExplanation", "")
            if text and warn and locale != "en":
                print(f"warning: KB entry {entry['id']!r} has no {locale} "
                      f"detailedExplanation — falling back to en (cross-language "
                      f"overlap is not meaningful)", file=sys.stderr)
        if text:
            kb[entry["id"]] = text
    return kb


def score_fixture(fixture_path: Path, kb_dir: Path, locale: str) -> int:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    items = data["items"]
    if not items:
        print(f"error: {fixture_path} contains no items", file=sys.stderr)
        return 1
    kb = load_kb(kb_dir, locale, warn=True)

    agree = 0
    rows = []
    for item in items:
        answer = item["ai_explanations"].get(locale)
        if not answer:
            answer = item["ai_explanations"]["en"]
            print(f"warning: {item['id']} has no {locale} answer — falling back "
                  f"to en (cross-language overlap is not meaningful)", file=sys.stderr)
        kb_text = kb.get(item.get("kb_concept_id") or "")
        if kb_text is not None and len(tokenize_v0(kb_text)) < MIN_KB_TOKENS:
            print(f"warning: {item['id']} KB entry tokenizes to fewer than "
                  f"{MIN_KB_TOKENS} tokens in {locale} — overlap denominator too "
                  f"small to be meaningful", file=sys.stderr)
        tier, score = classify(kb_text, answer)
        flagged = asserts_forbidden_claim(answer, item.get("forbidden_claims", []), locale)
        if flagged and kb_text is not None and tier == TIER_VALIDATED:
            # Forbidden-claim gate takes precedence over overlap (docs/SCHEMA.md):
            # an answer asserting a misconception is never validated.
            tier = TIER_LOW_CONFIDENCE
        expected = item["expected_source"]
        ok = tier == expected
        agree += ok
        blocked = ""
        if flagged:
            blocked = "BLOCKED: " + (flagged[:40] + "…" if len(flagged) > 40 else flagged)
        rows.append((item["id"], item["category"], f"{score:.2f}", tier, expected,
                     "OK" if ok else "MISS", blocked))

    widths = [6, 22, 5, 24, 24, 4, 20]
    header = ("id", "category", "ovl", "guard tier", "ground truth", "", "forbidden-claim gate")
    print(f"\nHallucinationGuard V0 baseline — {fixture_path.name} · locale={locale}\n")
    for row in (header, *rows):
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    n = len(items)
    print(f"\n  tier agreement with ground truth: {agree}/{n} = {agree / n:.4f}")
    if n >= 20:
        print(f"  (full 120-item set, documented production baseline: 0.9083 = 109/120)")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="benchmark fixture JSON")
    parser.add_argument("--locale", default="en", choices=LOCALES)
    parser.add_argument("--kb", type=Path, default=None,
                        help="KB directory (default: fixtures/kb-examples next to fixture)")
    args = parser.parse_args()
    kb_dir = args.kb or args.fixture.resolve().parent.parent / "kb-examples"
    if not kb_dir.is_dir():
        print(f"KB directory not found: {kb_dir}", file=sys.stderr)
        return 1
    try:
        return score_fixture(args.fixture, kb_dir, args.locale)
    except OSError as e:
        print(f"error: cannot read fixture {args.fixture}: {e}", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"error: {args.fixture} is not valid JSON: {e}", file=sys.stderr)
    except KeyError as e:
        print(f"error: {args.fixture} is missing required field {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Unit + regression tests for the V0 reference scorer. Stdlib only; exits
non-zero on violation.

  python3 tests/test_scorer.py

Covers: tokenizer (incl. CJK character bigrams), overlap metric, four-tier
classification, the forbidden-claim gate (locale claims, negation-cue guard,
pseudo-negation, mention-vs-assertion framing), CLI error paths, and a pinned
per-locale regression matrix over the shipped 20-item sample so any tier or
gate drift fails CI instead of passing silently.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))
from score_v0 import (  # noqa: E402
    TIER_LOW_CONFIDENCE, TIER_NO_KB, TIER_VALIDATED, TIER_VERIFIED,
    asserts_forbidden_claim, classify, load_kb, one_sided_overlap, tokenize_v0,
)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def unit_tests() -> None:
    # Tokenizer V0.1 — Latin behavior unchanged from V0 (punctuation attaches).
    check(tokenize_v0("Hello World.") == {"hello", "world."},
          "tokenize: latin whitespace split with attached punctuation")
    # Unspaced CJK becomes character bigrams instead of one giant token.
    ja = tokenize_v0("重ね合わせは基礎原理")
    check(len(ja) > 3, f"tokenize: ja must yield bigrams, got {ja}")
    check("重ね" in ja and "原理" in ja, "tokenize: expected ja bigrams missing")
    # Korean keeps whitespace eojeol tokens (documented V0 policy).
    check(tokenize_v0("중첩 상태") == {"중첩", "상태"}, "tokenize: ko eojeol tokens")
    # Mixed-script chunk splits into CJK bigrams + non-CJK remainder.
    mixed = tokenize_v0("状態state")
    check("状態" in mixed and "state" in mixed, f"tokenize: mixed-script split, got {mixed}")

    # Overlap metric.
    check(one_sided_overlap("", "anything") == 0.0, "overlap: empty KB -> 0.0")
    check(one_sided_overlap("a b c", "a b c d") == 1.0, "overlap: full recall -> 1.0")

    # Four-tier classification.
    check(classify(None, "x")[0] == TIER_NO_KB, "classify: no KB -> aiNoKBMatch")
    kb = "Measurement collapses the state onto an eigenstate of the observable."
    check(classify(kb, kb)[0] == TIER_VERIFIED,
          "classify: verbatim KB answer -> verifiedKnowledgeBase")
    check(classify(kb, "  " + kb + "\n")[0] == TIER_VERIFIED,
          "classify: whitespace-normalized verbatim -> verifiedKnowledgeBase")
    check(classify(kb, "Measurement collapses the state onto an eigenstate.")[0]
          == TIER_VALIDATED, "classify: high overlap -> aiValidatedAgainstKB")
    check(classify(kb, "Bananas are yellow.")[0] == TIER_LOW_CONFIDENCE,
          "classify: low overlap -> aiWithLowConfidence")
    # Containing every KB token plus extras is NOT verbatim service.
    check(classify("a b", "a b plus hallucinated extras")[0] == TIER_VALIDATED,
          "classify: superset answer must not count as verifiedKnowledgeBase")

    # Forbidden-claim gate: directionality.
    claim = ["Entanglement transmits information faster than light"]
    check(asserts_forbidden_claim(
        "Entanglement transmits information faster than light, enabling FTL phones.",
        claim) == claim[0], "gate: plain assertion must fire")
    check(asserts_forbidden_claim(
        "Entanglement never transmits information faster than light; no signal is carried.",
        claim) is None, "gate: correct denial must NOT fire (negation cue)")
    # Pseudo-negation: a negatively phrased claim, echoed, is an assertion.
    neg_claim = ["No further cryptanalysis is needed after NIST standardization"]
    check(asserts_forbidden_claim(
        "After NIST standardization no further cryptanalysis is needed at all.",
        neg_claim) == neg_claim[0], "gate: echoed negative claim must fire")
    # Mention-vs-assertion: warning frames are not assertions.
    check(asserts_forbidden_claim(
        "Ignoring caveats leads to the common overstatement that entanglement "
        "transmits information faster than light.", claim) is None,
        "gate: misconception mentioned inside a warning frame must NOT fire")
    # Locale handling: a legacy English-only list is inert for other locales.
    check(asserts_forbidden_claim("얽힘은 정보를 빛보다 빠르게 전송한다", claim, "ko") is None,
          "gate: legacy en-only list must not apply to non-en answers")
    loc_claims = {"en": claim, "ko": ["얽힘은 정보를 빛보다 빠르게 전송한다"]}
    check(asserts_forbidden_claim(
        "얽힘은 정보를 빛보다 빠르게 전송한다", loc_claims, "ko") is not None,
        "gate: localized dict claims must fire for ko")


# Pinned per-locale results on the shipped sample (schema v3.1 fixture,
# tokenizer V0.1). ko/ja/zh-Hans misses on HG004/HG005 and the de miss on HG005
# are the documented V0 naivety (agglutination / bigram granularity /
# punctuation); the blocked set is identical across locales by construction —
# the 7 recorded answers of each item are parallel translations asserting the
# same misconception.
EXPECTED_AGREEMENT = {"en": 20, "ko": 18, "ja": 18, "zh-Hans": 18, "es": 20, "fr": 20, "de": 19}
EXPECTED_BLOCKED = {"HG002", "HG023", "HG031", "HG039", "HG051", "HG073", "HG115"}
EXPECTED_MISSES = {
    "en": set(), "es": set(), "fr": set(),
    "ko": {"HG004", "HG005"}, "ja": {"HG004", "HG005"},
    "zh-Hans": {"HG004", "HG005"}, "de": {"HG005"},
}


def regression_matrix() -> None:
    from score_v0 import THRESHOLD  # noqa: F401  (import guards against rename)
    fixture = json.loads(
        (REPO / "fixtures/samples/hg120_sample_20.json").read_text(encoding="utf-8"))
    items = fixture["items"]
    for locale, expected_agree in EXPECTED_AGREEMENT.items():
        kb = load_kb(REPO / "fixtures/kb-examples", locale)
        agree, blocked, misses = 0, set(), set()
        for item in items:
            answer = item["ai_explanations"].get(locale) or item["ai_explanations"]["en"]
            kb_text = kb.get(item.get("kb_concept_id") or "")
            tier, _ = classify(kb_text, answer)
            flagged = asserts_forbidden_claim(
                answer, item.get("forbidden_claims", []), locale)
            if flagged and kb_text is not None and tier == TIER_VALIDATED:
                tier = TIER_LOW_CONFIDENCE
            if flagged:
                blocked.add(item["id"])
            if tier == item["expected_source"]:
                agree += 1
            else:
                misses.add(item["id"])
        check(agree == expected_agree,
              f"[{locale}] agreement {agree}/20 != pinned {expected_agree}/20")
        check(blocked == EXPECTED_BLOCKED,
              f"[{locale}] blocked set {sorted(blocked)} != pinned {sorted(EXPECTED_BLOCKED)}")
        check(misses == EXPECTED_MISSES[locale],
              f"[{locale}] miss set {sorted(misses)} != pinned {sorted(EXPECTED_MISSES[locale])}")


def cli_tests() -> None:
    scorer = str(REPO / "reference/score_v0.py")
    kb_dir = str(REPO / "fixtures/kb-examples")

    r = subprocess.run([sys.executable, scorer, "/nonexistent/fixture.json",
                        "--kb", kb_dir], capture_output=True, text=True)
    check(r.returncode == 1 and "cannot read fixture" in r.stderr,
          f"cli: missing fixture must exit 1 with diagnostic, got {r.returncode}: {r.stderr!r}")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        r = subprocess.run([sys.executable, scorer, str(bad), "--kb", kb_dir],
                           capture_output=True, text=True)
        check(r.returncode == 1 and "not valid JSON" in r.stderr,
              f"cli: invalid JSON must exit 1 with diagnostic, got {r.returncode}: {r.stderr!r}")

        empty = Path(td) / "empty.json"
        empty.write_text('{"items": []}', encoding="utf-8")
        r = subprocess.run([sys.executable, scorer, str(empty), "--kb", kb_dir],
                           capture_output=True, text=True)
        check(r.returncode == 1 and "no items" in r.stderr,
              f"cli: empty items must exit 1 with diagnostic, got {r.returncode}: {r.stderr!r}")

    # Bare-filename invocation from inside fixtures/samples must find the KB
    # via the resolved default path (fixture.resolve().parent.parent).
    r = subprocess.run([sys.executable, scorer, "hg120_sample_20.json", "--locale", "en"],
                       capture_output=True, text=True,
                       cwd=str(REPO / "fixtures/samples"))
    check(r.returncode == 0 and "20/20" in r.stdout,
          f"cli: bare-filename fixture must resolve default KB dir, got {r.returncode}")


def main() -> int:
    unit_tests()
    regression_matrix()
    cli_tests()
    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all scorer checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

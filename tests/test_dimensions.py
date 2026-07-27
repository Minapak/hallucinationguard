#!/usr/bin/env python3
"""Unit + regression tests for the multi-dimension layer (dimensions_v1).
Stdlib only; exits non-zero on violation.

  python3 tests/test_dimensions.py

Covers: structured-citation extraction and verification against the KB
reference corpus, the depth profile, and a pinned per-locale regression over
the 6-item HGX provenance extension fixture (computed flags must equal
expected_flags exactly; tiers pinned per locale).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))
import dimensions_v1 as dv  # noqa: E402
from score_v0 import TIER_LOW_CONFIDENCE, TIER_VALIDATED  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def unit_tests() -> None:
    # Structured-identifier extraction (DOI / arXiv / FIPS), normalization.
    cites = dv.extract_citations(
        "See doi:10.1103/PhysRevLett.79.325 and https://doi.org/10.1017/CBO9780511976667, "
        "also arXiv:2606.18119 and FIPS 203.")
    check("doi:10.1103/physrevlett.79.325" in cites, f"extract: PRL DOI missing from {cites}")
    check("doi:10.1017/cbo9780511976667" in cites, "extract: doi.org URL form missing")
    check("arxiv:2606.18119" in cites, "extract: arXiv id missing")
    check("fips 203" in cites, "extract: FIPS number missing")
    check(dv.extract_citations("no identifiers here, Smith et al. (2020)") == [],
          "extract: prose citations are out of scope by design")

    entries = dv.load_kb_entries(REPO / "fixtures/kb-examples")

    # Verification against a real KB reference corpus: grover's real DOI is
    # known; a fabricated PhysRevX DOI is not.
    grover = entries["grover_algorithm"]
    unknown, known = dv.unverified_citations(
        "Proven in Grover (1997), doi:10.1103/PhysRevLett.79.325.", grover)
    check(unknown == [] and len(known) == 1,
          f"citations: real Grover DOI must verify, got unknown={unknown}")
    unknown, known = dv.unverified_citations(
        "Proven in Grover & Chuang (2003), doi:10.1103/PhysRevX.99.123456.", grover)
    check(unknown == ["doi:10.1103/physrevx.99.123456"],
          f"citations: fabricated DOI must be unknown, got {unknown}")
    # ml_kem's en text names FIPS 203 inline; FIPS 209 exists nowhere.
    mlkem = entries["ml_kem"]
    check(dv.unverified_citations("Standardized as FIPS 203.", mlkem)[0] == [],
          "citations: FIPS 203 is in the ml_kem corpus")
    check(dv.unverified_citations("Standardized as FIPS 209.", mlkem)[0] == ["fips 209"],
          "citations: FIPS 209 must be unknown")
    # No KB entry -> no corpus -> unverifiable but never flagged.
    check(dv.unverified_citations("See doi:10.1000/xyz123.", None) == ([], ["doi:10.1000/xyz123"]),
          "citations: no KB entry means unverifiable, not hallucinated")

    # Depth profile: mentioned-but-undeveloped vs developed.
    hard = ["threshold theorem", "syndrome"]
    filler = "Errors happen. " * 30
    inverted = filler + "The threshold theorem and syndrome extraction handle the rest."
    prof = dv.depth_profile(inverted, hard)
    check(prof is not None and prof[0] == 1.0 and prof[1] < dv.DEPTH_SHARE_FLOOR,
          f"depth: inverted text must be full-coverage, low-share, got {prof}")
    balanced = ("The threshold theorem sets the critical rate. "
                "Syndrome extraction reveals errors without disturbing the state. "
                "Repeating syndrome measurements suppresses logical errors.")
    prof = dv.depth_profile(balanced, hard)
    check(prof is not None and prof[1] >= dv.DEPTH_SHARE_FLOOR,
          f"depth: balanced text must have high share, got {prof}")
    check(dv.depth_profile("anything", []) is None, "depth: unannotated -> None")


# Pinned regression for the HGX extension fixture. Flags must match
# expected_flags exactly in every locale; tiers are pinned as measured
# (expected_source follows the en-referenced convention, so ko/zh-Hans/de
# legitimately sit one tier lower on HGX03 — same convention as HG120).
V = TIER_VALIDATED
L = TIER_LOW_CONFIDENCE
EXPECTED_TIERS = {
    #            HGX01 HGX02 HGX03 HGX04 HGX05 HGX06
    "en":       (V, V, V, L, L, L),
    "ko":       (V, V, L, L, L, L),
    "ja":       (V, V, V, L, L, L),
    "zh-Hans":  (V, V, L, L, L, L),
    "es":       (V, V, V, L, L, L),
    "fr":       (V, V, V, L, L, L),
    "de":       (V, V, L, L, L, L),
}
ITEM_ORDER = ["HGX01", "HGX02", "HGX03", "HGX04", "HGX05", "HGX06"]


def regression() -> None:
    fixture = json.loads(
        (REPO / "fixtures/samples/hgx_provenance_6.json").read_text(encoding="utf-8"))
    items = {it["id"]: it for it in fixture["items"]}
    check(list(items) == ITEM_ORDER, f"fixture item order changed: {list(items)}")
    entries = dv.load_kb_entries(REPO / "fixtures/kb-examples")
    for locale, tiers in EXPECTED_TIERS.items():
        for hid, expected_tier in zip(ITEM_ORDER, tiers):
            it = items[hid]
            ans = it["ai_explanations"][locale]
            entry = entries.get(it["kb_concept_id"])
            tier, _ = dv.classify(dv.kb_text_for(entry, locale), ans)
            check(tier == expected_tier,
                  f"[{locale}] {hid} tier {tier} != pinned {expected_tier}")
            flags = dv.computed_flags(it, ans, entry, locale)
            check(flags == set(it["expected_flags"]),
                  f"[{locale}] {hid} flags {sorted(flags)} != expected {it['expected_flags']}")


def main() -> int:
    unit_tests()
    regression()
    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all dimension checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

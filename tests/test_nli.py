#!/usr/bin/env python3
"""NLI harness tests — SKIPPED (exit 0) when torch/transformers are absent.

The default CI matrix stays zero-dependency and never installs the harness
requirements, so this file must be a clean no-op there. With the harness venv
active (harness/requirements.txt), it verifies the M2 NLI backend end-to-end
against loose, stable pins:

  python3 tests/test_nli.py            # SKIP without deps
  .venv/bin/python tests/test_nli.py   # full run with deps

Pins are deliberately coarse (>=/<= bands, set membership) — exact
probabilities drift across torch/transformers versions; the published exact
numbers live in docs/NLI-GAP-2026-07-27.md with their version pins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))
sys.path.insert(0, str(REPO / "harness"))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    print("SKIP: torch/transformers not installed — NLI harness tests need "
          "harness/requirements.txt (the zero-dep CI matrix skips them by design)")
    raise SystemExit(0)

from nli_v1 import CONTRA_MIN, GATE_LOCALES, NLIChecker, nli_gate  # noqa: E402
from score_v0 import classify, load_kb  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def main() -> int:
    checker = NLIChecker()

    # 1. The headline pair: identical sentence entails; negated contradicts.
    a = "Measurement collapses the state."
    s_same = checker.scores(a, a)
    s_neg = checker.scores(a, "Measurement never collapses the state.")
    check(s_same["entailment"] > 0.85,
          f"identity pair must entail strongly, got {s_same['entailment']:.3f}")
    check(s_neg["contradiction"] > 0.90,
          f"negated pair must contradict strongly, got {s_neg['contradiction']:.3f}")

    # 2. HG002 en (the FTL misconception the lexical metric cannot see):
    #    sentence-by-sentence grounding must flag it.
    fixture = json.loads(
        (REPO / "fixtures/samples/hg120_sample_20.json").read_text(encoding="utf-8"))
    items = {it["id"]: it for it in fixture["items"]}
    kb = load_kb(REPO / "fixtures/kb-examples", "en")
    v = checker.check(kb["entanglement"], items["HG002"]["ai_explanations"]["en"])
    check(v.label == "nli_contradicted",
          f"HG002 en must be nli_contradicted, got {v.label}")
    check(v.max_conflict >= CONTRA_MIN,
          f"HG002 en max conflict {v.max_conflict:.2f} below threshold")

    # 3. Layering property (en): the union of the V0 forbidden-claim gate
    #    (blocks all 7 misconception items — pinned in test_scorer.py) and the
    #    NLI dimension must cover all 7; and NLI alone must catch at least 3
    #    (measured 4-6 per locale; loose floor so version drift can't flake).
    from score_v0 import asserts_forbidden_claim  # noqa: E402
    mis = ["HG002", "HG023", "HG031", "HG039", "HG051", "HG073", "HG115"]
    nli_caught, v0_caught = set(), set()
    for hid in mis:
        it = items[hid]
        ans = it["ai_explanations"]["en"]
        if asserts_forbidden_claim(ans, it["forbidden_claims"], "en"):
            v0_caught.add(hid)
        kb_text = kb.get(it.get("kb_concept_id") or "")
        if kb_text and checker.check(kb_text, ans).label == "nli_contradicted":
            nli_caught.add(hid)
    check(v0_caught == set(mis), f"V0 gate must block all 7, got {sorted(v0_caught)}")
    check(len(nli_caught) >= 3,
          f"NLI must catch >=3 misconceptions pattern-free, got {sorted(nli_caught)}")
    check((nli_caught | v0_caught) == set(mis), "layered union must cover all 7")

    # 4. V2 GATE — zero false blocks on faithful answers in the measured-clean
    #    scope, real blocks on wrong ones, and hard-off outside the scope.
    check(GATE_LOCALES == {"en", "es", "fr"},
          f"gate scope changed without a measurement pass: {sorted(GATE_LOCALES)}")
    good = ["HG001", "HG004", "HG005", "HG006"]
    for locale in ("en", "es", "fr"):
        kb_l = load_kb(REPO / "fixtures/kb-examples", locale)
        blocked, false_blocked = set(), set()
        for hid in mis + good:
            it = items[hid]
            ans = it["ai_explanations"][locale]
            kb_text = kb_l.get(it.get("kb_concept_id") or "")
            if not kb_text:
                continue
            tier, _ = classify(kb_text, ans)
            if nli_gate(tier, checker.check(kb_text, ans), locale):
                (false_blocked if hid in good else blocked).add(hid)
        check(false_blocked == set(),
              f"[{locale}] V2 gate FALSE-blocked faithful answers: {sorted(false_blocked)}")
        check(len(blocked) >= 3,
              f"[{locale}] V2 gate must block >=3 wrong answers, got {sorted(blocked)}")
    # Outside the measured-clean scope the gate must never fire, whatever the model says.
    kb_ko = load_kb(REPO / "fixtures/kb-examples", "ko")
    it = items["HG115"]
    ans_ko = it["ai_explanations"]["ko"]
    kb_text = kb_ko.get(it["kb_concept_id"])
    tier_ko, _ = classify(kb_text, ans_ko)
    check(nli_gate(tier_ko, checker.check(kb_text, ans_ko), "ko") is False,
          "gate must be hard-off outside GATE_LOCALES (ko)")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all NLI harness checks passed (NLI pattern-free catches: "
          f"{len(nli_caught)}/7 en; union with V0 gate: 7/7)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

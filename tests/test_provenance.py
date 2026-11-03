#!/usr/bin/env python3
"""Tests for reference/provenance_v1.py (Provenance Ink). Stdlib only.

  python3 tests/test_provenance.py

Pins: the five ink states and their precedence, the tier contract (the
whole-answer tier is score_v0's verdict, byte-for-byte), silence-as-a-state
(unanchored ≠ low), the forbidden-claim precedence, the injected-NLI
`disagree` rule (bidirectional, support-exempt, measured-clean locales only),
"no NLI → disagree can never fire", the receipt shape, the CLI, and a pinned
per-locale state matrix over the shipped 20-item sample.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))
from provenance_v1 import (  # noqa: E402
    ANCHOR_MIN, GATE_LOCALES, STATE_DISAGREE, STATE_LOW, STATE_UNANCHORED,
    STATE_VALIDATED, STATE_VERIFIED, STATES, anchor_for, provenance, sentence_state,
)
from score_v0 import (  # noqa: E402
    TIER_LOW_CONFIDENCE, TIER_NO_KB, TIER_VALIDATED, _sentences, asserts_forbidden_claim,
    classify, load_kb,
)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


KB = ("Superposition means a quantum system can exist in a coherent combination of basis states. "
      "Measurement collapses the state onto one outcome. "
      "Interference between components is the computational resource.")


def contradict_all(pairs):
    return [{"entailment": 0.01, "neutral": 0.01, "contradiction": 0.98} for _ in pairs]


def support_all(pairs):
    return [{"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01} for _ in pairs]


def one_sided(pairs):
    # Forward contradiction only: the reverse check (sentence as premise) returns neutral.
    out = []
    for p, h in pairs:
        fwd = p.startswith("Measurement") and "never" in h
        out.append({"entailment": 0.05, "neutral": 0.9, "contradiction": 0.05} if not fwd
                   else {"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90})
    return out


def unit_tests() -> None:
    # verbatim KB sentence → verified
    r = sentence_state("Measurement collapses the state onto one outcome.", _sentences(KB), "en")
    check(r["state"] == STATE_VERIFIED and r["overlap"] == 1.0, "verbatim KB sentence is verified")

    # paraphrase with ≥0.30 overlap → validated; anchor index points at the right KB sentence
    kbs = ["Superposition means a quantum system can exist in a coherent combination of basis states.",
           "Measurement collapses the state onto one outcome.",
           "Interference between components is the computational resource."]
    r = sentence_state("A quantum system can be in a coherent combination of basis states.", kbs, "en")
    check(r["state"] == STATE_VALIDATED and r["anchor"] == 0, f"paraphrase validated, anchor 0: {r}")

    # off-topic → unanchored (silence is a state, not low)
    r = sentence_state("The kitchen table is made of oak wood.", kbs, "en")
    check(r["state"] == STATE_UNANCHORED and "no_kb_anchor" in r["reasons"], f"off-topic is unanchored: {r}")

    # weak anchor (between ANCHOR_MIN and 0.30) → low, not unanchored
    r = sentence_state("The state is a resource here.", kbs, "en")
    check(ANCHOR_MIN <= r["overlap"] < 0.30 and r["state"] == STATE_LOW, f"weak anchor is low: {r}")

    # forbidden claim beats everything, even a high-overlap sentence
    claims = ["Measurement never collapses the state onto one outcome"]
    r = sentence_state("Measurement never collapses the state onto one outcome.", kbs, "en", claims)
    check(r["state"] == STATE_LOW and r["forbidden"], f"forbidden claim → low: {r}")

    # inversion WITHOUT NLI: lexical layer alone validates it (the documented blindness)
    inv = "Measurement never collapses the state onto one outcome."
    r = sentence_state(inv, kbs, "en")
    check(r["state"] == STATE_VALIDATED and r["nli"] is None, "no NLI → inversion passes lexically (blindness pinned)")
    check(STATE_DISAGREE not in {s["state"] for s in provenance(KB, inv)["sentences"]},
          "disagree can never fire without an NLI layer")

    # inversion WITH bidirectional NLI contradiction → disagree (en in scope)
    r = sentence_state(inv, kbs, "en", nli=contradict_all)
    check(r["state"] == STATE_DISAGREE and r["nli"]["conflict_rev"] >= 0.5, f"bidirectional contradiction → disagree: {r}")
    # same, out-of-scope locale → stays validated, reason recorded
    r = sentence_state(inv, kbs, "ko", nli=contradict_all)
    check(r["state"] == STATE_VALIDATED and "nli_contradiction_unmeasured_locale" in r["reasons"],
          f"unmeasured locale never disagrees: {r}")
    # one-sided contradiction (reverse neutral) → not disagree (rule R7)
    r = sentence_state(inv, kbs, "en", nli=one_sided)
    check(r["state"] == STATE_VALIDATED and r["nli"]["conflict_rev"] < 0.5, f"one-sided conflict is not disagree: {r}")
    # supported → validated with nli_supported
    r = sentence_state("A quantum system can be in a coherent combination of basis states.", kbs, "en", nli=support_all)
    check(r["state"] == STATE_VALIDATED and "nli_supported" in r["reasons"], "supported sentence stays validated")
    check(GATE_LOCALES == frozenset({"en", "es", "fr"}), "gate scope lockstep with harness/nli_v1.py")

    # anchor_for determinism
    check(anchor_for("Measurement collapses the state.", kbs) == anchor_for("Measurement collapses the state.", kbs),
          "anchor_for deterministic")
    check(anchor_for("zzz", kbs) == (None, 0.0), "anchor_for with no overlap")
    check(anchor_for("the is a of", kbs) == (None, 0.0), "function words alone never anchor")


def receipt_tests() -> None:
    ans = ("A quantum system can be in a coherent combination of basis states. "
           "Measurement never collapses the state onto one outcome. "
           "My cat likes sunny windows.")
    r = provenance(KB, ans, "en", nli=contradict_all)
    tier, ovl = classify(KB, ans)
    check(r["tier"] == tier and r["overlap"] == round(ovl, 4), "tier is score_v0's verdict, unchanged")
    check(r["version"] == "provenance-v1" and len(r["answer_sha256"]) == 64 and len(r["kb_sha256"]) == 64, "receipt hashes")
    check(r["checks"] == {"lexical": True, "forbidden_claim": True, "nli": True, "nli_scope": True}, f"checks: {r['checks']}")
    states = [s["state"] for s in r["sentences"]]
    check(states == [STATE_DISAGREE, STATE_DISAGREE, STATE_UNANCHORED] or states[2] == STATE_UNANCHORED,
          f"per-sentence states: {states}")
    check(sum(r["counts"].values()) == len(r["sentences"]) and set(r["counts"]) == set(STATES), "counts cover every sentence")
    check([s["index"] for s in r["sentences"]] == list(range(len(r["sentences"]))), "indexes sequential")

    # tier-vs-sentence agreement: a whole-answer LOW because of a forbidden claim marks that sentence low
    claims = ["Measurement never collapses the state onto one outcome"]
    r = provenance(KB, ans, "en", claims)
    check(r["tier"] == TIER_LOW_CONFIDENCE and r["blocked_claim"] and r["sentences"][1]["state"] == STATE_LOW,
          f"forbidden claim demotes tier and marks its sentence: {r['tier']} {[s['state'] for s in r['sentences']]}")
    check(asserts_forbidden_claim(ans, claims, "en") == r["blocked_claim"], "blocked_claim equals score_v0's")

    # no KB at all → tier aiNoKBMatch, every sentence unanchored with reason no_kb, nli check reported off
    r = provenance(None, ans, "en", nli=contradict_all)
    check(r["tier"] == TIER_NO_KB and all(s["state"] == STATE_UNANCHORED for s in r["sentences"])
          and r["checks"]["nli"] is False and r["kb_sha256"] is None, "no-KB receipt")

    # fragments are skipped
    r = provenance(KB, "Yes. Measurement collapses the state onto one outcome.", "en")
    check(len(r["sentences"]) == 1 and r["sentences"][0]["state"] == STATE_VERIFIED, "fragment skipped")

    # determinism
    check(provenance(KB, ans, "en") == provenance(KB, ans, "en"), "receipt deterministic")


def matrix_tests() -> None:
    """Pinned per-locale state totals over the 20-item sample (no NLI). Drift fails here."""
    fixture = REPO / "fixtures/samples/hg120_sample_20.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    pins_path = REPO / "tests/provenance_matrix_pins.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8")) if pins_path.exists() else None
    got: dict[str, dict[str, int]] = {}
    for loc in ["en", "ko", "ja", "zh-Hans", "es", "fr", "de"]:
        kb = load_kb(REPO / "fixtures/kb-examples", loc)
        tot = {st: 0 for st in STATES}
        for item in data["items"]:
            ans = item["ai_explanations"].get(loc) or item["ai_explanations"]["en"]
            r = provenance(kb.get(item.get("kb_concept_id") or ""), ans, loc, item.get("forbidden_claims", []))
            # contract: receipt tier == score_v0 tier + gate for every item
            t, _ = classify(kb.get(item.get("kb_concept_id") or ""), ans)
            fl = asserts_forbidden_claim(ans, item.get("forbidden_claims", []), loc)
            if fl and t == TIER_VALIDATED:
                t = TIER_LOW_CONFIDENCE
            check(r["tier"] == t, f"{item['id']} {loc}: receipt tier {r['tier']} != score_v0 {t}")
            check(STATE_DISAGREE not in {s["state"] for s in r["sentences"]}, f"{item['id']} {loc}: disagree without NLI")
            for st in STATES:
                tot[st] += r["counts"][st]
        got[loc] = tot
    if pins is None:
        pins_path.write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
        print(f"pinned matrix written to {pins_path}")
    else:
        check(got == pins, f"state matrix drift:\n got {json.dumps(got)}\n pin {json.dumps(pins)}")


def cli_tests() -> None:
    py = sys.executable
    out = subprocess.run([py, "reference/provenance_v1.py", "--kb", KB, "--answer",
                          "Measurement collapses the state onto one outcome. My cat sleeps.", "--json"],
                         cwd=REPO, capture_output=True, text=True)
    check(out.returncode == 0, f"cli --json exit {out.returncode}: {out.stderr}")
    j = json.loads(out.stdout)
    check(j["sentences"][0]["state"] == STATE_VERIFIED and j["sentences"][1]["state"] == STATE_UNANCHORED, "cli json states")
    out = subprocess.run([py, "reference/provenance_v1.py", "fixtures/samples/hg120_sample_20.json", "--locale", "ja"],
                         cwd=REPO, capture_output=True, text=True)
    check(out.returncode == 0 and "sentences" in out.stdout, f"cli fixture run: {out.stderr}")
    out = subprocess.run([py, "reference/provenance_v1.py"], cwd=REPO, capture_output=True, text=True)
    check(out.returncode == 2, "cli without args exits 2")


if __name__ == "__main__":
    unit_tests()
    receipt_tests()
    matrix_tests()
    cli_tests()
    if failures:
        print("\n".join(f"FAIL: {f}" for f in failures))
        sys.exit(1)
    print("all provenance checks passed")

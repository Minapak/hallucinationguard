#!/usr/bin/env python3
"""Unit + regression tests for the V0 reference scorer. Stdlib only; exits
non-zero on violation.

  python3 tests/test_scorer.py

Covers: tokenizer (incl. CJK character bigrams), overlap metric, four-tier
classification, the forbidden-claim gate (locale claims, negation-cue guard,
pseudo-negation, mention-vs-assertion framing), CLI error paths, and a pinned
per-locale regression matrix over the shipped 20-item sample so any tier or
gate drift fails CI instead of passing silently, plus a both-directions
matrix over the 3-item misconception extension (HGM01-03).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))
from score_v0 import (  # noqa: E402
    ATTRIBUTION_OVERLAP, TIER_LOW_CONFIDENCE, TIER_NO_KB, TIER_VALIDATED,
    TIER_VERIFIED, asserts_forbidden_claim, attribution_required, classify,
    load_kb, one_sided_overlap, tokenize_v0,
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
    # Formal copula negation (ko "아닙니다") and a comma-attached German
    # "nicht," ("bedeutet nicht, dass …") are denials too (added 2026-09-29).
    ko_claim = {"ko": ["중첩은 큐비트가 이미 정해진 값을 몰래 가지고 있는데 우리가 모를 뿐이라는 뜻이다"]}
    check(asserts_forbidden_claim(
        "중첩은 큐비트가 이미 정해진 값을 몰래 가지고 있는데 우리가 모를 뿐이라는 뜻이 아닙니다.",
        ko_claim, "ko") is None, "gate: ko 아닙니다 denial must NOT fire")
    de_claim = {"de": ["Superposition bedeutet, dass das Qubit insgeheim bereits einen festen Wert hat"]}
    check(asserts_forbidden_claim(
        "Superposition bedeutet nicht, dass das Qubit insgeheim bereits einen festen Wert hat.",
        de_claim, "de") is None, "gate: de 'nicht,' denial must NOT fire")

    # Attribution flag (provenance axis): fires only for generated answers that
    # nearly wholesale reproduce the KB; verbatim service has its own tier and
    # is exempt; the flag never touches the correctness tier.
    check(attribution_required(TIER_VALIDATED, ATTRIBUTION_OVERLAP),
          "attribution: VALIDATED at the threshold must flag")
    check(not attribution_required(TIER_VALIDATED, ATTRIBUTION_OVERLAP - 0.01),
          "attribution: below threshold must not flag")
    check(not attribution_required(TIER_VERIFIED, 1.0),
          "attribution: verbatim-served KB answers are exempt")
    check(not attribution_required(TIER_LOW_CONFIDENCE, 0.99),
          "attribution: only validated answers are in scope")
    kb2 = "a b c d e f g h i j"
    t2, s2 = classify(kb2, "intro sentence first. " + kb2)
    check(t2 == TIER_VALIDATED and attribution_required(t2, s2),
          "attribution: KB text plus padding must flag as near-verbatim")


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
        agree, blocked, misses, attributed = 0, set(), set(), set()
        for item in items:
            answer = item["ai_explanations"].get(locale) or item["ai_explanations"]["en"]
            kb_text = kb.get(item.get("kb_concept_id") or "")
            tier, score = classify(kb_text, answer)
            flagged = asserts_forbidden_claim(
                answer, item.get("forbidden_claims", []), locale)
            if flagged and kb_text is not None and tier == TIER_VALIDATED:
                tier = TIER_LOW_CONFIDENCE
            if flagged:
                blocked.add(item["id"])
            if attribution_required(tier, score):
                attributed.add(item["id"])
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
        # HG120 answers are recorded paraphrases (max overlap 0.53 across all
        # locales) — the attribution flag must never fire on the shipped sample.
        check(attributed == set(),
              f"[{locale}] attribution flag fired unexpectedly: {sorted(attributed)}")


# Misconception extension (fixtures/samples/hgm_misconceptions_3.json, synthetic,
# not HG120). These claims are unioned into the consuming apps' production gate,
# so besides "fires on the wrong answer" we pin the other direction: a correct
# denial of each claim, and correct explanations that share its vocabulary, must
# stay silent in every locale. The ko denials deliberately use the formal
# copula negation "아닙니다" and the de ones "nicht, dass" — both were missing
# from the negation-cue tables until 2026-09-29.
HGM_DENIALS = {
    "en": [
        "A qubit is not simply both 0 and 1 at the same time.",
        "A qubit is NOT like a spinning coin that is both heads and tails at once.",
        "Superposition does not mean the qubit secretly already has a definite value we are merely unaware of.",
        "A common misconception is that a qubit in superposition is simply both 0 and 1 at the same time.",
    ],
    "ko": [
        "중첩 상태의 큐비트는 그냥 0과 1이 동시에 있는 것이 아닙니다.",
        "큐비트는 앞면과 뒷면이 동시에 되는 회전하는 동전과 같지 않습니다.",
        "중첩은 큐비트가 이미 정해진 값을 몰래 가지고 있는데 우리가 모를 뿐이라는 뜻이 아닙니다.",
    ],
    "ja": [
        "重ね合わせ状態の量子ビットは単に0と1が同時に存在しているわけではありません。",
        "量子ビットは表と裏が同時に成り立つ回転するコインのようなものではありません。",
        "重ね合わせとは、量子ビットがすでに決まった値をひそかに持っていて、私たちが知らずにいるだけという意味ではありません。",
    ],
    "zh-Hans": [
        "处于叠加态的量子比特并非就是同时既是0又是1。",
        "量子比特并不像一枚同时是正面和反面的旋转硬币。",
        "叠加并非意味着量子比特早已暗中具有确定的值，只是我们不知道。",
    ],
    "es": [
        "Un qubit en superposición no es simplemente 0 y 1 al mismo tiempo.",
        "Un qubit no equivale a una moneda girando, cara y cruz simultáneamente.",
        "La superposición no significa que el qubit ya tiene en secreto un valor definido que simplemente desconocemos.",
    ],
    "fr": [
        "En superposition, un qubit ne vaut pas tout bêtement 0 et 1 simultanément.",
        "Un qubit n'est pas comme une pièce qui tourne, à la fois pile et face.",
        "La superposition ne signifie pas que le qubit possède déjà en secret une valeur définie que nous ignorons simplement.",
    ],
    "de": [
        "Ein Qubit in Superposition ist nicht einfach gleichzeitig 0 und 1.",
        "Ein Qubit ist nicht wie eine sich drehende Münze, die gleichzeitig Kopf und Zahl ist.",
        "Superposition bedeutet nicht, dass das Qubit insgeheim bereits einen festen Wert hat, der uns lediglich unbekannt ist.",
    ],
}

HGM_CORRECT = {
    "en": [
        "A qubit's state is a combination α|0⟩ + β|1⟩ of two complex amplitudes. Measuring it still gives just 0 or 1, with P(0) = |α|² and P(1) = |β|².",
        "A qubit in superposition is described by amplitudes for 0 and 1, and measuring it gives 0 or 1 at random with those probabilities.",
        "Superposition means the qubit is in a definite state α|0⟩ + β|1⟩; the value 0 or 1 appears only when we measure, with probabilities |α|² and |β|².",
        "A classical coin shows heads or tails. A qubit has amplitudes that can cancel, which a coin cannot do.",
        "Like light polarized at 45°, a qubit in superposition points in one definite direction between 0 and 1; a measurement gives 0 or 1, 50/50.",
    ],
    "ko": [
        "큐비트의 상태는 두 복소 진폭의 조합 α|0⟩ + β|1⟩입니다. 측정하면 여전히 0 또는 1만 나오며, P(0) = |α|², P(1) = |β|²입니다.",
        "중첩 상태의 큐비트는 0과 1의 진폭으로 기술되며, 측정하면 그 확률에 따라 0 또는 1이 나옵니다.",
        "45°로 편광된 빛처럼, 중첩 상태의 큐비트는 0과 1 사이의 한 방향을 가리킵니다. 측정하면 0 또는 1이 50:50으로 나옵니다.",
        "고전 동전은 앞면 또는 뒷면을 보여 줍니다. 큐비트의 진폭은 상쇄될 수 있습니다.",
    ],
    "ja": [
        "量子ビットの状態は2つの複素振幅の結合 α|0⟩ + β|1⟩ です。測定してもやはり0か1しか出ず、P(0) = |α|²、P(1) = |β|² です。",
        "重ね合わせ状態の量子ビットは0と1の振幅で記述され、測定するとその確率で0か1が出ます。",
        "45°に偏光した光のように、重ね合わせ状態の量子ビットは0と1の間の1つの方向を指します。",
    ],
    "zh-Hans": [
        "量子比特的状态是两个复振幅的组合 α|0⟩ + β|1⟩。测量时仍然只得到 0 或 1，且 P(0) = |α|²、P(1) = |β|²。",
        "处于叠加态的量子比特由0和1的振幅描述，测量时按这些概率得到0或1。",
        "就像45°偏振的光，处于叠加态的量子比特指向0和1之间的一个确定方向。",
    ],
    "es": [
        "El estado de un qubit es una combinación α|0⟩ + β|1⟩ de dos amplitudes complejas. Al medirlo sigue dando solo 0 o 1, con P(0) = |α|² y P(1) = |β|².",
        "Un qubit en superposición se describe con amplitudes para 0 y 1, y al medirlo da 0 o 1 con esas probabilidades.",
        "Una moneda clásica muestra cara o cruz; las amplitudes de un qubit pueden cancelarse.",
    ],
    "fr": [
        "L'état d'un qubit est une combinaison α|0⟩ + β|1⟩ de deux amplitudes complexes. Sa mesure donne toujours 0 ou 1, avec P(0) = |α|² et P(1) = |β|².",
        "Un qubit en superposition est décrit par des amplitudes pour 0 et 1, et sa mesure donne 0 ou 1 avec ces probabilités.",
        "Une pièce classique montre pile ou face ; les amplitudes d'un qubit peuvent s'annuler.",
    ],
    "de": [
        "Der Zustand eines Qubits ist eine Kombination α|0⟩ + β|1⟩ aus zwei komplexen Amplituden. Die Messung liefert trotzdem nur 0 oder 1, mit P(0) = |α|² und P(1) = |β|².",
        "Ein Qubit in Superposition wird durch Amplituden für 0 und 1 beschrieben, und die Messung liefert 0 oder 1 mit diesen Wahrscheinlichkeiten.",
        "Eine klassische Münze zeigt Kopf oder Zahl; die Amplituden eines Qubits können sich auslöschen.",
    ],
}


def misconception_matrix() -> None:
    fixture = json.loads(
        (REPO / "fixtures/samples/hgm_misconceptions_3.json").read_text(encoding="utf-8"))
    items = fixture["items"]
    locales = fixture["locales"]
    for locale in locales:
        kb = load_kb(REPO / "fixtures/kb-examples", locale)
        union = {locale: [c for it in items for c in it["forbidden_claims"][locale]]}
        for item in items:
            answer = item["ai_explanations"][locale]
            flagged = asserts_forbidden_claim(answer, item["forbidden_claims"], locale)
            check(flagged is not None,
                  f"[{locale}] {item['id']}: wrong answer must be blocked by its own claim")
            tier, _ = classify(kb.get(item["kb_concept_id"]), answer)
            if flagged and tier == TIER_VALIDATED:
                tier = TIER_LOW_CONFIDENCE
            check(tier == item["expected_source"],
                  f"[{locale}] {item['id']}: tier {tier} != expected {item['expected_source']}")
        for text in HGM_DENIALS[locale]:
            hit = asserts_forbidden_claim(text, union, locale)
            check(hit is None, f"[{locale}] correct denial blocked: {text!r} -> {hit!r}")
        for text in HGM_CORRECT[locale]:
            hit = asserts_forbidden_claim(text, union, locale)
            check(hit is None, f"[{locale}] correct explanation blocked: {text!r} -> {hit!r}")


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
    misconception_matrix()
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

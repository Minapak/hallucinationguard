# Test guide

Everything runs on stock CPython 3.9+ with zero dependencies — no pytest, no venv needed.
CI (`.github/workflows/ci.yml`) runs exactly the commands below on Python 3.9 and 3.14.

```bash
# 1. Fixture schema + certainty/parity/orthography gates
python3 tests/test_fixtures.py

# 2. Scorer unit tests + pinned per-locale regression matrix + CLI error paths
#    + both-directions misconception matrix (HGM extension)
python3 tests/test_scorer.py

# 3. Dimension-layer unit tests + per-locale flag & tier pins (HGX extension)
python3 tests/test_dimensions.py

# 4. End-to-end runs, all 7 locales, all three fixtures
for loc in en ko ja zh-Hans es fr de; do
  python3 reference/score_v0.py fixtures/samples/hg120_sample_20.json --locale "$loc" > /dev/null
  python3 reference/score_v0.py fixtures/samples/hgm_misconceptions_3.json --locale "$loc" > /dev/null
  python3 reference/dimensions_v1.py fixtures/samples/hgx_provenance_6.json --locale "$loc" --check-flags > /dev/null
done

# 5. Demo
python3 demo/metric_blindness_demo.py

# 6. OPTIONAL — NLI harness tests (M2 layer; skips with exit 0 when torch is absent,
#    so the zero-dep CI matrix never needs them; full run requires harness/requirements.txt)
python3 tests/test_nli.py
```

Both test scripts print `all … checks passed` and exit 0 on success; on violation they list every
failure and exit 1 (CPython exit-status convention; argparse usage errors exit 2).

## What each layer catches

| Layer | Guards against |
|---|---|
| `test_fixtures.py` | schema drift, missing locales, unmeaningful `sourceRef` on `established`, undated `timebound`, metadata corruption (preskill_part/ai_confidence bounds, kb_category mismatch), `expected_keywords` absent from the en answer, non-localized `forbidden_claims`, stub translations (parity floors: es/fr/de ≥ 50%, ko ≥ 40%, ja ≥ 25%, zh-Hans ≥ 18% of en chars), unaccented es/fr paragraphs, sample-distribution/category miscounts |
| `test_scorer.py` unit tests | tokenizer regressions (Latin punctuation attachment is *asserted*, CJK bigrams required, ko eojeol preserved), overlap edge cases, four-tier classification incl. verbatim-vs-superset distinction, gate directionality (assertion fires; denial, mention-frame, legacy-list-on-non-en do not; pseudo-negation still fires) |
| `test_scorer.py` regression matrix | any change to tier agreement (en/es/fr 20/20 · de 19/20 · ko/ja/zh-Hans 18/20), per-locale miss sets, or the blocked set {HG002, HG023, HG031, HG039, HG051, HG073, HG115} — identical across all 7 locales by construction |
| `test_scorer.py` CLI tests | missing/malformed/empty fixture diagnostics + exit codes, bare-filename `--kb` default resolution |
| `test_fixtures.py` (v3.2 block) | HGX extension schema: HGX id namespace, `adversarial_class` values, `expected_flags` within the known-flag set, 7-locale `hard_step_concepts` on depth items |
| `test_fixtures.py` (HGM block) | misconception extension: schema v3.1, HGM id namespace, tier/KB/metadata bounds, `forbidden_claims` in all 7 locales, and **no claim may contain a negation cue** (pseudo-negation would switch off the denial guard for that claim) |
| `test_scorer.py` misconception matrix | HGM01–03 × 7 locales in both directions: each claim fires on its authored wrong answer; correct denials (ko `아닙니다`, de `nicht, dass`) and correct explanations sharing its vocabulary stay silent against the union of all HGM claims. Also unit-pins the ko `아닙` / de `" nicht,"` negation cues added in 0.5.1 |
| `test_scorer.py` attribution | flag threshold edges, verbatim-service exemption, KB-plus-padding case; HG120 sample pinned attribution-clean in all 7 locales (max overlap 0.53 < 0.85) |
| `test_dimensions.py` | citation extraction/verification (real Grover DOI verifies; fabricated `PhysRevX` DOI and `FIPS 209` flag; no-KB items are unverifiable-not-flagged), depth profile (inverted vs balanced), and per-locale pins: computed flags == `expected_flags` and tiers match `EXPECTED_TIERS` for all 6 HGX items × 7 locales |
| `test_nli.py` (optional, M2) | headline-pair separation (identity entails > 0.85, negation contradicts > 0.90), the HG002 pattern-free catch, the layered-union property (V0 gate ∪ NLI = 7/7 en), and the **V2 gate pins**: `GATE_LOCALES == {en, es, fr}` (changing the scope requires a measurement pass), zero false blocks + ≥3 catches per active locale, hard-off outside the scope — drift-tolerant; exact numbers live in docs/NLI-GAP-2026-07-27.md with version pins |

## Updating the pins

The pinned numbers in `tests/test_scorer.py` (`EXPECTED_AGREEMENT`, `EXPECTED_BLOCKED`,
`EXPECTED_MISSES`) and `tests/test_dimensions.py` (`EXPECTED_TIERS`, per-item
`expected_flags`) are a *ratchet*, not a target. If a legitimate change moves them
(new fixture items, tokenizer change, gate tuning):

1. Re-run the matrix and inspect **which** items flipped and **why** — a flip must be explainable
   in terms of the change you made, not accepted because the total looks close.
2. Update the pins **and** the README results table **and** note the move in `CHANGELOG.md` in the
   same commit, including the direction of the trade-off (see the 0.2.0 entry's ja/zh note for the
   expected style: a number that goes down for an honest reason is stated as such).
3. Never weaken a pin to make CI pass on an unexplained flip.

## Known-limitation tests

Some tests assert *documented naivety* rather than desirable behavior — e.g.
`tokenize: latin whitespace split with attached punctuation` asserts `"world."` stays one token.
These exist so that "fixing" a documented limitation is a conscious schema/docs decision
(it changes the published baseline), never an accidental side effect.

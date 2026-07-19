# Test guide

Everything runs on stock CPython 3.9+ with zero dependencies — no pytest, no venv needed.
CI (`.github/workflows/ci.yml`) runs exactly the commands below on Python 3.9 and 3.14.

```bash
# 1. Fixture schema + certainty/parity/orthography gates
python3 tests/test_fixtures.py

# 2. Scorer unit tests + pinned per-locale regression matrix + CLI error paths
python3 tests/test_scorer.py

# 3. End-to-end scorer run, all 7 locales
for loc in en ko ja zh-Hans es fr de; do
  python3 reference/score_v0.py fixtures/samples/hg120_sample_20.json --locale "$loc" > /dev/null
done

# 4. Demo
python3 demo/metric_blindness_demo.py
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

## Updating the pins

The pinned numbers in `tests/test_scorer.py` (`EXPECTED_AGREEMENT`, `EXPECTED_BLOCKED`,
`EXPECTED_MISSES`) are a *ratchet*, not a target. If a legitimate change moves them
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

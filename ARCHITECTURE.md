# Architecture

This repository is a **benchmark artifact + reference implementation**, not a service. There is no
deployment surface, no runtime infrastructure, and no ops/monitoring component here — the production
guard runs inside the SwiftQuantum app's commercial pipeline, from which this public artifact is
being extracted (see [README — Provenance](README.md#provenance--transparency)). Everything in this
repo runs offline with zero dependencies on CPython 3.9+.

```
hallucinationguard/
├── fixtures/
│   ├── samples/hg120_sample_20.json   # 20 of 120 benchmark items (schema hg_benchmark_item_v3.1)
│   │                                  #   7-locale prompts + recorded model answers (frozen artifacts),
│   │                                  #   7-locale forbidden_claims, ground-truth tier per item
│   └── kb-examples/*.json             # 12 verified KB concepts: certainty grade, DOI sourceRef,
│                                      #   7-locale translations, verifier record
├── reference/score_v0.py              # the production V0 baseline scorer (single file, stdlib only)
├── demo/metric_blindness_demo.py      # 5-second reproduction of the core finding
├── tests/
│   ├── test_fixtures.py               # schema + certainty gates + parity/orthography gates
│   └── test_scorer.py                 # unit tests + pinned per-locale regression matrix
├── docs/                              # SCHEMA.md · TEST_GUIDE.md · AUDIT-2026-07-19.md
└── .github/workflows/ci.yml           # SHA-pinned, least-privilege, Python 3.9+3.14 matrix
```

## Data flow of the scorer

```
fixture item ──► answer (active locale, en fallback + warning)
                     │
KB dir ──► load_kb ──┤  per-field locale fallback, stderr warning on cross-language comparison
                     ▼
              tokenize_v0 (V0.1)        whitespace split, case-folded; unspaced CJK runs → char bigrams
                     │
              classify()                verbatim string equality → verifiedKnowledgeBase
                     │                  overlap |KB ∩ answer|/|KB| ≥ 0.30 → aiValidatedAgainstKB
                     │                  else aiWithLowConfidence; no KB entry → aiNoKBMatch
                     ▼
       asserts_forbidden_claim()        locale claims, token recall ≥ 0.60,
                     │                  minus mention-framing and negation-cue suppression
                     ▼
              gate precedence           flagged + would-validate → demoted to aiWithLowConfidence
```

Design invariants (enforced by `tests/`, documented in [docs/SCHEMA.md](docs/SCHEMA.md) and
[docs/AUDIT-2026-07-19.md](docs/AUDIT-2026-07-19.md)):

1. **Recorded answers are frozen.** `ai_explanations` are real model outputs, errors included;
   correcting them would falsify the benchmark. Verified content lives only in `fixtures/kb-examples/`.
2. **The V0 scorer is deliberately naive, and the naivety is pinned.** Its documented failures
   (semantic inversion, Korean agglutination, punctuation attachment) are the benchmark's subject.
   The per-locale regression matrix in `tests/test_scorer.py` pins agreement, miss sets, and
   blocked sets so any drift is a deliberate, reviewed change.
3. **Certainty is never auto-promoted**, translations carry parity/orthography CI gates, and
   time-sensitive claims must be date-qualified — see the schema.
4. **Zero dependencies is a contract.** Anything requiring a model or a segmenter (MeCab, NLI
   checkers) belongs to the M2 harness, reported *alongside* — not replacing — the V0 baseline.

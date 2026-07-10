# Schema — one page

## Benchmark item (`hg_benchmark_item_v3`)

Each item in `fixtures/samples/hg120_sample_20.json → items[]`:

```jsonc
{
  "id": "HG002",                        // stable ID, HG001–HG120
  "category": "entanglement",           // 1 of 16 benchmark categories
  "kb_category": "entanglement",        // category inside the verified KB
  "kb_concept_id": "entanglement",      // KB entry this item validates against (see fixtures/kb-examples/)
  "preskill_part": 3,                   // Preskill CS219/Ph219 lecture-note part (1–24) the topic maps to
  "key_concepts": ["entanglement"],     // concept lookup keys used by the guard
  "ai_confidence": 0.62,                // model self-reported confidence for this recorded answer
  "expected_source": "aiWithLowConfidence",  // ground-truth tier (see below)
  "expected_keywords": [],              // tokens a faithful answer should contain
  "forbidden_claims": [                 // misconception patterns that must never be asserted
    "Entanglement transmits information faster than light"
  ],
  "difficulty": "intermediate",         // basic | intermediate | advanced
  "prompts":         { "en": "...", "ko": "...", "ja": "...", "zh-Hans": "...", "es": "...", "fr": "...", "de": "..." },
  "ai_explanations": { "en": "...", /* same 7 locales */ }
}
```

7 prompt fields + 7 answer fields = **14 multilingual fields per item**; 120 items = **1,680 fields**.

### Ground-truth tiers (`expected_source`)

| Value | Meaning |
|---|---|
| `verifiedKnowledgeBase` | answer served verbatim from the verified KB |
| `aiValidatedAgainstKB` | generated answer; one-sided overlap with KB entry ≥ 0.30 → passes |
| `aiWithLowConfidence` | KB entry exists but overlap < 0.30 → flag, offer KB answer instead |
| `aiNoKBMatch` | no KB entry for the topic → pass through with explicit caveat |

### Scoring (production V0 baseline, `reference/score_v0.py`)

- Tokenizer V0: case-folded whitespace split — no stemming, no punctuation stripping.
- Metric: one-sided overlap `|KB ∩ answer| / |KB|`; threshold **0.30**.
- Forbidden-claim check: an answer that asserts a `forbidden_claims` pattern fails regardless of overlap.

## KB entry (`fixtures/kb-examples/*.json`)

```jsonc
{
  "id": "superposition",
  "certainty": "established",           // established | verify | open | timebound | null (= needs-review)
  "certaintyNote": "Superposition principle per Dirac (1930); textbook formalization N&C.",
  "sourceRef": "Nielsen & Chuang (2010), doi:10.1017/CBO9780511976667",
  "category": "foundations",
  "translations": { /* 7 locales × {title, shortDefinition, detailedExplanation, analogies,
                       commonMisconceptions, keyTakeaways, practiceQuestions} */ },
  "relatedFormulas": [ { "latex": "...", "plainTextDescription": {...}, "variables": [...] } ],
  "verifiedBy": { "verifierName": "...", "verifierCredentials": "...", "verificationDate": "YYYY-MM-DD" },
  "citationReferences": ["..."]
}
```

### Certainty gates (CI-enforced, never auto-promoted)

- `established` → `sourceRef` must be meaningful: a DOI, URL, or citation carrying a year. Empty shells rejected.
- `timebound` → must carry a valid `asOfDate` (`YYYY-MM-DD`). Example excerpt from the production KB:

  ```json
  { "certainty": "timebound", "asOfDate": "2026-07-02",
    "sourceRef": "Khaneja et al., J. Magn. Reson. 172, 296 (2005), doi:10.1016/j.jmr.2004.11.004" }
  ```

- `verify`, `open`, `null` → no mandatory metadata; `null` renders as *needs-review*, never hidden.

### Locale terminology & dated-sentence policy

- **KO terms follow the Korean Physical Society (KPS) 2020 glossary.** Where common usage differs from the official term, the official term comes first with the common variant in parentheses at first mention per entry — e.g. decoherence = 결깨짐(결어긋남); KPS maps 결어긋남 to *incoherence*. Official-term confirmations are recorded in `certaintyNote`.
- **An `established` entry may contain experiment- or ecosystem-status sentences only if they are explicitly year- or date-qualified in the text** (the pattern used by `measurement`, `quantum_error_correction`, `ml_kem`). Undated time-sensitive claims must instead be graded `timebound` with `asOfDate` — mixing undated status claims into an `established` entry is a gate violation.

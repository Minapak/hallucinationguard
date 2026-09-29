# Schema — one page

## Benchmark item (`hg_benchmark_item_v3.1`)

Each item in `fixtures/samples/hg120_sample_20.json → items[]`:

```jsonc
{
  "id": "HG002",                        // stable ID, HG001–HG120
  "category": "entanglement",           // 1 of 16 benchmark categories
  "kb_category": "entanglement",        // category inside the verified KB (metadata; must equal the KB entry's category — CI-checked)
  "kb_concept_id": "entanglement",      // KB entry this item validates against (see fixtures/kb-examples/)
  "preskill_part": 3,                   // Preskill CS219/Ph219 lecture-note part (1–24) — annotation metadata, bounds CI-checked
  "key_concepts": ["entanglement"],     // concept lookup keys used by the production app's guard;
                                        // the V0 reference scorer resolves KB entries via kb_concept_id only
  "ai_confidence": 0.62,                // model self-reported confidence for this recorded answer (metadata, 0–1 CI-checked; not consumed by the V0 scorer)
  "expected_source": "aiWithLowConfidence",  // ground-truth tier (see below)
  "expected_keywords": [],              // tokens a faithful answer should contain — each must appear
                                        // (case-insensitive substring) in the en answer; CI-checked, not consumed by the V0 scorer
  "forbidden_claims": {                 // misconception patterns that must never be asserted —
    "en": ["Entanglement transmits information faster than light"],
    "ko": ["얽힘을 통해 빛보다 빠른 속도로 정보 전송이 이루어진다"]
    /* … all 7 locales. v3 shipped an English-only list, which left the gate
       silently inert for 6 of 7 locales; v3.1 localizes the patterns. A legacy
       list is still accepted by the scorer but only applies to en answers. */
  },
  "difficulty": "intermediate",         // basic | intermediate | advanced
  "prompts":         { "en": "...", "ko": "...", "ja": "...", "zh-Hans": "...", "es": "...", "fr": "...", "de": "..." },
  "ai_explanations": { "en": "...", /* same 7 locales */ }
}
```

7 prompt fields + 7 answer fields = **14 multilingual fields per item**; 120 items = **1,680 fields**.
The recorded `ai_explanations` are frozen benchmark artifacts (real model answers, including their
errors and their pre-KPS terminology); they are never "corrected" — correcting them would falsify the benchmark.

### Ground-truth tiers (`expected_source`)

| Value | Meaning | How the V0 scorer detects it |
|---|---|---|
| `verifiedKnowledgeBase` | answer served verbatim from the verified KB | normalized string equality with the KB entry (a serving-path fact — an answer merely *containing* every KB token also reaches overlap 1.0 and must not count) |
| `aiValidatedAgainstKB` | generated answer; one-sided overlap with KB entry ≥ 0.30 → passes | overlap threshold |
| `aiWithLowConfidence` | KB entry exists but overlap < 0.30 → flag, offer KB answer instead | overlap threshold |
| `aiNoKBMatch` | no KB entry for the topic → pass through with explicit caveat | KB lookup miss |

### Scoring (production V0 baseline, `reference/score_v0.py`)

- **Tokenizer V0.1**: case-folded whitespace split — no stemming, no punctuation stripping —
  plus **character bigrams for unspaced CJK runs** (Han, Hiragana, Katakana). Whitespace splitting
  alone collapses Japanese/Chinese text to ~1 pseudo-token (*scriptio continua*), which made every
  ja/zh overlap score vacuous (matches were math symbols like `=` and `+`). The character-n-gram
  remedy follows chrF (Popović, WMT 2015), the WMT17 metrics-task practice of character-level
  Chinese scoring, and SacreBLEU's `zh` tokenizer. Korean keeps whitespace *eojeol* tokens:
  spacing exists in Korean, and not splitting agglutinative morphology is part of the documented
  V0 naivety (it costs ko two tier agreements on the sample — see the pinned matrix in
  `tests/test_scorer.py`).
- **Metric**: one-sided overlap `|KB ∩ answer| / |KB|`; threshold **0.30**. A KB side that
  tokenizes below 5 tokens triggers a stderr warning (denominator too small to be meaningful).
- **Forbidden-claim gate** (locale-aware): a claim fires when its token recall inside the answer
  is ≥ 0.60, **unless** the answer's top-overlap sentence either
  (a) *mentions* the claim inside a warning frame ("… the common overstatement that …") —
  denial-framing cues, the zero-dependency approximation of the stance-detection
  mention-vs-assertion distinction (FEVER; FNC-1 refuting-word features); or
  (b) carries a locale negation cue — a NegEx-style check (Chapman et al., J Biomed Inform 2001)
  so an answer that correctly *denies* a misconception is not flagged. The negation guard is
  skipped for claims that are themselves negatively phrased ("No further cryptanalysis is
  needed …"): echoing that negation is assertion (NegEx's pseudo-negation problem).
- **Gate precedence**: a flagged answer is never classed `aiValidatedAgainstKB` — it is demoted to
  `aiWithLowConfidence` (the guard serves the KB answer instead). For topics with **no KB entry**,
  the tier remains `aiNoKBMatch` and the block is reported alongside it (there is no KB answer to
  serve; the pass-through caveat becomes a block notice).
- **Known limits, stated plainly**: the gate is lexical. Paraphrases below the token-recall
  threshold escape it; a negation or framing cue anywhere in the top sentence suppresses it even
  if the cue belongs to a different clause; cue lists are small and curated. NegEx itself reports
  sensitivity 77.8% / PPV 84.5% on clinical text — rule-based cue matching is a cheap floor, not a
  solution. Measuring where this floor fails is this benchmark's purpose; NLI-based checkers
  (AlignScore, MiniCheck) are the documented upgrade path (see README).

## Provenance/exposition extension items (`hg_benchmark_item_v3.2`, `fixtures/samples/hgx_provenance_6.json`)

Six **synthetic adversarial items** (HGX01–HGX06) authored 2026-07-27 for failure classes that are
*orthogonal to correctness* — an answer can be physically right and still fail a learner or an
editorial standard. They are **not** part of the frozen HG120 production set and are **not**
counted in the 1,680 verified fields. Evidence base: the First Proof second batch (uncited
line-by-line reuse that "would have been flagged for plagiarism"; a system that "hallucinates that
a key result exists in a previous paper"; meticulous routine detail while "glossing over the most
difficult steps") and Tao's ICM 2026 lecture (exposition that "dwells at length on trivialities,
while passing very briefly through … the most interesting and novel portions").

v3.2 adds three fields on top of v3.1:

```jsonc
{
  "adversarial_class": "uncitedNearVerbatim",   // uncitedNearVerbatim | hallucinatedCitation | depthInverted | control
  "expected_flags": ["attributionRequired"],    // dimension flags the guard must raise (may be [] for controls)
  "hard_step_concepts": { "en": ["threshold theorem", "syndrome", "stabilizer"], /* 7 locales */ }
                                                // only for depth items: the locale's renderings of the
                                                // concepts that constitute the hard step, substring-matchable
}
```

### Dimension flags (computed by `reference/dimensions_v1.py`; CI-checked via `--check-flags`)

| Flag | Fires when | Detector floor, stated plainly |
|---|---|---|
| `attributionRequired` | a generated (non-verbatim-served) answer covers ≥ 0.85 of the KB entry's tokens | token coverage, not sequence match; the tier is never changed — provenance is a separate axis from correctness |
| `hallucinatedCitation` | the answer mentions a structured identifier (DOI / arXiv / FIPS) absent from the KB entry's reference corpus (sourceRef + citationReferences + certaintyNote + en text + reviewedSources) | only structured identifiers; prose citations ("Smith et al. showed…") pass unseen |
| `depthInverted` | annotated hard-step concepts are *mentioned* (coverage ≥ 0.5) but *undeveloped* (< 20% of answer chars in hard-step sentences) | crude annotated-coverage proxy with hand-set thresholds; replaced by the M2 NLI/judge backend behind the same `computed_flags()` interface |

Dimensions are reported **separately and never blended** into one score (HELM's multi-metric
design rationale). Goodhart's law (Tao's ICM warning: "all metrics, when excessively optimized
for, are at risk") applies to these detectors as to every metric in this repo — the extension
items were tuned so the detectors fire on them, which is exactly why the flags are a floor, not
a definition of the failure class.

## Misconception extension items (`hg_benchmark_item_v3.1`, `fixtures/samples/hgm_misconceptions_3.json`)

Three **synthetic items** (HGM01–HGM03) authored 2026-09-29 for intro-level superposition
misconceptions: the spinning-coin picture, "simply both 0 and 1 at the same time", and "the
qubit already has a definite value we don't know". Same v3.1 item schema as HG120, no extra
fields. The `ai_explanations` are authored wrong answers that assert each misconception, not
recorded model outputs. They are **not** part of HG120 and **not** counted in the 1,680 fields.

Extra constraints (checked by `tests/test_fixtures.py`): ids use the `HGM` namespace,
`forbidden_claims` carry all 7 locales, and no claim contains a locale negation cue. Consuming
apps union every `fixtures/samples/*.json` claim into their production gate, so claims are kept
content-word dominant; the file's `description` records the false-block measurement done before
adding them.

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

### Translation-parity and orthography gates (CI-enforced)

- Every locale's `detailedExplanation` must reach a script-density-aware share of the en text:
  **es/fr/de ≥ 50%, ko ≥ 40%, ja ≥ 25%, zh-Hans ≥ 18%** of en characters. Motivation: stub
  translations shrink the overlap denominator until stopwords cross the 0.30 threshold — the
  audited failure was a hallucinated Grover/NP answer validating against a 13-token es stub.
- es/fr `detailedExplanation` longer than 200 chars must contain at least one accented character —
  unaccented text is not a verified translation.
- Translation passes are recorded in `certaintyNote` and remain **pending native-speaker review**
  (ROADMAP M3); `verifiedBy` records only human verification and is never updated by a
  translation pass.

### Locale terminology & dated-sentence policy

- **KB entries follow the Korean Physical Society (KPS) 2020 glossary.** Where common usage differs
  from the official term, the official term comes first with the common variant in parentheses at
  first mention per entry — e.g. decoherence = 결깨짐(결어긋남); KPS maps 결어긋남 to *incoherence*.
  Official-term confirmations are recorded in `certaintyNote`. The policy is scoped to KB entries:
  recorded benchmark answers (`ai_explanations`) are frozen artifacts and keep whatever
  terminology the model actually produced.
- **An `established` entry may contain experiment- or ecosystem-status sentences only if they are
  explicitly year- or date-qualified in the text** (the pattern used by `measurement`,
  `quantum_error_correction`, `ml_kem`). Undated time-sensitive claims must instead be graded
  `timebound` with `asOfDate` — mixing undated status claims into an `established` entry is a
  gate violation. Translations must preserve the date qualifications verbatim-equivalently.

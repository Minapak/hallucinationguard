# Primary literature vendored or ported into this repository

Only material whose licence allows redistribution is stored here in full. Everything else is cited by DOI/arXiv id
with the exact rule we took from it. Checked 2026-11-03.

| Source | Licence | What is here | What we ported |
|---|---|---|---|
| Farquhar, Kossen, Kuhn, Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 625–630 (2024). doi:10.1038/s41586-024-07421-0 | CC BY 4.0 | [full text](farquhar-2024-semantic-entropy.md) (PMC11186750, text only) | `harness/semantic_entropy_v1.py` — bidirectional-entailment clustering + discrete semantic entropy (Methods, "Principles of semantic uncertainty"); the bidirectional test behind `disagree` in `reference/provenance_v1.py` |
| Kalai, Nachum, Vempala, Zhang. *Why Language Models Hallucinate.* arXiv:2509.04664 (2025) | CC BY 4.0 | [full text](kalai-2025-why-language-models-hallucinate.txt) (pdftotext) | The scoring argument (confident error must score below abstention) — applied to the guard: `disagree` is reported, never resolved by picking a side |
| *Evaluating large language models for accuracy incentivizes hallucinations.* Nature (2026). doi:10.1038/s41586-026-10549-w | not open (login wall at fetch time) | citation only | same argument, journal version — not reproduced |
| Laban, Schnabel, Bennett, Hearst. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL 10 (2022). Code github.com/tingofurro/summac | paper: CC BY 4.0 (TACL); code: Apache-2.0 | citation only | sentence-level premise/hypothesis granularity (`harness/nli_v1.py`, `reference/provenance_v1.py`) |
| Tang, Laban, Durrett. *MiniCheck.* EMNLP 2024. Code github.com/liyan06/minicheck | code: Apache-2.0 (the Bespoke-MiniCheck-7B weights are non-commercial) | citation only | none (English-only; this repo's contract is 7 locales) |
| Zha, Yang, Li, Hu. *AlignScore.* ACL 2023. Code github.com/yuh-zha/AlignScore | MIT | citation only | none (English-only) |
| Kuhn, Gal, Farquhar. *Semantic Uncertainty.* ICLR 2023. Code github.com/jlko/semantic_uncertainty | BSD-3-Clause-Clear | citation only | algorithm re-implemented from the Nature paper's Methods; no code copied |
| Coleman, Marfatia. *Gatekeepers and Hallucinations: A Layered Evaluation Framework for LLM-Driven Quantum Circuit Generation.* arXiv:2606.18422 (2026) | arXiv non-exclusive licence (not redistributable) | citation only | the run-to-run "design entropy" idea → `consistency` dimension (same discrete-entropy estimator, applied to regenerated circuits) |
| NIST FIPS 203 (ML-KEM), Nielsen & Chuang (2010), Preskill lecture notes, Mermin (2007) | KB primary sources (see `fixtures/kb-examples/`) | citation only | the verified KB itself |

Attribution for the two vendored texts is in each file's header. If a licence above is wrong, open an issue and the file is removed.

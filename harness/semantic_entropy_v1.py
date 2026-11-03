#!/usr/bin/env python3
"""HallucinationGuard — discrete semantic entropy over bidirectional-entailment clusters (V1).

Ported from the method section of Farquhar, Kossen, Kuhn & Gal, "Detecting
hallucinations in large language models using semantic entropy", Nature 630,
625–630 (2024), https://doi.org/10.1038/s41586-024-07421-0 — CC BY 4.0. The
full text is vendored at docs/literature/farquhar-2024-semantic-entropy.md.
This file is our own implementation of the published procedure, not their code.

The procedure (paper, Methods):
  1. Sample M answers to the same question.
  2. Cluster them by meaning: two answers share a cluster iff each entails the
     other ("bidirectional entailment"); an answer that entails no existing
     cluster's representative starts a new cluster.
  3. Discrete semantic entropy = entropy of the cluster-size distribution
     (the "discrete" variant, for when sequence probabilities are not available —
     our on-device and API paths). High entropy = the model's answers vary in
     MEANING = likely confabulation; low = consistent meaning.

Where it plugs into HallucinationGuard: it is a *consistency* dimension beside
the KB-anchored verdict — the KB says whether an answer matches the physics;
this says whether the model even has one answer. The receipt reports it as
`consistency` when samples are available; the tier is never touched.

Zero dependencies in this file; the entailment judge is injected as
`scores_batch([(premise, hypothesis), ...]) -> [{"entailment": p, ...}]`
(harness/nli_v1.NLIChecker.scores_batch fits). Entailment threshold 0.50,
same as the NLI layer.

Usage:
  python3 harness/semantic_entropy_v1.py --answers "a" "b" "c"      # needs the harness deps
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

ENTAIL_MIN = 0.50
SE_VERSION = "semantic-entropy-discrete-v1"

ScoresBatch = Callable[[list], list]


def bidirectional_entails(nli: ScoresBatch, a: str, b: str, threshold: float = ENTAIL_MIN) -> bool:
    rows = nli([(a, b), (b, a)])
    return all(float(r.get("entailment", 0.0)) >= threshold for r in rows)


def cluster_by_meaning(answers: List[str], nli: ScoresBatch,
                       threshold: float = ENTAIL_MIN) -> List[List[int]]:
    """Greedy clustering as in the paper: each answer is compared with the first
    member of every existing cluster; joins the first cluster it bidirectionally
    entails, else opens a new one. Deterministic in input order."""
    clusters: List[List[int]] = []
    for i, ans in enumerate(answers):
        for cl in clusters:
            if bidirectional_entails(nli, answers[cl[0]], ans, threshold):
                cl.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def discrete_semantic_entropy(clusters: List[List[int]], n: int) -> float:
    """-Σ p_c log p_c with p_c = |cluster| / n (natural log, as in the paper's estimator)."""
    if n == 0:
        return 0.0
    return -sum((len(c) / n) * math.log(len(c) / n) for c in clusters if c)


def consistency(answers: List[str], nli: ScoresBatch, threshold: float = ENTAIL_MIN) -> Dict[str, Any]:
    """Receipt fragment: clusters, discrete semantic entropy, and the normalized
    entropy in [0, 1] (divided by log M, so 0 = one meaning, 1 = every answer
    means something different)."""
    cl = cluster_by_meaning(answers, nli, threshold)
    se = discrete_semantic_entropy(cl, len(answers))
    norm = se / math.log(len(answers)) if len(answers) > 1 else 0.0
    return {"version": SE_VERSION, "samples": len(answers), "clusters": cl,
            "semantic_entropy": round(se, 4), "normalized": round(norm, 4),
            "entail_min": threshold}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--answers", nargs="+", required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from nli_v1 import NLIChecker
    except Exception as e:  # torch/transformers absent
        print(f"harness deps missing ({e}); install harness/requirements.txt", file=sys.stderr)
        return 1
    import json
    print(json.dumps(consistency(a.answers, NLIChecker().scores_batch), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

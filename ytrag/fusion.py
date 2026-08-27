"""Reciprocal Rank Fusion.

RRF combines rankings by *position* rather than by score, which is what makes it
safe to fuse BM25 (unbounded, corpus-dependent) with cosine similarity (bounded)
without calibrating either.  A document that both retrievers place near the top
outranks one that a single retriever loves -- exactly the property wanted when a
lexical hit on a rare word and a semantic hit on a paraphrase should reinforce
each other.

Cormack, Clarke & Buettcher (SIGIR 2009).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[tuple[Hashable, float]]:
    """Fuse ranked id lists into one ranking of ``(id, score)``.

    Args:
        rankings: one ranked list of ids per retriever, best first.
        k: damping constant; larger values flatten the advantage of rank 1.
        weights: per-retriever weights, defaulting to equal.
        limit: keep only the top N results.

    Ties are broken by id so the output is deterministic across runs.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights has {len(weights)} entries but there are {len(rankings)} rankings"
        )

    scores: dict[Hashable, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for position, doc_id in enumerate(ranking):
            scores[doc_id] += weight / (k + position + 1)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ordered[:limit] if limit is not None else ordered

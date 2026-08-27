"""Consensus-Weighted Retrieval (CWR).

Ordinary RAG retrieves the top-k *chunks* nearest a query. Over a comment
section that is the wrong unit of evidence, for two reasons:

1. **Redundancy.** If 200 people make the same point, top-k returns five
   near-identical copies of it and the remaining 195 are invisible. The model
   sees five documents and has no way to know whether that view is universal or
   fringe.
2. **Proportion.** "What do people think of this video?" is a question about a
   *distribution*. No individual comment answers it.

CWR retrieves **opinion clusters** instead. Each cluster carries the social
proof behind it -- how many people said it (``support``) and how many likes
those comments drew (``endorsement``) -- so the prompt can state that a view is
held by 38% of commenters rather than quoting four chunks and hoping.

Clusters are ranked by

    score(C, q) = relevance(C, q) × social(C)

    relevance(C, q) = (1 - γ) + γ·saliencẽ(C, q)
    social(C)       = 1 + α·support̃(C) + β·endorsement̃(C)

Social proof **multiplies** relevance rather than adding to it, and that choice
is the whole point. Under an additive score a sufficiently popular cluster wins
every query, including ones it has nothing to do with -- ask about a minor
character and you get handed the video's most-liked joke. Multiplying makes an
irrelevant cluster score near zero no matter how many likes it carries, while
among clusters that *are* relevant the widely-held view rises. Relevance decides
*whether*; consensus decides *which*.

The three inputs are log-damped (so one enormous "first!" cluster cannot win on
bulk alone) and then rescaled into ``[0, 1]`` across the candidates:

    salience(C, q)  = cosine between q and the cluster centroid
    support(C)      = log(1 + |C|)
    endorsement(C)  = log(1 + Σ likes)

Salience is measured against the **centroid** rather than the best-matching
member, because taking a maximum over members is not size-neutral: a cluster of
six gets six draws at that maximum and a cluster of three gets three, so the
larger group scores higher on noise alone. Since support and endorsement
already reward size deliberately, letting salience reward it accidentally as
well double-counts it. One vector per cluster, whatever its size.

γ controls how much relevance narrows the field: at γ=1 an off-topic cluster is
annihilated, at γ=0 relevance is ignored entirely, which is what the CONSENSUS
weights below lean towards when the question is about the distribution itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ytrag.embed import query_tokens
from ytrag.fusion import reciprocal_rank_fusion
from ytrag.models import Evidence, OpinionCluster
from ytrag.store import BM25Index, HybridStore


@dataclass(frozen=True, slots=True)
class ConsensusWeights:
    """Mixing weights for the three ranking signals.

    The defaults keep relevance in charge (γ=0.6). :mod:`ytrag.router` swaps in
    consensus-heavy weights for questions that are actually about the
    distribution of opinion rather than about a topic.
    """

    #: How sharply relevance narrows the field; must stay within [0, 1] since
    #: it is used as a mixing fraction, not an unbounded coefficient.
    salience: float = 0.6
    support: float = 0.2
    endorsement: float = 0.2

    def validate(self) -> None:
        if self.salience < 0 or self.support < 0 or self.endorsement < 0:
            raise ValueError("consensus weights must be non-negative")
        if self.salience > 1:
            raise ValueError("salience weight is a fraction and must not exceed 1")
        if self.salience + self.support + self.endorsement <= 0:
            raise ValueError("at least one consensus weight must be greater than zero")


CONSENSUS_WEIGHTS = ConsensusWeights(salience=0.25, support=0.4, endorsement=0.35)
"""Weights for 'what does everyone think' questions: proportion over relevance."""


def cluster_opinions(
    store: HybridStore,
    threshold: float | None = None,
    max_quotes: int = 5,
) -> list[OpinionCluster]:
    """Group comments that say substantially the same thing.

    Uses greedy leader clustering, walking comments in descending like order.
    Two properties follow from that ordering and both are wanted:

    * the most-endorsed comment in a group becomes its leader, so the quote
      shown to the user is the one the crowd actually upvoted;
    * the result is deterministic, unlike k-means with a random seed.

    ``threshold`` defaults to the embedder's own recommendation, because a
    cosine of 0.4 means something different for hashed n-grams than it does for
    a trained sentence encoder.
    """
    if threshold is None:
        threshold = getattr(store.embedder, "cluster_threshold", 0.5)

    order = sorted(
        range(store.total_comments),
        key=lambda i: (-store.comments[i].likes, store.cids[i]),
    )
    vectors = store.vectors
    leaders: list[int] = []
    members: list[list[int]] = []

    for i in order:
        best_leader = -1
        best_similarity = -np.inf
        for slot, leader in enumerate(leaders):
            similarity = float(vectors[i] @ vectors[leader])
            if similarity > best_similarity:
                best_similarity, best_leader = similarity, slot
        if best_leader >= 0 and best_similarity >= threshold:
            members[best_leader].append(i)
        else:
            leaders.append(i)
            members.append([i])

    clusters: list[OpinionCluster] = []
    for cluster_id, (leader, group) in enumerate(zip(leaders, members, strict=True)):
        group_comments = [store.comments[i] for i in group]
        ranked = sorted(group_comments, key=lambda c: (-c.likes, c.cid))
        clusters.append(
            OpinionCluster(
                cluster_id=cluster_id,
                member_cids=[c.cid for c in ranked],
                representative_cid=store.comments[leader].cid,
                representative_text=store.comments[leader].text
                or store.comments[leader].emojis,
                support=len(group),
                endorsement=sum(c.likes for c in group_comments),
                valence=float(np.mean([c.valence for c in group_comments])),
                centroid=_unit(vectors[group].mean(axis=0)),
                exemplar_cids=[c.cid for c in ranked[:max_quotes]],
            )
        )
    return clusters


def _unit(vector: np.ndarray) -> np.ndarray:
    """L2-normalise, so a centroid can be compared by plain dot product."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


#: Damping constant for cluster-level rank fusion. Smaller than the usual 60
#: because there are far fewer clusters than documents, and at k=60 the gap
#: between rank 1 and rank 4 all but vanishes.
_CLUSTER_RRF_K = 10


def _hybrid_salience(
    clusters: Sequence[OpinionCluster],
    store: HybridStore,
    query: str,
) -> list[float]:
    """Relevance of each cluster to ``query``, from both retrievers.

    Dense similarity to the cluster centroid and BM25 over the cluster's
    concatenated member text are fused by **reciprocal rank fusion**, then
    scaled against the best cluster.

    Fusing by rank rather than by magnitude is the important part, and it is
    worth recording why, because the obvious alternative fails. Taking the
    larger of the two normalised scores looks reasonable and is measurably
    wrong: asked "who wants to see Luffy fight Buggy" on the sample corpus, the
    hashed embedder scores the entirely unrelated Vegapunk cluster at 0.23
    against the correct cluster's 0.292 -- near parity, pure noise -- while BM25
    scores them 0.0 and 3.83. Any scheme that reads those cosines as magnitudes
    lets the noise through; scaling them against the best candidate actively
    amplifies it, turning 0.23-out-of-0.292 into "79% as relevant".

    Rank agreement is robust to exactly that. A cluster both retrievers place
    near the top outranks one that only the noisy retriever likes, without
    either having to be calibrated against the other -- which is the property
    RRF exists for.
    """
    if not clusters:
        return []

    # No content words means no topic to be relevant to. Relevance has no
    # opinion, and must not impose an arbitrary order on the clusters.
    if not query_tokens(query):
        return [1.0] * len(clusters)

    query_vector = store.embedder.encode_query(query)
    dense = [(c.cluster_id, float(np.dot(c.centroid, query_vector))) for c in clusters]
    dense_ranking = [
        cid for cid, score in sorted(dense, key=lambda kv: -kv[1]) if score > 0
    ]

    lexical_index = BM25Index(
        [
            (
                c.cluster_id,
                " ".join(
                    (store.get(cid).text if store.get(cid) else "")
                    for cid in c.member_cids
                ),
            )
            for c in clusters
        ]
    )
    lexical_ranking = [
        cid for cid, _ in lexical_index.search(query, limit=len(clusters))
    ]

    fused = dict(
        reciprocal_rank_fusion(
            [lexical_ranking, dense_ranking], k=_CLUSTER_RRF_K
        )
    )
    return _ratio_to_best([fused.get(c.cluster_id, 0.0) for c in clusters])


def score_evidence(
    clusters: Sequence[OpinionCluster],
    store: HybridStore,
    query: str,
    weights: ConsensusWeights | None = None,
    limit: int = 6,
    max_quotes: int = 3,
) -> list[Evidence]:
    """Rank clusters for ``query`` and attach the social proof behind each."""
    weights = weights or ConsensusWeights()
    weights.validate()
    if not clusters:
        return []

    similarities = store.similarity_to(query)
    total_comments = max(1, store.total_comments)
    total_likes = max(1, store.total_likes)

    saliences = _hybrid_salience(clusters, store, query)
    supports = [math.log1p(c.support) for c in clusters]
    endorsements = [math.log1p(c.endorsement) for c in clusters]

    # The three signals live on unrelated scales -- cosine is compressed into a
    # narrow band near zero while log-likes saturate -- so mixing them raw lets
    # a large cluster outrank an exactly-on-topic one no matter what gamma says.
    # Rescaling each against the best candidate makes the weights mean what they
    # claim: gamma=0.6 really is "relevance decides 60% of the ranking".
    salience_hat = _ratio_to_best(saliences)
    support_hat = _ratio_to_best(supports)
    endorsement_hat = _ratio_to_best(endorsements)

    scored: list[Evidence] = []
    for index, cluster in enumerate(clusters):
        salience = saliences[index]
        relevance = (1.0 - weights.salience) + weights.salience * salience_hat[index]
        social = (
            1.0
            + weights.support * support_hat[index]
            + weights.endorsement * endorsement_hat[index]
        )
        score = relevance * social
        # Quotes are chosen by how well each member answers *this* question,
        # while the cluster as a whole is ranked by its centroid. Both are
        # wanted: an honest ranking, and the most pertinent thing anyone in
        # that camp actually said.
        on_topic = sorted(
            cluster.exemplar_cids,
            key=lambda cid: (-similarities.get(cid, 0.0), cid),
        )
        quotes = [
            text
            for text in (
                (store.get(cid).text or store.get(cid).emojis)
                for cid in on_topic[:max_quotes]
            )
            if text
        ] or [cluster.representative_text]

        scored.append(
            Evidence(
                cluster=cluster,
                salience=float(salience),
                score=float(score),
                support_share=cluster.support / total_comments,
                endorsement_share=cluster.endorsement / total_likes,
                quotes=quotes,
            )
        )

    scored.sort(key=lambda e: (-e.score, e.cluster.cluster_id))
    return scored[:limit]


def _ratio_to_best(values: Sequence[float]) -> list[float]:
    """Scale a signal to ``[0, 1]`` as a fraction of the best value present.

    Deliberately *not* min-max. Min-max maps the weakest candidate to 0 and the
    second-strongest to roughly 0.5 regardless of the absolute numbers, so a
    cluster with a cosine of 0.02 against a best of 0.30 comes out at ~0.5 --
    "half as relevant as the best match" -- when it is not relevant at all.
    Measured on the sample corpus that was enough for a large off-topic cluster
    to beat an exactly-on-topic singleton, because a spurious 0.5 relevance
    multiplied by a real consensus boost outweighed real relevance.

    Dividing by the maximum keeps zero meaning zero: irrelevant stays
    irrelevant, and only the genuinely comparable get comparable scores.
    """
    if not values:
        return []
    clamped = [max(0.0, v) for v in values]
    best = max(clamped)
    if best <= 1e-12:
        # Nothing matched at all, so relevance has no opinion and must not veto
        # the social signals.
        return [1.0] * len(values)
    return [v / best for v in clamped]


def coverage(evidence: Sequence[Evidence], store: HybridStore) -> float:
    """Share of the corpus the selected evidence speaks for.

    Reported alongside every answer. A confident-sounding answer built from 2%
    of the comments is a different object from one built from 80%, and the user
    is entitled to know which they were handed.
    """
    if not evidence or store.total_comments == 0:
        return 0.0
    seen = {cid for e in evidence for cid in e.cluster.member_cids}
    return len(seen) / store.total_comments

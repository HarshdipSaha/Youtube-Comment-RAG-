# ADR 0004 — Fuse cluster relevance by rank, not by magnitude

**Status:** accepted (this is the third attempt — see History)

## Context

Cluster relevance to a query has two sources: dense cosine similarity to the
cluster centroid, and BM25 over the cluster's concatenated member text. They
must be combined into one number, which is then multiplied by the consensus
boost (ADR 0002).

## Decision

Rank the clusters by each retriever, fuse the two rankings with reciprocal rank
fusion (k=10, lower than the usual 60 because there are far fewer clusters than
documents), then scale the fused scores against the best cluster.

## Rationale

Magnitudes cannot be trusted here, and the failure is concrete. Asked "who wants
to see Luffy fight Buggy" on the sample corpus, the hashed embedder scored the
entirely unrelated Vegapunk cluster at **0.23** against the correct cluster's
**0.292** — near parity, pure noise — while BM25 scored them **0.0** and
**3.83**. Any combination rule that reads those cosines as magnitudes lets the
noise through.

Scaling against the best candidate actively amplifies it: 0.23 out of 0.292
becomes "79% as relevant as the best match". Multiplied by a real consensus
boost, that was enough for a large off-topic cluster to outrank an exactly
on-topic one.

Rank agreement is robust to exactly this. A cluster both retrievers place near
the top outranks one only the noisy retriever likes, and neither retriever has to
be calibrated against the other — which is what RRF is for. It also finally puts
the BM25 index on the answer path, where it earns its place: a rare proper noun
like "Mihawk" is precisely what hashed embeddings blur and lexical matching nails.

## Consequences

- Relevance is ordinal, so "how relevant" is only meaningful relative to the
  other clusters for the same query. Acceptable — ranking is all that is needed.
- A query with no content tokens returns uniform relevance rather than an
  arbitrary order, so consensus decides. Deliberate; see ADR 0003.
- Two fusion strategies now coexist: RRF at comment level
  (`HybridStore.search`) and at cluster level. The k values differ because the
  candidate-set sizes differ by orders of magnitude.

## History

1. **Min-max rescaling of raw scores.** Maps the weakest candidate to 0 and the
   second-strongest to ~0.5 regardless of absolute values, inventing relevance
   for clusters that had none.
2. **Ratio-to-best, taking the max of the dense and lexical ratios.** Better —
   zero stayed zero — but still magnitude-based, so it inherited the dense noise
   documented above. Improved hand-labelled R-precision from 50% to 83%.
3. **Rank fusion (current).** 100% on the same measure.

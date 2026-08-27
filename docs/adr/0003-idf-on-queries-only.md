# ADR 0003 — IDF weights queries, never documents

**Status:** accepted (arrived at after two failures — see History)

## Context

The offline `HashingEmbedder` exists so the pipeline runs with no model download
and no network. Unweighted, every token counts equally, so two comments sharing
only "the", "of" and "this" look as similar as two sharing "pacing" and
"terrible". IDF is the standard remedy.

## Decision

Three rules, each fixing a distinct failure:

1. **Documents carry no IDF; queries do.** This is the `lnc.ltc` weighting from
   the SMART retrieval system.
2. **Query stopwords are removed from a fixed list**, not by IDF.
3. **Query IDF is sqrt-damped**, so one rare term sharpens a query without
   single-handedly deciding it.

## Rationale

IDF measures rarity *against this corpus*, and this corpus is one video's
comments. That breaks both symmetric weighting and the obvious stopword strategy.

**Why documents get no IDF.** If 40% of comments praise Vegapunk, then
"vegapunk", "best" and "character" are common, so IDF down-weights precisely the
terms that make that opinion a coherent group. Measured on the sample corpus,
IDF-weighting documents dropped within-cluster similarity from 0.32 to 0.18
while cross-cluster similarity stayed at 0.23 — the clusters stopped being
separable at all. IDF is built for discriminating against a large background
corpus, not for clustering within a small topical one.

**Why stopwords need a list.** The same pathology inverts on the query side.
"think", "people" and "say" are rare *in comments* and common *in questions*, so
IDF rates them as highly informative — while the entity being asked about is, by
virtue of being discussed, common and therefore down-weighted. Measured: "what
do people think of vegapunk" retrieved the *pacing* complaints, because "think"
and "of" outweighed "vegapunk". Corpus statistics cannot fix this, because the
problem is precisely that the query and document distributions differ. A fixed
English function-word list can.

A query that is *nothing but* stopwords yields no tokens, and that is the correct
outcome: "what do people think" has no topic to be relevant to, so relevance
abstains and consensus decides the ranking.

## Consequences

- The embedder gains an optional `fit()`; `HybridStore` calls it on both the
  build and load paths, so queries land in the space the stored vectors were
  built in.
- The stopword list is English-only. A non-English comment section still works —
  nothing is removed — but query noise is not suppressed.
- BM25 applies the same query-side filter, since its IDF has the same blind spot.

# ADR 0001 — Retrieve opinion clusters, not comments

**Status:** accepted

## Context

The unit of retrieval determines what questions can be answered. Standard RAG
retrieves chunks; a comment section punishes that in two ways.

*Redundancy.* Comment sections are massively repetitive. If 200 people make the
same point, top-*k* returns *k* near-identical copies and the other 200 − *k*
are invisible. The model sees *k* documents.

*Proportion.* "What do people think of this?" is a question about a
distribution. No individual comment answers it, so no set of individual comments
answers it either — not without the reader knowing what fraction of the whole
they represent, which retrieval does not report.

## Decision

Cluster comments into opinions and make the **cluster** the retrieval unit. Each
cluster carries `support` (how many people said it) and `endorsement` (how many
likes those comments drew), and both are passed into the prompt as shares of the
corpus.

Clustering is greedy leader clustering over cosine similarity, walking comments
in descending like order. Two properties fall out of that ordering, and both are
wanted:

- the most-endorsed comment in a group becomes its leader, so the quote shown to
  the user is the one the crowd upvoted;
- the result is deterministic, unlike k-means with a random seed.

## Consequences

- "38% of commenters (142 comments, 8,312 likes) say X" becomes a statement the
  model can make from its context rather than infer.
- Coverage becomes measurable and is reported on every answer.
- Cost: clustering is O(n · clusters) per corpus. Acceptable at comment-section
  scale (thousands), and it happens once at index time, not per query.
- The threshold at which two comments count as "the same opinion" is
  embedder-specific, so it lives on the embedder rather than at the call site.
  See ADR 0003.

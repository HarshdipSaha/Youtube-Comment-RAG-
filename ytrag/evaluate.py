"""Measure this pipeline against the design it replaces.

Two comparisons, deliberately kept separate because they answer different
questions and one of them is much weaker evidence than the other.

**Exact questions** (:func:`exact_accuracy`). Ground truth is computed by
exhaustive scan, then each system is asked the question and its answer is
checked for the correct figure. The baseline -- top-k dense retrieval, which is
what the original project did -- loses these almost completely, and it is worth
being precise about *why* that is not an unfair fight: the correct answer to
"which comment has the most likes" is one row out of N, and nothing about that
row is semantically similar to the question. No amount of retrieval tuning
reaches it. This measures a structural limitation, not a quality gap, and it
should not be read as a general RAG benchmark.

**Retrieval questions** (:func:`retrieval_precision`). This one is a fair fight.
Both systems are asked topical questions against a hand-labelled corpus and
scored by **R-precision**: for a question with ``R`` relevant comments, each
system retrieves exactly ``R`` comments and is scored on how many of them are
relevant. Both face an identical budget and both can score 1.0, which is what
makes it symmetric -- an earlier version of this benchmark gave the baseline
four comments and the clustered system one cluster, and flattered the baseline
badly as a result. Here the comparison is BM25 + dense + clustering versus
dense-only, with no structural advantage on either side.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ytrag.aggregate import AggregateEngine
from ytrag.cluster import cluster_opinions, score_evidence
from ytrag.engine import CommentRAG
from ytrag.models import Comment
from ytrag.store import HybridStore


@dataclass(slots=True)
class Case:
    """One evaluation question with its ground truth."""

    question: str
    expected: str
    label: str = ""


@dataclass(slots=True)
class Result:
    """How one system scored.

    ``correct`` and ``total`` count items for exact questions and are summed
    over questions for R-precision, so ``accuracy`` reads as a proportion in
    both cases.
    """

    system: str
    correct: float
    total: float
    misses: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


class NaiveTopKRAG:
    """The design being replaced: dense top-k retrieval, then quote it.

    This is the original project's architecture, minus its bugs -- the likes are
    parsed as integers here and the module imports. Kept faithful in the way
    that matters: one dense retriever, ``k`` documents, no routing, no
    aggregation, no clustering, no notion of how much of the corpus was seen.
    """

    def __init__(self, store: HybridStore, k: int = 4) -> None:
        self.store = store
        self.k = k

    def retrieve(self, question: str) -> list[Comment]:
        hits = self.store.dense_search(question, limit=self.k)
        return [c for c in (self.store.get(cid) for cid, _ in hits) if c is not None]

    def ask(self, question: str) -> str:
        retrieved = self.retrieve(question)
        if not retrieved:
            return "I don't know."
        # Stuff the documents, exactly as `create_stuff_documents_chain` did.
        return " ".join(
            f'"{c.text}" ({c.likes} likes, {c.author})' for c in retrieved
        )


def exact_cases(store: HybridStore) -> list[Case]:
    """Build exact-answer cases whose ground truth is computed, not hard-coded.

    Derived from the corpus so the benchmark stays correct if the sample data
    changes, rather than rotting into a set of stale literals.
    """
    aggregates = AggregateEngine(store)
    stats = aggregates.stats()
    top = aggregates.top_liked(1)[0]

    # Pick a term that actually appears, so the mention case is meaningful.
    term = max(
        ("luffy", "vegapunk", "one piece", "chapter"),
        key=aggregates.count_mentions,
    )
    mentions = aggregates.count_mentions(term)

    return [
        Case("which comment has the most likes?", f"{top.likes:,}", "top-liked"),
        Case("how many comments are there?", f"{stats['comments']:,}", "corpus size"),
        Case(f"how many comments mention {term}?", f"{mentions:,}", "mention count"),
        Case("what is the longest comment?", str(aggregates.longest(1)[0].cid), "longest"),
        Case("who commented the most?", aggregates.top_authors(1)[0]["author"], "top author"),
        Case(
            "what is the average number of likes?",
            f"{stats['mean_likes']:.1f}",
            "mean likes",
        ),
    ]


def _score(cases: Sequence[Case], answer_fn: Callable[[str], str], system: str) -> Result:
    correct, misses = 0, []
    for case in cases:
        answer = answer_fn(case.question) or ""
        if case.expected.lower() in answer.lower():
            correct += 1
        else:
            misses.append(case.label or case.question)
    return Result(system=system, correct=correct, total=len(cases), misses=misses)


def exact_accuracy(store: HybridStore) -> list[Result]:
    """Score both systems on questions with one right answer."""
    cases = exact_cases(store)
    rag = CommentRAG(store)
    naive = NaiveTopKRAG(store)
    return [
        _score(cases, lambda q: rag.ask(q).text, "consensus-weighted (this project)"),
        _score(cases, naive.ask, "naive top-k (original design)"),
    ]


def _cluster_top_comments(
    store: HybridStore, clusters: Sequence[Any], question: str, k: int
) -> list[str]:
    """The k comments the clustered system would surface, best cluster first."""
    surfaced: list[str] = []
    for item in score_evidence(clusters, store, question, limit=len(clusters)):
        for cid in item.cluster.member_cids:
            if cid not in surfaced:
                surfaced.append(cid)
            if len(surfaced) >= k:
                return surfaced
    return surfaced


def retrieval_precision(
    store: HybridStore,
    labelled: Sequence[tuple[str, set[str]]],
) -> list[Result]:
    """R-precision on hand-labelled topical questions.

    ``labelled`` pairs each question with the set of comment ids a human judged
    relevant. Each system retrieves exactly as many comments as there are
    relevant ones, so the budgets match and a perfect score is reachable.
    """
    clusters = cluster_opinions(store)

    consensus_hits = naive_hits = 0.0
    considered = 0.0
    consensus_misses: list[str] = []
    naive_misses: list[str] = []

    for question, relevant in labelled:
        r = len(relevant)
        if r == 0:
            continue
        considered += r

        surfaced = set(_cluster_top_comments(store, clusters, question, r))
        hit = len(surfaced & relevant)
        consensus_hits += hit
        if hit < r:
            consensus_misses.append(f"{question} ({hit}/{r})")

        retrieved = {c.cid for c in NaiveTopKRAG(store, k=r).retrieve(question)}
        naive_hit = len(retrieved & relevant)
        naive_hits += naive_hit
        if naive_hit < r:
            naive_misses.append(f"{question} ({naive_hit}/{r})")

    return [
        Result("consensus-weighted (this project)", consensus_hits, considered, consensus_misses),
        Result("naive top-k (original design)", naive_hits, considered, naive_misses),
    ]


def report(results: Sequence[Result]) -> str:
    """Format results as a table."""
    width = max(len(r.system) for r in results) if results else 10
    lines = [f"{'system'.ljust(width)}  score   accuracy"]
    lines.append("-" * (width + 20))
    for r in results:
        lines.append(
            f"{r.system.ljust(width)}  {r.correct:g}/{r.total:g}     {r.accuracy:6.1%}"
        )
        if r.misses:
            lines.append(f"{''.ljust(width)}  missed: {', '.join(r.misses)}")
    return "\n".join(lines)


def run(store: HybridStore, labelled: Sequence[tuple[str, set[str]]] | None = None) -> dict[str, Any]:
    """Run every comparison and return both the results and their rendering."""
    exact = exact_accuracy(store)
    output = ["EXACT-ANSWER QUESTIONS", report(exact)]
    payload: dict[str, Any] = {"exact": exact}

    if labelled:
        retrieval = retrieval_precision(store, labelled)
        payload["retrieval"] = retrieval
        output += ["", "TOPICAL RETRIEVAL (hand-labelled)", report(retrieval)]

    payload["text"] = "\n".join(output)
    return payload

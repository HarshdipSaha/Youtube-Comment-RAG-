"""The public interface: build a knowledge base, then ask it questions.

Everything below this line is one of five moves, and which ones run depends on
what kind of question arrived:

    question
       |
       route ................. AGGREGATE / SEMANTIC / CONSENSUS / HYBRID
       |
       +-- exact ............. SQL over every comment      (AGGREGATE, HYBRID)
       |
       +-- retrieve .......... BM25 + dense, fused by RRF  (SEMANTIC, CONSENSUS, HYBRID)
       |     rank ............ consensus-weighted clusters
       |
       +-- generate .......... LLM, or the extractive composer
       |
       +-- verify ............ citations resolve, figures were computed
       |
    Answer(text, evidence, citations, coverage, exact, warnings)

The ``coverage`` figure on every answer is the share of the comment section the
answer actually saw. It is reported because "62% of comments informed this" and
"0.4% of comments informed this" are very different claims, and a RAG system
that does not say which one it made is asking to be over-trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ytrag.aggregate import AggregateEngine
from ytrag.citations import CitationGuard
from ytrag.cluster import ConsensusWeights, cluster_opinions, coverage, score_evidence
from ytrag.embed import Embedder, get_embedder
from ytrag.ingest import from_csv, from_records, from_youtube
from ytrag.llm import LLM, ExtractiveLLM, get_llm
from ytrag.models import Answer, Comment
from ytrag.prompt import SYSTEM_PROMPT, evidence_numbers, render_context
from ytrag.router import route
from ytrag.store import HybridStore

DEFAULT_INDEX = "kb_index"

#: Overview ranking ignores relevance entirely -- there is no question to be
#: relevant to, only the shape of the conversation.
_OVERVIEW_WEIGHTS = ConsensusWeights(salience=0.0, support=0.55, endorsement=0.45)


class CommentRAG:
    """Consensus-aware question answering over one video's comments."""

    def __init__(
        self,
        store: HybridStore,
        llm: LLM | None = None,
        cluster_threshold: float | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.aggregates = AggregateEngine(store)
        self.clusters = cluster_opinions(store, threshold=cluster_threshold)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_comments(
        cls,
        comments: Sequence[Comment],
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        cluster_threshold: float | None = None,
    ) -> CommentRAG:
        embedder = embedder or get_embedder("hashing")
        return cls(HybridStore.build(comments, embedder), llm, cluster_threshold)

    @classmethod
    def from_youtube(
        cls,
        url: str,
        limit: int = 500,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> CommentRAG:
        return cls.from_comments(from_youtube(url, limit=limit), embedder, llm)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> CommentRAG:
        return cls.from_comments(from_csv(path), embedder, llm)

    @classmethod
    def from_records(
        cls, records, embedder: Embedder | None = None, llm: LLM | None = None
    ) -> CommentRAG:
        return cls.from_comments(from_records(records), embedder, llm)

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_INDEX,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> CommentRAG:
        embedder = embedder or get_embedder("hashing")
        return cls(HybridStore.load(path, embedder), llm)

    def save(self, path: str | Path = DEFAULT_INDEX) -> Path:
        return self.store.save(path)

    # -- asking -----------------------------------------------------------

    def ask(self, question: str, evidence_limit: int = 5, max_quotes: int = 3) -> Answer:
        """Answer ``question`` and report how it was answered."""
        if not (question or "").strip():
            return Answer(
                question=question,
                kind="SEMANTIC",
                text="Ask a question about the comments.",
                warnings=["empty question"],
            )

        decision = route(question)
        exact = self.aggregates.answer(decision)

        # An exact result is the whole answer unless the question also asked for
        # something to be read out of the comments.
        evidence = []
        if decision.kind != "AGGREGATE":
            evidence = score_evidence(
                self.clusters,
                self.store,
                question,
                weights=decision.weights,
                limit=evidence_limit,
                max_quotes=max_quotes,
            )

        # The extractive composer reads the pipeline's own output rather than a
        # rendered prompt, so it is rebuilt per question with this evidence.
        llm = self.llm
        if llm is None or isinstance(llm, ExtractiveLLM):
            llm = ExtractiveLLM(evidence, exact=(exact or {}).get("text", ""))

        context = render_context(evidence, self.store, exact=exact)
        raw = llm.complete(SYSTEM_PROMPT, f"{context}\n\nQUESTION: {question}")

        allowed_ids = {c.cid for c in self.store.comments}
        guard = CitationGuard(
            allowed_ids=allowed_ids,
            allowed_numbers=evidence_numbers(evidence, self.store, exact=exact),
        )
        report = guard.verify(raw)

        warnings = list(report.warnings)
        answer_coverage = coverage(evidence, self.store)
        if decision.kind == "AGGREGATE":
            # Exact answers are computed over the whole corpus by definition.
            answer_coverage = 1.0
        elif evidence and answer_coverage < 0.05:
            warnings.append(
                f"This answer draws on {answer_coverage * 100:.1f}% of the comment "
                "section; treat it as a sample, not a summary."
            )

        return Answer(
            question=question,
            kind=decision.kind,
            text=report.text or raw,
            evidence=evidence,
            citations=report.citations,
            coverage=answer_coverage,
            exact=(exact or {}).get("data", {}),
            warnings=warnings,
        )

    # -- inspection -------------------------------------------------------

    def overview(self) -> dict:
        """A standing summary of the comment section, independent of any question."""
        ranked = score_evidence(
            self.clusters, self.store, "", weights=_OVERVIEW_WEIGHTS, limit=8
        )
        return {
            "stats": self.aggregates.stats(),
            "sentiment": self.aggregates.sentiment(),
            "top_liked": [c.to_dict() for c in self.aggregates.top_liked(5)],
            "clusters": [
                {
                    "cid": e.cluster.representative_cid,
                    "text": e.cluster.representative_text,
                    "support": e.cluster.support,
                    "endorsement": e.cluster.endorsement,
                    "support_share": e.support_share,
                    "valence": e.cluster.valence,
                }
                for e in ranked
            ],
        }


def build_index(
    url: str,
    path: str | Path = DEFAULT_INDEX,
    limit: int = 500,
    embedder: Embedder | None = None,
) -> CommentRAG:
    """Download a video's comments and save a queryable knowledge base."""
    rag = CommentRAG.from_youtube(url, limit=limit, embedder=embedder)
    rag.save(path)
    return rag


def load_index(
    path: str | Path = DEFAULT_INDEX,
    embedder: Embedder | None = None,
    llm_backend: str | None = None,
) -> CommentRAG:
    """Load a saved knowledge base, choosing an LLM from the environment."""
    return CommentRAG.load(path, embedder=embedder, llm=get_llm(llm_backend))

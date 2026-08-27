"""Core domain types.

The vocabulary here is the project's domain language; test names and interfaces
should use these words.  A *comment* is one YouTube comment with its social
metadata.  An *opinion cluster* is a group of comments that say substantially
the same thing, and it is the unit this system retrieves -- not the individual
comment -- because a single comment carries no information about how widely its
view is held.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

QueryKind = Literal["AGGREGATE", "SEMANTIC", "CONSENSUS", "HYBRID"]


@dataclass(frozen=True, slots=True)
class Comment:
    """One YouTube comment.

    ``cid`` is a stable identifier used in citations; callers may pass the
    platform id or let :func:`ytrag.ingest.build_corpus` assign ``c0, c1, ...``.
    """

    cid: str
    text: str
    author: str = ""
    likes: int = 0
    published: str = ""
    published_ts: float | None = None
    reply_count: int = 0
    is_reply: bool = False
    emojis: str = ""
    valence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OpinionCluster:
    """A group of near-duplicate opinions plus the social proof behind them.

    ``support`` is how many people said it; ``endorsement`` is how many likes
    those comments attracted.  Together they are what lets an answer say "most
    commenters think X" instead of quoting four arbitrary chunks.
    """

    cluster_id: int
    member_cids: list[str]
    representative_cid: str
    representative_text: str
    support: int
    endorsement: int
    valence: float = 0.0
    #: Excluded from equality: numpy arrays make `==` ambiguous, and two
    #: clusters with the same members are the same opinion regardless.
    centroid: Any = field(default=None, compare=False)
    exemplar_cids: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.member_cids)


@dataclass(slots=True)
class Evidence:
    """A scored cluster, ready to be rendered into an LLM prompt."""

    cluster: OpinionCluster
    salience: float
    score: float
    support_share: float
    endorsement_share: float
    quotes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"C{self.cluster.cluster_id}"


@dataclass(slots=True)
class Answer:
    """The result of :meth:`ytrag.engine.CommentRAG.ask`."""

    question: str
    kind: QueryKind
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    coverage: float = 0.0
    exact: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "kind": self.kind,
            "answer": self.text,
            "citations": self.citations,
            "coverage": round(self.coverage, 4),
            "exact": self.exact,
            "warnings": self.warnings,
            "evidence": [
                {
                    "label": e.label,
                    "representative": e.cluster.representative_text,
                    "support": e.cluster.support,
                    "endorsement": e.cluster.endorsement,
                    "support_share": round(e.support_share, 4),
                    "score": round(e.score, 4),
                }
                for e in self.evidence
            ],
        }

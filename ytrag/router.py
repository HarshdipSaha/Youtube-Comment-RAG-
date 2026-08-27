"""Decide what kind of question was asked, before trying to answer it.

Three kinds of question arrive at a comment section and they want three
different machines:

``AGGREGATE``
    "Which comment has the most likes?", "How many people mention Vegapunk?"
    These have exact answers that live in a table. Retrieval cannot find them:
    the most-liked comment is one row in thousands, and no amount of semantic
    similarity to the word "likes" will surface it. Routed to SQL.

``CONSENSUS``
    "What do people think of this?" This is a question about a *distribution*,
    so it is ranked by how widely a view is held rather than by how well any
    one comment matches the words of the question.

``SEMANTIC``
    "What are people saying about Vegapunk?" Ordinary topical retrieval, with
    relevance in charge.

``HYBRID``
    Both at once -- "how many mention pacing, and what do they say?" -- which
    gets the exact count *and* the reading.

The rules are deliberately explicit rather than an LLM classifier: routing runs
on every question, must be fast, must be testable offline, and a
misclassification here silently degrades every answer downstream. An LLM
classifier can be layered on later for the ambiguous tail; it cannot be the
foundation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ytrag.cluster import CONSENSUS_WEIGHTS, ConsensusWeights
from ytrag.models import QueryKind


class Intent(Enum):
    """The specific exact-answer computation an AGGREGATE question wants."""

    TOP_LIKED = "top_liked"
    COUNT_ALL = "count_all"
    MENTION_COUNT = "mention_count"
    TOP_AUTHORS = "top_authors"
    LONGEST = "longest"
    STATS = "stats"
    TIMELINE = "timeline"
    SENTIMENT = "sentiment"


@dataclass(frozen=True, slots=True)
class Route:
    """Where a question is being sent, and why."""

    kind: QueryKind
    weights: ConsensusWeights
    reason: str
    intent: Intent | None = None
    term: str = ""


# Ordered: the first pattern to match wins, so put the specific before the general.
_AGGREGATE_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    (
        Intent.MENTION_COUNT,
        re.compile(
            r"how many\s+(?:comments?|people|users|viewers|of them)?\s*"
            r"(?:mention|mentions|mentioned|talk about|talks about|talked about|"
            r"say|says|said|reference|discuss|discusses)\s+(?P<term>.+?)"
            r"(?:\s+and\s+.*)?[?.]*$"
        ),
    ),
    (
        Intent.TOP_LIKED,
        re.compile(
            r"(most|highest|top)\s+(liked|likes|upvoted|rated|popular|voted)"
            r"|(most|highest|top)\s+\w*\s*(comment|one)\b"
            r"|\b(top|best)\s+comments?\b"
            r"|comment.*\b(most|highest)\s+(likes|votes)"
        ),
    ),
    (
        Intent.TOP_AUTHORS,
        re.compile(
            r"who\s+(commented|posted|wrote)\s+(the\s+)?most"
            r"|most\s+(active|prolific)\s+(user|commenter|author|person)"
            r"|which\s+(user|author|commenter)\s+.*most"
        ),
    ),
    (
        Intent.LONGEST,
        re.compile(r"\b(longest|shortest|biggest|wordiest)\s+comment"),
    ),
    (
        Intent.SENTIMENT,
        re.compile(
            r"(sentiment|valence)\s+(breakdown|split|distribution|ratio)"
            r"|how many.*(positive|negative)"
            r"|(positive|negative)\s+(vs|versus|to)\s+(negative|positive)"
        ),
    ),
    (
        Intent.TIMELINE,
        re.compile(r"when\s+(were|was|did).*(post|comment|written)|comment\s+timeline"),
    ),
    (
        Intent.STATS,
        re.compile(
            r"\b(average|mean|median|typical)\b.*\b(likes?|comments?|length)\b"
            r"|\b(statistics|stats)\b"
        ),
    ),
    (
        Intent.COUNT_ALL,
        re.compile(
            r"how many comments?\b(?!\s+(?:mention|talk|say))"
            r"|(total|overall)\s+(number|count)\s+of\s+comments?"
            r"|\bcomment count\b"
            r"|how many people commented"
        ),
    ),
]

_CONSENSUS_RE = re.compile(
    r"\b(overall|in general|generally|consensus|majority|most people|everyone|"
    r"general (opinion|reaction|sentiment|feeling|mood)|"
    r"public opinion|what do (people|they|viewers|fans) think|"
    r"how do (people|they|viewers|fans) feel|"
    r"sentiment|reaction|vibe|mood|"
    r"summar(y|ise|ize)|tl;?dr|gist|"
    r"(positive|negative|mixed)\b.*\b(or|vs)\b)"
)

# Markers that a question wants prose about the comments, not just a number.
_ELABORATION_RE = re.compile(
    r"\band\s+what\b|\band\s+why\b|\band\s+how\b|what do they (say|think)"
    r"|\bexplain\b|\bdescribe\b|\bsummar(y|ise|ize)\b"
)

_SEMANTIC_WEIGHTS = ConsensusWeights()

_STOP_TERM_WORDS = {"the", "a", "an", "this", "that", "it", "about"}


def _clean_term(term: str) -> str:
    """Reduce a captured phrase to the thing being counted."""
    term = re.sub(r"[?.!,]+$", "", term.strip().lower())
    words = [w for w in term.split() if w not in _STOP_TERM_WORDS]
    return " ".join(words) if words else term


def route(question: str) -> Route:
    """Classify ``question`` and choose the ranking weights that suit it."""
    text = (question or "").strip().lower()
    if not text:
        return Route(
            kind="SEMANTIC",
            weights=_SEMANTIC_WEIGHTS,
            reason="empty question; defaulting to semantic retrieval",
        )

    intent: Intent | None = None
    term = ""
    for candidate, pattern in _AGGREGATE_PATTERNS:
        match = pattern.search(text)
        if match:
            intent = candidate
            if "term" in (match.groupdict() or {}) and match.group("term"):
                term = _clean_term(match.group("term"))
            break

    wants_consensus = bool(_CONSENSUS_RE.search(text))

    if intent is not None:
        # An aggregate question that also asks for elaboration needs both the
        # exact number and something read out of the comments themselves.
        if _ELABORATION_RE.search(text) or wants_consensus:
            return Route(
                kind="HYBRID",
                weights=CONSENSUS_WEIGHTS if wants_consensus else _SEMANTIC_WEIGHTS,
                reason=f"exact {intent.value} plus a reading of the comments",
                intent=intent,
                term=term,
            )
        return Route(
            kind="AGGREGATE",
            weights=_SEMANTIC_WEIGHTS,
            reason=f"computable exactly from the comment table ({intent.value})",
            intent=intent,
            term=term,
        )

    if wants_consensus:
        return Route(
            kind="CONSENSUS",
            weights=CONSENSUS_WEIGHTS,
            reason="asks about the distribution of opinion, not a single topic",
        )

    return Route(
        kind="SEMANTIC",
        weights=_SEMANTIC_WEIGHTS,
        reason="topical question; relevance leads the ranking",
    )

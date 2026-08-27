"""Exact answers over the whole corpus.

This module is the direct answer to the original project's central bug. Its
README promised users could ask "which comment has the most likes", and the
implementation shipped a top-k vector search. Those two things are incompatible,
and not by a small margin:

* retrieval returns the ``k`` comments most *similar to the question*, and the
  most-liked comment has no particular reason to mention likes at all;
* with 4 documents retrieved from 5,000, a "how many people said X" question is
  answered from 0.08% of the evidence;
* the like counts were strings, so even a model that saw them would rank
  ``"9"`` above ``"1.2K"``.

So questions with exact answers never touch the retriever. They run as SQL over
every row, and the number that comes back is the number -- not a plausible one.
The LLM's job downstream is to phrase the result, never to compute it.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from ytrag.models import Comment
from ytrag.router import Intent, Route
from ytrag.store import HybridStore

_POSITIVE_CUTOFF = 0.2
_NEGATIVE_CUTOFF = -0.2


def _fmt(number: float | int) -> str:
    """Thousands separators: '1000 likes' reads as noise, '1,000 likes' reads."""
    return f"{number:,.0f}"


def _pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def _quote(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class AggregateEngine:
    """Deterministic answers computed from the comment table."""

    def __init__(self, store: HybridStore) -> None:
        self.store = store
        self._db = store.db

    # -- primitives -------------------------------------------------------

    def top_liked(self, n: int = 5) -> list[Comment]:
        rows = self._db.execute(
            "SELECT cid FROM comments ORDER BY likes DESC, cid ASC LIMIT ?", (n,)
        ).fetchall()
        return [self.store.get(row["cid"]) for row in rows]

    def longest(self, n: int = 5) -> list[Comment]:
        rows = self._db.execute(
            "SELECT cid FROM comments ORDER BY length DESC, cid ASC LIMIT ?", (n,)
        ).fetchall()
        return [self.store.get(row["cid"]) for row in rows]

    def count_all(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) AS n FROM comments").fetchone()["n"])

    def _mention_pattern(self, term: str) -> re.Pattern[str]:
        """Whole-word match, so 'rain' does not match 'trained'.

        Done in Python rather than SQL ``LIKE`` because ``LIKE '%rain%'`` has no
        word-boundary notion and would silently over-count.
        """
        return re.compile(rf"\b{re.escape(term.strip().lower())}\b")

    def _mentioning(self, term: str) -> list[Comment]:
        if not term.strip():
            return []
        pattern = self._mention_pattern(term)
        return [c for c in self.store.comments if pattern.search(c.text.lower())]

    def count_mentions(self, term: str) -> int:
        return len(self._mentioning(term))

    def mention_report(self, term: str) -> dict[str, Any]:
        matches = self._mentioning(term)
        total = max(1, self.store.total_comments)
        return {
            "term": term,
            "count": len(matches),
            "share": len(matches) / total,
            "likes": sum(c.likes for c in matches),
            "cids": [c.cid for c in matches],
        }

    def top_authors(self, n: int = 5) -> list[dict[str, Any]]:
        rows = self._db.execute(
            """
            SELECT author, COUNT(*) AS comments, SUM(likes) AS likes
            FROM comments WHERE author != ''
            GROUP BY author ORDER BY comments DESC, likes DESC, author ASC LIMIT ?
            """,
            (n,),
        ).fetchall()
        return [
            {"author": r["author"], "comments": int(r["comments"]), "likes": int(r["likes"] or 0)}
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        likes = [c.likes for c in self.store.comments]
        total_likes = sum(likes)
        return {
            "comments": len(likes),
            "total_likes": total_likes,
            "mean_likes": (total_likes / len(likes)) if likes else 0.0,
            "median_likes": statistics.median(likes) if likes else 0,
            "max_likes": max(likes) if likes else 0,
            "mean_length": (
                statistics.mean(len(c.text) for c in self.store.comments)
                if self.store.comments else 0.0
            ),
            "top_comment_like_share": (max(likes) / total_likes) if total_likes else 0.0,
        }

    def sentiment(self) -> dict[str, Any]:
        """Split by emoji valence, both by head-count and by likes.

        Both views are reported because they routinely disagree. Ten calm
        comments and one furious one with 5,000 likes is a positive comment
        section by count and a negative one by attention.
        """
        breakdown = {
            "positive": 0, "negative": 0, "neutral": 0,
            "like_weighted_positive": 0, "like_weighted_negative": 0,
            "like_weighted_neutral": 0,
        }
        for comment in self.store.comments:
            if comment.valence >= _POSITIVE_CUTOFF:
                bucket = "positive"
            elif comment.valence <= _NEGATIVE_CUTOFF:
                bucket = "negative"
            else:
                bucket = "neutral"
            breakdown[bucket] += 1
            breakdown[f"like_weighted_{bucket}"] += comment.likes
        return breakdown

    def timeline(self, buckets: int = 6) -> list[dict[str, Any]]:
        """Bucket comments into equal time spans, oldest first."""
        stamped = [c for c in self.store.comments if c.published_ts is not None]
        if not stamped or buckets < 1:
            return [
                {"label": "unknown", "count": self.store.total_comments,
                 "likes": self.store.total_likes, "start": None, "end": None}
            ]

        oldest = min(c.published_ts for c in stamped)
        newest = max(c.published_ts for c in stamped)
        span = max(newest - oldest, 1e-6)
        width = span / buckets

        rows: list[dict[str, Any]] = [
            {"label": f"bucket {i + 1}", "count": 0, "likes": 0,
             "start": oldest + i * width, "end": oldest + (i + 1) * width}
            for i in range(buckets)
        ]
        for comment in self.store.comments:
            if comment.published_ts is None:
                index = buckets - 1  # undated comments join the most recent bucket
            else:
                index = min(buckets - 1, int((comment.published_ts - oldest) / width))
            rows[index]["count"] += 1
            rows[index]["likes"] += comment.likes

        # Label by the human-readable time of the oldest comment in each bucket.
        for row in rows:
            in_bucket = [
                c for c in stamped
                if row["start"] <= c.published_ts <= row["end"] and c.published
            ]
            if in_bucket:
                row["label"] = max(in_bucket, key=lambda c: c.published_ts).published
        return rows

    # -- rendering --------------------------------------------------------

    def compute(self, intent: Intent, term: str = "") -> dict[str, Any] | None:
        """Run one exact computation and render it as text plus structured data."""
        handler = {
            Intent.TOP_LIKED: self._render_top_liked,
            Intent.COUNT_ALL: self._render_count_all,
            Intent.MENTION_COUNT: self._render_mentions,
            Intent.TOP_AUTHORS: self._render_authors,
            Intent.LONGEST: self._render_longest,
            Intent.STATS: self._render_stats,
            Intent.TIMELINE: self._render_timeline,
            Intent.SENTIMENT: self._render_sentiment,
        }.get(intent)
        return handler(term) if handler else None

    def answer(self, decision: Route) -> dict[str, Any] | None:
        """Answer ``decision`` exactly, or return ``None`` if it is not exact."""
        if decision.kind not in ("AGGREGATE", "HYBRID") or decision.intent is None:
            return None
        return self.compute(decision.intent, term=decision.term)

    def _render_top_liked(self, term: str = "") -> dict[str, Any]:
        top = self.top_liked(3)
        if not top:
            return {"text": "There are no comments to rank.", "citations": [], "data": {}}
        best = top[0]
        lines = [
            f'The most-liked comment has {_fmt(best.likes)} likes, by {best.author or "an unknown user"}'
            f'{" " + best.published if best.published else ""}: "{_quote(best.text or best.emojis)}" [{best.cid}]'
        ]
        if len(top) > 1:
            runners = ", ".join(
                f'"{_quote(c.text or c.emojis, 60)}" ({_fmt(c.likes)} likes) [{c.cid}]'
                for c in top[1:]
            )
            lines.append(f"Next highest: {runners}.")
        return {
            "text": " ".join(lines),
            "citations": [c.cid for c in top],
            "data": {"top": [c.to_dict() for c in top]},
        }

    def _render_count_all(self, term: str = "") -> dict[str, Any]:
        total = self.count_all()
        likes = self.store.total_likes
        return {
            "text": (
                f"There are {_fmt(total)} comments in this knowledge base, "
                f"carrying {_fmt(likes)} likes between them."
            ),
            "citations": [],
            "data": {"comments": total, "total_likes": likes},
        }

    def _render_mentions(self, term: str = "") -> dict[str, Any]:
        if not term:
            return self._render_count_all()
        report = self.mention_report(term)
        if report["count"] == 0:
            return {
                "text": f'No comment mentions "{term}".',
                "citations": [],
                "data": report,
            }
        return {
            "text": (
                f'{_fmt(report["count"])} of {_fmt(self.store.total_comments)} comments '
                f'({_pct(report["share"])}) mention "{term}", '
                f'drawing {_fmt(report["likes"])} likes in total.'
            ),
            "citations": report["cids"][:10],
            "data": report,
        }

    def _render_authors(self, term: str = "") -> dict[str, Any]:
        authors = self.top_authors(5)
        if not authors:
            return {"text": "No authors recorded.", "citations": [], "data": {}}
        top = authors[0]
        listing = ", ".join(
            f'{a["author"]} ({a["comments"]} comments, {_fmt(a["likes"])} likes)'
            for a in authors
        )
        return {
            "text": (
                f'{top["author"]} posted the most, with {top["comments"]} '
                f'comment{"s" if top["comments"] != 1 else ""}. Most active: {listing}.'
            ),
            "citations": [],
            "data": {"authors": authors},
        }

    def _render_longest(self, term: str = "") -> dict[str, Any]:
        longest = self.longest(1)
        if not longest:
            return {"text": "There are no comments.", "citations": [], "data": {}}
        c = longest[0]
        return {
            "text": (
                f'The longest comment is {len(c.text)} characters, by '
                f'{c.author or "an unknown user"}: "{_quote(c.text, 300)}" [{c.cid}]'
            ),
            "citations": [c.cid],
            "data": {"cid": c.cid, "length": len(c.text)},
        }

    def _render_stats(self, term: str = "") -> dict[str, Any]:
        s = self.stats()
        return {
            "text": (
                f'{_fmt(s["comments"])} comments, {_fmt(s["total_likes"])} likes in total. '
                f'Mean {s["mean_likes"]:.1f} likes per comment, median {_fmt(s["median_likes"])} '
                f'-- the gap between those two is the usual comment-section skew. '
                f'The single top comment holds {_pct(s["top_comment_like_share"])} of all likes.'
            ),
            "citations": [],
            "data": s,
        }

    def _render_timeline(self, term: str = "") -> dict[str, Any]:
        rows = self.timeline(6)
        busiest = max(rows, key=lambda r: r["count"])
        described = ", ".join(f'{r["label"]}: {r["count"]}' for r in rows)
        return {
            "text": (
                f'Comment activity by period -- {described}. '
                f'The busiest period was "{busiest["label"]}" with {busiest["count"]} comments.'
            ),
            "citations": [],
            "data": {"timeline": rows},
        }

    def _render_sentiment(self, term: str = "") -> dict[str, Any]:
        b = self.sentiment()
        total = max(1, self.store.total_comments)
        return {
            "text": (
                f'By emoji signal: {b["positive"]} positive ({_pct(b["positive"] / total)}), '
                f'{b["negative"]} negative ({_pct(b["negative"] / total)}), '
                f'{b["neutral"]} neutral. Weighted by likes, positive comments hold '
                f'{_fmt(b["like_weighted_positive"])} likes against '
                f'{_fmt(b["like_weighted_negative"])} for negative ones. '
                f'Note this reads emoji only, not tone in text.'
            ),
            "citations": [],
            "data": b,
        }

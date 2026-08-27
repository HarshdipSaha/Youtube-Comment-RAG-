"""Exact answers, computed over the whole corpus rather than retrieved from it.

Every assertion here is a number that can be checked by hand against the
fixture. That is the point: these answers are arithmetic, not inference, and
they must be right every single time rather than usually.
"""

import pytest

from ytrag.aggregate import AggregateEngine
from ytrag.router import Intent, route
from ytrag.store import HybridStore


@pytest.fixture
def engine(comments, embedder):
    return AggregateEngine(HybridStore.build(comments, embedder))


class TestTopLiked:
    def test_finds_the_single_most_liked_comment(self, engine):
        """c4 and c6 both have 1000 likes; ties break by id, so c4 wins."""
        top = engine.top_liked(1)
        assert len(top) == 1
        assert top[0].likes == 1000

    def test_returns_them_in_descending_like_order(self, engine):
        likes = [c.likes for c in engine.top_liked(4)]
        assert likes == [1000, 1000, 900, 73]

    def test_asking_for_more_than_exist_is_not_an_error(self, engine):
        assert len(engine.top_liked(500)) == 11

    def test_the_answer_does_not_depend_on_retrieval(self, engine):
        """The whole reason this module exists.

        A vector search for 'likes' would rank comments that use the *word*
        'likes'. None of these do, and the right answer still comes back.
        """
        assert engine.top_liked(1)[0].cid in {"c4", "c6"}


class TestCounts:
    def test_counts_the_corpus(self, engine):
        assert engine.count_all() == 11

    def test_counts_comments_mentioning_a_term(self, engine):
        assert engine.count_mentions("vegapunk") == 6

    def test_mention_counting_is_case_insensitive(self, engine):
        assert engine.count_mentions("VEGAPUNK") == engine.count_mentions("vegapunk")

    def test_mention_counting_matches_whole_words(self, engine):
        assert engine.count_mentions("arc") == 5  # c1, c5, c6, c7, c8

    def test_a_term_inside_a_longer_word_does_not_count(self, engine):
        """'rain' appears inside 'trained' (c10) but nobody mentioned rain."""
        assert engine.count_mentions("rain") == 0

    def test_a_term_nobody_used_counts_zero(self, engine):
        assert engine.count_mentions("cryptocurrency") == 0

    def test_mention_count_reports_the_share_too(self, engine):
        result = engine.mention_report("vegapunk")
        assert result["count"] == 6
        assert result["share"] == pytest.approx(6 / 11)
        assert result["likes"] == 2020


class TestAuthors:
    def test_ranks_authors_by_comment_count_then_likes(self, engine):
        top = engine.top_authors(3)
        assert top[0]["author"]
        assert top[0]["comments"] >= top[-1]["comments"]

    def test_every_author_in_the_fixture_posted_once(self, engine):
        assert all(row["comments"] == 1 for row in engine.top_authors(20))


class TestStats:
    def test_reports_totals_and_averages(self, engine):
        stats = engine.stats()
        assert stats["comments"] == 11
        assert stats["total_likes"] == 3142
        assert stats["mean_likes"] == pytest.approx(3142 / 11)
        assert stats["max_likes"] == 1000
        assert stats["median_likes"] == 40  # sorted: 9 12 13 15 20 [40] 60 73 900 1000 1000

    def test_reports_the_share_of_likes_held_by_the_top_comment(self, engine):
        """Comment sections are extremely top-heavy; say so."""
        assert engine.stats()["top_comment_like_share"] == pytest.approx(1000 / 3142)


class TestLongest:
    def test_finds_the_longest_comment(self, engine):
        assert engine.longest(1)[0].cid == "c9"


class TestSentiment:
    def test_splits_the_corpus_by_valence(self, engine):
        breakdown = engine.sentiment()
        assert breakdown["positive"] == 2   # c3, c4
        assert breakdown["negative"] == 2   # c6, c7
        assert breakdown["neutral"] == 7
        assert sum(breakdown[k] for k in ("positive", "negative", "neutral")) == 11

    def test_reports_a_like_weighted_view_as_well(self, engine):
        """One comment with 1000 likes is not one person's opinion."""
        breakdown = engine.sentiment()
        assert breakdown["like_weighted_positive"] == 1900
        assert breakdown["like_weighted_negative"] == 1060


class TestTimeline:
    def test_buckets_comments_over_time(self, engine):
        timeline = engine.timeline(buckets=3)
        assert len(timeline) == 3
        assert sum(b["count"] for b in timeline) == 11
        assert all("label" in b for b in timeline)


class TestAnswerDispatch:
    """The engine takes a Route and produces a rendered exact answer."""

    def test_answers_a_top_liked_question(self, engine):
        answer = engine.answer(route("which comment has the most likes"))
        assert "1,000 likes" in answer["text"]
        assert answer["citations"]

    def test_answers_a_count_question(self, engine):
        answer = engine.answer(route("how many comments are there"))
        assert "11" in answer["text"]

    def test_answers_a_mention_question_with_the_share(self, engine):
        answer = engine.answer(route("how many comments mention vegapunk"))
        assert "6" in answer["text"]
        assert "55" in answer["text"] or "54" in answer["text"]  # ~54.5%

    def test_a_non_aggregate_route_returns_nothing(self, engine):
        assert engine.answer(route("what are people saying about buggy")) is None

    def test_every_intent_is_handled(self, engine):
        """No intent may route to an exact answer the engine cannot compute."""
        for intent in Intent:
            result = engine.compute(intent, term="vegapunk")
            assert result is not None, f"{intent} has no implementation"
            assert result["text"]

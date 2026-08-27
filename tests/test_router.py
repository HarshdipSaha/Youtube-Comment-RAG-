"""Query routing: deciding what kind of question was actually asked.

The original project sent every question down the same top-k path. "Which
comment has the most likes?" cannot be answered that way -- the answer is one
row out of thousands and retrieval will almost never surface it -- so the first
job is to notice which questions need counting rather than reading.
"""

import pytest

from ytrag.router import Intent, route


class TestAggregateRouting:
    @pytest.mark.parametrize(
        "question",
        [
            "which comment has the most likes?",
            "what is the most liked comment",
            "top comment by likes",
            "show me the highest rated comment",
            "most popular comment please",
        ],
    )
    def test_top_liked_questions_are_aggregate(self, question):
        decision = route(question)
        assert decision.kind == "AGGREGATE"
        assert decision.intent is Intent.TOP_LIKED

    @pytest.mark.parametrize(
        "question",
        ["how many comments are there", "total number of comments", "comment count"],
    )
    def test_counting_the_corpus_is_aggregate(self, question):
        assert route(question).intent is Intent.COUNT_ALL

    def test_counting_mentions_extracts_the_term(self):
        decision = route("how many comments mention Vegapunk")
        assert decision.kind == "AGGREGATE"
        assert decision.intent is Intent.MENTION_COUNT
        assert decision.term == "vegapunk"

    def test_counting_mentions_handles_talk_about_phrasing(self):
        decision = route("how many people talk about the pacing")
        assert decision.intent is Intent.MENTION_COUNT
        assert "pacing" in decision.term

    def test_most_active_author_is_aggregate(self):
        assert route("who commented the most").intent is Intent.TOP_AUTHORS

    def test_longest_comment_is_aggregate(self):
        assert route("what is the longest comment").intent is Intent.LONGEST

    def test_average_likes_is_aggregate(self):
        assert route("what is the average number of likes").intent is Intent.STATS

    def test_timeline_question_is_aggregate(self):
        assert route("when were these comments posted").intent is Intent.TIMELINE


class TestConsensusRouting:
    @pytest.mark.parametrize(
        "question",
        [
            "what do people think of this video",
            "what is the overall opinion",
            "what is the general sentiment",
            "is the reaction positive or negative",
            "summarize the comments",
            "what is the consensus",
        ],
    )
    def test_distribution_questions_are_consensus(self, question):
        assert route(question).kind == "CONSENSUS"

    def test_consensus_weighting_favours_proportion_over_relevance(self):
        decision = route("what do people think overall")
        assert decision.weights.support > decision.weights.salience


class TestSemanticRouting:
    @pytest.mark.parametrize(
        "question",
        [
            "what are people saying about Vegapunk",
            "any theories about the next arc",
            "did anyone mention the animation quality",
        ],
    )
    def test_topic_questions_are_semantic(self, question):
        assert route(question).kind == "SEMANTIC"

    def test_semantic_weighting_keeps_relevance_in_charge(self):
        decision = route("what are people saying about Vegapunk")
        assert decision.weights.salience > decision.weights.support


class TestHybridRouting:
    def test_a_question_that_counts_and_summarises_is_hybrid(self):
        """Needs the exact number *and* a reading of what was said."""
        decision = route("how many people mention pacing and what do they say about it")
        assert decision.kind == "HYBRID"
        assert decision.intent is Intent.MENTION_COUNT


class TestRobustness:
    def test_routing_is_case_insensitive(self):
        assert route("MOST LIKED COMMENT").intent is Intent.TOP_LIKED

    def test_empty_question_falls_back_to_semantic(self):
        assert route("").kind == "SEMANTIC"
        assert route("   ").kind == "SEMANTIC"

    def test_decision_reports_why_it_chose(self):
        """A router that cannot explain itself cannot be debugged."""
        assert route("most liked comment").reason

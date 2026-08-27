"""End-to-end behaviour of the public interface.

The first class here is the important one: it asks the questions the original
README promised and the original code could not answer.
"""

import pytest

from ytrag.engine import CommentRAG
from ytrag.ingest import from_records, to_csv


@pytest.fixture
def rag(comments):
    return CommentRAG.from_comments(comments, cluster_threshold=0.30)


class TestTheQuestionsTheOriginalCouldNotAnswer:
    def test_which_comment_has_the_most_likes(self, rag):
        answer = rag.ask("which comment has the most likes?")
        assert answer.kind == "AGGREGATE"
        assert "1,000 likes" in answer.text
        assert answer.coverage == 1.0  # computed over every comment, not a sample

    def test_how_many_comments_mention_a_topic(self, rag):
        answer = rag.ask("how many comments mention vegapunk")
        assert "6" in answer.text
        assert "54.5%" in answer.text

    def test_what_do_people_think_overall(self, rag):
        answer = rag.ask("what do people think of this video overall")
        assert answer.kind == "CONSENSUS"
        assert answer.evidence
        assert "%" in answer.text  # states proportion, not just quotes

    def test_a_topical_question_finds_the_right_camp(self, rag):
        answer = rag.ask("what are people saying about the pacing")
        assert answer.kind == "SEMANTIC"
        assert "pacing" in answer.text.lower()

    def test_a_hybrid_question_gets_both_the_count_and_the_reading(self, rag):
        answer = rag.ask("how many comments mention pacing and what do they say")
        assert answer.kind == "HYBRID"
        assert "3" in answer.text          # the exact count
        assert answer.evidence             # and the retrieved opinion


class TestGrounding:
    def test_every_answer_reports_its_coverage(self, rag):
        for question in [
            "what do people think",
            "what about buggy",
            "how many comments are there",
        ]:
            assert 0.0 <= rag.ask(question).coverage <= 1.0

    def test_semantic_answers_cite_the_comments_they_used(self, rag):
        answer = rag.ask("what do people think about vegapunk")
        assert answer.citations
        assert all(rag.store.get(cid) is not None for cid in answer.citations)

    def test_answers_never_cite_a_comment_that_does_not_exist(self, rag):
        answer = rag.ask("what do people think")
        known = {c.cid for c in rag.store.comments}
        assert set(answer.citations) <= known

    def test_the_default_backend_produces_a_grounded_answer(self, rag):
        answer = rag.ask("what do people think overall")
        assert not [w for w in answer.warnings if "invent" in w.lower()]


class TestRobustness:
    def test_an_empty_question_does_not_crash(self, rag):
        assert rag.ask("").text
        assert rag.ask("   ").warnings

    def test_a_question_about_nothing_in_the_corpus_is_answered_honestly(self, rag):
        answer = rag.ask("what do people say about quarterly tax filings")
        assert answer.text  # says something rather than raising

    def test_a_single_comment_corpus_works(self):
        rag = CommentRAG.from_records([{"text": "great video", "votes": "5", "author": "@a"}])
        assert rag.ask("what do people think").text
        assert rag.ask("most liked comment").text

    def test_a_corpus_with_no_likes_at_all_works(self):
        rag = CommentRAG.from_records(
            [{"text": f"comment {i}", "votes": 0, "author": f"@u{i}"} for i in range(5)]
        )
        assert rag.ask("which comment has the most likes").text

    def test_comments_that_are_only_emoji_are_kept(self):
        """The original pipeline deleted these entirely."""
        rag = CommentRAG.from_records(
            [{"text": "🔥🔥🔥", "votes": "500", "author": "@a"},
             {"text": "actual words here", "votes": "1", "author": "@b"}]
        )
        assert rag.store.total_comments == 2
        assert "500" in rag.ask("most liked comment").text


class TestOverview:
    def test_summarises_the_comment_section_without_a_question(self, rag):
        overview = rag.overview()
        assert overview["stats"]["comments"] == 11
        assert overview["clusters"]
        assert overview["top_liked"][0]["likes"] == 1000

    def test_clusters_are_ordered_by_how_widely_held_they_are(self, rag):
        supports = [c["support"] for c in rag.overview()["clusters"]]
        assert supports[0] == max(supports)


class TestPersistence:
    def test_a_saved_knowledge_base_answers_identically(self, comments, tmp_path):
        original = CommentRAG.from_comments(comments, cluster_threshold=0.30)
        original.save(tmp_path / "kb")

        restored = CommentRAG.load(tmp_path / "kb")
        question = "what do people think about vegapunk"
        assert restored.ask(question).text == original.ask(question).text

    def test_exact_answers_survive_a_round_trip(self, comments, tmp_path):
        CommentRAG.from_comments(comments).save(tmp_path / "kb")
        assert "1,000 likes" in CommentRAG.load(tmp_path / "kb").ask(
            "most liked comment"
        ).text


class TestIngest:
    def test_reads_back_a_csv_it_wrote(self, comments, tmp_path):
        path = to_csv(comments, tmp_path / "out.csv")
        rag = CommentRAG.from_csv(path)
        assert rag.store.total_comments == 11
        assert rag.store.total_likes == 3142

    def test_reads_the_original_projects_merged_csv_format(self, tmp_path):
        """An index built by the old code must still be readable."""
        path = tmp_path / "legacy.csv"
        path.write_text(
            "Merged Comment\n"
            "\"comment is 'Vegapunk is the best' with likes= 20 with user_id "
            "'@Gleeblevr' and published(time)  '8 hours ago'\"\n"
            "\"comment is 'One piece forever' with likes= 1.2K with user_id "
            "'@Letfreakinggo' and published(time)  '8 hours ago'\"\n",
            encoding="utf-8",
        )
        rag = CommentRAG.from_csv(path)
        assert rag.store.total_comments == 2
        assert rag.store.get("c1").likes == 1200  # '1.2K' parsed, not sorted as text
        assert rag.store.get("c0").author == "@Gleeblevr"

    def test_records_get_stable_sequential_ids(self):
        comments = from_records([{"text": "a"}, {"text": "b"}])
        assert [c.cid for c in comments] == ["c0", "c1"]

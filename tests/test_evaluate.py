"""The benchmark itself has to be trustworthy, so it is tested too."""

import pytest

from ytrag.evaluate import (
    NaiveTopKRAG,
    exact_accuracy,
    exact_cases,
    report,
    retrieval_precision,
    run,
)
from ytrag.store import HybridStore

#: Hand-labelled relevance over the fixture corpus. These are the judgements a
#: person would make reading the eleven comments, written down once.
LABELLED = [
    ("what do people say about the pacing", {"c6", "c7", "c8"}),
    ("what do people think of vegapunk", {"c0", "c1", "c2", "c3", "c4", "c5"}),
    ("anything about Mihawk or Crocodile", {"c10"}),
    ("who wants to see Luffy fight Buggy", {"c9", "c10"}),
]


@pytest.fixture
def store(comments, embedder):
    return HybridStore.build(comments, embedder)


class TestNaiveBaseline:
    def test_retrieves_exactly_k_comments(self, store):
        assert len(NaiveTopKRAG(store, k=4).retrieve("vegapunk")) == 4

    def test_quotes_what_it_retrieved(self, store):
        answer = NaiveTopKRAG(store).ask("what about the pacing")
        assert "likes" in answer

    def test_cannot_answer_a_question_about_the_maximum(self, store):
        """Not a bug in the baseline -- a property of top-k retrieval.

        The most-liked comment is one row of eleven and says nothing about
        likes, so similarity to the question cannot find it.
        """
        answer = NaiveTopKRAG(store).ask("which comment has the most likes?")
        assert "1000 likes" not in answer.replace(",", "")


class TestExactCases:
    def test_ground_truth_is_derived_from_the_corpus(self, store):
        cases = exact_cases(store)
        assert any(c.expected == "1,000" for c in cases)
        assert any(c.expected == "11" for c in cases)

    def test_every_case_has_a_label_and_an_expected_value(self, store):
        assert all(c.label and c.expected for c in exact_cases(store))


class TestExactAccuracy:
    def test_this_project_answers_the_exact_questions(self, store):
        results = {r.system: r for r in exact_accuracy(store)}
        consensus = next(r for k, r in results.items() if "this project" in k)
        assert consensus.accuracy >= 0.8, consensus.misses

    def test_the_original_design_cannot(self, store):
        results = {r.system: r for r in exact_accuracy(store)}
        naive = next(r for k, r in results.items() if "original" in k)
        assert naive.accuracy <= 0.5

    def test_this_project_scores_strictly_higher(self, store):
        consensus, naive = exact_accuracy(store)
        assert consensus.accuracy > naive.accuracy


class TestRetrievalPrecision:
    def test_both_systems_are_scored(self, store):
        results = retrieval_precision(store, LABELLED)
        assert len(results) == 2
        # R-precision: the denominator is the number of relevant comments across
        # all questions, and both systems must face the identical budget.
        expected = sum(len(relevant) for _, relevant in LABELLED)
        assert all(r.total == expected for r in results)

    def test_a_question_with_no_relevant_comments_is_skipped(self, store):
        results = retrieval_precision(store, [("nothing relevant here", set())])
        assert all(r.total == 0 for r in results)
        assert all(r.accuracy == 0.0 for r in results)

    def test_the_clustered_system_is_at_least_as_good(self, store):
        """A fair comparison: no structural advantage on either side here."""
        consensus, naive = retrieval_precision(store, LABELLED)
        assert consensus.accuracy >= naive.accuracy, consensus.misses


class TestReporting:
    def test_renders_a_table(self, store):
        text = report(exact_accuracy(store))
        assert "accuracy" in text
        assert "%" in text

    def test_run_returns_both_comparisons(self, store):
        payload = run(store, LABELLED)
        assert "exact" in payload and "retrieval" in payload
        assert "EXACT-ANSWER" in payload["text"]
        assert "TOPICAL RETRIEVAL" in payload["text"]

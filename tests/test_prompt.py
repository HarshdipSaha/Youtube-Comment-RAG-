"""How evidence is rendered for a language model.

The context block is where this system's advantage is either communicated or
thrown away. A model handed five raw comments cannot know whether they are the
whole conversation or a fringe of it; a model handed "142 comments (38%),
8,312 likes" can reason about proportion.
"""

import pytest

from ytrag.cluster import cluster_opinions, score_evidence
from ytrag.prompt import SYSTEM_PROMPT, evidence_numbers, render_context
from ytrag.store import HybridStore


@pytest.fixture
def evidence(comments, embedder):
    store = HybridStore.build(comments, embedder)
    clusters = cluster_opinions(store, threshold=0.30)
    return score_evidence(clusters, store, "vegapunk", limit=3), store


class TestRenderContext:
    def test_every_block_is_labelled_with_a_citable_id(self, evidence):
        items, store = evidence
        context = render_context(items, store)
        for item in items:
            assert item.cluster.representative_cid in context

    def test_states_how_many_people_hold_each_view(self, evidence):
        items, store = evidence
        context = render_context(items, store)
        assert "6 comments" in context
        assert "54.5%" in context

    def test_states_the_likes_behind_each_view(self, evidence):
        items, store = evidence
        assert "2,020" in render_context(items, store)

    def test_includes_the_actual_quotes(self, evidence):
        items, store = evidence
        context = render_context(items, store)
        assert "vegapunk" in context.lower()

    def test_reports_corpus_totals_so_shares_can_be_checked(self, evidence):
        items, store = evidence
        context = render_context(items, store)
        assert "11" in context and "3,142" in context

    def test_empty_evidence_says_so_rather_than_rendering_nothing(self, evidence):
        _, store = evidence
        assert "no " in render_context([], store).lower()

    def test_exact_results_are_marked_as_authoritative(self, evidence):
        items, store = evidence
        context = render_context(
            items, store, exact={"text": "The top comment has 1,000 likes."}
        )
        assert "1,000 likes" in context
        assert "EXACT" in context


class TestEvidenceNumbers:
    """The figures the citation guard is allowed to accept."""

    def test_collects_supports_endorsements_and_shares(self, evidence):
        items, store = evidence
        numbers = evidence_numbers(items, store)
        assert 6.0 in numbers          # support of the praise cluster
        assert 2020.0 in numbers       # its endorsement
        assert 11.0 in numbers         # corpus size
        assert 3142.0 in numbers       # total likes
        assert any(abs(n - 54.5) < 0.1 for n in numbers)  # its share as a percentage

    def test_includes_individual_like_counts(self, evidence):
        items, store = evidence
        assert 1000.0 in evidence_numbers(items, store)

    def test_includes_numbers_from_an_exact_result(self, evidence):
        items, store = evidence
        numbers = evidence_numbers(
            items, store, exact={"data": {"count": 42, "share": 0.5}}
        )
        assert 42.0 in numbers
        assert 50.0 in numbers  # share rendered as a percentage


class TestSystemPrompt:
    def test_tells_the_model_to_cite(self):
        assert "cite" in SYSTEM_PROMPT.lower()

    def test_forbids_inventing_figures(self):
        assert "invent" in SYSTEM_PROMPT.lower() or "do not" in SYSTEM_PROMPT.lower()

    def test_explains_that_support_means_people_not_relevance(self):
        assert "support" in SYSTEM_PROMPT.lower()

    def test_does_not_leak_the_old_merged_string_format(self):
        """The original prompt taught the model to parse 'comment is ... with likes='."""
        assert "with likes=" not in SYSTEM_PROMPT

"""Opinion clustering and consensus weighting -- the core of the approach.

A single retrieved comment tells you someone said something. It cannot tell you
how many people agreed. These tests pin down the behaviour that difference buys.
"""

import pytest

from ytrag.cluster import (
    ConsensusWeights,
    cluster_opinions,
    coverage,
    score_evidence,
)
from ytrag.store import HybridStore


@pytest.fixture
def store(comments, embedder):
    return HybridStore.build(comments, embedder)


@pytest.fixture
def clusters(store):
    return cluster_opinions(store, threshold=0.30)


def _cluster_containing(clusters, cid):
    return next(c for c in clusters if cid in c.member_cids)


class TestClusterOpinions:
    def test_paraphrases_of_one_opinion_land_together(self, clusters):
        praise = _cluster_containing(clusters, "c0")
        assert set(praise.member_cids) == {"c0", "c1", "c2", "c3", "c4", "c5"}

    def test_a_different_opinion_forms_its_own_cluster(self, clusters):
        pacing = _cluster_containing(clusters, "c6")
        assert set(pacing.member_cids) == {"c6", "c7", "c8"}

    def test_every_comment_belongs_to_exactly_one_cluster(self, clusters, comments):
        assigned = [cid for c in clusters for cid in c.member_cids]
        assert sorted(assigned) == sorted(c.cid for c in comments)

    def test_representative_is_the_most_liked_member(self, clusters):
        """The quote shown to the user should be the one the crowd endorsed."""
        praise = _cluster_containing(clusters, "c0")
        assert praise.representative_cid == "c4"  # 1000 likes, the camp's highest
        assert "vegapunk" in praise.representative_text.lower()

    def test_support_counts_people_and_endorsement_counts_likes(self, clusters):
        praise = _cluster_containing(clusters, "c0")
        assert praise.support == 6
        assert praise.endorsement == 20 + 15 + 12 + 900 + 1000 + 73

    def test_cluster_valence_averages_its_members(self, clusters):
        assert _cluster_containing(clusters, "c6").valence < 0
        assert _cluster_containing(clusters, "c0").valence > 0

    def test_a_high_threshold_splits_clusters_apart(self, store):
        assert len(cluster_opinions(store, threshold=0.99)) == store.total_comments

    def test_a_low_threshold_merges_everything(self, store):
        assert len(cluster_opinions(store, threshold=-1.0)) == 1

    def test_clustering_is_deterministic(self, store):
        first = cluster_opinions(store, threshold=0.30)
        second = cluster_opinions(store, threshold=0.30)
        assert [c.member_cids for c in first] == [c.member_cids for c in second]

    def test_threshold_defaults_to_the_embedders_recommendation(self, store):
        """No magic constant at the call site; the embedder knows its own scale."""
        assert cluster_opinions(store) == cluster_opinions(
            store, threshold=store.embedder.cluster_threshold
        )


class TestScoreEvidence:
    def test_the_relevant_camp_ranks_first(self, clusters, store):
        evidence = score_evidence(clusters, store, "what about the pacing of the arc")
        assert "c6" in evidence[0].cluster.member_cids

    def test_shares_are_reported_against_the_whole_corpus(self, clusters, store):
        evidence = score_evidence(clusters, store, "vegapunk")
        praise = next(e for e in evidence if "c0" in e.cluster.member_cids)
        assert praise.support_share == pytest.approx(6 / 11)
        assert praise.endorsement_share == pytest.approx(2020 / store.total_likes)

    def test_consensus_breaks_a_tie_between_equally_relevant_clusters(
        self, clusters, store
    ):
        """With salience switched off, the widely-held view must come first."""
        weights = ConsensusWeights(salience=0.0, support=1.0, endorsement=0.0)
        evidence = score_evidence(clusters, store, "anything", weights=weights)
        assert evidence[0].cluster.support == 6

    def test_endorsement_weighting_prefers_the_liked_camp(self, clusters, store):
        weights = ConsensusWeights(salience=0.0, support=0.0, endorsement=1.0)
        evidence = score_evidence(clusters, store, "anything", weights=weights)
        assert evidence[0].cluster.endorsement >= evidence[-1].cluster.endorsement

    def test_relevance_still_dominates_by_default(self, clusters, store):
        """A big cluster must not drown out a small, exactly-on-topic one."""
        evidence = score_evidence(clusters, store, "Mihawk Croc training")
        assert "c10" in evidence[0].cluster.member_cids

    def test_salience_is_not_inflated_by_cluster_size(self, clusters, store):
        """Regression: `max` over members gave big clusters more draws at the top.

        The praise camp (6 comments) must not out-score the pacing camp
        (3 comments) on a pacing question just by being larger.
        """
        evidence = score_evidence(clusters, store, "the pacing of the arc is slow")
        pacing = next(e for e in evidence if "c6" in e.cluster.member_cids)
        praise = next(e for e in evidence if "c0" in e.cluster.member_cids)
        assert pacing.salience > praise.salience
        assert pacing.score > praise.score

    def test_a_hugely_popular_cluster_cannot_hijack_an_unrelated_question(
        self, clusters, store
    ):
        """Social proof multiplies relevance; it never substitutes for it."""
        evidence = score_evidence(clusters, store, "Mihawk")
        assert "c10" in evidence[0].cluster.member_cids

    def test_quotes_are_ordered_by_relevance_to_this_question(self, clusters, store):
        evidence = score_evidence(clusters, store, "egghead", max_quotes=3)
        praise = next(e for e in evidence if "c0" in e.cluster.member_cids)
        assert "egghead" in praise.quotes[0].lower()

    def test_scores_are_ordered_and_finite(self, clusters, store):
        evidence = score_evidence(clusters, store, "vegapunk")
        scores = [e.score for e in evidence]
        assert scores == sorted(scores, reverse=True)
        assert all(s == s for s in scores)  # no NaN

    def test_limit_caps_the_evidence_returned(self, clusters, store):
        assert len(score_evidence(clusters, store, "arc", limit=2)) == 2

    def test_quotes_come_from_the_cluster_and_are_capped(self, clusters, store):
        evidence = score_evidence(clusters, store, "vegapunk", max_quotes=3)
        praise = next(e for e in evidence if "c0" in e.cluster.member_cids)
        assert 1 <= len(praise.quotes) <= 3
        assert all(isinstance(q, str) and q for q in praise.quotes)

    def test_weights_must_not_all_be_zero(self, clusters, store):
        with pytest.raises(ValueError, match="at least one"):
            score_evidence(
                clusters, store, "x",
                weights=ConsensusWeights(salience=0.0, support=0.0, endorsement=0.0),
            )


class TestCoverage:
    def test_coverage_is_the_share_of_the_corpus_the_answer_saw(self, clusters, store):
        evidence = score_evidence(clusters, store, "vegapunk", limit=1)
        assert coverage(evidence, store) == pytest.approx(6 / 11)

    def test_all_evidence_covers_the_whole_corpus(self, clusters, store):
        evidence = score_evidence(clusters, store, "x", limit=len(clusters))
        assert coverage(evidence, store) == pytest.approx(1.0)

    def test_no_evidence_is_zero_coverage(self, store):
        assert coverage([], store) == 0.0

"""Reciprocal Rank Fusion: how lexical and dense rankings are combined."""

import pytest

from ytrag.fusion import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_document_ranked_first_by_everyone_wins(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
        assert fused[0][0] == "a"

    def test_agreement_beats_a_single_first_place(self):
        """`b` is 2nd on both lists; `a` and `x` are 1st on only one each."""
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["x", "b", "y"]], k=1)
        assert fused[0][0] == "b"

    def test_scores_descend(self):
        fused = reciprocal_rank_fusion([["a", "b", "c", "d"]])
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_union_of_all_rankings_is_returned(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
        assert {doc for doc, _ in fused} == {"a", "b", "c"}

    def test_weights_shift_the_winner(self):
        unweighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[1.0, 1.0])
        weighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[0.0, 1.0])
        assert {d for d, _ in unweighted} == {"a", "b"}
        assert weighted[0][0] == "b"

    def test_empty_rankings_give_empty_result(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_limit_truncates(self):
        fused = reciprocal_rank_fusion([["a", "b", "c", "d"]], limit=2)
        assert len(fused) == 2

    def test_mismatched_weights_are_rejected(self):
        with pytest.raises(ValueError, match="weights"):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

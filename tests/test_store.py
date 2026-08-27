"""The hybrid store seam: lexical + dense retrieval over one corpus."""

import pytest

from ytrag.store import BM25Index, HybridStore


class TestBM25Index:
    def test_finds_the_document_containing_a_rare_word(self, comments):
        index = BM25Index([(c.cid, c.text) for c in comments])
        top = index.search("Mihawk", limit=1)
        assert top[0][0] == "c10"

    def test_ranks_by_relevance_not_insertion_order(self, comments):
        index = BM25Index([(c.cid, c.text) for c in comments])
        results = [cid for cid, _ in index.search("pacing slow terrible", limit=3)]
        assert set(results) <= {"c6", "c7", "c8"}

    def test_query_with_no_matching_term_returns_nothing(self, comments):
        index = BM25Index([(c.cid, c.text) for c in comments])
        assert index.search("cryptocurrency taxation", limit=5) == []

    def test_empty_corpus_is_safe(self):
        assert BM25Index([]).search("anything") == []


class TestHybridStore:
    def test_retrieves_the_relevant_camp(self, comments, embedder):
        store = HybridStore.build(comments, embedder)
        hits = [cid for cid, _ in store.search("what do people say about pacing", limit=4)]
        assert any(cid in {"c6", "c7", "c8"} for cid in hits)

    def test_hybrid_beats_dense_alone_on_a_rare_proper_noun(self, comments, embedder):
        """Lexical recall is the reason BM25 is fused in at all."""
        store = HybridStore.build(comments, embedder)
        hits = [cid for cid, _ in store.search("Mihawk", limit=3)]
        assert "c10" in hits

    def test_search_returns_at_most_limit(self, comments, embedder):
        store = HybridStore.build(comments, embedder)
        assert len(store.search("vegapunk", limit=2)) == 2

    def test_comments_are_retrievable_by_id(self, comments, embedder):
        store = HybridStore.build(comments, embedder)
        assert store.get("c4").author == "@fan4"
        assert store.get("nope") is None

    def test_vectors_align_with_comment_order(self, comments, embedder):
        store = HybridStore.build(comments, embedder)
        assert store.vectors.shape == (len(comments), embedder.dim)
        assert store.cids == [c.cid for c in comments]

    def test_totals_are_exposed_for_share_calculations(self, comments, embedder):
        store = HybridStore.build(comments, embedder)
        assert store.total_comments == 11
        assert store.total_likes == sum(c.likes for c in comments)

    def test_empty_corpus_is_rejected_with_a_clear_message(self, embedder):
        with pytest.raises(ValueError, match="no comments"):
            HybridStore.build([], embedder)


class TestPersistence:
    def test_roundtrips_through_disk(self, comments, embedder, tmp_path):
        original = HybridStore.build(comments, embedder)
        original.save(tmp_path / "kb")

        restored = HybridStore.load(tmp_path / "kb", embedder)
        assert restored.cids == original.cids
        assert restored.total_likes == original.total_likes
        assert restored.get("c6").text == original.get("c6").text
        assert [cid for cid, _ in restored.search("pacing", limit=3)] == [
            cid for cid, _ in original.search("pacing", limit=3)
        ]

    def test_loading_a_missing_index_is_a_clear_error(self, embedder, tmp_path):
        with pytest.raises(FileNotFoundError):
            HybridStore.load(tmp_path / "nothing-here", embedder)

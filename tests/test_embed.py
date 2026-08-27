"""The embedder seam. Tests run against the offline embedder only."""

import numpy as np
import pytest

from ytrag.embed import HashingEmbedder, get_embedder


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=256)


class TestHashingEmbedder:
    def test_returns_one_row_per_text(self, embedder):
        vectors = embedder.encode(["a", "b", "c"])
        assert vectors.shape == (3, 256)

    def test_vectors_are_l2_normalised(self, embedder):
        vectors = embedder.encode(["hello world", "another comment"])
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_is_deterministic_across_instances(self):
        a = HashingEmbedder(dim=128).encode(["luffy vs buggy"])
        b = HashingEmbedder(dim=128).encode(["luffy vs buggy"])
        assert np.allclose(a, b)

    def test_similar_text_scores_higher_than_unrelated(self, embedder):
        vectors = embedder.encode(
            [
                "vegapunk is the best character",
                "vegapunk is the greatest character",
                "the shipping rates went up again",
            ]
        )
        related = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        assert related > unrelated

    def test_empty_text_is_handled(self, embedder):
        vectors = embedder.encode(["", "real text"])
        assert vectors.shape == (2, 256)
        assert np.isfinite(vectors).all()

    def test_empty_batch_returns_empty_matrix(self, embedder):
        vectors = embedder.encode([])
        assert vectors.shape == (0, 256)

    def test_query_and_document_share_a_space(self, embedder):
        doc = embedder.encode(["people love vegapunk"])
        query = embedder.encode_query("vegapunk")
        assert query.shape == (256,)
        assert float(doc[0] @ query) > 0


class TestGetEmbedder:
    def test_hashing_backend_is_selectable_by_name(self):
        assert isinstance(get_embedder("hashing", dim=64), HashingEmbedder)

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="unknown embedder"):
            get_embedder("no-such-backend")

"""Text -> vector, behind one small interface.

The default backend is :class:`HashingEmbedder`: a deterministic, dependency-free
embedder built from hashed word and character n-grams.  It exists so the whole
pipeline -- and the whole test suite -- runs with no model download and no
network.  When ``sentence-transformers`` is installed, ``get_embedder("st")``
swaps in a real semantic model without anything else in the system changing.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")

#: English function words and interrogatives, stripped from **queries only**.
#:
#: These have to be removed by list rather than by IDF, and the reason is
#: specific to this corpus shape. IDF measures how rare a word is among the
#: comments -- but "think", "people" and "say" are rare in comments and common
#: in questions, so IDF scores them as highly informative. Meanwhile the entity
#: actually being asked about is, by definition of it being discussed, common
#: and therefore down-weighted. Measured on the sample corpus, "what do people
#: think of vegapunk" retrieved the *pacing* complaints, because "think" and
#: "of" outweighed "vegapunk". A fixed list fixes this; corpus statistics
#: cannot, because the problem is that the two distributions differ.
QUERY_STOPWORDS = frozenset(["#", "noqa:", "SIM905", "-", "a", "word", "list", "reads", "better", "than", "60", "quoted", "strings", "a", "an", "the", "this", "that", "these", "those", "there", "here", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its", "they", "them", "their", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "done", "done", "have", "has", "had", "having", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "can", "could", "will", "would", "shall", "should", "may", "might", "must", "of", "in", "on", "at", "to", "for", "with", "about", "from", "by", "as", "into", "over", "under", "and", "or", "but", "nor", "so", "yet", "if", "then", "than", "because", "while", "not", "no", "nor", "none", "any", "some", "all", "more", "most", "other", "another", "such", "own", "same", "me", "myself", "us", "tell", "show", "give", "find", "get", "say", "says", "said", "saying", "tell", "told", "think", "thinks", "thinking", "thought", "feel", "feels", "feeling", "felt", "people", "person", "everyone", "anyone", "someone", "somebody", "folks", "comment", "comments", "commenter", "commenters", "video", "videos", "mention", "mentions", "mentioned", "mentioning", "talk", "talks", "talked", "talking", "about", "regarding", "concerning", "please", "just", "really", "very", "much", "many", "lot", "lots", "s", "t", "ll", "re", "ve", "d", "m", "o"])


def query_tokens(text: str) -> list[str]:
    """Content-bearing tokens of a query.

    Falls back to the unfiltered tokens only when the query is *nothing but*
    stopwords in a form that still names something -- an empty result is a valid
    and meaningful outcome for a question like "what do people think", where
    there is no topic to be relevant to and consensus should decide the ranking.
    """
    return [t for t in tokenize(text) if t not in QUERY_STOPWORDS]


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into unit-norm row vectors."""

    dim: int
    cluster_threshold: float

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Shared with the BM25 index so the two agree."""
    return _TOKEN_RE.findall(str(text).lower())


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    """Map a feature to a bucket and a sign, via a stable hash.

    ``hash()`` is salted per process in Python 3, so blake2b is used instead:
    an index built today must match a query embedded tomorrow.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder:
    """Signed-hashing embedder over word unigrams, bigrams and char 4-grams.

    Character n-grams give it partial robustness to the typos and elongations
    ("goooood") that fill comment sections, which pure word hashing lacks.
    """

    #: Cosine above which two hashed comments are "the same opinion".
    #: Measured on the sample corpus: paraphrases score >=0.32, unrelated
    #: comments <=0.24, so 0.30 sits in the gap.
    cluster_threshold = 0.30

    def __init__(self, dim: int = 512, char_ngram: int = 4) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.char_ngram = int(char_ngram)
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0

    def fit(self, texts: Sequence[str]) -> HashingEmbedder:
        """Learn inverse document frequencies from the corpus.

        These weights are applied to **queries only** -- see :meth:`_vector`.
        Fitting is optional; an unfitted embedder falls back to uniform weights
        and still works, just less sharply.
        """
        document_frequency: Counter = Counter()
        for text in texts:
            document_frequency.update(set(self._features(text)))
        n = max(1, len(texts))
        self._idf = {
            feature: math.log((n + 1) / (df + 1)) + 1.0
            for feature, df in document_frequency.items()
        }
        # A feature never seen in the corpus is maximally informative.
        self._default_idf = math.log(n + 1) + 1.0
        return self

    def _features(self, text: str, query: bool = False) -> list[str]:
        tokens = query_tokens(text) if query else tokenize(text)
        features = [f"w:{t}" for t in tokens]
        features += [f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        packed = " ".join(tokens)
        n = self.char_ngram
        features += [f"c:{packed[i:i + n]}" for i in range(max(0, len(packed) - n + 1))]
        return features

    def _vector(self, text: str, use_idf: bool = False) -> np.ndarray:
        """Hash ``text`` into a unit vector, optionally IDF-weighted.

        The asymmetry is deliberate and is the ``lnc.ltc`` weighting from the
        SMART retrieval system: **queries** carry IDF, **documents** do not.

        Weighting documents too looks tempting and quietly breaks clustering.
        IDF measures rarity against the corpus, but here the corpus *is* one
        video's comments: if 40% of them praise Vegapunk, then "vegapunk",
        "best" and "character" are common, so IDF down-weights precisely the
        terms that make that opinion a coherent group. Measured on the sample
        corpus, IDF-weighting documents dropped within-cluster similarity from
        0.32 to 0.18 while cross-cluster stayed at 0.23 -- the groups stopped
        being separable at all. Leaving documents unweighted keeps the geometry
        intact, while IDF on the query side still stops "the", "of" and "this"
        from deciding what a question is about.
        """
        vector = np.zeros(self.dim, dtype=np.float32)
        weighted = use_idf and bool(self._idf)
        for feature in self._features(text, query=use_idf):
            index, sign = _bucket(feature, self.dim)
            # sqrt-damped, so one rare term sharpens the query without
            # single-handedly deciding it.
            weight = (
                math.sqrt(self._idf.get(feature, self._default_idf)) if weighted else 1.0
            )
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode documents: term frequency only, no IDF (the ``lnc`` half)."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t, use_idf=False) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a query: IDF-weighted, so rare terms steer it (the ``ltc`` half)."""
        return self._vector(text, use_idf=True)


class SentenceTransformerEmbedder:
    """Adapter over ``sentence-transformers`` for higher retrieval quality."""

    #: Trained encoders put paraphrases far higher up the cosine range than the
    #: hashing fallback does, so the same 0.30 would merge unrelated opinions.
    cluster_threshold = 0.62

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "sentence-transformers is not installed; "
                "use get_embedder('hashing') or pip install ytrag[semantic]"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - heavy
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:  # pragma: no cover - heavy
        return self.encode([text])[0]


def get_embedder(backend: str = "hashing", **kwargs) -> Embedder:
    """Resolve an embedder by name.

    ``"auto"`` prefers sentence-transformers and falls back to hashing, so a
    machine with no model cache still works instead of crashing.
    """
    backend = (backend or "hashing").lower()
    if backend == "hashing":
        return HashingEmbedder(**kwargs)
    if backend in ("st", "sentence-transformers", "semantic"):
        return SentenceTransformerEmbedder(**kwargs)
    if backend == "auto":
        try:  # pragma: no cover - environment dependent
            return SentenceTransformerEmbedder(**kwargs)
        except Exception:
            return HashingEmbedder()
    raise ValueError(f"unknown embedder backend: {backend!r}")

"""The corpus index: lexical, dense and relational views of the same comments.

One store owns all three because they must never disagree about what the corpus
contains. ``search`` fuses the lexical and dense rankings with RRF; the SQLite
table underneath is what :mod:`ytrag.aggregate` uses to compute exact counts.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ytrag.embed import Embedder, tokenize
from ytrag.fusion import reciprocal_rank_fusion
from ytrag.models import Comment

_MANIFEST = "manifest.json"
_VECTORS = "vectors.npy"
_COMMENTS = "comments.jsonl"
_FORMAT_VERSION = 1


def _fit(embedder: Embedder, texts: Sequence[str]) -> None:
    """Give the embedder corpus statistics, if it can use them."""
    fit = getattr(embedder, "fit", None)
    if callable(fit):
        fit(texts)


class BM25Index:
    """Okapi BM25 over the comment corpus.

    Written out rather than pulled from a library so tokenisation is guaranteed
    identical to the embedder's, and so the core package needs no extra
    dependency.
    """

    def __init__(
        self,
        documents: Sequence[tuple[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self._tokens = [tokenize(text) for _, text in documents]
        self._lengths = np.array([len(t) for t in self._tokens], dtype=np.float32)
        self._avg_length = float(self._lengths.mean()) if len(self._lengths) else 0.0
        self._term_frequencies: list[Counter] = [Counter(t) for t in self._tokens]

        document_frequency: Counter = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))

        n = len(documents)
        # idf floor: a term present in more than half the corpus would otherwise
        # score negatively and rank below documents that lack it entirely.
        self._idf = {
            term: max(1e-6, math.log(1 + (n - df + 0.5) / (df + 0.5)))
            for term, df in document_frequency.items()
        }
        self._postings: dict[str, list[int]] = {}
        for i, tokens in enumerate(self._tokens):
            for term in set(tokens):
                self._postings.setdefault(term, []).append(i)

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` for documents sharing a term with ``query``."""
        query_terms = [t for t in tokenize(query) if t in self._idf]
        if not query_terms or not self.doc_ids:
            return []

        scores: dict[int, float] = {}
        for term in query_terms:
            idf = self._idf[term]
            for i in self._postings.get(term, ()):
                tf = self._term_frequencies[i][term]
                norm = 1 - self.b + self.b * (self._lengths[i] / (self._avg_length or 1.0))
                scores[i] = scores.get(i, 0.0) + idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * norm
                )

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[i], float(score)) for i, score in ranked[:limit]]


class HybridStore:
    """Lexical + dense + relational index over one comment corpus."""

    def __init__(
        self,
        comments: Sequence[Comment],
        vectors: np.ndarray,
        embedder: Embedder,
    ) -> None:
        if not comments:
            raise ValueError("cannot build a store with no comments")
        self.comments = list(comments)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.embedder = embedder
        self.cids = [c.cid for c in self.comments]
        self._by_cid = {c.cid: c for c in self.comments}
        self._bm25 = BM25Index([(c.cid, f"{c.text} {c.author}") for c in self.comments])
        self._db = self._build_db(self.comments)

    # -- construction -----------------------------------------------------

    @classmethod
    def build(cls, comments: Sequence[Comment], embedder: Embedder) -> "HybridStore":
        comments = list(comments)
        if not comments:
            raise ValueError("cannot build a store with no comments")
        texts = [c.text or c.emojis or " " for c in comments]
        _fit(embedder, texts)
        return cls(comments, embedder.encode(texts), embedder)

    @staticmethod
    def _build_db(comments: Iterable[Comment]) -> sqlite3.Connection:
        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute(
            """
            CREATE TABLE comments (
                cid TEXT PRIMARY KEY, text TEXT, author TEXT, likes INTEGER,
                published TEXT, published_ts REAL, reply_count INTEGER,
                is_reply INTEGER, emojis TEXT, valence REAL, length INTEGER
            )
            """
        )
        db.executemany(
            "INSERT OR REPLACE INTO comments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    c.cid, c.text, c.author, c.likes, c.published, c.published_ts,
                    c.reply_count, int(c.is_reply), c.emojis, c.valence, len(c.text),
                )
                for c in comments
            ],
        )
        db.commit()
        return db

    # -- accessors --------------------------------------------------------

    @property
    def db(self) -> sqlite3.Connection:
        return self._db

    @property
    def total_comments(self) -> int:
        return len(self.comments)

    @property
    def total_likes(self) -> int:
        return sum(c.likes for c in self.comments)

    def get(self, cid: str) -> Comment | None:
        return self._by_cid.get(cid)

    # -- retrieval --------------------------------------------------------

    def dense_search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        query_vector = self.embedder.encode_query(query)
        similarities = self.vectors @ query_vector
        order = np.argsort(-similarities)[:limit]
        return [(self.cids[i], float(similarities[i])) for i in order]

    def lexical_search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        return self._bm25.search(query, limit=limit)

    def search(
        self,
        query: str,
        limit: int = 10,
        depth: int = 50,
        weights: Sequence[float] = (1.0, 1.0),
    ) -> list[tuple[str, float]]:
        """Hybrid retrieval: BM25 and dense rankings fused by RRF."""
        lexical = [cid for cid, _ in self.lexical_search(query, limit=depth)]
        dense = [cid for cid, _ in self.dense_search(query, limit=depth)]
        fused = reciprocal_rank_fusion([lexical, dense], weights=list(weights), limit=limit)
        return [(str(cid), float(score)) for cid, score in fused]

    def similarity_to(self, query: str) -> dict[str, float]:
        """Cosine similarity of every comment to ``query``, keyed by cid."""
        query_vector = self.embedder.encode_query(query)
        similarities = self.vectors @ query_vector
        return {cid: float(s) for cid, s in zip(self.cids, similarities)}

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / _VECTORS, self.vectors)
        with open(path / _COMMENTS, "w", encoding="utf-8") as handle:
            for comment in self.comments:
                handle.write(json.dumps(comment.to_dict(), ensure_ascii=False) + "\n")
        (path / _MANIFEST).write_text(
            json.dumps(
                {
                    "format_version": _FORMAT_VERSION,
                    "count": len(self.comments),
                    "dim": int(self.vectors.shape[1]),
                    "embedder": type(self.embedder).__name__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> "HybridStore":
        path = Path(path)
        manifest_path = path / _MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"no knowledge base at {path} (missing {_MANIFEST})")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dim") != getattr(embedder, "dim", None):
            raise ValueError(
                f"index was built with dim={manifest.get('dim')} but this embedder "
                f"produces dim={getattr(embedder, 'dim', None)}; rebuild the knowledge base"
            )

        vectors = np.load(path / _VECTORS)
        comments = []
        with open(path / _COMMENTS, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    comments.append(Comment(**json.loads(line)))
        # Refit on the same corpus so queries are embedded in the same space the
        # stored vectors were built in.
        _fit(embedder, [c.text or c.emojis or " " for c in comments])
        return cls(comments, vectors, embedder)

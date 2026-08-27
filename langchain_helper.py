"""Backwards-compatible shim for the original module's two functions.

The original ``langchain_helper`` could not be imported at all: line 15 read

    llm = ChatGoogleGenerativeAI(model=..., google_api_key=apikey)

with ``apikey`` never defined, so ``import langchain_helper`` raised
``NameError`` before anything ran. It also read ``HUGGINGFACEHUB_API_TOKEN``
with ``os.environ[...]`` at import time, raising ``KeyError`` on any machine
without one, and loaded the CSV from a hard-coded
``H:\\data science roadmap\\langchain\\youtubeproj\\`` path.

The two public functions are preserved here so existing scripts keep working,
implemented on top of :mod:`ytrag`. New code should use :class:`ytrag.CommentRAG`
directly -- it returns evidence, coverage and citations, which this interface
has no way to express.
"""

from __future__ import annotations

import warnings
from typing import Any

from ytrag.engine import DEFAULT_INDEX, CommentRAG
from ytrag.llm import get_llm

vectordb_file_path = DEFAULT_INDEX


def create_vector_db(Url: str, limit: int = 500, path: str = DEFAULT_INDEX) -> CommentRAG:
    """Download a video's comments and save a knowledge base.

    Unlike the original, this raises on failure instead of printing the
    exception and continuing on to index a file that was never written.
    """
    rag = CommentRAG.from_youtube(Url, limit=limit)
    rag.save(path)
    return rag


class _ChainShim:
    """Mimics the LangChain ``.invoke({"input": ...})`` -> ``{"answer": ...}`` contract."""

    def __init__(self, rag: CommentRAG) -> None:
        self.rag = rag

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        answer = self.rag.ask(payload.get("input", ""))
        return {
            "input": answer.question,
            "answer": answer.text,
            "context": [e.cluster.representative_text for e in answer.evidence],
            "coverage": answer.coverage,
            "citations": answer.citations,
        }


def get_qa_chain(path: str = DEFAULT_INDEX, llm_backend: str | None = None) -> _ChainShim:
    """Load a saved knowledge base and return an invokable chain."""
    warnings.warn(
        "langchain_helper is a compatibility shim; use ytrag.CommentRAG for "
        "evidence, coverage and citations.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _ChainShim(CommentRAG.load(path, llm=get_llm(llm_backend)))


if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain.invoke({"input": "which comment has the most likes?"})["answer"])

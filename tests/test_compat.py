"""The original module's public surface still works.

The point of these is narrow but real: the original `langchain_helper` could not
be imported at all, so anything that depended on it was dead. It imports now,
and its two functions behave as their names promise.
"""

import pytest

from ytrag.engine import CommentRAG


def test_the_module_imports_without_any_api_keys(monkeypatch):
    """The original raised KeyError on import when HUGGINGFACEHUB_API_TOKEN was unset."""
    for env in ("HUGGINGFACEHUB_API_TOKEN", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    import importlib

    import langchain_helper

    importlib.reload(langchain_helper)
    assert hasattr(langchain_helper, "create_vector_db")
    assert hasattr(langchain_helper, "get_qa_chain")


def test_get_qa_chain_returns_the_expected_contract(comments, tmp_path):
    import langchain_helper

    CommentRAG.from_comments(comments, cluster_threshold=0.30).save(tmp_path / "kb")

    with pytest.deprecated_call():
        chain = langchain_helper.get_qa_chain(path=str(tmp_path / "kb"))

    response = chain.invoke({"input": "which comment has the most likes?"})
    assert "answer" in response
    assert "1,000 likes" in response["answer"]


def test_the_chain_also_exposes_the_new_grounding_fields(comments, tmp_path):
    import langchain_helper

    CommentRAG.from_comments(comments, cluster_threshold=0.30).save(tmp_path / "kb")
    with pytest.deprecated_call():
        chain = langchain_helper.get_qa_chain(path=str(tmp_path / "kb"))

    response = chain.invoke({"input": "what do people think overall"})
    assert 0.0 <= response["coverage"] <= 1.0
    assert isinstance(response["citations"], list)

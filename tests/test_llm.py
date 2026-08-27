"""Provider selection and the offline extractive backend.

No test here touches the network. The hosted providers are checked only for
their selection and error behaviour, which is where they actually go wrong.
"""

import pytest

from ytrag.cluster import cluster_opinions, score_evidence
from ytrag.llm import ExtractiveLLM, available_providers, get_llm
from ytrag.prompt import SYSTEM_PROMPT
from ytrag.store import HybridStore


@pytest.fixture
def evidence(comments, embedder):
    store = HybridStore.build(comments, embedder)
    return score_evidence(cluster_opinions(store, threshold=0.30), store, "vegapunk", limit=3)


class TestExtractiveLLM:
    def test_states_the_size_of_the_leading_view(self, evidence):
        answer = ExtractiveLLM(evidence).complete(SYSTEM_PROMPT, "what do people think")
        assert "6 comments" in answer
        assert "54.5%" in answer

    def test_cites_the_cluster_it_quotes(self, evidence):
        answer = ExtractiveLLM(evidence).complete(SYSTEM_PROMPT, "q")
        assert "[c4]" in answer

    def test_quotes_a_real_comment(self, evidence):
        answer = ExtractiveLLM(evidence).complete(SYSTEM_PROMPT, "q")
        assert "vegapunk" in answer.lower()

    def test_mentions_more_than_one_camp(self, evidence):
        answer = ExtractiveLLM(evidence).complete(SYSTEM_PROMPT, "q")
        assert "A further" in answer

    def test_leads_with_an_exact_result_when_there_is_one(self, evidence):
        answer = ExtractiveLLM(evidence, exact="The top comment has 1,000 likes.").complete(
            SYSTEM_PROMPT, "q"
        )
        assert answer.startswith("The top comment has 1,000 likes.")

    def test_says_so_when_there_is_no_evidence(self):
        answer = ExtractiveLLM([]).complete(SYSTEM_PROMPT, "q")
        assert "no comments" in answer.lower()

    def test_an_exact_result_alone_is_still_an_answer(self):
        answer = ExtractiveLLM([], exact="There are 11 comments.").complete(SYSTEM_PROMPT, "q")
        assert answer == "There are 11 comments."

    def test_is_deterministic(self, evidence):
        first = ExtractiveLLM(evidence).complete(SYSTEM_PROMPT, "q")
        second = ExtractiveLLM(evidence).complete(SYSTEM_PROMPT, "q")
        assert first == second

    def test_never_states_a_figure_it_was_not_given(self, evidence):
        """It builds from pipeline output, so hallucinated numbers are impossible."""
        from ytrag.citations import CitationGuard

        answer = ExtractiveLLM(evidence).complete(SYSTEM_PROMPT, "q")
        allowed_ids = {e.cluster.representative_cid for e in evidence}
        # Numbers it can emit are exactly support/endorsement/share per cluster.
        allowed = set()
        for e in evidence:
            allowed |= {
                float(e.cluster.support),
                float(e.cluster.endorsement),
                round(e.support_share * 100, 1),
            }
        report = CitationGuard(allowed_ids, allowed).verify(answer)
        assert report.grounded, report.warnings


class TestProviderSelection:
    def test_extractive_is_selectable_by_name(self):
        assert get_llm("extractive").name == "extractive"

    def test_auto_falls_back_to_extractive_with_no_keys(self, monkeypatch):
        """The app must start on a machine with no credentials at all."""
        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HUGGINGFACEHUB_API_TOKEN"):
            monkeypatch.delenv(env, raising=False)
        assert get_llm("auto").name == "extractive"
        assert get_llm(None).name == "extractive"

    def test_unknown_backend_is_rejected_with_the_valid_options(self):
        with pytest.raises(ValueError, match="unknown LLM backend"):
            get_llm("gpt-5-turbo-ultra")

    def test_a_named_provider_without_its_key_fails_loudly(self, monkeypatch):
        """Explicitly asking for Claude with no key must not silently downgrade."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises((ValueError, ImportError)):
            get_llm("anthropic")

    def test_available_providers_always_includes_extractive(self, monkeypatch):
        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HUGGINGFACEHUB_API_TOKEN"):
            monkeypatch.delenv(env, raising=False)
        assert available_providers() == ["extractive"]

    def test_available_providers_reflects_the_environment(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert "anthropic" in available_providers()

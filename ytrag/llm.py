"""Answer generation, behind one interface with a working offline default.

The default backend is :class:`ExtractiveLLM`, which needs no API key and no
network. It is not a fallback stub: it composes a real proportional summary from
the evidence -- "a majority view (54.5%, 6 comments, 2,020 likes) says X, while
27.3% say Y" -- which is most of what a comment-section question actually wants.
Because the numbers come from the pipeline rather than a model, it cannot
hallucinate them at all.

Setting an API key upgrades phrasing and multi-step reasoning. It does not
change what the system can answer, which is the point: the retrieval work is
where the accuracy lives, and the model is there to write it up.

The original project required ``HUGGINGFACEHUB_API_TOKEN`` at import time, so
``import langchain_helper`` failed on any machine without one -- before a single
question had been asked.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from ytrag.models import Evidence

_MAX_TOKENS = 800


@runtime_checkable
class LLM(Protocol):
    """Anything that can turn a system prompt and a user message into text."""

    name: str

    def complete(self, system: str, user: str) -> str: ...


class ExtractiveLLM:
    """Compose an answer from the evidence, with no model in the loop.

    Deterministic, free, offline, and incapable of inventing a figure. The
    trade-off is register: it summarises the distribution rather than reasoning
    about it, so a question like "why do people dislike the pacing" gets the
    relevant camps and their sizes rather than an explanation.
    """

    name = "extractive"

    def __init__(self, evidence: list[Evidence] | None = None, exact: str = "") -> None:
        self.evidence = evidence or []
        self.exact = exact

    def complete(self, system: str, user: str) -> str:
        parts: list[str] = []
        if self.exact:
            parts.append(self.exact)

        if not self.evidence:
            if not parts:
                return (
                    "No comments in this knowledge base match that question. "
                    "Try rephrasing it, or rebuild the knowledge base with more comments."
                )
            return " ".join(parts)

        leader = self.evidence[0]
        parts.append(
            f"The most prominent view here is held by {leader.cluster.support:,} "
            f"comment{'s' if leader.cluster.support != 1 else ''} "
            f"({leader.support_share * 100:.1f}% of the comment section) with "
            f"{leader.cluster.endorsement:,} likes behind it "
            f"[{leader.cluster.representative_cid}]: "
            f'"{leader.quotes[0]}"'
        )

        for item in self.evidence[1:4]:
            parts.append(
                f"A further {item.cluster.support:,} comment"
                f"{'s' if item.cluster.support != 1 else ''} "
                f"({item.support_share * 100:.1f}%, {item.cluster.endorsement:,} likes) "
                f'[{item.cluster.representative_cid}]: "{item.quotes[0]}"'
            )

        return " ".join(parts)


class AnthropicLLM:
    """Claude, via the Anthropic SDK."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("pip install anthropic") from exc
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - network
        response = self._client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAILLM:
    """GPT models, via the OpenAI SDK."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("pip install openai") from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model
        self._client = OpenAI(api_key=key)

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - network
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class HuggingFaceLLM:
    """Hosted inference, keeping the original project's Mistral endpoint working."""

    name = "huggingface"

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        api_key: str | None = None,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("pip install huggingface_hub") from exc
        key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")
        self.model = model
        self._client = InferenceClient(model=model, token=key)

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - network
        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=_MAX_TOKENS,
        )
        return response.choices[0].message.content or ""


_PROVIDERS = {
    "extractive": ExtractiveLLM,
    "anthropic": AnthropicLLM,
    "claude": AnthropicLLM,
    "openai": OpenAILLM,
    "huggingface": HuggingFaceLLM,
    "hf": HuggingFaceLLM,
}

_AUTO_ORDER = [
    ("ANTHROPIC_API_KEY", AnthropicLLM),
    ("OPENAI_API_KEY", OpenAILLM),
    ("HUGGINGFACEHUB_API_TOKEN", HuggingFaceLLM),
]


def available_providers() -> list[str]:
    """Provider names that could actually run right now."""
    ready = ["extractive"]
    ready += [
        provider.name for env, provider in _AUTO_ORDER if os.getenv(env)
    ]
    return ready


def get_llm(backend: str | None = None, **kwargs) -> LLM:
    """Resolve a backend by name, or auto-detect one from the environment.

    Auto-detection never raises: with no key set it returns the extractive
    backend, so the application always starts.
    """
    if backend in (None, "", "auto"):
        for env, provider in _AUTO_ORDER:
            if os.getenv(env):
                try:
                    return provider(**kwargs)
                except Exception:  # pragma: no cover - defensive
                    continue
        return ExtractiveLLM(**kwargs)

    key = backend.lower()
    if key not in _PROVIDERS:
        raise ValueError(
            f"unknown LLM backend: {backend!r}; choose from {sorted(set(_PROVIDERS))}"
        )
    return _PROVIDERS[key](**kwargs)

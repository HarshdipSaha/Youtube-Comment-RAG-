"""Rendering evidence for a language model.

The original project handed the model rows of

    comment is '...' with likes= 20 with user_id '...' and published(time) '...'

and spent its system prompt teaching the model to parse that string back into
fields. That is work the pipeline should have done, and it left the model with
four arbitrary comments and no idea what fraction of the conversation they
represented.

Here each block is an *opinion*, already carrying its own social proof:

    [c4] 6 comments (54.5% of all comments), 2,020 likes (64.3% of all likes)
      - "best character is vegapunk for sure"
      - "Vegapunk really is the best character here"

Now "most people think X" is a statement the model can make from what it was
given, and the citation guard can check the figures afterwards because they were
computed rather than described.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ytrag.models import Evidence
from ytrag.store import HybridStore

SYSTEM_PROMPT = """\
You answer questions about a YouTube video's comment section.

You are given OPINION CLUSTERS, not individual comments. Each cluster is a group
of people who said substantially the same thing, and it is labelled with:
  - an id in square brackets, e.g. [c4] -- cite it as [c4]
  - support: how many people expressed that view, and what share of all comments
  - endorsement: how many likes those comments drew, and what share of all likes
  - a few representative quotes

How to use them:
  - Support is how many people held a view. It is NOT how relevant the view is
    to the question, and a cluster appearing first does not make it a majority.
  - Quantify when the numbers allow it: "roughly two thirds of commenters (68%)"
    beats "many people". Use only the figures given to you.
  - Likes and comment counts measure different things. A view held by three
    people with 9,000 likes is popular; a view held by 300 people is common.
    If they disagree, say so rather than picking one.
  - A block marked EXACT was computed over every comment, not retrieved. Its
    figures are authoritative -- use them verbatim and never recompute them.

Rules:
  - Cite the cluster id for every claim you make, e.g. "viewers loved it [c4]".
  - Never invent a number. If a figure is not in the context, do not state it.
  - If the context does not answer the question, say so plainly. Do not offer
    "the nearest answer" -- a confident wrong answer is worse than none.
  - Answer in prose, in the register of someone summarising a comment section.
"""


def _pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def render_context(
    evidence: Sequence[Evidence],
    store: HybridStore,
    exact: dict[str, Any] | None = None,
) -> str:
    """Render evidence into the context block handed to the model."""
    lines: list[str] = [
        f"CORPUS: {store.total_comments:,} comments carrying "
        f"{store.total_likes:,} likes in total.",
        "",
    ]

    if exact and exact.get("text"):
        lines += [
            "EXACT RESULT (computed over every comment; authoritative):",
            f"  {exact['text']}",
            "",
        ]

    if not evidence:
        lines.append("OPINION CLUSTERS: no comments matched this question.")
        return "\n".join(lines)

    lines.append("OPINION CLUSTERS (ranked by relevance and consensus):")
    for item in evidence:
        cluster = item.cluster
        lines.append(
            f"\n[{cluster.representative_cid}] "
            f"{cluster.support:,} comment{'s' if cluster.support != 1 else ''} "
            f"({_pct(item.support_share)} of all comments), "
            f"{cluster.endorsement:,} likes ({_pct(item.endorsement_share)} of all likes)"
        )
        for quote in item.quotes:
            lines.append(f'  - "{quote}"')

    return "\n".join(lines)


def evidence_numbers(
    evidence: Sequence[Evidence],
    store: HybridStore,
    exact: dict[str, Any] | None = None,
) -> set[float]:
    """Every figure the answer is entitled to state.

    Handed to :class:`~ytrag.citations.CitationGuard`, which reports any
    percentage or count in the answer that is not in this set.
    """
    numbers: set[float] = {float(store.total_comments), float(store.total_likes)}

    for item in evidence:
        numbers.add(float(item.cluster.support))
        numbers.add(float(item.cluster.endorsement))
        numbers.add(round(item.support_share * 100, 1))
        numbers.add(round(item.endorsement_share * 100, 1))
        for cid in item.cluster.member_cids:
            comment = store.get(cid)
            if comment is not None:
                numbers.add(float(comment.likes))

    if exact:
        for value in (exact.get("data") or {}).values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numbers.add(float(value))
                # Fractions are usually restated as percentages in prose.
                if 0.0 <= float(value) <= 1.0:
                    numbers.add(round(float(value) * 100, 1))
            elif isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        numbers.update(
                            float(v) for v in row.values()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                        )
    return numbers

"""Figures for the comment section.

Three charts, each answering a question the numbers alone answer badly:

* **Opinion share** -- how the comment section divides, by people and by likes
  side by side. These two routinely disagree, and the disagreement is the
  finding: a view held by many people with few likes is a different thing from
  a view held by few people with many.
* **Like distribution** -- comment sections are extremely top-heavy, and a log
  scale is the only way to see both the one comment with 40,000 likes and the
  four hundred with two.
* **Activity over time** -- when the conversation actually happened.

matplotlib is an optional dependency; :func:`charts_available` lets callers
degrade gracefully rather than crash.
"""

from __future__ import annotations

from typing import Any

# A neutral, colour-blind-safe palette. Positive/negative use blue/orange
# rather than green/red so the charts stay readable for the ~8% of viewers with
# red-green colour vision deficiency.
POSITIVE = "#2E7DB8"
NEGATIVE = "#E08A26"
NEUTRAL = "#8C8C8C"
ACCENT = "#4B4B8F"
GRID = "#D9D9D9"


def charts_available() -> bool:
    """Whether matplotlib can be imported in this environment."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _figure(width: float, height: float):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(width, height), constrained_layout=True)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="x", color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    return figure, axes


def _valence_colour(valence: float) -> str:
    if valence >= 0.2:
        return POSITIVE
    if valence <= -0.2:
        return NEGATIVE
    return NEUTRAL


def opinion_share(overview: dict[str, Any], top: int = 8):
    """Horizontal bars: share of commenters vs share of likes, per opinion."""
    import numpy as np

    clusters = (overview.get("clusters") or [])[:top]
    if not clusters:
        return None

    total_likes = max(1, overview["stats"]["total_likes"])
    labels = [
        (c["text"][:52] + "…") if len(c["text"]) > 52 else (c["text"] or "(emoji only)")
        for c in clusters
    ]
    people = [c["support_share"] * 100 for c in clusters]
    likes = [c["endorsement"] / total_likes * 100 for c in clusters]

    figure, axes = _figure(9, 0.62 * len(clusters) + 1.6)
    positions = np.arange(len(clusters))
    height = 0.38

    axes.barh(positions + height / 2, people, height, label="% of commenters", color=ACCENT)
    axes.barh(
        positions - height / 2, likes, height, label="% of likes",
        color=[_valence_colour(c["valence"]) for c in clusters],
    )

    axes.set_yticks(positions, labels, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlabel("share of the comment section (%)")
    axes.set_title("How the comment section divides", loc="left", fontsize=12)
    axes.legend(frameon=False, loc="lower right", fontsize=9)
    return figure


def like_distribution(comments, bins: int = 24):
    """Histogram of likes on a log-ish axis, because the tail is the story."""
    import numpy as np

    likes = np.array([c.likes for c in comments], dtype=float)
    if likes.size == 0:
        return None

    figure, axes = _figure(7, 3.6)
    axes.hist(np.log10(likes + 1), bins=bins, color=ACCENT, edgecolor="white", linewidth=0.6)

    highest = int(likes.max())
    ticks = [t for t in (0, 1, 10, 100, 1_000, 10_000, 100_000) if t <= max(highest, 1)]
    axes.set_xticks([np.log10(t + 1) for t in ticks], [f"{t:,}" for t in ticks])
    axes.set_xlabel("likes per comment")
    axes.set_ylabel("comments")
    axes.set_title(
        f"Like distribution — median {int(np.median(likes)):,}, top comment {highest:,}",
        loc="left", fontsize=12,
    )
    return figure


def activity_timeline(timeline: list[dict[str, Any]]):
    """Comments and likes per time bucket, oldest on the left."""
    import numpy as np

    if not timeline:
        return None

    figure, axes = _figure(7.5, 3.4)
    positions = np.arange(len(timeline))
    axes.bar(positions, [b["count"] for b in timeline], color=ACCENT, width=0.66)
    axes.set_xticks(
        positions,
        [str(b["label"])[:16] for b in timeline],
        rotation=30, ha="right", fontsize=8,
    )
    axes.set_ylabel("comments")
    axes.set_title("When the conversation happened", loc="left", fontsize=12)

    likes_axis = axes.twinx()
    likes_axis.plot(
        positions, [b["likes"] for b in timeline],
        color=NEGATIVE, marker="o", linewidth=1.8, markersize=4,
    )
    likes_axis.set_ylabel("likes", color=NEGATIVE)
    likes_axis.tick_params(axis="y", colors=NEGATIVE)
    likes_axis.spines[["top"]].set_visible(False)
    return figure


def save(figure, path: str) -> str:
    """Write a figure to disk and close it."""
    import matplotlib.pyplot as plt

    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path

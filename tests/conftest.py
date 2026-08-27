"""Shared fixtures. Everything here is offline and deterministic."""

import pytest

from ytrag.embed import HashingEmbedder
from ytrag.models import Comment

NOW = 1_700_000_000.0


def _c(cid, text, likes, author, hours_ago, valence=0.0, emojis=""):
    return Comment(
        cid=cid,
        text=text,
        author=author,
        likes=likes,
        published=f"{hours_ago} hours ago",
        published_ts=NOW - hours_ago * 3600,
        emojis=emojis,
        valence=valence,
    )


@pytest.fixture
def comments():
    """A small corpus with a deliberate opinion split.

    Six commenters praise Vegapunk (2,020 likes between them), three complain
    about the pacing (1,100 likes), and two talk about Buggy.  The praise camp
    is larger by count; the pacing camp has the single most-liked comment.  That
    tension is what the consensus layer has to represent honestly.
    """
    return [
        _c("c0", "Vegapunk is the best character in egghead", 20, "@Gleeblevr", 8, 0.0),
        _c("c1", "Vegapunk is the greatest character of the arc", 15, "@fan1", 9),
        _c("c2", "vegapunk best character no debate", 12, "@fan2", 10),
        _c("c3", "Vegapunk really is the best character here", 900, "@fan3", 11, 1.0, "🔥"),
        _c("c4", "best character is vegapunk for sure", 1000, "@fan4", 12, 1.0, "❤"),
        _c("c5", "vegapunk the best character in this arc", 73, "@fan5", 13),
        _c("c6", "The pacing of this arc is terrible and slow", 1000, "@critic1", 20, -1.0, "😡"),
        _c("c7", "pacing is so slow and terrible this arc", 60, "@critic2", 21, -1.0, "💀"),
        _c("c8", "terrible slow pacing ruins the arc", 40, "@critic3", 22),
        _c("c9", "I would love to see Luffy vs Buggy that would be funny", 13, "@JaydonHilton", 8),
        _c("c10", "Imagine Buggy being trained by Croc and Mihawk", 9, "@Ridlay_", 8),
    ]


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=256)

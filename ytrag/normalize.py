"""Turn raw downloader dictionaries into :class:`~ytrag.models.Comment` records.

Two decisions here differ from the original project and matter downstream:

1. Likes arrive as display strings (``"1.2K"``), so they are parsed to integers.
   Sorting text lexicographically puts ``"9"`` above ``"1.2K"``, which is how a
   naive implementation reports the wrong top comment.
2. Emoji are *extracted and scored*, not deleted.  On YouTube a wall of 🔥 or 💀
   is often the entire opinion, so dropping it discards the signal.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any

from ytrag.models import Comment

_SUFFIXES = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
_LIKES_RE = re.compile(r"^([0-9][0-9,.\s]*)\s*([kmb])?$", re.IGNORECASE)

_UNITS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
    "month": 2_629_746.0,   # mean Gregorian month
    "year": 31_556_952.0,   # mean Gregorian year
}
_RELATIVE_RE = re.compile(
    r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago", re.IGNORECASE
)

# Valence weights for the emoji that actually carry opinion in comment sections.
# Anything unlisted is treated as neutral rather than guessed at.
_POSITIVE = "❤🧡💛💚💙💜🖤🤍🤎💖💕💗💓💞💘😍🥰😘🤩😊😁😄😃🙂😂🤣😹👍👏🙌🔥💯✨⭐🌟🏆🥇💪🫶🤝🎉🎊😻🥹"
_NEGATIVE = "😡🤬😠👎💀☠😢😭😞😔😩😫🤮🤢🙄😒😑😐🥱💩⚰🚮❌⛔😤😰😱"


def parse_likes(raw: Any) -> int:
    """Parse a YouTube like count into an integer.

    Accepts ints, ``"1,234"``, ``"1.2K"``, ``"3.4M"``.  Anything unparseable is
    0 -- a missing count must never be allowed to look like a large one.
    """
    if raw is None:
        return 0
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, (int, float)):
        return max(0, int(raw))

    text = str(raw).strip()
    if not text:
        return 0

    match = _LIKES_RE.match(text)
    if not match:
        return 0

    number, suffix = match.groups()
    number = number.replace(",", "").replace(" ", "")
    try:
        value = float(number)
    except ValueError:
        return 0

    if suffix:
        value *= _SUFFIXES[suffix.lower()]
    return max(0, int(round(value)))


def _is_emoji(char: str) -> bool:
    if char in ("\u200d", "\ufe0f", "\ufe0e"):
        return True
    if "\U0001F1E6" <= char <= "\U0001F1FF":  # regional indicators (flags)
        return True
    if unicodedata.category(char) == "So":
        return True
    return "\U0001F000" <= char <= "\U0001FAFF"


def split_emoji(text: str) -> tuple[str, str]:
    """Split ``text`` into (prose, emoji) instead of discarding the emoji."""
    if not text:
        return "", ""
    prose_chars: list[str] = []
    emoji_chars: list[str] = []
    for char in text:
        (emoji_chars if _is_emoji(char) else prose_chars).append(char)
    prose = re.sub(r"\s+", " ", "".join(prose_chars)).strip()
    emojis = "".join(c for c in emoji_chars if c not in ("\u200d", "\ufe0f", "\ufe0e"))
    return prose, emojis


def emoji_valence(emojis: str) -> float:
    """Score an emoji run in ``[-1, 1]``.

    Uses the *proportion* of polarised emoji rather than a raw sum so that a
    single 🔥 and fifty 🔥 both land inside the bound.
    """
    if not emojis:
        return 0.0
    positive = sum(1 for c in emojis if c in _POSITIVE)
    negative = sum(1 for c in emojis if c in _NEGATIVE)
    polarised = positive + negative
    if polarised == 0:
        return 0.0
    return (positive - negative) / polarised


def parse_relative_time(raw: str | None, now: float | None = None) -> float | None:
    """Convert ``"8 hours ago"`` into a POSIX timestamp.

    Returns ``None`` when the string carries no parseable age, so callers can
    tell "unknown" apart from "right now".
    """
    if not raw:
        return None
    now = time.time() if now is None else now
    match = _RELATIVE_RE.search(str(raw))
    if not match:
        return None
    amount, unit = match.groups()
    return now - int(amount) * _UNITS[unit.lower()]


def normalize_comment(raw: dict[str, Any], cid: str, now: float | None = None) -> Comment:
    """Build a :class:`Comment` from one downloader dictionary."""
    prose, emojis = split_emoji(str(raw.get("text") or ""))
    published = str(raw.get("time") or "")
    return Comment(
        cid=cid,
        text=prose,
        author=str(raw.get("author") or ""),
        likes=parse_likes(raw.get("votes", raw.get("likes"))),
        published=published,
        published_ts=parse_relative_time(published, now=now),
        reply_count=parse_likes(raw.get("replies")),
        is_reply=bool(raw.get("reply")),
        emojis=emojis,
        valence=emoji_valence(emojis),
    )

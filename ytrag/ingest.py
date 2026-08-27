"""Getting comments in: from YouTube, from CSV, or from a list of dicts.

Three entry points, because the original project had one and it was the fragile
one. ``from_youtube`` needs the network and a working downloader;
``from_csv`` and ``from_records`` do not, which is what makes the pipeline
testable and what lets the repository ship a sample corpus that anyone can run
against immediately.

The CSV reader also understands the original project's single-column
``Merged Comment`` format, so an index built by the old code is still readable.
"""

from __future__ import annotations

import csv
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from ytrag.models import Comment
from ytrag.normalize import normalize_comment

#: Matches the original project's merged-comment line:
#: ``comment is '...' with likes= 20 with user_id '...' and published(time) '...'``
_LEGACY_RE = re.compile(
    r"comment is '(?P<text>.*)' with likes=\s*(?P<likes>[^\s]+)\s+"
    r"with user_id '(?P<author>.*?)' and published\(time\)\s*'(?P<time>.*?)'",
    re.DOTALL,
)


def from_records(
    records: Iterable[dict[str, Any]], now: float | None = None
) -> list[Comment]:
    """Normalise raw downloader dictionaries into comments with stable ids."""
    return [
        normalize_comment(record, cid=f"c{index}", now=now)
        for index, record in enumerate(records)
    ]


def from_youtube(
    url: str,
    limit: int = 500,
    sort_by_popular: bool = True,
    language: str | None = None,
) -> list[Comment]:
    """Download comments for ``url``.

    ``limit`` exists because the downloader is an unbounded generator: a popular
    video has hundreds of thousands of comments and the original code would keep
    pulling until something broke.
    """
    try:
        from youtube_comment_downloader import (
            SORT_BY_POPULAR,
            SORT_BY_RECENT,
            YoutubeCommentDownloader,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pip install youtube-comment-downloader to fetch comments from YouTube"
        ) from exc

    if not str(url).strip():
        raise ValueError("a YouTube video URL is required")

    downloader = YoutubeCommentDownloader()
    stream = downloader.get_comments_from_url(
        url,
        sort_by=SORT_BY_POPULAR if sort_by_popular else SORT_BY_RECENT,
        language=language,
    )

    records: list[dict[str, Any]] = []
    for record in stream:
        records.append(record)
        if len(records) >= limit:
            break

    if not records:
        raise ValueError(
            f"no comments found for {url} -- the video may have comments disabled, "
            "be private, or the URL may be wrong"
        )
    return from_records(records)


def _parse_legacy_line(line: str) -> dict[str, Any] | None:
    match = _LEGACY_RE.search(line)
    if not match:
        return None
    return {
        "text": match.group("text"),
        "votes": match.group("likes"),
        "author": match.group("author"),
        "time": match.group("time"),
    }


def from_csv(path: str | Path, now: float | None = None) -> list[Comment]:
    """Read comments from CSV, in either the new or the original format.

    New format: columns ``text``/``comment``, ``likes``/``votes``, ``author``,
    ``time``. Original format: one ``Merged Comment`` column holding a sentence
    that has to be parsed back apart.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such CSV: {path}")

    # Comment text routinely exceeds the default 128 KB field limit.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    records: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [f.strip().lower() for f in (reader.fieldnames or [])]

        if len(fieldnames) == 1 and "merged" in fieldnames[0]:
            handle.seek(0)
            raw = csv.reader(handle)
            next(raw, None)  # header
            for row in raw:
                if row and (parsed := _parse_legacy_line(row[0])):
                    records.append(parsed)
            return from_records(records, now=now)

        for row in reader:
            lower = {(k or "").strip().lower(): v for k, v in row.items()}
            records.append(
                {
                    "text": lower.get("text") or lower.get("comment")
                    or lower.get("comment text") or "",
                    "votes": lower.get("likes") or lower.get("votes") or 0,
                    "author": lower.get("author") or lower.get("user_id") or "",
                    "time": lower.get("time") or lower.get("published") or "",
                }
            )
    return from_records(records, now=now)


def to_csv(comments: Sequence[Comment], path: str | Path) -> Path:
    """Write comments as real columns, one field per column.

    The original code merged every field into one prose string and then taught
    the LLM to parse it back out. Storing structured data as a sentence is what
    made counting and sorting impossible downstream.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cid", "text", "author", "likes", "time", "emojis", "valence"])
        for c in comments:
            writer.writerow(
                [c.cid, c.text, c.author, c.likes, c.published, c.emojis, f"{c.valence:.3f}"]
            )
    return path


def iter_batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield ``items`` in chunks, for progress reporting during long ingests."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]

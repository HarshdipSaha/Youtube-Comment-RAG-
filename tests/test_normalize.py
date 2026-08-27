"""Normalisation is the seam between raw downloader output and the corpus."""

import math

import pytest

from ytrag.normalize import (
    emoji_valence,
    normalize_comment,
    parse_likes,
    parse_relative_time,
    split_emoji,
)


class TestParseLikes:
    """YouTube serves likes as display strings, not integers."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0, 0),
            (42, 42),
            ("", 0),
            (None, 0),
            ("7", 7),
            ("1,234", 1234),
            ("1.2K", 1200),
            ("15K", 15000),
            ("3.4M", 3_400_000),
            ("2B", 2_000_000_000),
            ("  8  ", 8),
            ("garbage", 0),
        ],
    )
    def test_parses_display_counts(self, raw, expected):
        assert parse_likes(raw) == expected


class TestSplitEmoji:
    def test_separates_emoji_from_prose(self):
        text, emojis = split_emoji("One piece forever ❤🔥")
        assert text == "One piece forever"
        assert "❤" in emojis and "🔥" in emojis

    def test_leaves_plain_text_untouched(self):
        text, emojis = split_emoji("plain text")
        assert text == "plain text"
        assert emojis == ""

    def test_comment_of_only_emoji_keeps_them_as_signal(self):
        text, emojis = split_emoji("😂😂😂")
        assert text == ""
        assert len(emojis) == 3


class TestEmojiValence:
    """Emoji are signal, not noise -- the original project deleted them."""

    def test_positive_emoji_score_above_zero(self):
        assert emoji_valence("❤🔥") > 0

    def test_negative_emoji_score_below_zero(self):
        assert emoji_valence("💀😡") < 0

    def test_no_emoji_is_neutral(self):
        assert emoji_valence("") == 0.0

    def test_valence_is_bounded(self):
        assert -1.0 <= emoji_valence("🔥" * 50) <= 1.0


class TestParseRelativeTime:
    """`youtube_comment_downloader` returns '8 hours ago', not a timestamp."""

    def test_hours_ago_is_before_now(self):
        now = 1_700_000_000.0
        assert parse_relative_time("8 hours ago", now=now) == now - 8 * 3600

    def test_understands_plural_and_singular(self):
        now = 1_700_000_000.0
        assert parse_relative_time("1 day ago", now=now) == now - 86400
        assert parse_relative_time("2 days ago", now=now) == now - 2 * 86400

    def test_edited_suffix_is_ignored(self):
        now = 1_700_000_000.0
        assert parse_relative_time("3 months ago (edited)", now=now) is not None

    def test_unparseable_returns_none(self):
        assert parse_relative_time("whenever", now=1.0) is None


class TestNormalizeComment:
    def test_builds_a_comment_with_derived_fields(self):
        c = normalize_comment(
            {"text": "Vegapunk is the best 🔥", "votes": "1.2K",
             "author": "@Gleeblevr", "time": "8 hours ago"},
            cid="c0",
            now=1_700_000_000.0,
        )
        assert c.cid == "c0"
        assert c.text == "Vegapunk is the best"
        assert c.likes == 1200
        assert c.author == "@Gleeblevr"
        assert c.emojis == "🔥"
        assert c.valence > 0
        assert c.published_ts == 1_700_000_000.0 - 8 * 3600

    def test_missing_fields_do_not_raise(self):
        c = normalize_comment({}, cid="c1")
        assert c.cid == "c1"
        assert c.text == ""
        assert c.likes == 0
        assert not math.isnan(c.valence)

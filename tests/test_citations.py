"""The citation guard: catching claims the evidence does not support.

RAG's characteristic failure is not inventing text out of nothing -- it is
producing a fluent, well-cited-looking answer whose *numbers* were never in the
context. Those are the claims users trust most and check least, so they are the
ones worth verifying mechanically.
"""

import pytest

from ytrag.citations import CitationGuard


@pytest.fixture
def guard():
    return CitationGuard(
        allowed_ids={"c0", "c1", "c6"},
        allowed_numbers={6.0, 11.0, 54.5, 2020.0, 1000.0},
    )


class TestExtractingCitations:
    def test_finds_bracketed_ids(self, guard):
        assert guard.extract("People love it [c0] though some disagree [c6]") == ["c0", "c6"]

    def test_is_case_insensitive_about_the_prefix(self, guard):
        assert guard.extract("as in [C1]") == ["c1"]

    def test_deduplicates_while_keeping_order(self, guard):
        assert guard.extract("[c6] and again [c6] and [c0]") == ["c6", "c0"]

    def test_finds_nothing_in_plain_prose(self, guard):
        assert guard.extract("just an opinion") == []


class TestInvalidCitations:
    def test_flags_a_citation_that_does_not_exist(self, guard):
        report = guard.verify("A confident claim [c99]")
        assert "c99" in report.invalid
        assert report.warnings

    def test_keeps_valid_citations(self, guard):
        report = guard.verify("Grounded [c0] and [c6]")
        assert report.citations == ["c0", "c6"]
        assert report.invalid == []
        assert not report.warnings

    def test_strips_invented_citations_from_the_text(self, guard):
        """A fake citation is worse than none -- it manufactures credibility."""
        report = guard.verify("Real [c0] and fake [c42].")
        assert "[c42]" not in report.text
        assert "[c0]" in report.text


class TestUnsupportedNumbers:
    def test_accepts_a_percentage_present_in_the_evidence(self, guard):
        assert guard.verify("54.5% of commenters agree [c0]").unsupported == []

    def test_tolerates_sensible_rounding_of_a_percentage(self, guard):
        """54.5 rounded to 55 is honest; 80 is not."""
        assert guard.verify("55% of commenters agree [c0]").unsupported == []

    def test_flags_a_percentage_nobody_computed(self, guard):
        report = guard.verify("80% of commenters agree [c0]")
        assert "80%" in report.unsupported
        assert report.warnings

    def test_flags_an_invented_like_count(self, guard):
        report = guard.verify("The top comment has 9,999 likes [c0]")
        assert any("9,999" in u or "9999" in u for u in report.unsupported)

    def test_accepts_a_like_count_from_the_evidence(self, guard):
        assert guard.verify("It drew 2,020 likes [c0]").unsupported == []

    def test_flags_an_invented_comment_count(self, guard):
        assert guard.verify("500 comments discuss this [c0]").unsupported

    def test_ignores_numbers_that_are_not_quantitative_claims(self, guard):
        """'Luffy vs Buggy' and 'season 2' are not statistics."""
        assert guard.verify("They discuss season 2 and chapter 1089 [c0]").unsupported == []


class TestOverallVerdict:
    def test_a_clean_answer_is_grounded(self, guard):
        assert guard.verify("6 comments say so [c0]").grounded

    def test_an_answer_with_a_bad_number_is_not_grounded(self, guard):
        assert not guard.verify("90% say so [c0]").grounded

    def test_an_answer_with_no_citations_at_all_is_warned_about(self, guard):
        report = guard.verify("People generally liked it.")
        assert not report.citations
        assert any("no citation" in w.lower() for w in report.warnings)

    def test_empty_allowed_numbers_disables_the_numeric_check(self):
        """When there is nothing to check against, do not invent failures."""
        guard = CitationGuard(allowed_ids={"c0"}, allowed_numbers=set())
        assert guard.verify("42% agree [c0]").unsupported == []

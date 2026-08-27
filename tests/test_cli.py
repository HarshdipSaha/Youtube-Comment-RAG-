"""The CLI seam: argument handling and exit codes.

Driven through ``main()`` with argv, so these exercise the same path a user
takes rather than calling the command functions directly.
"""

import json

import pytest

from ytrag.cli import main
from ytrag.engine import CommentRAG
from ytrag.ingest import to_csv


@pytest.fixture
def index(comments, tmp_path):
    CommentRAG.from_comments(comments, cluster_threshold=0.30).save(tmp_path / "kb")
    return str(tmp_path / "kb")


class TestBuild:
    def test_builds_an_index_from_a_csv(self, comments, tmp_path, capsys):
        csv_path = to_csv(comments, tmp_path / "in.csv")
        code = main(["--index", str(tmp_path / "kb"), "build", "--csv", str(csv_path)])
        assert code == 0
        assert (tmp_path / "kb" / "manifest.json").exists()
        assert "11 comments" in capsys.readouterr().out

    def test_can_export_normalised_comments(self, comments, tmp_path):
        csv_path = to_csv(comments, tmp_path / "in.csv")
        main([
            "--index", str(tmp_path / "kb"), "build",
            "--csv", str(csv_path), "--export-csv", str(tmp_path / "out.csv"),
        ])
        assert (tmp_path / "out.csv").exists()

    def test_building_with_neither_url_nor_csv_is_an_error(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--index", str(tmp_path / "kb"), "build"])


class TestAsk:
    def test_answers_a_question(self, index, capsys):
        assert main(["--index", index, "ask", "which comment has the most likes"]) == 0
        assert "1,000 likes" in capsys.readouterr().out

    def test_reports_the_route_and_coverage(self, index, capsys):
        main(["--index", index, "ask", "what do people think overall"])
        out = capsys.readouterr().out
        assert "CONSENSUS" in out
        assert "coverage" in out

    def test_json_output_is_valid_json(self, index, capsys):
        main(["--index", index, "--json", "ask", "how many comments are there"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["kind"] == "AGGREGATE"
        assert "11" in payload["answer"]

    def test_json_output_carries_the_evidence(self, index, capsys):
        main(["--index", index, "--json", "ask", "what about vegapunk"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["evidence"]
        assert "support" in payload["evidence"][0]


class TestOverview:
    def test_summarises_the_corpus(self, index, capsys):
        assert main(["--index", index, "overview"]) == 0
        out = capsys.readouterr().out
        assert "11 comments" in out
        assert "Most widely-held views" in out

    def test_json_overview_is_valid_json(self, index, capsys):
        main(["--index", index, "--json", "overview"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["stats"]["comments"] == 11


class TestErrorHandling:
    def test_a_missing_index_explains_how_to_build_one(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            main(["--index", str(tmp_path / "absent"), "ask", "hello"])
        assert "build" in str(excinfo.value)

    def test_an_unknown_llm_backend_exits_nonzero(self, index, capsys):
        assert main(["--index", index, "--llm", "nope", "ask", "hi"]) == 1
        assert "unknown LLM backend" in capsys.readouterr().err

    def test_no_subcommand_is_rejected(self):
        with pytest.raises(SystemExit):
            main([])

    def test_help_lists_the_available_providers(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        assert "extractive" in capsys.readouterr().out


class TestEval:
    def test_reports_both_comparisons(self, index, capsys):
        assert main(["--index", index, "eval"]) == 0
        out = capsys.readouterr().out
        assert "EXACT-ANSWER QUESTIONS" in out
        assert "naive top-k" in out

    def test_states_that_the_exact_comparison_is_structural(self, index, capsys):
        """A benchmark that oversells itself is worse than none."""
        main(["--index", index, "eval"])
        assert "structural" in capsys.readouterr().out

    def test_json_eval_is_valid_json(self, index, capsys):
        main(["--index", index, "--json", "eval"])
        payload = json.loads(capsys.readouterr().out)
        assert "exact" in payload
        assert payload["exact"][0]["accuracy"] >= 0.8


class TestNonAsciiOutput:
    """Regression: the CLI crashed on any comment containing an emoji.

    `print()` encodes with the console's codepage, which on a default Windows
    terminal is cp1252 and cannot represent U+2764. Every command that echoes
    comment text died with a UnicodeEncodeError -- and the capsys-based tests
    above could not see it, because capsys captures text before encoding.
    """

    @pytest.fixture
    def emoji_index(self, tmp_path):
        rag = CommentRAG.from_records(
            [
                # Emoji are split out of `text` into `emojis`, so a comment
                # with prose *and* emoji only exercises the JSON path. An
                # emoji-only comment has emoji in the text position too, which
                # is what the human-readable paths print.
                {"text": "One piece forever ❤", "votes": "7", "author": "@Letfreakinggo"},
                {"text": "🔥🔥🔥", "votes": "9", "author": "@hype"},
                {"text": "plain comment", "votes": "1", "author": "@b"},
            ]
        )
        rag.save(tmp_path / "kb")
        return str(tmp_path / "kb")

    @staticmethod
    def _cp1252_stdout(monkeypatch):
        import io
        import sys

        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
        monkeypatch.setattr(sys, "stdout", stream)
        return stream

    def test_ask_survives_a_cp1252_console(self, emoji_index, monkeypatch):
        self._cp1252_stdout(monkeypatch)
        assert main(["--index", emoji_index, "ask", "most liked comment"]) == 0

    def test_overview_survives_a_cp1252_console(self, emoji_index, monkeypatch):
        self._cp1252_stdout(monkeypatch)
        assert main(["--index", emoji_index, "overview"]) == 0

    def test_json_output_survives_a_cp1252_console(self, emoji_index, monkeypatch):
        self._cp1252_stdout(monkeypatch)
        assert main(["--index", emoji_index, "--json", "overview"]) == 0

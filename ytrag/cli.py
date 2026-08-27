"""Command line interface.

    ytrag build "https://youtu.be/..."     download comments, save a knowledge base
    ytrag ask "which comment has the most likes?"
    ytrag overview                          summarise the comment section
    ytrag repl                              ask questions interactively

Built on ``argparse`` so the package stays dependency-light; ``--json`` on any
command makes the output machine-readable for piping.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from ytrag.embed import get_embedder
from ytrag.engine import DEFAULT_INDEX, CommentRAG
from ytrag.ingest import to_csv
from ytrag.llm import available_providers, get_llm


def _load(args) -> CommentRAG:
    path = Path(args.index)
    if not (path / "manifest.json").exists():
        raise SystemExit(
            f"No knowledge base at '{path}'. Build one first:\n"
            f'  ytrag build "https://www.youtube.com/watch?v=..."\n'
            f"  ytrag build --csv youtube_comments.csv"
        )
    return CommentRAG.load(
        path,
        embedder=get_embedder(args.embedder),
        llm=get_llm(args.llm),
    )


def _print_answer(answer, as_json: bool) -> None:
    if as_json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
        return

    print(f"\n{answer.text}\n")
    print(f"  route     {answer.kind}")
    print(f"  coverage  {answer.coverage:.1%} of the comment section")
    if answer.citations:
        print(f"  cited     {', '.join(answer.citations)}")
    for warning in answer.warnings:
        print(f"  warning   {warning}")
    if answer.evidence:
        print("\n  evidence:")
        for item in answer.evidence:
            print(
                f"    [{item.cluster.representative_cid}] "
                f"{item.cluster.support} comments ({item.support_share:.1%}), "
                f"{item.cluster.endorsement:,} likes - "
                f'"{item.cluster.representative_text[:70]}"'
            )


def cmd_build(args) -> int:
    embedder = get_embedder(args.embedder)
    if args.csv:
        print(f"Reading {args.csv} ...")
        rag = CommentRAG.from_csv(args.csv, embedder=embedder)
    else:
        if not args.url:
            raise SystemExit("give a video URL, or --csv to build from a file")
        print(f"Downloading up to {args.limit} comments from {args.url} ...")
        rag = CommentRAG.from_youtube(args.url, limit=args.limit, embedder=embedder)

    path = rag.save(args.index)
    if args.export_csv:
        to_csv(rag.store.comments, args.export_csv)
        print(f"Wrote {args.export_csv}")

    print(
        f"Indexed {rag.store.total_comments:,} comments "
        f"({rag.store.total_likes:,} likes) into {len(rag.clusters):,} opinion clusters."
    )
    print(f"Saved to {path}/")
    return 0


def cmd_ask(args) -> int:
    _print_answer(_load(args).ask(args.question), args.json)
    return 0


def cmd_overview(args) -> int:
    overview = _load(args).overview()
    if args.json:
        print(json.dumps(overview, indent=2, ensure_ascii=False))
        return 0

    stats, sentiment = overview["stats"], overview["sentiment"]
    print(f"\n{stats['comments']:,} comments, {stats['total_likes']:,} likes")
    print(
        f"  mean {stats['mean_likes']:.1f} likes, median {stats['median_likes']:.0f}; "
        f"the top comment alone holds {stats['top_comment_like_share']:.1%} of all likes"
    )
    print(
        f"  emoji sentiment: {sentiment['positive']} positive / "
        f"{sentiment['negative']} negative / {sentiment['neutral']} neutral"
    )
    print("\nMost widely-held views:")
    for cluster in overview["clusters"]:
        print(
            f"  {cluster['support_share']:>6.1%}  {cluster['support']:>4} comments  "
            f"{cluster['endorsement']:>7,} likes  "
            f"[{cluster['cid']}] \"{cluster['text'][:64]}\""
        )
    return 0


def cmd_repl(args) -> int:
    rag = _load(args)
    print(
        f"{rag.store.total_comments:,} comments in {len(rag.clusters):,} opinion clusters. "
        "Ask a question, or Ctrl-D to quit."
    )
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question in ("exit", "quit"):
            return 0
        if question:
            _print_answer(rag.ask(question), args.json)


def cmd_eval(args) -> int:
    from ytrag.evaluate import run

    rag = _load(args)
    payload = run(rag.store)
    if args.json:
        print(json.dumps(
            {
                key: [
                    {"system": r.system, "correct": r.correct,
                     "total": r.total, "accuracy": r.accuracy, "misses": r.misses}
                    for r in results
                ]
                for key, results in payload.items() if key != "text"
            },
            indent=2,
        ))
    else:
        print(payload["text"])
        print(
            "\nExact questions are a structural comparison, not a quality one:\n"
            "top-k retrieval cannot reach a corpus-wide maximum. "
            "See ytrag/evaluate.py."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ytrag",
        description="Consensus-aware question answering over YouTube comments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  ytrag build "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --limit 1000\n'
            "  ytrag build --csv youtube_comments.csv\n"
            '  ytrag ask "which comment has the most likes?"\n'
            "  ytrag overview\n"
            "  ytrag eval\n\n"
            f"LLM providers ready in this environment: {', '.join(available_providers())}\n"
            "  (set ANTHROPIC_API_KEY / OPENAI_API_KEY / HUGGINGFACEHUB_API_TOKEN for more)"
        ),
    )
    parser.add_argument("--index", default=DEFAULT_INDEX, help="knowledge base directory")
    parser.add_argument(
        "--embedder",
        default="hashing",
        choices=["hashing", "st", "auto"],
        help="hashing (offline, default) or st (sentence-transformers)",
    )
    parser.add_argument("--llm", default="auto", help="extractive, anthropic, openai, huggingface")
    parser.add_argument("--json", action="store_true", help="machine-readable output")

    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="download comments and index them")
    build.add_argument("url", nargs="?", help="YouTube video URL")
    build.add_argument("--csv", help="build from a CSV instead of downloading")
    build.add_argument("--limit", type=int, default=500, help="max comments to fetch")
    build.add_argument("--export-csv", help="also write the normalised comments here")
    build.set_defaults(func=cmd_build)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    overview = sub.add_parser("overview", help="summarise the comment section")
    overview.set_defaults(func=cmd_overview)

    repl = sub.add_parser("repl", help="ask questions interactively")
    repl.set_defaults(func=cmd_repl)

    evaluate = sub.add_parser(
        "eval", help="benchmark this pipeline against naive top-k retrieval"
    )
    evaluate.set_defaults(func=cmd_eval)

    return parser


def _force_utf8_stdout() -> None:
    """Make stdout able to carry comment text.

    Comments are full of emoji and non-Latin script, and ``print`` encodes using
    the console's codepage -- cp1252 on a default Windows terminal, which cannot
    represent U+2764. Without this, every command that echoes a comment dies
    with a ``UnicodeEncodeError``, and because ``UnicodeEncodeError`` subclasses
    ``ValueError`` it was being swallowed by the handler below and reported as a
    generic error.

    ``backslashreplace`` is the fallback rather than ``replace`` so that a
    console which genuinely cannot display a character shows an escape rather
    than silently substituting "?" for someone's words.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            # A stream that refuses to be reconfigured is not worth failing over;
            # the commands still work, they just cannot print exotic characters.
            with contextlib.suppress(OSError, ValueError):  # pragma: no cover
                reconfigure(encoding="utf-8", errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

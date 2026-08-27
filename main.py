"""Streamlit app: ask a YouTube comment section questions.

    streamlit run main.py

Differences from the original that a user actually notices:

* it starts with no API keys set, and answers questions in that state;
* the knowledge base is cached, so asking a second question does not re-embed
  every comment;
* every answer shows how it was routed, what share of the comment section it
  drew on, and which comments back it.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ytrag import charts
from ytrag.embed import get_embedder
from ytrag.engine import CommentRAG
from ytrag.llm import available_providers, get_llm

st.set_page_config(page_title="YouTube Comment RAG", page_icon="▶", layout="wide")

SAMPLE_CSV = Path(__file__).parent / "youtube_comments.csv"

EXAMPLE_QUESTIONS = [
    "What do people think of this video overall?",
    "Which comment has the most likes?",
    "How many comments mention the ending?",
    "What are people complaining about?",
    "Who commented the most?",
    "What is the average number of likes?",
]


@st.cache_resource(show_spinner=False)
def _build_from_youtube(url: str, limit: int, embedder_name: str) -> CommentRAG:
    return CommentRAG.from_youtube(url, limit=limit, embedder=get_embedder(embedder_name))


@st.cache_resource(show_spinner=False)
def _build_from_csv(path: str, embedder_name: str) -> CommentRAG:
    return CommentRAG.from_csv(path, embedder=get_embedder(embedder_name))


def _sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.subheader("Settings")

        providers = available_providers()
        backend = st.selectbox(
            "Answer style",
            providers,
            help=(
                "extractive needs no API key and composes the answer from the "
                "retrieved evidence. The others write it up with a language model."
            ),
        )
        if providers == ["extractive"]:
            st.caption(
                "No API keys detected. Everything works; set ANTHROPIC_API_KEY, "
                "OPENAI_API_KEY or HUGGINGFACEHUB_API_TOKEN for generated prose."
            )

        embedder = st.selectbox(
            "Embeddings",
            ["hashing", "st"],
            help=(
                "hashing runs instantly and offline. st downloads a "
                "sentence-transformers model once, and retrieves better."
            ),
        )
        st.divider()
        st.caption(
            "Comments are grouped into opinion clusters. Each answer reports the "
            "share of the comment section it drew on."
        )
    return backend, embedder


def _render_answer(rag: CommentRAG, question: str, backend: str) -> None:
    rag.llm = get_llm(backend) if backend != "extractive" else None
    with st.spinner("Reading the comments..."):
        answer = rag.ask(question)

    st.markdown(f"### {answer.text}")

    left, middle, right = st.columns(3)
    left.metric("Route", answer.kind)
    middle.metric("Coverage", f"{answer.coverage:.0%}", help="share of comments behind this answer")
    right.metric("Comments cited", len(answer.citations))

    for warning in answer.warnings:
        st.warning(warning, icon="⚠")

    if answer.evidence:
        st.markdown("#### Evidence")
        for item in answer.evidence:
            cluster = item.cluster
            with st.expander(
                f"[{cluster.representative_cid}] {cluster.support} comments "
                f"({item.support_share:.1%}) · {cluster.endorsement:,} likes — "
                f"{cluster.representative_text[:80]}"
            ):
                for cid in cluster.member_cids[:8]:
                    comment = rag.store.get(cid)
                    if comment is None:
                        continue
                    st.markdown(
                        f"**{comment.author or 'unknown'}** · {comment.likes:,} likes · "
                        f"{comment.published or 'unknown time'}  \n"
                        f"{comment.text or comment.emojis} `[{comment.cid}]`"
                    )
                if cluster.support > 8:
                    st.caption(f"...and {cluster.support - 8} more saying the same thing.")


def _render_overview(rag: CommentRAG) -> None:
    overview = rag.overview()
    stats = overview["stats"]

    columns = st.columns(4)
    columns[0].metric("Comments", f"{stats['comments']:,}")
    columns[1].metric("Likes", f"{stats['total_likes']:,}")
    columns[2].metric("Opinion clusters", f"{len(rag.clusters):,}")
    columns[3].metric(
        "Top comment's share of likes", f"{stats['top_comment_like_share']:.0%}"
    )

    if not charts.charts_available():
        st.info("Install matplotlib to see the charts.")
        return

    left, right = st.columns([3, 2])
    with left:
        figure = charts.opinion_share(overview, top=8)
        if figure is not None:
            st.pyplot(figure, use_container_width=True)
    with right:
        figure = charts.like_distribution(rag.store.comments)
        if figure is not None:
            st.pyplot(figure, use_container_width=True)
        figure = charts.activity_timeline(rag.aggregates.timeline(6))
        if figure is not None:
            st.pyplot(figure, use_container_width=True)


def main() -> None:
    st.title("YouTube Comment RAG")
    st.caption(
        "Ask a comment section what it thinks — and get the proportion, "
        "not four cherry-picked comments."
    )

    backend, embedder = _sidebar()

    source = st.radio(
        "Comments from",
        ["YouTube URL", "Sample video", "Upload CSV"],
        horizontal=True,
    )

    rag: CommentRAG | None = None
    if source == "YouTube URL":
        url = st.text_input("Video link", placeholder="https://www.youtube.com/watch?v=...")
        limit = st.slider("Comments to fetch", 100, 5000, 500, step=100)
        if st.button("Build knowledge base", type="primary") and url:
            try:
                rag = _build_from_youtube(url, limit, embedder)
                st.session_state["rag"] = rag
            except Exception as exc:
                st.error(f"Could not fetch comments: {exc}")
    elif source == "Sample video":
        if SAMPLE_CSV.exists():
            st.session_state["rag"] = _build_from_csv(str(SAMPLE_CSV), embedder)
        else:
            st.error(f"Sample file not found at {SAMPLE_CSV}")
    else:
        upload = st.file_uploader("CSV of comments", type=["csv"])
        if upload is not None:
            temp = Path(st.session_state.get("_tmpdir", ".")) / "uploaded.csv"
            temp.write_bytes(upload.getvalue())
            st.session_state["rag"] = _build_from_csv(str(temp), embedder)

    rag = st.session_state.get("rag")
    if rag is None:
        st.info("Load a comment section to begin.")
        return

    overview_tab, ask_tab = st.tabs(["Overview", "Ask"])
    with overview_tab:
        _render_overview(rag)
    with ask_tab:
        chosen = st.selectbox("Try one of these", [""] + EXAMPLE_QUESTIONS)
        question = st.text_input("Your question", value=chosen)
        if question:
            _render_answer(rag, question, backend)


if __name__ == "__main__":
    main()

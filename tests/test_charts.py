"""Chart generation. Rendered on the Agg backend, so no display is needed."""

import pytest

from ytrag import charts
from ytrag.aggregate import AggregateEngine
from ytrag.engine import CommentRAG
from ytrag.store import HybridStore

pytestmark = pytest.mark.skipif(
    not charts.charts_available(), reason="matplotlib is not installed"
)


@pytest.fixture
def rag(comments):
    return CommentRAG.from_comments(comments, cluster_threshold=0.30)


class TestOpinionShare:
    def test_draws_one_pair_of_bars_per_opinion(self, rag):
        overview = rag.overview()
        figure = charts.opinion_share(overview, top=4)
        assert figure is not None
        axes = figure.axes[0]
        assert len(axes.patches) == 8  # 4 clusters x (people, likes)

    def test_labels_each_bar_with_the_opinion(self, rag):
        figure = charts.opinion_share(rag.overview(), top=3)
        labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
        assert len(labels) == 3
        assert all(labels)

    def test_no_clusters_produces_no_figure(self):
        assert charts.opinion_share({"clusters": [], "stats": {"total_likes": 0}}) is None


class TestLikeDistribution:
    def test_renders_for_a_normal_corpus(self, comments):
        figure = charts.like_distribution(comments)
        assert figure is not None
        assert "median" in figure.axes[0].get_title(loc="left")

    def test_handles_a_corpus_where_nobody_has_likes(self, comments):
        from dataclasses import replace

        zeroed = [replace(c, likes=0) for c in comments]
        assert charts.like_distribution(zeroed) is not None

    def test_empty_corpus_produces_no_figure(self):
        assert charts.like_distribution([]) is None


class TestActivityTimeline:
    def test_renders_a_bar_per_bucket(self, comments, embedder):
        timeline = AggregateEngine(HybridStore.build(comments, embedder)).timeline(4)
        figure = charts.activity_timeline(timeline)
        assert figure is not None
        assert len(figure.axes[0].patches) == 4

    def test_empty_timeline_produces_no_figure(self):
        assert charts.activity_timeline([]) is None


class TestSaving:
    def test_writes_a_png(self, rag, tmp_path):
        figure = charts.opinion_share(rag.overview(), top=3)
        path = charts.save(figure, str(tmp_path / "share.png"))
        assert (tmp_path / "share.png").stat().st_size > 1000
        assert path.endswith("share.png")

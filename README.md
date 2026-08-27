# YouTube Comment RAG

Ask a comment section what it thinks — and get the **proportion**, not four cherry-picked comments.

```bash
pip install -e .
ytrag --index sample_index ask "which comment has the most likes?"
```

```
The most-liked comment has 21 likes, by @dfile7598 8 hours ago: "My thoughts &
opinions on the chapter (so far): 1. When reading about Vegapunk mentioning the
One Piece…" [c10]

  route     AGGREGATE
  coverage  100.0% of the comment section
  cited     c10, c0, c1
```

---

## The problem this solves

A comment section is not a document corpus. It is a **distribution of opinion**
with structured metadata attached, and standard RAG — embed everything, retrieve
the top *k*, stuff it into a prompt — mishandles it in three specific ways.

**1. Aggregate questions are unanswerable by retrieval.** "Which comment has the
most likes?" has exactly one right answer, and that comment has no reason to be
semantically similar to the question. Retrieval ranks by similarity, so it never
looks at the row it needs. This is structural: no embedding model, no reranker
and no value of *k* fixes it.

**2. Redundancy destroys proportion.** If 200 people make the same point,
top-*k* returns five near-identical copies of it and the other 195 are invisible.
The model sees five documents and cannot tell whether that view is universal or
fringe — so "what do people think of this video?" gets answered from five
comments out of five thousand, and nothing in the pipeline says so.

**3. Structured data stored as prose stops being usable.** The original code
merged every field into one sentence:

```
comment is 'Vegapunk is the best' with likes= 1.2K with user_id '@x' and published(time) '8 hours ago'
```

and then spent its system prompt teaching the model to parse it back apart.
Likes stayed strings, so `"9"` sorts above `"1.2K"`.

## The approach: Consensus-Weighted Retrieval

Retrieve **opinion clusters** carrying their own social proof, and route
questions that have exact answers away from the retriever entirely.

```
question
   │
   ├─ route ───────── AGGREGATE │ SEMANTIC │ CONSENSUS │ HYBRID
   │
   ├─ exact ───────── SQL over every comment          (AGGREGATE, HYBRID)
   │
   ├─ retrieve ────── BM25 + dense, fused by RRF      (SEMANTIC, CONSENSUS, HYBRID)
   │   └─ rank ────── consensus-weighted clusters
   │
   ├─ generate ────── LLM, or the built-in extractive composer
   │
   └─ verify ──────── citations resolve, figures were computed
```

### 1. Route before retrieving

Questions with exact answers never touch the retriever. `which comment has the
most likes` becomes `ORDER BY likes DESC LIMIT 1` over a SQLite table holding
every comment. The number is *the* number, and the LLM's job is to phrase it,
never to compute it.

### 2. Retrieve opinions, not comments

Comments are grouped by greedy leader clustering, walking in descending like
order so the most-endorsed comment in each group becomes its representative —
the quote you see is the one the crowd actually upvoted. Each cluster carries:

- **support** — how many people said it
- **endorsement** — how many likes those comments drew

Clusters are ranked by

```
score(C, q) = relevance(C, q) × social(C)

relevance(C, q) = (1 − γ) + γ · saliencẽ(C, q)
social(C)       = 1 + α · support̃(C) + β · endorsement̃(C)
```

Social proof **multiplies** relevance rather than adding to it, and that is the
load-bearing choice. Under an additive score a sufficiently popular cluster wins
every query including ones it has nothing to do with — ask about a minor
character, get handed the video's most-liked joke. Multiplying drives an
irrelevant cluster to near zero however many likes it carries, while among
clusters that *are* relevant the widely-held view rises. Relevance decides
*whether*; consensus decides *which*.

### 3. Tell the model the proportions

The context block hands over opinions with their arithmetic already done:

```
CORPUS: 5,412 comments carrying 48,901 likes in total.

OPINION CLUSTERS (ranked by relevance and consensus):

[c412] 142 comments (2.6% of all comments), 8,312 likes (17.0% of all likes)
  - "the pacing in this arc is unbearable"
  - "why is this arc so slow"
```

Now "most commenters think X" is a claim the model can make *from what it was
given*, and the figures can be checked afterwards because they were computed
rather than described.

### 4. Verify the answer

Every generated answer is checked: citations must name comments that were
actually retrieved (invented ones are stripped, not just reported), and
percentages, like counts and comment counts must match figures the pipeline
computed. Every answer also reports **coverage** — the share of the comment
section behind it. "62% of comments informed this" and "0.4% of comments
informed this" are very different claims, and a system that does not say which
one it made is asking to be over-trusted.

## Results

`ytrag eval` runs both comparisons on your own index. On the bundled sample:

**Exact-answer questions** (ground truth computed by exhaustive scan)

| system | score |
|---|---|
| Consensus-weighted (this project) | **6/6 — 100%** |
| Naive top-*k* (the original design) | 1/6 — 17% |

**Topical retrieval** — R-precision on a hand-labelled corpus. For a question
with *R* relevant comments each system retrieves exactly *R*, so the budgets
match and both can reach 1.0.

| system | score |
|---|---|
| Consensus-weighted (this project) | **12/12 — 100%** |
| Naive top-*k* (the original design) | 11/12 — 92% |

Read the first table for what it is: a **structural** comparison, not a quality
one. Top-*k* retrieval cannot reach a corpus-wide maximum, so losing it is not
evidence that the baseline retrieves badly. The second table is the fair fight —
same budget, same corpus, no structural advantage either way.

## Install

```bash
pip install -e .              # core engine — numpy only, fully offline
pip install -e ".[all]"       # + Streamlit UI, charts, YouTube download, LLM providers
```

`numpy` is the only hard dependency. **No API key is required**, and none of the
245 tests touch the network or download a model.

## Use

### Command line

```bash
ytrag build "https://www.youtube.com/watch?v=..." --limit 1000
ytrag build --csv youtube_comments.csv        # or from a file

ytrag ask "what do people think of this video?"
ytrag ask "how many comments mention the ending?"
ytrag overview                                 # the shape of the conversation
ytrag repl                                     # interactive
ytrag eval                                     # benchmark vs naive top-k
ytrag --json ask "..."                         # machine-readable
```

### Streamlit

```bash
streamlit run main.py
```

Loads the sample video with one click, shows the opinion split as a chart, and
reports the route, coverage and evidence behind every answer.

### Python

```python
from ytrag import CommentRAG

rag = CommentRAG.from_youtube("https://youtu.be/...", limit=1000)
rag.save("kb_index")

answer = rag.ask("what are people complaining about?")
print(answer.text)
print(f"{answer.coverage:.0%} of comments informed this")

for e in answer.evidence:
    print(f"{e.cluster.support} people ({e.support_share:.0%}), "
          f"{e.cluster.endorsement:,} likes: {e.cluster.representative_text}")
```

## Configuration

| Variable | Effect |
|---|---|
| *(none)* | Works. Uses the offline hashing embedder and the extractive composer. |
| `ANTHROPIC_API_KEY` | Claude writes the answers. |
| `OPENAI_API_KEY` | GPT writes the answers. |
| `HUGGINGFACEHUB_API_TOKEN` | Hosted Mistral writes the answers. |

Auto-detection never fails: with no key set it falls back to the extractive
composer, which builds a real proportional summary from the pipeline's own
output and therefore **cannot hallucinate a figure at all**. Setting a key
improves phrasing, not what the system can answer — the accuracy lives in the
retrieval, not the model.

`--embedder st` swaps in `sentence-transformers` for better retrieval, at the
cost of a one-time model download.

## Layout

```
ytrag/
  models.py      Comment, OpinionCluster, Evidence, Answer
  normalize.py   parse "1.2K" likes; keep emoji as valence signal
  embed.py       Embedder protocol; offline hashing embedder (lnc.ltc weighting)
  store.py       BM25 + dense + SQLite over one corpus
  fusion.py      reciprocal rank fusion
  cluster.py     opinion clustering and consensus weighting  ← the core
  router.py      AGGREGATE / SEMANTIC / CONSENSUS / HYBRID
  aggregate.py   exact answers via SQL
  prompt.py      context blocks carrying social proof
  llm.py         provider registry; offline extractive default
  citations.py   citation and figure verification
  engine.py      CommentRAG — the public interface
  evaluate.py    benchmark vs the design this replaces
  charts.py      opinion share, like distribution, timeline
  cli.py         ytrag command
main.py                Streamlit app
langchain_helper.py    compatibility shim for the original API
tests/                 245 tests, all offline
docs/adr/              why the load-bearing decisions went the way they did
```

## Notes on the rewrite

Three things worth knowing if you used the previous version:

- **`langchain_helper` could not be imported at all.** Line 15 referenced an
  undefined `apikey`, so `import langchain_helper` raised `NameError` before
  anything ran; it also read `os.environ["HUGGINGFACEHUB_API_TOKEN"]` at import
  time and loaded the CSV from a hard-coded `H:\data science roadmap\…` path.
  `langchain_helper.py` is now a working shim over the new engine, so existing
  scripts keep running.

- **Emoji are signal, not noise.** `emoji_remove.py` stripped them. On YouTube a
  wall of 🔥 or 💀 is frequently the entire opinion, so they are now extracted
  and scored for valence in `ytrag/normalize.py`.

- **Sentiment is reported two ways.** By head-count *and* weighted by likes,
  because they routinely disagree: ten calm comments and one furious one with
  5,000 likes is a positive comment section by count and a negative one by
  attention. Note it reads emoji only, not tone in text.

## Development

```bash
pip install -e ".[dev]"
pytest                              # 245 tests, offline, ~6s
pytest --cov=ytrag                  # 93% coverage
ruff check ytrag tests
```

CI runs the suite on Python 3.10–3.12 across Linux and Windows, plus a CLI smoke
test against the sample corpus.

## Limitations

- Sentiment comes from emoji only. Text-based sentiment would need a model and
  is not implemented; `ytrag overview` says so rather than implying otherwise.
- The clustering threshold is calibrated per embedder. A corpus whose comments
  are unusually short or unusually long may need `cluster_threshold` tuned.
- `youtube_comment_downloader` is an unofficial scraper. It has no reply-thread
  depth, so thread structure is not modelled.
- Relative timestamps ("8 hours ago") are converted at ingest time, so a saved
  index records when it was *built*, not absolute publication times.

## License

MIT — see `LICENSE`.

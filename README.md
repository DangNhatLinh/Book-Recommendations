# Classical Book Recommender & Retrieval-Model Comparison

A content-based recommendation and search system for classical literature, built
over **200 Project Gutenberg books**. It does two things:

1. **Recommends books** — search by keywords (`monster horror revenge`) or find
   similar titles ("because you liked *Frankenstein*, try...").
2. **Benchmarks retrieval models** — compares **TF-IDF vs. BM25 vs. LSA** on a
   held-out query set using standard IR metrics (Precision@5, MRR, nDCG@10).

Every recommendation is **explainable** (it shows the words that drove the match).
Started as a CS 242 (Purdue) TF-IDF assignment; extended into a reproducible,
evaluated information-retrieval study with an interactive web app.

## Quickstart

```bash
pip install -r requirements.txt

# 0. (Optional) rebuild/expand the corpus from Project Gutenberg
python src/download_books.py --target 200

# 1. Interactive web app (keyword search with a model selector)
streamlit run app.py

# 2. Retrieval-model comparison on the held-out query set (+ bar chart)
python src/benchmark.py

# 3. Recommendation quality vs. genre labels (+ similarity heatmap)
python src/evaluate.py

# 4. Command-line recommender demo
python src/recommender.py
```

## Data

The corpus is downloaded from **Project Gutenberg**, whose texts are in the
**US public domain** (copyright expired), so they are free to download, use, and
redistribute. `src/download_books.py` fetches the most popular English books via
the [Gutendex](https://gutendex.com) API, saves the plain-text files to `data/`,
and writes `data/metadata.json` (author + Gutenberg subjects/bookshelves). This
makes the whole dataset reproducible from one command rather than hand-collected.

> Note: TF-IDF + cosine similarity has **no learned parameters**, so there is
> nothing to "overfit" in the usual ML sense. With a modest corpus the main risk
> is over-tuning hyperparameters (e.g. `min_df`) to this specific set, which is
> why the defaults are kept conservative. Metrics below are a directional sanity
> check, not a large-scale benchmark.

## How it works

1. **Clean** each book: strip Project Gutenberg boilerplate, jump to the first
   preface/chapter, drop headings and transcriber notes.
2. **Tokenize**: lowercase, keep `a–z` words, remove English stopwords and
   domain junk.
3. **Vectorize** (from scratch with NumPy):
   - TF = term count / total terms in the book
   - IDF = `log(N / (1 + df))`
   - TF-IDF = TF × IDF, then L2-normalized so a dot product is cosine similarity
4. **Filter** the vocabulary with `min_df = 3` (a word must appear in at least 3
   books). This removes book-specific character names that otherwise dominate the
   vectors and measurably improves thematic accuracy.
5. **Recommend** via nearest neighbors. Keyword queries are vectorized with the
   same TF-IDF weights and compared against every book.

## Retrieval-model comparison

The keyword-search task is evaluated on a **held-out query set** (`src/queries.py`):
14 themed queries (e.g. *"detective murder mystery investigation"*) whose relevant
books are defined by genre labels — independent of any model. Three classical IR
models rank the corpus and are scored with Precision@5, MRR, and nDCG@10.

| Model | P@5 | MRR | nDCG@10 |
|-------|-----|-----|---------|
| **TF-IDF** | **0.51** | **0.78** | **0.55** |
| BM25 | 0.50 | 0.73 | 0.52 |
| LSA (SVD, 100 dims) | 0.34 | 0.53 | 0.35 |

**Takeaway:** for whole-book documents with short genre queries, the simple
TF-IDF baseline is competitive with (slightly ahead of) BM25, and LSA's
dimensionality reduction *hurts* keyword precision — a good reminder that a more
complex model isn't automatically better. `python src/benchmark.py` regenerates
this table and saves `model_comparison.png`.

## Recommendation quality

Book-to-book recommendations are separately scored against genre labels (a hit =
shares a genre). Labels combine hand-assigned genres for the original books with
Project Gutenberg "Category" bookshelves for the rest.

`python src/evaluate.py` reports Precision@3 / MAP@5 vs. a random baseline and
saves `similarity_heatmap.png`.

> Note: TF-IDF, BM25 and LSA are **unsupervised** — there are no trained weights
> on labeled data, so there is nothing to "overfit" in the usual ML sense. The
> main risk with a modest corpus is over-tuning hyperparameters (`min_df`, SVD
> dimensions) to this specific set, so the defaults are kept conservative.

## Project structure

```
├── app.py                 # Streamlit web app (keyword search + model selector)
├── requirements.txt
├── data/                  # Project Gutenberg .txt books + metadata.json
└── src/
    ├── download_books.py  # reproducibly fetch the corpus from Gutenberg
    ├── recommender.py     # BookRecommender: cleaning + TF-IDF + recommend
    ├── models.py          # TF-IDF, BM25 (from scratch), LSA retrievers
    ├── queries.py         # held-out evaluation query set
    ├── benchmark.py       # model comparison (P@5 / MRR / nDCG@10) + chart
    ├── evaluate.py        # recommendation quality (Precision@k / MAP) + heatmap
    ├── parsing.ipynb      # original exploratory notebooks
    └── calculation.ipynb
```

## References
- Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed., draft), Ch. 11.
- CS 242 Staff. *Project 1: TF-IDF Handout*.

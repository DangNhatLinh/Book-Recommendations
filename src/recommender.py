"""
TF-IDF book recommender for classical (Project Gutenberg) texts.

This module turns the original CS 242 notebooks into a small, reusable library.
It builds TF-IDF vectors for each book from scratch (NumPy/pandas, no black-box
vectorizer) and supports two kinds of recommendations:

    1. recommend_by_book(title)      -> "because you liked X, try these"
    2. recommend_by_keywords(query)  -> type keywords, get matching classics

Both use cosine similarity in the shared TF-IDF space, and both can explain a
recommendation by listing the terms that contributed most to the match.
"""

from __future__ import annotations

import glob
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- Gutenberg / text cleaning (kept from the original parsing notebook) -----

WORDS_ONLY = re.compile(r"[a-z]+")

START = re.compile(
    r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
END = re.compile(
    r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*",
    re.IGNORECASE | re.DOTALL,
)
# Content usually starts at the first "preface" or "chapter <n>" line.
RAW_CONTENT = re.compile(
    r"^(?:preface|chapter\s+(?:\d+|[ivxlcdm]+))\b",
    re.IGNORECASE | re.MULTILINE,
)
HEAD_START = re.compile(
    r"^\s*(chapter\b|contents\b|epilogue\b|preface\b|prologue\b|etymology\b)",
    re.IGNORECASE,
)

# Domain junk that survives stopword removal.
BUZZWORDS = {
    "chapter", "chap", "book", "preface", "contents", "page", "project",
    "gutenberg", "ebook", "transcriber", "pgdp", "illustration", "copyright",
    "ll", "mr", "mrs", "dr",
}
# English stopwords + domain buzzwords. Removing these is the main quality fix:
# without it, "the/and/to" dominate every book and drown out real signal.
STOPWORDS = set(ENGLISH_STOP_WORDS) | BUZZWORDS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n").replace("\r", "\n")


def strip_content(text: str) -> str:
    """Drop the Project Gutenberg header/footer boilerplate."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    m = START.search(text)
    if m:
        text = text[m.end():]
    m = END.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def starting_point(text: str) -> str:
    """Skip front matter by jumping to the first preface/chapter line."""
    m = RAW_CONTENT.search(text)
    if m:
        line_start = text.rfind("\n", 0, m.start()) + 1
        return text[line_start:].lstrip()
    return text


def additional_removals(text: str) -> str:
    kept = [ln for ln in text.splitlines() if not HEAD_START.match(ln.strip())]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def remove_transcriber(text: str) -> str:
    bad = ("transcriber", "proofreading", "pgdp", "proofreaders",
           "illustration", "copyright")
    return "\n".join(
        ln for ln in text.splitlines()
        if not any(b in ln.lower() for b in bad)
    )


def tokenize(text: str) -> list[str]:
    """Lowercase, keep a-z words only, drop stopwords and 1-char tokens."""
    toks = WORDS_ONLY.findall(text.lower())
    return [t for t in toks if len(t) > 1 and t not in STOPWORDS]


def clean(raw: str) -> list[str]:
    text = strip_content(raw)
    text = starting_point(text)
    text = additional_removals(text)
    text = remove_transcriber(text)
    return tokenize(text)


@dataclass
class Recommendation:
    title: str
    score: float
    terms: list[str] = field(default_factory=list)  # terms driving the match


class BookRecommender:
    """Builds a TF-IDF matrix over a folder of .txt books and recommends.

    Parameters
    ----------
    data_dir : folder containing Project Gutenberg .txt files.
    min_df : keep a term only if it appears in at least this many books.
        Terms in a single book are almost always character names or typos;
        dropping them stops those book-specific tokens from dominating the
        vectors and roughly doubles thematic recommendation accuracy.
    """

    def __init__(self, data_dir: str | Path = DATA_DIR, min_df: int = 3):
        self.data_dir = Path(data_dir)
        self.min_df = min_df
        self.titles: list[str] = []
        self.vocab: list[str] = []
        self._term_index: dict[str, int] = {}
        self.tfidf: np.ndarray | None = None       # books x terms, L2-normalized
        self.idf: np.ndarray | None = None

    # -- fitting -------------------------------------------------------------

    def fit(self) -> "BookRecommender":
        paths = sorted(glob.glob(str(self.data_dir / "*.txt")))
        if not paths:
            raise FileNotFoundError(f"No .txt files found in {self.data_dir}")

        tokens_per_book = {Path(p).stem: clean(load_text(p)) for p in paths}
        self.titles = sorted(tokens_per_book)

        # Build vocabulary using a document-frequency filter: a term must show
        # up in at least `min_df` books to be kept.
        doc_freq: Counter[str] = Counter()
        for toks in tokens_per_book.values():
            doc_freq.update(set(toks))
        self.vocab = sorted(t for t, c in doc_freq.items() if c >= self.min_df)
        self._term_index = {t: i for i, t in enumerate(self.vocab)}

        # Term frequency: counts per book, normalized to sum to 1 per row.
        n_books, n_terms = len(self.titles), len(self.vocab)
        tf = np.zeros((n_books, n_terms), dtype=np.float64)
        for r, title in enumerate(self.titles):
            counts = Counter(t for t in tokens_per_book[title] if t in self._term_index)
            for term, c in counts.items():
                tf[r, self._term_index[term]] = c
        row_sums = tf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        tf /= row_sums

        # IDF (same formula as the original handout): log(N / (1 + df)).
        df = (tf > 0).sum(axis=0)
        self.idf = np.log(n_books / (1.0 + df))

        tfidf = tf * self.idf
        self.tfidf = self._l2_normalize(tfidf)
        return self

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return matrix / norm

    def _check_fitted(self) -> None:
        if self.tfidf is None:
            raise RuntimeError("Call fit() before recommending.")

    # -- query handling ------------------------------------------------------

    def resolve_title(self, query: str) -> str:
        """Match a book by exact stem or a case-insensitive substring."""
        if query in self.titles:
            return query
        low = query.lower()
        matches = [t for t in self.titles if low in t.lower()]
        if not matches:
            raise KeyError(f"No book matches {query!r}. Known titles: {self.titles}")
        return matches[0]

    def _query_vector(self, keywords: str | list[str]) -> np.ndarray:
        """Turn free-text keywords into a normalized TF-IDF vector."""
        self._check_fitted()
        text = keywords if isinstance(keywords, str) else " ".join(keywords)
        toks = [t for t in tokenize(text) if t in self._term_index]
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        if not toks:
            return vec
        for term, c in Counter(toks).items():
            vec[self._term_index[term]] = c
        vec /= vec.sum()          # term frequency
        vec *= self.idf           # weight by rarity
        return self._l2_normalize(vec.reshape(1, -1))[0]

    def _top_terms(self, vec_a: np.ndarray, vec_b: np.ndarray, n: int = 5) -> list[str]:
        """Terms contributing most to the cosine similarity of two vectors."""
        contrib = vec_a * vec_b
        idx = np.argsort(contrib)[::-1]
        return [self.vocab[i] for i in idx[:n] if contrib[i] > 0]

    # -- public recommendation API ------------------------------------------

    def recommend_by_book(self, title: str, k: int = 5) -> list[Recommendation]:
        """Books most similar to a given book ('because you liked ...')."""
        self._check_fitted()
        title = self.resolve_title(title)
        row = self.titles.index(title)
        sims = self.tfidf @ self.tfidf[row]
        order = np.argsort(sims)[::-1]
        out: list[Recommendation] = []
        for i in order:
            if i == row:
                continue
            out.append(Recommendation(
                title=self.titles[i],
                score=float(sims[i]),
                terms=self._top_terms(self.tfidf[row], self.tfidf[i]),
            ))
            if len(out) == k:
                break
        return out

    def recommend_by_keywords(self, keywords: str | list[str], k: int = 5) -> list[Recommendation]:
        """Books best matching free-text keywords (the main feature)."""
        self._check_fitted()
        q = self._query_vector(keywords)
        if not np.any(q):
            return []
        sims = self.tfidf @ q
        order = np.argsort(sims)[::-1]
        out: list[Recommendation] = []
        for i in order[:k]:
            out.append(Recommendation(
                title=self.titles[i],
                score=float(sims[i]),
                terms=self._top_terms(q, self.tfidf[i]),
            ))
        return out

    def similarity_matrix(self) -> np.ndarray:
        """Book-by-book cosine similarity matrix."""
        self._check_fitted()
        return self.tfidf @ self.tfidf.T


def _demo() -> None:
    rec = BookRecommender().fit()
    print(f"Loaded {len(rec.titles)} books, vocabulary of {len(rec.vocab)} terms.\n")

    print('Keyword search: "monster horror death"')
    for r in rec.recommend_by_keywords("monster horror death", k=3):
        print(f"  {r.score:.3f}  {r.title}  (via: {', '.join(r.terms)})")

    print('\nKeyword search: "love marriage sisters"')
    for r in rec.recommend_by_keywords("love marriage sisters", k=3):
        print(f"  {r.score:.3f}  {r.title}  (via: {', '.join(r.terms)})")

    sample = rec.resolve_title("Frankenstein")
    print(f'\nBecause you liked: {sample}')
    for r in rec.recommend_by_book(sample, k=3):
        print(f"  {r.score:.3f}  {r.title}  (via: {', '.join(r.terms)})")


if __name__ == "__main__":
    _demo()

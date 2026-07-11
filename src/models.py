"""
Retrieval models for the book-recommendation benchmark.

All models share the same tokenization (from recommender.py) and the same
document-term matrix, so the only thing that differs is the ranking method.
Each model exposes the same tiny interface:

    model = SomeRetriever(titles, docs_tokens).fit()
    model.search("keywords here", k=10)  -> [(title, score), ...]

Implemented:
  * TfidfRetriever  - TF-IDF vectors, cosine similarity (the baseline)
  * BM25Retriever   - Okapi BM25 (from scratch), the standard ranking upgrade
  * LSARetriever    - Latent Semantic Analysis (TF-IDF -> TruncatedSVD)
"""

from __future__ import annotations

import glob
from collections import Counter
from pathlib import Path

import numpy as np

from recommender import DATA_DIR, clean, load_text, tokenize


def load_corpus(data_dir: str | Path = DATA_DIR) -> tuple[list[str], list[list[str]]]:
    """Return (titles, token_lists) for every .txt in data_dir."""
    paths = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    titles, docs_tokens = [], []
    for p in paths:
        titles.append(Path(p).stem)
        docs_tokens.append(clean(load_text(p)))
    return titles, docs_tokens


class _CountMatrix:
    """Shared vocabulary + document-term count matrix (books x terms)."""

    def __init__(self, docs_tokens: list[list[str]], min_df: int = 3):
        doc_freq: Counter[str] = Counter()
        for toks in docs_tokens:
            doc_freq.update(set(toks))
        self.vocab = sorted(t for t, c in doc_freq.items() if c >= min_df)
        self.index = {t: i for i, t in enumerate(self.vocab)}

        n_docs, n_terms = len(docs_tokens), len(self.vocab)
        self.counts = np.zeros((n_docs, n_terms), dtype=np.float64)
        for r, toks in enumerate(docs_tokens):
            for term, c in Counter(toks).items():
                j = self.index.get(term)
                if j is not None:
                    self.counts[r, j] = c
        self.doc_len = self.counts.sum(axis=1)
        self.df = (self.counts > 0).sum(axis=0)
        self.n_docs = n_docs

    def query_counts(self, query: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        for term, c in Counter(tokenize(query)).items():
            j = self.index.get(term)
            if j is not None:
                vec[j] = c
        return vec


class BaseRetriever:
    name = "base"

    def __init__(self, titles: list[str], docs_tokens: list[list[str]], min_df: int = 3):
        self.titles = titles
        self.cm = _CountMatrix(docs_tokens, min_df=min_df)

    def fit(self) -> "BaseRetriever":
        raise NotImplementedError

    def _rank(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        order = np.argsort(scores)[::-1][:k]
        return [(self.titles[i], float(scores[i])) for i in order]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        raise NotImplementedError


class TfidfRetriever(BaseRetriever):
    """TF-IDF with cosine similarity (baseline). IDF = log(N / (1 + df))."""

    name = "TF-IDF"

    def fit(self) -> "TfidfRetriever":
        cm = self.cm
        tf = cm.counts / np.where(cm.doc_len[:, None] == 0, 1, cm.doc_len[:, None])
        self.idf = np.log(cm.n_docs / (1.0 + cm.df))
        tfidf = tf * self.idf
        self.doc_vecs = self._l2(tfidf)
        return self

    @staticmethod
    def _l2(m: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return m / norm

    def _query_vec(self, query: str) -> np.ndarray:
        q = self.cm.query_counts(query)
        if q.sum() == 0:
            return q
        q = (q / q.sum()) * self.idf
        n = np.linalg.norm(q)
        return q / n if n else q

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = self._query_vec(query)
        if not np.any(q):
            return []
        return self._rank(self.doc_vecs @ q, k)


class BM25Retriever(BaseRetriever):
    """Okapi BM25, implemented from scratch.

    score(d, q) = sum_t idf(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1 - b + b*|d|/avgdl))
    idf(t) = log((N - df + 0.5) / (df + 0.5) + 1)
    """

    name = "BM25"

    def __init__(self, titles, docs_tokens, min_df: int = 3, k1: float = 1.5, b: float = 0.75):
        super().__init__(titles, docs_tokens, min_df=min_df)
        self.k1 = k1
        self.b = b

    def fit(self) -> "BM25Retriever":
        cm = self.cm
        self.idf = np.log((cm.n_docs - cm.df + 0.5) / (cm.df + 0.5) + 1.0)
        self.avgdl = cm.doc_len.mean() if cm.n_docs else 0.0
        # Precompute the length-normalization denominator term per document.
        self._len_norm = self.k1 * (1 - self.b + self.b * cm.doc_len / (self.avgdl or 1.0))
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = self.cm.query_counts(query)
        q_terms = np.nonzero(q)[0]
        if q_terms.size == 0:
            return []
        f = self.cm.counts[:, q_terms]                     # docs x query-terms
        idf = self.idf[q_terms]
        denom = f + self._len_norm[:, None]
        scores = ((f * (self.k1 + 1)) / denom) @ idf
        return self._rank(scores, k)


class LSARetriever(BaseRetriever):
    """Latent Semantic Analysis: TF-IDF then TruncatedSVD (dense topics)."""

    name = "LSA"

    def __init__(self, titles, docs_tokens, min_df: int = 3, n_components: int = 100):
        super().__init__(titles, docs_tokens, min_df=min_df)
        self.n_components = n_components

    def fit(self) -> "LSARetriever":
        from sklearn.decomposition import TruncatedSVD

        cm = self.cm
        tf = cm.counts / np.where(cm.doc_len[:, None] == 0, 1, cm.doc_len[:, None])
        self.idf = np.log(cm.n_docs / (1.0 + cm.df))
        self._tfidf = tf * self.idf

        n_comp = min(self.n_components, min(self._tfidf.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
        self.doc_vecs = self._l2(self.svd.fit_transform(self._tfidf))
        return self

    @staticmethod
    def _l2(m: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return m / norm

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = self.cm.query_counts(query)
        if q.sum() == 0:
            return []
        q = (q / q.sum()) * self.idf
        latent = self.svd.transform(q.reshape(1, -1))[0]
        n = np.linalg.norm(latent)
        if n == 0:
            return []
        latent /= n
        return self._rank(self.doc_vecs @ latent, k)


MODELS = {
    "tfidf": TfidfRetriever,
    "bm25": BM25Retriever,
    "lsa": LSARetriever,
}


def build_all(titles, docs_tokens, min_df: int = 3) -> dict[str, BaseRetriever]:
    return {key: cls(titles, docs_tokens, min_df=min_df).fit() for key, cls in MODELS.items()}


if __name__ == "__main__":
    titles, docs = load_corpus()
    models = build_all(titles, docs)
    q = "detective murder mystery investigation"
    print(f"Query: {q!r}\n")
    for key, model in models.items():
        print(f"[{model.name}]")
        for title, score in model.search(q, k=3):
            print(f"  {score:.3f}  {title}")
        print()

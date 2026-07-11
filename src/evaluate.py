"""
Evaluate the TF-IDF recommender.

Because there are no user ratings, we evaluate against genre labels: a "good"
recommendation shares a genre with the query book. Labels come from two places:

  * the original 11 books use hand-assigned genres (GENRES below), and
  * any book downloaded via download_books.py uses its Project Gutenberg
    "Category: X" bookshelves (data/metadata.json).

We report Precision@k and Mean Average Precision (MAP) over every labeled book,
compare against a random baseline, and save a similarity heatmap.

Run:  python src/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recommender import BookRecommender

METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "metadata.json"

# Hand-labeled genres for the original 11 books.
GENRES: dict[str, set[str]] = {
    "Alice's Adventures in Wonderland": {"fantasy", "children"},
    "Beowulf- An Anglo-Saxon Epic Poem": {"epic", "adventure", "monster"},
    "Dracula": {"gothic", "horror", "monster"},
    "Frankenstein; Or, The Modern Prometheus": {"gothic", "horror", "monster"},
    "Grimms' Fairy Tales": {"fantasy", "children"},
    "Little Women; Or, Meg, Jo, Beth, and Amy": {"romance", "domestic"},
    "Moby Dick; Or, The Whale": {"adventure", "sea"},
    "Nora's twin sister": {"romance", "domestic", "children"},
    "Pride and Prejudice": {"romance", "domestic"},
    "Sense and Sensibility": {"romance", "domestic"},
    "Wuthering Heights": {"gothic", "romance"},
}

# Gutenberg categories too broad to be meaningful genre signal.
GENERIC_LABELS = {
    "novels", "short stories", "classics of literature", "british literature",
    "american literature", "french literature", "russian literature",
    "german literature", "italian literature", "movie books",
}


def _labels_from_metadata() -> dict[str, set[str]]:
    if not METADATA_PATH.exists():
        return {}
    catalog = json.loads(METADATA_PATH.read_text())
    labels: dict[str, set[str]] = {}
    for book in catalog:
        genres = set()
        for shelf in book.get("bookshelves", []):
            if shelf.startswith("Category:"):
                g = shelf.split("Category:", 1)[1].strip().lower()
                if g and g not in GENERIC_LABELS:
                    genres.add(g)
        if genres:
            labels[book["title"]] = genres
    return labels


def build_labels() -> dict[str, set[str]]:
    """Combine hand labels (original 11) with Gutenberg-derived labels."""
    labels = dict(GENRES)
    labels.update(_labels_from_metadata())
    return labels


def _relevant(labels: dict[str, set[str]], a: str, b: str) -> bool:
    return bool(labels.get(a, set()) & labels.get(b, set()))


def precision_at_k(rec: BookRecommender, labels: dict[str, set[str]], k: int = 3) -> float:
    hits = total = 0
    for title in rec.titles:
        if not labels.get(title):
            continue  # only score books we have labels for
        for r in rec.recommend_by_book(title, k=k):
            total += 1
            if _relevant(labels, title, r.title):
                hits += 1
    return hits / total if total else 0.0


def mean_average_precision(rec: BookRecommender, labels: dict[str, set[str]], k: int = 5) -> float:
    aps = []
    for title in rec.titles:
        if not labels.get(title):
            continue
        hits = 0
        precisions = []
        for rank, r in enumerate(rec.recommend_by_book(title, k=k), start=1):
            if _relevant(labels, title, r.title):
                hits += 1
                precisions.append(hits / rank)
        aps.append(np.mean(precisions) if precisions else 0.0)
    return float(np.mean(aps)) if aps else 0.0


def random_baseline(rec: BookRecommender, labels: dict[str, set[str]], k: int = 3, trials: int = 5000) -> float:
    rng = np.random.default_rng(0)
    labeled = [t for t in rec.titles if labels.get(t)]
    scores = []
    for _ in range(trials):
        q = rng.choice(labeled)
        others = [t for t in rec.titles if t != q]
        picks = rng.choice(others, size=k, replace=False)
        scores.append(np.mean([_relevant(labels, q, p) for p in picks]))
    return float(np.mean(scores))


def save_heatmap(rec: BookRecommender, out_path: str | Path = "similarity_heatmap.png") -> Path:
    import matplotlib.pyplot as plt

    sim = rec.similarity_matrix()
    short = [t.split(";")[0].split("-")[0].strip()[:20] for t in rec.titles]
    n = len(short)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.22), max(7.5, n * 0.22)))
    im = ax.imshow(sim, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fs = 8 if n <= 20 else 5
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(short, fontsize=fs)
    ax.set_title("Book-to-book TF-IDF cosine similarity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    rec = BookRecommender().fit()
    labels = build_labels()
    n_labeled = sum(1 for t in rec.titles if labels.get(t))
    k = 3

    p_at_k = precision_at_k(rec, labels, k=k)
    mapk = mean_average_precision(rec, labels, k=5)
    baseline = random_baseline(rec, labels, k=k)

    print(f"Corpus: {len(rec.titles)} books ({n_labeled} labeled), "
          f"{len(rec.vocab)} vocabulary terms\n")
    print(f"Precision@{k}     : {p_at_k:.3f}")
    print(f"MAP@5           : {mapk:.3f}")
    print(f"Random baseline : {baseline:.3f}  (Precision@{k})")
    lift = p_at_k / baseline if baseline else float("inf")
    print(f"Lift over random: {lift:.2f}x\n")

    path = save_heatmap(rec, Path(__file__).resolve().parent.parent / "similarity_heatmap.png")
    print(f"Saved similarity heatmap -> {path}")


if __name__ == "__main__":
    main()

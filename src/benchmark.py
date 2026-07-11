"""
Benchmark retrieval models on the held-out query set.

Compares TF-IDF, BM25 and LSA on the keyword-search task using standard IR
ranking metrics (Precision@5, MRR, nDCG@10), prints a comparison table, and
saves a bar chart.

Run:  python src/benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models import build_all, load_corpus
from queries import build_query_set


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(t in relevant for t in top) / k if top else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, t in enumerate(ranked, start=1):
        if t in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum((1.0 if t in relevant else 0.0) / np.log2(i + 1)
              for i, t in enumerate(ranked[:k], start=1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_model(model, query_set, k_p: int = 5, k_ndcg: int = 10) -> dict[str, float]:
    p, mrr, ndcg = [], [], []
    for q in query_set:
        ranked = [title for title, _ in model.search(q["query"], k=max(k_p, k_ndcg))]
        rel = q["relevant"]
        p.append(precision_at_k(ranked, rel, k_p))
        mrr.append(reciprocal_rank(ranked, rel))
        ndcg.append(ndcg_at_k(ranked, rel, k_ndcg))
    return {"P@5": float(np.mean(p)), "MRR": float(np.mean(mrr)), "nDCG@10": float(np.mean(ndcg))}


def save_chart(results: dict[str, dict[str, float]], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    metrics = ["P@5", "MRR", "nDCG@10"]
    model_names = list(results)
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_title("Retrieval model comparison (held-out query set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    titles, docs = load_corpus()
    query_set = build_query_set()
    models = build_all(titles, docs)

    print(f"Corpus: {len(titles)} books | Queries: {len(query_set)} "
          f"| Vocabulary: {len(next(iter(models.values())).cm.vocab)}\n")

    results = {model.name: evaluate_model(model, query_set) for model in models.values()}

    metrics = ["P@5", "MRR", "nDCG@10"]
    header = f"{'Model':<10}" + "".join(f"{m:>10}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        print(f"{name:<10}" + "".join(f"{scores[m]:>10.3f}" for m in metrics))

    best = max(results, key=lambda n: results[n]["nDCG@10"])
    print(f"\nBest by nDCG@10: {best}")

    out = save_chart(results, Path(__file__).resolve().parent.parent / "model_comparison.png")
    print(f"Saved chart -> {out}")


if __name__ == "__main__":
    main()

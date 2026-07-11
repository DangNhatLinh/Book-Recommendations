"""
Held-out evaluation queries.

Each entry is a natural keyword query plus the genre label(s) that make a book
"relevant". Relevance judgments come from the same label source as evaluate.py
(hand labels for the original books + Project Gutenberg categories for the rest),
so the queries are independent of any model -- this is our held-out test set for
comparing retrieval models on the actual keyword-search task.
"""

from __future__ import annotations

from evaluate import build_labels

# (query text, set of genre labels that count as relevant)
QUERIES: list[tuple[str, set[str]]] = [
    ("detective murder mystery investigation clue", {"crime, thrillers and mystery"}),
    ("love marriage courtship heart wedding", {"romance"}),
    ("voyage adventure journey quest danger", {"adventure"}),
    ("space future science planet machine", {"science-fiction & fantasy"}),
    ("soul virtue reason truth wisdom", {"philosophy & ethics"}),
    ("god faith prayer heaven sin", {"religion/spirituality"}),
    ("king battle war soldier medieval", {"historical novels", "historical fiction"}),
    ("poem verse song rhyme", {"poetry"}),
    ("life memoir born father childhood", {"biographies"}),
    ("ghost horror haunted fear terror", {"gothic", "horror"}),
    ("monster creature beast blood", {"monster", "horror", "gothic"}),
    ("fairy magic princess enchanted", {"fantasy", "children"}),
    ("sea ship ocean sailor whale", {"sea", "adventure"}),
    ("sisters family home mother daughters", {"domestic", "romance"}),
]


def relevant_for(labels: dict[str, set[str]], target_labels: set[str]) -> set[str]:
    return {title for title, labs in labels.items() if labs & target_labels}


def build_query_set() -> list[dict]:
    """Return usable queries with their ground-truth relevant titles."""
    labels = build_labels()
    out = []
    for text, target in QUERIES:
        relevant = relevant_for(labels, target)
        if relevant:  # skip queries whose genre isn't present in the corpus
            out.append({"query": text, "labels": target, "relevant": relevant})
    return out


if __name__ == "__main__":
    for q in build_query_set():
        print(f"{q['query']:<45} -> {len(q['relevant'])} relevant books")

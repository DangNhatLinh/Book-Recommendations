"""
Streamlit demo for the book recommender / retrieval-model comparison.

Run:  streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from models import build_all, load_corpus  # noqa: E402
from recommender import BookRecommender  # noqa: E402


def pretty(title: str) -> str:
    """Trim the long Gutenberg filename into a readable title."""
    return title.split(";")[0].split("- An")[0].strip()


@st.cache_resource(show_spinner="Building retrieval models...")
def load_everything():
    titles, docs = load_corpus()
    models = build_all(titles, docs)
    token_sets = {t: set(toks) for t, toks in zip(titles, docs)}
    recommender = BookRecommender().fit()
    return models, token_sets, recommender


st.set_page_config(page_title="Classical Book Recommender", page_icon="📚")
st.title("📚 Classical Book Recommender")
st.caption("Compare TF-IDF, BM25 and LSA retrieval over Project Gutenberg classics.")

models, token_sets, rec = load_everything()
model_by_name = {m.name: m for m in models.values()}

tab_keywords, tab_similar = st.tabs(["Search by keywords", "Find similar books"])

with tab_keywords:
    st.subheader("What are you in the mood for?")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter keywords",
            value="monster horror revenge",
            help="e.g. 'detective murder mystery', 'love marriage sisters', 'space future science'",
        )
    with col2:
        model_name = st.selectbox("Model", list(model_by_name))
    k = st.slider("Number of recommendations", 1, 10, 5, key="kw_k")

    if query.strip():
        q_tokens = set(query.lower().split())
        results = model_by_name[model_name].search(query, k=k)
        if not results:
            st.warning("None of those keywords are in the vocabulary. Try others.")
        for title, score in results:
            st.markdown(f"**{pretty(title)}**  —  score `{score:.3f}`")
            matched = sorted(q_tokens & token_sets.get(title, set()))
            if matched:
                st.caption("Matched: " + ", ".join(matched))

with tab_similar:
    st.subheader("Because you liked...")
    choice = st.selectbox("Pick a book", options=rec.titles, format_func=pretty)
    k2 = st.slider("Number of recommendations", 1, 10, 5, key="sim_k")
    for r in rec.recommend_by_book(choice, k=k2):
        st.markdown(f"**{pretty(r.title)}**  —  similarity `{r.score:.3f}`")
        if r.terms:
            st.caption("Shared themes/words: " + ", ".join(r.terms))

with st.sidebar:
    st.header("About")
    st.write(
        "Type keywords and pick a retrieval model to see how classical IR methods "
        "rank the same corpus differently:\n\n"
        "- **TF-IDF** — term frequency weighted by rarity, cosine similarity\n"
        "- **BM25** — the standard probabilistic ranking function\n"
        "- **LSA** — TF-IDF compressed into latent topics via SVD\n\n"
        "Run `python src/benchmark.py` for a full metric comparison."
    )
    st.metric("Books indexed", len(rec.titles))
    st.metric("Vocabulary size", f"{len(rec.vocab):,}")

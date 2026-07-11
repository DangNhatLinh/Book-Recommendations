"""
Download public-domain books from Project Gutenberg (reproducibly).

Uses the Gutendex API (https://gutendex.com) to find the most popular English
books, downloads their plain-text UTF-8 files into ../data, and writes a
data/metadata.json catalog (title, author, Gutenberg subjects/bookshelves).
Those subjects double as genre labels for evaluation, so the whole corpus is
reproducible from a single command instead of hand-collected files.

All Project Gutenberg texts used here are in the US public domain.

Run:  python src/download_books.py --target 60
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

GUTENDEX = "https://gutendex.com/books/"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_PATH = DATA_DIR / "metadata.json"
HEADERS = {"User-Agent": "book-recommender/1.0 (educational project)"}
PLAIN_TEXT_KEYS = ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain")


def sanitize_filename(title: str) -> str:
    """Make a title safe to use as a filename stem."""
    title = title.replace("\n", " ").strip()
    title = re.sub(r'[\\/:*?"<>|]', "", title)   # illegal path chars
    return re.sub(r"\s+", " ", title)[:120].strip()


def pick_text_url(formats: dict[str, str]) -> str | None:
    """Prefer a UTF-8 plain-text download; fall back to other plain text."""
    for key in PLAIN_TEXT_KEYS:
        url = formats.get(key)
        if url and not url.endswith(".zip"):
            return url
    for mime, url in formats.items():
        if mime.startswith("text/plain") and not url.endswith(".zip"):
            return url
    return None


def existing_title_stems() -> set[str]:
    return {p.stem.lower() for p in DATA_DIR.glob("*.txt")}


def download_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public-domain books from Project Gutenberg.")
    parser.add_argument("--target", type=int, default=60, help="how many books to end up with in data/")
    parser.add_argument("--sleep", type=float, default=0.5, help="delay between downloads (be polite)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    have = existing_title_stems()
    catalog: list[dict] = []
    if METADATA_PATH.exists():
        catalog = json.loads(METADATA_PATH.read_text())

    print(f"Starting with {len(have)} books in {DATA_DIR}. Target: {args.target}.")
    url = f"{GUTENDEX}?languages=en&sort=popular"
    added = 0

    while url and len(have) < args.target:
        page = requests.get(url, headers=HEADERS, timeout=60).json()
        for book in page["results"]:
            if len(have) >= args.target:
                break

            title = book["title"]
            stem = sanitize_filename(title)
            if not stem or stem.lower() in have:
                continue  # skip duplicates / books we already have

            text_url = pick_text_url(book.get("formats", {}))
            if not text_url:
                continue

            try:
                text = download_text(text_url)
            except Exception as exc:  # network hiccup, keep going
                print(f"  skip {title!r}: {exc}")
                continue

            (DATA_DIR / f"{stem}.txt").write_text(text, encoding="utf-8")
            have.add(stem.lower())
            added += 1
            catalog.append({
                "id": book.get("id"),
                "title": stem,
                "authors": [a["name"] for a in book.get("authors", [])],
                "subjects": book.get("subjects", []),
                "bookshelves": book.get("bookshelves", []),
            })
            print(f"  [{len(have):>3}] {stem}")
            time.sleep(args.sleep)

        url = page.get("next")

    METADATA_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(f"\nDone. Added {added} books. Corpus now has {len(have)} books.")
    print(f"Catalog written to {METADATA_PATH}")


if __name__ == "__main__":
    main()

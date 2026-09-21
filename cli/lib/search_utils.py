import json
from typing import TypedDict
import string
from pathlib import Path
from nltk.stem import PorterStemmer


class Movie(TypedDict):
    id: int
    title: str
    description: str


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MOVIES_PATH = BASE_DIR / "data" / "movies.json"
STOPWORDS_PATH = BASE_DIR / "data" / "stopwords.txt"

MOVIES: list[Movie] = json.loads(MOVIES_PATH.read_text())["movies"]
STOPWORDS: set[str] = set(STOPWORDS_PATH.read_text().splitlines())

stemmer = PorterStemmer()


def tokenize(s: str) -> list[str]:
    s = s.strip().lower()
    for punc in string.punctuation:
        s = s.replace(punc, "")
    return [
        stemmed
        for token in s.split()
        if token not in STOPWORDS
        if isinstance(stemmed := stemmer.stem(token), str)
    ]


def keyword_search(query: str) -> list[Movie]:
    query_tokens = tokenize(query)

    results: list[Movie] = []
    for movie in MOVIES:
        title_tokens = tokenize(movie["title"])
        if any(
            q_token in t_token for q_token in query_tokens for t_token in title_tokens
        ):
            results.append(movie)
    return results


def print_results(results: list[Movie]) -> None:
    for i, movie in enumerate(results):
        print(f"{i + 1}. {movie['title']}")

import json
from pathlib import Path
import string
from typing import TypedDict
from nltk import defaultdict
from nltk.stem import PorterStemmer
import pickle
from typing import Self
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
MOVIES_PATH = BASE_DIR / "data" / "movies.json"
STOPWORDS_PATH = BASE_DIR / "data" / "stopwords.txt"
CACHE_PATH = BASE_DIR / "cache"

STOPWORDS: set[str] = set(STOPWORDS_PATH.read_text().splitlines())


class Movie(TypedDict):
    id: int
    title: str
    description: str


class InvertedIndex:
    def __init__(self: Self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.index_path: Path = CACHE_PATH / "index.pkl"
        self.docmap_path: Path = CACHE_PATH / "docmap.pkl"

    def __add_document(self: Self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_documents(self: Self, term: str) -> list[int]:
        ids = list(self.index.get(term, set()))
        ids.sort()
        return ids

    def build(self: Self) -> None:
        movies = load_movies()
        for movie in tqdm(movies):
            id = movie["id"]
            self.__add_document(id, f"{movie['title']} {movie['description']}")
            self.docmap[id] = movie

    def save(self: Self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self: Self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not self.docmap_path.exists():
            raise FileNotFoundError(f"Docmap file not found: {self.docmap_path}")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str) -> None:
    index = InvertedIndex()
    index.load()

    print(f"Searching for: {query}")
    print_results(keyword_search(query, index))


def load_movies() -> list[Movie]:
    return json.loads(MOVIES_PATH.read_text())["movies"]


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


def keyword_search(query: str, index: InvertedIndex) -> list[Movie]:
    query_tokens = tokenize(query)

    results: list[Movie] = []
    for token in query_tokens:
        doc_ids = index.get_documents(token)
        results.extend(index.docmap[doc_id] for doc_id in doc_ids)
        if len(results) >= 5:
            break

    return results[:5]


def print_results(results: list[Movie]) -> None:
    for i, movie in enumerate(results):
        print(f"{i + 1}. {movie['title']} (ID: {movie['id']})")

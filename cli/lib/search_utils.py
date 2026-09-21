import json
from pathlib import Path
import string
from typing import TypedDict
from nltk import defaultdict
from nltk.stem import PorterStemmer
import pickle
from typing import Self
from nltk.text import Counter
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
        self.index_path: Path = CACHE_PATH / "index.pkl"
        self.docmap: dict[int, Movie] = {}
        self.docmap_path: Path = CACHE_PATH / "docmap.pkl"
        self.term_frequencies: dict[int, Counter[str]] = defaultdict(Counter)
        self.tf_path: Path = CACHE_PATH / "term_frequencies.pkl"

    def __add_document(self: Self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self: Self, token: str) -> list[int]:
        if token not in self.index:
            return []
        ids = list(self.index[token])
        ids.sort()
        return ids

    def get_tf(self: Self, doc_id: int, word: str) -> int:
        token = tokenize_word(word)
        if doc_id not in self.term_frequencies:
            raise ValueError(f"Document ID {doc_id} not found in term frequencies")
        return self.term_frequencies[doc_id].get(token, 0)

    def build(self: Self) -> None:
        movies = load_movies()
        for movie in tqdm(movies):
            id = movie["id"]
            self.docmap[id] = movie
            self.__add_document(id, f"{movie['title']} {movie['description']}")

    def save(self: Self) -> None:
        BASE_DIR.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self: Self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        if not self.docmap_path.exists():
            raise FileNotFoundError(f"Docmap file not found: {self.docmap_path}")
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        if not self.tf_path.exists():
            raise FileNotFoundError(f"Term frequencies file not found: {self.tf_path}")
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str) -> None:
    print(f"Searching for: {query}")
    index = InvertedIndex()
    index.load()
    print_results(keyword_search(query, index))


def tf_command(doc_id: int, word: str) -> None:
    index = InvertedIndex()
    index.load()
    print(index.get_tf(doc_id, word))


def load_movies() -> list[Movie]:
    return json.loads(MOVIES_PATH.read_text())["movies"]


stemmer = PorterStemmer()


def tokenize_text(s: str) -> list[str]:
    s = s.strip().lower()
    for punc in string.punctuation:
        s = s.replace(punc, "")
    return [
        stemmed
        for token in s.split()
        if token not in STOPWORDS
        if isinstance(stemmed := stemmer.stem(token), str)
    ]


def tokenize_word(word: str) -> str:
    tokens = tokenize_text(word)
    if len(tokens) != 1:
        raise ValueError(f"Expected a single token, got: {tokens}")
    return tokens[0]


def keyword_search(query: str, index: InvertedIndex) -> list[Movie]:
    query_tokens = tokenize_text(query)

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

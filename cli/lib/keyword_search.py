import math
from pathlib import Path
import string
from nltk.stem import PorterStemmer
from typing import Self
from tqdm import tqdm
from collections import defaultdict, Counter
from .search_utils import (
    CACHE_DIR,
    BM25_K1,
    BM25_B,
    Movie,
    load_movies,
    load_pickle,
    load_stopwords,
    save_pickle,
)

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self: Self) -> None:
        self.__stopwords: set[str] = load_stopwords()

        self.index: dict[str, set[int]] = defaultdict(set)
        self.index_path: Path = CACHE_DIR / "index.pkl"
        self.docmap: dict[int, Movie] = {}
        self.docmap_path: Path = CACHE_DIR / "docmap.pkl"
        self.term_frequencies: dict[int, Counter[str]] = defaultdict(Counter)
        self.tf_path: Path = CACHE_DIR / "term_frequencies.pkl"
        self.doc_lengths: dict[int, int] = defaultdict(int)
        self.doc_lengths_path: Path = CACHE_DIR / "doc_lengths.pkl"

    def __add_document(self: Self, doc_id: int, text: str) -> None:
        tokens = self.__tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id].update(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self: Self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def __tokenize_term(self: Self, term: str) -> str:
        tokens = self.__tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError(f"Expected a single token, got: {tokens}")
        return tokens[0]

    def __tokenize_text(self: Self, text: str) -> list[str]:
        text = text.strip().lower()
        for punc in string.punctuation:
            text = text.replace(punc, "")
        return [
            stemmed
            for token in text.split()
            if token not in self.__stopwords
            if isinstance(stemmed := stemmer.stem(token), str)
        ]

    def build(self: Self) -> None:
        for movie in tqdm(load_movies(), desc="Building inverted index"):
            id = movie["id"]
            self.docmap[id] = movie
            self.__add_document(id, f"{movie['title']} {movie['description']}")

    def save(self: Self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_pickle(self.index_path, self.index)
        save_pickle(self.docmap_path, self.docmap)
        save_pickle(self.tf_path, self.term_frequencies)
        save_pickle(self.doc_lengths_path, self.doc_lengths)

    def load(self: Self) -> None:
        self.index = load_pickle(self.index_path)
        self.docmap = load_pickle(self.docmap_path)
        self.term_frequencies = load_pickle(self.tf_path)
        self.doc_lengths = load_pickle(self.doc_lengths_path)

    def keyword_search(self: Self, query: str) -> list[Movie]:
        seen_ids: set[int] = set()
        movies: list[Movie] = []
        for token in self.__tokenize_text(query):
            doc_ids = self.get_documents(token)
            for doc_id in doc_ids:
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    movies.append(self.docmap[doc_id])
        return movies

    def get_documents(self: Self, token: str) -> list[int]:
        if token not in self.index:
            return []
        ids = list(self.index[token])
        ids.sort()
        return ids

    def get_tf(self: Self, doc_id: int, term: str) -> int:
        token = self.__tokenize_term(term)
        if doc_id not in self.term_frequencies:
            raise ValueError(f"Document ID {doc_id} not found in term frequencies")
        return self.term_frequencies[doc_id].get(token, 0)

    def get_idf(self: Self, term: str) -> float:
        token = self.__tokenize_term(term)
        doc_count = len(self.docmap)
        match_doc_count = len(self.get_documents(token))
        return math.log((doc_count + 1) / (match_doc_count + 1))

    def get_tfidf(self: Self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25idf(self: Self, term: str) -> float:
        token = self.__tokenize_term(term)
        doc_count = len(self.docmap)
        match_doc_count = len(self.get_documents(token))
        return math.log(
            (doc_count - match_doc_count + 0.5) / (match_doc_count + 0.5) + 1
        )

    def get_bm25tf(
        self: Self, doc_id: int, term: str, k1: float | None, b: float | None
    ) -> float:
        tf = self.get_tf(doc_id, term)
        if k1 is None:
            k1 = BM25_K1
        if b is None:
            b = BM25_B
        if doc_id not in self.doc_lengths:
            raise ValueError(f"Document ID {doc_id} not found in document lengths")
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = (1 - b) + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(
        self: Self, doc_id: int, term: str, k1: float | None, b: float | None
    ) -> float:
        return self.get_bm25tf(doc_id, term, k1=k1, b=b) * self.get_bm25idf(term)

    def bm25_search(
        self: Self,
        query: str,
        limit: int | None = None,
        k1: float | None = None,
        b: float | None = None,
    ) -> list[tuple[Movie, float]]:
        scores: dict[int, float] = defaultdict(float)
        for doc_id in self.docmap:
            for token in self.__tokenize_text(query):
                scores[doc_id] += self.bm25(doc_id, token, k1=k1, b=b)
        head = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if limit is not None:
            head = head[:limit]
        return [(self.docmap[doc_id], score) for doc_id, score in head]


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str) -> None:
    print(f"Searching for: {query}")
    index = InvertedIndex()
    index.load()
    results = index.keyword_search(query)
    results.sort(key=lambda movie: movie["id"])
    for i, movie in enumerate(results[:5]):
        print(f"{i + 1}. {movie['title']} (ID: {movie['id']})")


def tf_command(doc_id: int, term: str) -> None:
    index = InvertedIndex()
    index.load()
    print(index.get_tf(doc_id, term))


def idf_command(term: str) -> None:
    index = InvertedIndex()
    index.load()
    idf = index.get_idf(term)
    print(f"Inverse document frequency for '{term}': {idf:.2f}")


def tfidf_command(doc_id: int, term: str) -> None:
    index = InvertedIndex()
    index.load()
    tf_idf = index.get_tfidf(doc_id, term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def bm25idf_command(term: str) -> None:
    index = InvertedIndex()
    index.load()
    bm25idf = index.get_bm25idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")


def bm25tf_command(
    doc_id: int, term: str, k1: float | None = None, b: float | None = None
) -> None:
    index = InvertedIndex()
    index.load()
    bm25tf = index.get_bm25tf(doc_id, term, k1=k1, b=b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")


def bm25search_command(
    query: str,
    limit: int | None = None,
    k1: float | None = None,
    b: float | None = None,
) -> None:
    index = InvertedIndex()
    index.load()
    results = index.bm25_search(query, limit=limit, k1=k1, b=b)
    for i, (movie, score) in enumerate(results):
        print(f"{i + 1}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")

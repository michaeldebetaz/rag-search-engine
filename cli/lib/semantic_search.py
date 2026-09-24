import os
import re
from typing import Self
import numpy as np
from sentence_transformers import SentenceTransformer
from lib.search_utils import (
    MOVIE_EMBEDDINGS_PATH,
    Movie,
    SemanticSearchResult,
    cosine_similarity,
    load_movies,
    save_embeddings,
)

MODEL = "all-MiniLM-L6-v2"


class SemanticSearch:
    def __init__(self: Self):
        self.model: SentenceTransformer = SentenceTransformer(
            MODEL, device="cpu", token=os.getenv("HUGGING_FACE_ACCESS_TOKEN")
        )
        self.embeddings: np.ndarray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def build_embeddings(self: Self, documents: list[Movie]) -> None:
        self.documents = documents
        texts: list[str] = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            texts.append(f"{doc['title']} {doc['description']}")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        save_embeddings(MOVIE_EMBEDDINGS_PATH, self.embeddings)

    def load_or_create_embeddings(self: Self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if MOVIE_EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if self.embeddings is None:
                raise ValueError("Failed to load embeddings from file.")
            if len(self.embeddings) != len(documents):
                print("Embeddings file is outdated. Rebuilding embeddings...")
                self.build_embeddings(documents)
        else:
            self.build_embeddings(documents)

        if self.embeddings is None:
            raise ValueError("Failed to load or create embeddings.")
        return self.embeddings

    def generate_embedding(self: Self, text: str) -> np.ndarray:
        text = text.strip()
        if text == "":
            raise ValueError("Input text cannot be empty.")
        embedding = self.model.encode(text)
        return embedding

    def search(self: Self, query: str, limit: int) -> list[SemanticSearchResult]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        if self.documents is None:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        scores: list[tuple[float, Movie]] = []
        for i, doc in enumerate(self.documents):
            similarity_score = cosine_similarity(query_embedding, self.embeddings[i])
            scores.append((similarity_score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)

        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in scores[:limit]
        ]


def verify_model() -> None:
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape}")


def search(query: str, limit: int) -> None:
    search = SemanticSearch()
    documents = load_movies()
    search.load_or_create_embeddings(documents)
    for i, result in enumerate(search.search(query, limit)):
        print(f"{i + 1}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description'][:100]}...\n")


def chunk(text: str, chunk_size: int, overlap: int) -> None:
    chunks: list[str] = []
    curr_chunk: list[str] = []
    for word in text.split():
        curr_chunk.append(word)
        if len(curr_chunk) >= chunk_size:
            chunks.append(" ".join(curr_chunk))
            curr_chunk = curr_chunk[-overlap:] if overlap > 0 else []
    if len(curr_chunk) > overlap:
        chunks.append(" ".join(curr_chunk))

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {''.join(chunk)}")


def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> None:
    chunks: list[str] = []
    curr_chunk: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        curr_chunk.append(sentence)
        if len(curr_chunk) >= max_chunk_size:
            chunks.append(" ".join(curr_chunk))
            curr_chunk = curr_chunk[-overlap:] if overlap > 0 else []
    if len(curr_chunk) > overlap:
        chunks.append(" ".join(curr_chunk))

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {''.join(chunk)}")

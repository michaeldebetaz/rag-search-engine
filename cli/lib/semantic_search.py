import json
import os
import numpy as np
from typing import final
from sentence_transformers import SentenceTransformer
from lib.search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    MOVIE_EMBEDDINGS_PATH,
    SCORE_PRECISION,
    ChunkMetadata,
    ChunkScore,
    ChunkedSemanticSearchResult,
    Movie,
    SemanticSearchResult,
    cosine_similarity,
    load_movies,
    save_chunk_metadata,
    semantic_chunk,
)


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model: SentenceTransformer = SentenceTransformer(
            model_name, device="cpu", token=os.getenv("HUGGING_FACE_ACCESS_TOKEN")
        )
        self.embeddings: np.ndarray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def build_embeddings(self, documents: list[Movie]) -> None:
        self.documents = documents
        texts: list[str] = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            texts.append(f"{doc['title']} {doc['description']}")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)

    def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in self.documents}

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

    def generate_embedding(self, text: str) -> np.ndarray:
        text = text.strip()
        if text == "":
            raise ValueError("Input text cannot be empty.")
        embedding = self.model.encode(text)
        return embedding

    def search(self, query: str, limit: int) -> list[SemanticSearchResult]:
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
            {
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"][:100],
                "score": round(score, SCORE_PRECISION),
            }
            for score, doc in scores[:limit]
        ]


@final
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> None:
        self.documents = documents
        chunks: list[str] = []
        self.chunk_metadata = []
        for doc in self.documents:
            if doc["description"] == "":
                continue
            desc_chunks = semantic_chunk(
                doc["description"], max_chunk_size=4, overlap=1
            )
            chunks.extend(desc_chunks)
            for chunk_idx in range(len(desc_chunks)):
                self.chunk_metadata.append(
                    {
                        "movie_idx": doc["id"],
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(desc_chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        save_chunk_metadata(CHUNK_METADATA_PATH, self.chunk_metadata, len(chunks))

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in self.documents}
        if CHUNK_EMBEDDINGS_PATH.exists() and CHUNK_METADATA_PATH.exists():
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
        else:
            self.build_chunk_embeddings(documents)
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("Failed to load or create chunk embeddings.")
        return self.chunk_embeddings

    def search_chunks(
        self, query: str, limit: int
    ) -> list[ChunkedSemanticSearchResult]:
        search = SemanticSearch()
        if self.documents is None:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )
        search.load_or_create_embeddings(self.documents)
        embed_query = search.generate_embedding(query)
        chunk_scores: list[ChunkScore] = []
        if self.chunk_embeddings is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )
        if self.chunk_metadata is None:
            raise ValueError(
                "No chunk metadata loaded. Call `load_or_create_chunk_embeddings` first."
            )
        for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(embed_query, chunk_embedding)
            metadata = self.chunk_metadata[chunk_idx] or {}
            chunk_scores.append(
                {
                    "chunk_idx": chunk_idx,
                    "movie_idx": self.chunk_metadata[chunk_idx]["movie_idx"],
                    "score": score,
                }
            )

        movie_scores: dict[int, ChunkScore] = {}
        for score in chunk_scores:
            movie_idx = score["movie_idx"]
            prev_score: float | None = None
            if movie_idx in movie_scores:
                prev_score = movie_scores[movie_idx]["score"]
            curr_score = score["score"]
            if prev_score is None or curr_score > prev_score:
                movie_scores[movie_idx] = score

        movie_scores_sorted = sorted(
            movie_scores.items(), key=lambda item: item[1]["score"], reverse=True
        )
        results: list[ChunkedSemanticSearchResult] = []
        for movie_idx, score in movie_scores_sorted[:limit]:
            metadata = self.chunk_metadata[score["chunk_idx"]] or {}
            doc = self.document_map[movie_idx]
            results.append(
                {
                    "id": metadata["movie_idx"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score["score"], SCORE_PRECISION),
                    "metadata": metadata,
                }
            )
        return results


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


def semantic_chunk_command(text: str, max_chunk_size: int, overlap: int) -> None:
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(semantic_chunk(text, max_chunk_size, overlap)):
        print(f"{i + 1}. {''.join(chunk)}")


def embed_chunks() -> None:
    search = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int) -> None:
    search = ChunkedSemanticSearch()
    documents = load_movies()
    search.load_or_create_chunk_embeddings(documents)
    results = search.search_chunks(query, limit)
    for i, result in enumerate(results):
        print(f"\n{i + 1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")

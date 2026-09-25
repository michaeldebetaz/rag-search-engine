import re
from dotenv import load_dotenv
import json
from pathlib import Path
import pickle
import string
from typing import Any, TypedDict
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
DOTENV_PATH = BASE_DIR / ".env"
MOVIES_PATH = BASE_DIR / "data" / "movies.json"
STOPWORDS_PATH = BASE_DIR / "data" / "stopwords.txt"
CACHE_DIR = BASE_DIR / "cache"
MOVIE_EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = CACHE_DIR / "chunk_metadata.json"

BM25_K1: float = 1.5
BM25_B: float = 0.75
SCORE_PRECISION: int = 4
LIMIT_MULTIPLIER: int = 500


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SemanticSearchResult(TypedDict):
    id: int
    score: float
    title: str
    description: str


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class ChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float


class ChunkedSemanticSearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: ChunkMetadata


class WeightedSearchResult(TypedDict):
    document: Movie
    bm25_score: float
    semantic_score: float
    hybrid_score: float


class RRFSearchResult(TypedDict):
    document: Movie
    bm25_rank: int
    semantic_rank: int
    rrf_score: float


def load_env() -> None:
    if not DOTENV_PATH.exists():
        raise FileNotFoundError(f"Environment file not found: {DOTENV_PATH}")
    if not load_dotenv(dotenv_path=DOTENV_PATH):
        raise RuntimeError(f"Failed to load environment variables from {DOTENV_PATH}")


def save_pickle(path: Path, data: dict[Any, Any]) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> dict[Any, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_movies() -> list[Movie]:
    if not MOVIES_PATH.exists():
        raise FileNotFoundError(f"Movies file not found: {MOVIES_PATH}")
    with open(MOVIES_PATH, "r") as f:
        return json.load(f)["movies"]


def load_stopwords() -> set[str]:
    if not STOPWORDS_PATH.exists():
        raise FileNotFoundError(f"Stopwords file not found: {STOPWORDS_PATH}")
    stopwords = STOPWORDS_PATH.read_text().splitlines()
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].strip().lower()
        for punc in string.punctuation:
            stopwords[i] = stopwords[i].replace(punc, "")
    return set(stopwords)


def save_chunk_metadata(
    path: Path, chunk_metadata: list[ChunkMetadata], total_chunks: int
) -> None:
    with open(path, "w") as f:
        json.dump(
            {"chunks": chunk_metadata, "total_chunks": total_chunks},
            f,
            indent=2,
        )


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if text == "":
        return []
    sentences: list[str] = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        return sentences

    chunks: list[str] = []
    curr_chunk: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if sentence == "":
            continue
        curr_chunk.append(sentence)
        if len(curr_chunk) >= max_chunk_size:
            chunks.append(" ".join(curr_chunk))
            curr_chunk = curr_chunk[-overlap:] if overlap > 0 else []
    if len(curr_chunk) > overlap:
        chunks.append(" ".join(curr_chunk))

    return chunks


def normalize(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_scrore(bm25_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int) -> float:
    return 1 / (k + rank)

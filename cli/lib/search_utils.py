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

BM25_K1: float = 1.5
BM25_B: float = 0.75


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


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


def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    np.save(path, embeddings)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

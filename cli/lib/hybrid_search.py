from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.search_utils import (
    LIMIT_MULTIPLIER,
    Movie,
    RRFSearchResult,
    WeightedSearchResult,
    hybrid_scrore,
    load_movies,
    normalize,
    rrf_score,
)


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents: list[Movie] = documents
        self.semantic_search: ChunkedSemanticSearch = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(self.documents)

        self.idx: InvertedIndex = InvertedIndex()
        if not self.idx.index_path.exists():
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[Movie, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def _bm25_normalized_scores(self, query: str, limit: int) -> dict[int, float]:
        bm25_results = self._bm25_search(query, limit * LIMIT_MULTIPLIER)
        bm25_normalized_scores = normalize([score for _, score in bm25_results])
        return {
            movie["id"]: bm25_normalized_scores[i]
            for i, (movie, _) in enumerate(bm25_results)
        }

    def _bm25_ranked(self, query: str, limit: int) -> dict[int, int]:
        bm25_results = self._bm25_search(query, limit * LIMIT_MULTIPLIER)
        _sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)
        return {movie["id"]: rank for rank, (movie, _) in enumerate(_sorted)}

    def _chunked_semantic_normalized_scores(
        self, query: str, limit: int
    ) -> dict[int, float]:
        results = self.semantic_search.search_chunks(query, limit * LIMIT_MULTIPLIER)
        normalized_scores = normalize([result["score"] for result in results])
        return {result["id"]: normalized_scores[i] for i, result in enumerate(results)}

    def _chunked_semantic_ranked(self, query: str, limit: int) -> dict[int, int]:
        results = self.semantic_search.search_chunks(query, limit * LIMIT_MULTIPLIER)
        _sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        return {result["id"]: rank for rank, result in enumerate(_sorted)}

    def weighted_search(
        self, query: str, alpha: float, limit: int
    ) -> list[WeightedSearchResult]:
        bm25_scores = self._bm25_normalized_scores(query, limit * LIMIT_MULTIPLIER)
        semantic_scores = self._chunked_semantic_normalized_scores(
            query, limit * LIMIT_MULTIPLIER
        )
        tmp: dict[int, WeightedSearchResult] = {}
        for doc_id, doc in self.semantic_search.document_map.items():
            bm25_score = bm25_scores[doc_id]
            semantic_score = semantic_scores[doc_id]
            tmp[doc_id] = {
                "document": doc,
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
                "hybrid_score": hybrid_scrore(bm25_score, semantic_score, alpha),
            }
        results: list[tuple[int, WeightedSearchResult]] = sorted(
            tmp.items(), key=lambda x: x[1]["hybrid_score"], reverse=True
        )
        return [result for _, result in results[:limit]]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[RRFSearchResult]:
        bm25_ranked = self._bm25_ranked(query, limit * LIMIT_MULTIPLIER)
        semantic_ranked = self._chunked_semantic_ranked(query, limit * LIMIT_MULTIPLIER)

        rrf_scores: dict[int, RRFSearchResult] = {}
        for doc_id, doc in self.semantic_search.document_map.items():
            bm25_rank = bm25_ranked[doc_id]
            bm25_rrf = rrf_score(bm25_rank, k)
            semantic_rank = semantic_ranked[doc_id]
            semantic_rrf = rrf_score(semantic_rank, k)
            rrf_scores[doc_id] = {
                "document": doc,
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
                "rrf_score": bm25_rrf + semantic_rrf,
            }
        return [
            result
            for _, result in sorted(
                rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
            )[:limit]
        ]


def normalize_command(scores: list[float]) -> None:
    for score in normalize(scores):
        print(f"* {score:.4f}")


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['document']['title']}")
        print(f"  Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"  BM25: {result['bm25_score']:.4f}")
        print(f"  {result['document']['description'][:100]}...")


def rrf_search_command(query: str, k: int, limit: int) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k, limit)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['document']['title']}")
        print(f"  RRF Score: {result['rrf_score']:.4f}")
        print(
            f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
        )
        print(f"  {result['document']['description'][:100]}...")

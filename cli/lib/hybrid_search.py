import time
from sentence_transformers import CrossEncoder

from lib.llm_utils import (
    correct_query,
    expand_query,
    rank_batch_rrf_results,
    rank_invdividual_rrf_result,
    rewrite_query,
)
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.search_utils import (
    LIMIT_MULTIPLIER,
    CrossEncoderRRFSearchResult,
    Movie,
    RRFSearchResult,
    RerankRRFSearchResult,
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


def enhance_query(query: str, method: str) -> str:
    enhanced_query: str = query
    match method:
        case "spell":
            enhanced_query = correct_query(query)
        case "rewrite":
            enhanced_query = rewrite_query(query)
        case "expand":
            enhanced_query = expand_query(query)
        case _:
            raise ValueError(f"Unknown enhancement method: {method}")
    if enhanced_query != query:
        print(f"Enhanced Query ({method}): '{query}' -> '{enhanced_query}'\n")
    return enhanced_query


def rerank_results(
    query: str, rrf_results: list[RRFSearchResult], method: str, limit: int
) -> None:
    match method:
        case "individual":
            rerank_results: list[RerankRRFSearchResult] = []
            for i, result in enumerate(rrf_results[:limit]):
                rerank_result = rank_invdividual_rrf_result(query, result)
                rerank_results.append(rerank_result)
                if i < len(rrf_results) - 1:
                    time.sleep(3)
            rerank_results = sorted(
                rerank_results, key=lambda x: x["rerank_score"], reverse=True
            )
            for i, result in enumerate(rerank_results):
                print(f"{i + 1}. {result['document']['title']}")
                print(f"  Re-rank Score: {result['rerank_score']}/10")
                print(f"  RRF Score: {result['rrf_score']:.4f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {result['document']['description'][:100]}...")

        case "batch":
            ranked_ids = rank_batch_rrf_results(query, rrf_results)
            id_to_result = {result["document"]["id"]: result for result in rrf_results}
            for i, doc_id in enumerate(ranked_ids[:limit]):
                result = id_to_result[doc_id]
                print(f"{i + 1}. {result['document']['title']}")
                print(f"  Re-rank Rank: {i + 1}")
                print(f"  RRF Score: {result['rrf_score']:.4f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {result['document']['description'][:100]}...")

        case "cross_encoder":
            pairs: list[tuple[str, str]] = []
            for result in rrf_results:
                doc = result["document"]
                doc_str = f"{doc['title']} - {doc['description']}"
                pairs.append((query, doc_str))

            cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu"
            )
            scores = cross_encoder.predict(pairs)
            cross_encoder_results: list[CrossEncoderRRFSearchResult] = [
                {
                    "document": result["document"],
                    "cross_encoder_score": score,
                    "rrf_score": result["rrf_score"],
                    "bm25_rank": result["bm25_rank"],
                    "semantic_rank": result["semantic_rank"],
                }
                for result, score in zip(rrf_results, scores)
            ]
            cross_encoder_results = sorted(
                cross_encoder_results,
                key=lambda x: x["cross_encoder_score"],
                reverse=True,
            )
            for i, result in enumerate(cross_encoder_results[:limit]):
                print(f"{i + 1}. {result['document']['title']}")
                print(f"  Cross Encoder Score: {result['cross_encoder_score']:.4f}")
                print(f"  RRF Score: {result['rrf_score']:.4f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {result['document']['description'][:100]}...")

        case _:
            raise ValueError(f"Unknown rerank method: {rerank_method}")


def rrf_search_command(
    query: str,
    k: int,
    limit: int,
    enhance_method: str | None,
    rerank_method: str | None,
) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    if enhance_method is not None:
        query = enhance_query(query, enhance_method)

    limit_multiplier: int = 1
    if rerank_method is not None:
        limit_multiplier = 5

    rrf_results = hybrid_search.rrf_search(query, k, limit * limit_multiplier)
    if rerank_method is not None:
        rerank_results(query, rrf_results, rerank_method, limit)
    else:
        for i, result in enumerate(rrf_results):
            print(f"{i + 1}. {result['document']['title']}")
            print(f"  RRF Score: {result['rrf_score']:.4f}")
            print(
                f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
            )
            print(f"  {result['document']['description'][:100]}...")

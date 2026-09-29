import json

from .hybrid_search import HybridSearch
from .search_utils import GOLDEN_DATASET_PATH, EvaluationResult, TestCase, load_movies


def evaluation_command(k: int, limit: int) -> None:
    documents = load_movies()
    search = HybridSearch(documents)

    results: list[EvaluationResult] = []
    for test_case in load_test_cases():
        query = test_case["query"]
        rrf_results = search.rrf_search(query, k, limit)
        retrieved_titles = [result["document"]["title"] for result in rrf_results]
        score = 0
        relevant_titles = test_case["relevant_docs"]
        for title in retrieved_titles:
            if title in relevant_titles:
                score += 1
        precision = score / len(retrieved_titles)
        results.append(
            {
                "query": query,
                "precision_at_k": precision,
                "retrieved_titles": retrieved_titles,
                "relevant_titles": relevant_titles,
            }
        )

    for result in results:
        print(f"- Query: {result['query']}")
        print(f"  - Precision@{limit}: {result['precision_at_k']:.4f}")
        print(f"  - Retrieved: {', '.join(result['retrieved_titles'])}")
        print(f"  - Relevant: {', '.join(result['relevant_titles'])}")


def load_test_cases() -> list[TestCase]:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)["test_cases"]

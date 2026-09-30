import json

from .hybrid_search import HybridSearch
from .search_utils import (
    GOLDEN_DATASET_PATH,
    EvaluationResult,
    TestCase,
    load_movies,
)


def evaluation_command(k: int, limit: int) -> None:
    documents = load_movies()
    search = HybridSearch(documents)

    results: list[EvaluationResult] = []
    for test_case in load_test_cases():
        query = test_case["query"]
        rrf_results = search.rrf_search(query, k, limit)
        retrieved_titles = [result["document"]["title"] for result in rrf_results]
        precision_at_k = precision(retrieved_titles, test_case)
        recall_at_k = recall(retrieved_titles, test_case)
        f1_score = f1(precision_at_k, recall_at_k)
        results.append(
            {
                "query": query,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "f1_score": f1_score,
                "retrieved_titles": retrieved_titles,
                "relevant_titles": test_case["relevant_docs"],
            }
        )

    for result in results:
        print(f"- Query: {result['query']}")
        print(f"  - Precision@{limit}: {result['precision_at_k']:.4f}")
        print(f"  - Recall@{limit}: {result['recall_at_k']:.4f}")
        print(f"  - F1 Score: {result['f1_score']:.4f}")
        print(f"  - Retrieved: {', '.join(result['retrieved_titles'])}")
        print(f"  - Relevant: {', '.join(result['relevant_titles'])}")


def precision(retrieved_titles: list[str], test_case: TestCase) -> float:
    if len(retrieved_titles) == 0:
        return 0.0
    relevant_titles = test_case["relevant_docs"]
    relevant_retrieved = sum(
        1 for title in retrieved_titles if title in relevant_titles
    )
    return relevant_retrieved / len(retrieved_titles)


def recall(retrieved_titles: list[str], test_case: TestCase) -> float:
    relevant_titles = test_case["relevant_docs"]
    total_relevant = len(relevant_titles)
    if total_relevant == 0:
        return 0.0
    relevant_retrieved = sum(
        1 for title in retrieved_titles if title in relevant_titles
    )
    return relevant_retrieved / total_relevant


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def load_test_cases() -> list[TestCase]:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)["test_cases"]

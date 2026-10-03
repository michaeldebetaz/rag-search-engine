from .llm_utils import (
    answer_question,
    answer_with_citations,
    rag_query,
    summarize_results,
)
from .hybrid_search import HybridSearch, format_rrf_results
from .search_utils import RRFSearchResult, load_movies


def rag_command(query: str, k: int, limit: int) -> None:
    documents = load_movies()
    search = HybridSearch(documents)
    rrf_results = search.rrf_search(query, k, limit)
    formatted_rrf_results = format_rrf_results(rrf_results)
    answer = rag_query(query, formatted_rrf_results)

    print_search_results(rrf_results)
    print("\nRAG Response:")
    print(answer)

    pass


def summarize_command(query: str, k: int, limit: int) -> None:
    documents = load_movies()
    search = HybridSearch(documents)
    rrf_results = search.rrf_search(query, k, limit)
    formatted_rrf_results = format_rrf_results(rrf_results)
    summary = summarize_results(query, formatted_rrf_results)

    print_search_results(rrf_results)
    print("\nLMM Summary:")
    print(summary)


def citations_command(query: str, k: int, limit: int) -> None:
    documents = load_movies()
    search = HybridSearch(documents)
    rrf_results = search.rrf_search(query, k, limit=limit)
    formatted_rrf_results = format_rrf_results(rrf_results)
    answer = answer_with_citations(query, formatted_rrf_results)
    print_search_results(rrf_results)
    print("\nLLM Answer:")
    print(answer)


def question_command(query: str, k: int, limit: int) -> None:
    documents = load_movies()
    search = HybridSearch(documents)
    rrf_results = search.rrf_search(query, k, limit=limit)
    formatted_rrf_results = format_rrf_results(rrf_results)
    answer = answer_question(query, formatted_rrf_results)
    print_search_results(rrf_results)
    print("\nAnswer:")
    print(answer)


def print_search_results(rrf_results: list[RRFSearchResult]) -> None:
    print("Search Results:")
    for result in rrf_results:
        print(f"- {result['document']['title']}")

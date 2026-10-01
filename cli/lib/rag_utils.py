from .llm_utils import rag_query, summarize_results
from .hybrid_search import HybridSearch, format_rrf_results
from .search_utils import load_movies


def rag_command(query: str, k: int, limit: int) -> None:
    documents = load_movies()
    search = HybridSearch(documents)
    rrf_results = search.rrf_search(query, k, limit)
    formatted_rrf_results = format_rrf_results(rrf_results)
    answer = rag_query(query, formatted_rrf_results)

    print("Search Results:")
    for result in rrf_results:
        print(f"- {result['document']['title']}")
    print("\nRAG Response:")
    print(answer)

    pass


def summarize_command(query: str, k: int, limit: int) -> None:
    documents = load_movies()
    search = HybridSearch(documents)
    rrf_results = search.rrf_search(query, k, limit)
    formatted_rrf_results = format_rrf_results(rrf_results)
    summary = summarize_results(query, formatted_rrf_results)

    print("Search Results:")
    for result in rrf_results:
        print(f"- {result['document']['title']}")
    print("\nLMM Summary:")
    print(summary)

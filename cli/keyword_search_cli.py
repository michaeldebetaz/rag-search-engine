import argparse
from lib.search_utils import BM25_B, BM25_K1
from lib.keyword_search import (
    bm25idf_command,
    bm25search_command,
    bm25tf_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index")

    # Search
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    # Term frequencie
    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency of a term in a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    # IDF
    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency of a term"
    )
    idf_parser.add_argument("term", type=str, help="Term to search for")

    # TF-IDF
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score of a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")

    # BM25-IDF
    bm25idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25-IDF score of a term in a document"
    )
    bm25idf_parser.add_argument("term", type=str, help="Term to search for")

    # BM25-TF
    bm25tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25-TF score of a term in a document"
    )
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Term to search for")
    bm25tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    # BM25 Search
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Limit the number of results"
    )
    bm25search_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25search_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25idf_command(args.term)
        case "bm25tf":
            bm25tf_command(args.doc_id, args.term, args.k1, args.b)
        case "bm25search":
            bm25search_command(
                args.query,
                args.limit,
                args.k1,
                args.b,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

import argparse

from lib.hybrid_search import (
    normalize_command,
    rrf_search_command,
    weighted_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="*", type=float, help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform a weighted hybrid search"
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="Query string for the search"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 score (default: 0.5)",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default: 5)"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform a Reciprocal Rank Fusion (RRF) hybrid search"
    )
    rrf_search_parser.add_argument(
        "query", type=str, help="Query string for the search"
    )
    rrf_search_parser.add_argument(
        "-k", type=int, default=60, help="RRF parameter k (default: 60)"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

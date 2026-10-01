import argparse

from lib.rag_utils import rag_command, summarize_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="Number of top documents to retrieve (default: 60)",
    )
    rag_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to return (default: 5)",
    )

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize multi-documents search results"
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarize_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="Number of top documents to retrieve (default: 60)",
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to summarize (default: 5)",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query, args.k, args.limit)
        case "summarize":
            summarize_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

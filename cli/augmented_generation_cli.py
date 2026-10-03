import argparse

from lib.rag_utils import (
    citations_command,
    question_command,
    rag_command,
    summarize_command,
)


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

    citations_parser = subparsers.add_parser(
        "citations", help="Answer a query with citations from the retrieved documents"
    )
    citations_parser.add_argument(
        "query", type=str, help="Query for answering with citations"
    )
    citations_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="Number of top documents to retrieve (default: 60)",
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to return (default: 5)",
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question based on the retrieved documents"
    )
    question_parser.add_argument("query", type=str, help="Question to answer")
    question_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="Number of top documents to retrieve (default: 60)",
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to return (default: 5)",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query, args.k, args.limit)
        case "summarize":
            summarize_command(args.query, args.k, args.limit)
        case "citations":
            citations_command(args.query, args.k, args.limit)
        case "question":
            question_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

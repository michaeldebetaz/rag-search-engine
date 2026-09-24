import argparse
from lib.search_utils import load_env
from lib.semantic_search import (
    chunk,
    embed_query_text,
    embed_text,
    search,
    semantic_chunk,
    verify_embeddings,
    verify_model,
)

load_env()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model is loaded correctly")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for a given text"
    )
    embed_text_parser.add_argument(
        "text", type=str, help="Text to generate embedding for"
    )

    _ = subparsers.add_parser(
        "verify_embeddings", help="Verify that embeddings are loaded correctly"
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate embedding for a given query"
    )
    embed_query_parser.add_argument(
        "query", type=str, help="Query text to generate embedding for"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search for a query in the movie dataset"
    )
    search_parser.add_argument(
        "query", type=str, help="Query text to search for in the movie dataset"
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit the number of search results (default: 5)",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk a given text into smaller pieces"
    )
    chunk_parser.add_argument(
        "text", type=str, help="Text to chunk into smaller pieces"
    )
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Size of each chunk in characters (default: 200)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping words between chunks (default: 0)",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk a given text into semantically meaningful pieces"
    )
    semantic_chunk_parser.add_argument(
        "text", type=str, help="Text to chunk into semantically meaningful pieces"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences in each chunk (default: 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping sentences between chunks (default: 0)",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search(args.query, limit=args.limit)
        case "chunk":
            chunk(args.text, chunk_size=args.chunk_size, overlap=args.overlap)
        case "semantic_chunk":
            semantic_chunk(
                args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
